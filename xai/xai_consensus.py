"""
Real-time-oriented variant for Jetson Xavier AGX deployment.

Design principle: LIME and KernelSHAP are fundamentally sample-hungry (hundreds
of sequential model calls per frame) and cannot be made real-time on any
hardware, let alone an embedded Jetson. This file drops them from the live
path and replaces them with a single batched occlusion pass, which gives a
similar "what region matters" signal at a small fraction of the cost.

Use THIS file for the deployed/live loop. Use xai_lane_consensus.py (the
LIME+SHAP+Grad-CAM version) as an offline/periodic audit tool run on a
workstation, not per-frame on the device.
"""
import os
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms


# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
GRADCAM_LAYER_NAME = "layer3"
OCCLUSION_GRID = (4, 6)     # (rows, cols) -> 24 blocks total
OCCLUSION_SCALE = 0.5      # run occlusion at this fraction of full resolution, then upsample
OCCLUSION_CHUNK = 8        # process the block batch in chunks of this size (bounds peak memory)
ROAD_KEEP_PERCENTILE = 60

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)


# ----------------------------------------------------------------------------
# Model (identical architecture to the validated checkpoint: smp Unet/resnet34)
# ----------------------------------------------------------------------------
class LaneDetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                             in_channels=3, classes=1)
        self.encoder = self.net.encoder

    def forward(self, x):
        out = self.net(x)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out


def load_checkpoint(model, model_path):
    if not os.path.exists(model_path):
        print(f"[warn] checkpoint not found at {model_path} -> random weights")
        return
    state = torch.load(model_path, map_location='cpu')
    missing, unexpected = model.net.load_state_dict(state, strict=False)
    n = len(model.net.state_dict())
    print(f"[checkpoint] loaded {n - len(missing)}/{n} parameter tensors")
    if len(missing) > n * 0.3:
        print("[warn] checkpoint does not match this architecture")


def normalize01(x):
    x = x.astype(np.float32)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def scalar_score(model, batch_tensor):
    out = model(batch_tensor.to(DEVICE))
    return out.view(batch_tensor.shape[0], -1).mean(dim=1).cpu()


# ----------------------------------------------------------------------------
# Grad-CAM -- unchanged cost profile: 1 forward + 1 backward pass, real-time safe
# ----------------------------------------------------------------------------
def compute_gradcam(model, img_tensor, layer_name=GRADCAM_LAYER_NAME):
    target_layer = getattr(model.encoder, layer_name)
    activations, gradients = {}, {}

    def fwd_hook(module, inp, out):
        activations['value'] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    img_tensor = img_tensor.clone().to(DEVICE).requires_grad_(True)
    out = model(img_tensor)
    out.mean().backward()
    h1.remove()
    h2.remove()

    acts = activations['value'].detach()[0]
    grads = gradients['value'].detach()[0]
    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * acts).sum(0)).cpu().numpy()
    cam = cv2.resize(cam, (img_tensor.shape[-1], img_tensor.shape[-2]))
    return normalize01(cam)


# ----------------------------------------------------------------------------
# Fast batched occlusion sensitivity -- replaces LIME + KernelSHAP for live use.
# ONE batched forward call covering the whole grid, instead of hundreds of
# sequential calls. This is the key latency fix for Jetson deployment.
# ----------------------------------------------------------------------------
def compute_fast_occlusion(model, img_tensor, grid=OCCLUSION_GRID, fill_value=0.5,
                            scale=OCCLUSION_SCALE, chunk=OCCLUSION_CHUNK):
    full_h, full_w = img_tensor.shape[-2:]
    small_h, small_w = int(full_h * scale), int(full_w * scale)
    small_tensor = F.interpolate(img_tensor, size=(small_h, small_w), mode='bilinear',
                                  align_corners=False)

    gh, gw = grid
    bh, bw = small_h // gh, small_w // gw

    with torch.no_grad():
        # baseline (unoccluded) image goes in as element 0 of the SAME batch --
        # saves a separate forward call vs computing it up front.
        occluded_batch = [small_tensor[0]]
        for i in range(gh):
            for j in range(gw):
                occ = small_tensor.clone()[0]
                occ[:, i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] = fill_value
                occluded_batch.append(occ)

        # GPU: one shot (batch is tiny, memory is not the constraint, kernel-launch
        # overhead is). CPU: stay chunked to bound peak memory.
        effective_chunk = len(occluded_batch) if DEVICE.type in ("cuda", "mps") else chunk

        scores = []
        for start in range(0, len(occluded_batch), effective_chunk):
            batch_t = torch.stack(occluded_batch[start:start + effective_chunk])
            scores.append(scalar_score(model, batch_t).numpy())
        scores = np.concatenate(scores)

    baseline_score, block_scores = scores[0], scores[1:]
    importance = baseline_score - block_scores  # drop in score when a block is occluded
    imp_map = np.zeros((small_h, small_w), dtype=np.float32)
    idx = 0
    for i in range(gh):
        for j in range(gw):
            imp_map[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] = importance[idx]
            idx += 1

    imp_map = cv2.resize(imp_map, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    return normalize01(np.clip(imp_map, 0, None))


# ----------------------------------------------------------------------------
# Fast per-frame consensus: Grad-CAM + occlusion. 2 methods, ~25 total forward
# passes (1 + 1 + 24), all real-time-safe on an accelerated Jetson pipeline
# (see TensorRT notes below for the remaining raw-inference speedup).
# ----------------------------------------------------------------------------
def explain_frame_fast(model, frame_rgb, road_mask=None):
    img_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0)

    gradcam_map = compute_gradcam(model, img_tensor)
    occlusion_map = compute_fast_occlusion(model, img_tensor)
    consensus_map = (gradcam_map + occlusion_map) / 2.0

    result = {"gradcam": gradcam_map, "occlusion": occlusion_map, "consensus": consensus_map}

    if road_mask is not None:
        road_vals = consensus_map[road_mask == 1]
        if road_vals.size > 0:
            threshold_value = np.percentile(road_vals, ROAD_KEEP_PERCENTILE)
            binary_mask = ((consensus_map >= threshold_value) & (road_mask == 1)).astype(np.uint8)
        else:
            binary_mask = np.zeros_like(consensus_map, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        voted_mask = cv2.morphologyEx(binary_mask * 255, cv2.MORPH_CLOSE, kernel)
        voted_mask = cv2.dilate(voted_mask, kernel, iterations=2)
        inter = np.logical_and(voted_mask > 0, road_mask > 0)
        union = np.logical_or(voted_mask > 0, road_mask > 0)
        score = np.sum(inter) / (np.sum(union) + 1e-8)
        result["voted_mask"] = voted_mask
        result["agreement_score"] = score

    return result


if __name__ == "__main__":
    print(f"[diagnostics] DEVICE={DEVICE}")
    print(f"[diagnostics] torch.get_num_threads()={torch.get_num_threads()}")
    print(f"[diagnostics] mps available={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
    print(f"[diagnostics] cuda available={torch.cuda.is_available()}")

    model = LaneDetectionModel().eval().to(DEVICE)
    load_checkpoint(model, "/home/vgtu/Masters-Project/models/vil100_model/best_model.pth")

    img_path = "/home/vgtu/Masters-Project/datasets/VIL100/JPEGImages/1269_Road022_Trim002_frames/00297.jpg"
    bgr = cv2.imread(img_path)
    frame = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (854, 480))

    road_mask = np.zeros((480, 854), dtype=np.uint8)
    cv2.fillPoly(road_mask,
                 [np.array([[42, 345], [375, 240], [478, 240], [811, 345]], dtype=np.int32)], 1)

    img_tensor = transforms.ToTensor()(frame).unsqueeze(0)

    # Warm-up (first call includes lazy CUDA/MPS kernel compilation -- exclude from timing)
    explain_frame_fast(model, frame, road_mask)

    # Per-stage breakdown so we know WHERE time is going, not just the total
    t0 = time.time()
    for _ in range(5):
        _ = compute_gradcam(model, img_tensor)
    gradcam_ms = (time.time() - t0) / 5 * 1000

    t0 = time.time()
    for _ in range(5):
        _ = compute_fast_occlusion(model, img_tensor)
    occlusion_ms = (time.time() - t0) / 5 * 1000

    t0 = time.time()
    for _ in range(5):
        with torch.no_grad():
            _ = scalar_score(model, img_tensor)
    single_forward_ms = (time.time() - t0) / 5 * 1000

    print(f"[timing] single forward pass:  {single_forward_ms:.1f} ms")
    print(f"[timing] Grad-CAM (1 fwd+bwd): {gradcam_ms:.1f} ms")
    print(f"[timing] Occlusion (24 fwd, chunked): {occlusion_ms:.1f} ms")

    n_runs = 10
    t0 = time.time()
    for _ in range(n_runs):
        result = explain_frame_fast(model, frame, road_mask)
    elapsed = (time.time() - t0) / n_runs
    print(f"[timing] TOTAL: {elapsed*1000:.1f} ms/frame -> ~{1/elapsed:.1f} FPS")
    print(f"[result] agreement score: {result['agreement_score']:.4f}")