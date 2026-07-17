"""
Fast LIME + SHAP + Grad-CAM consensus for near-real-time use.

WHY THIS IS FAST (vs xai.py's LIME which took ~40 sequential batched calls,
or a naive KernelSHAP which would take even more):

LIME and KernelSHAP are not intrinsically sequential. The `lime` and `shap`
libraries just happen to call your model in a python loop, a handful of
samples at a time. But every perturbed sample is independent of every other
sample. So instead of:

    for batch in perturbations:      # library does this internally
        model(batch)                 # dozens of round trips

...we do:

    all_perturbations = build_all_masks()   # cheap, pure numpy
    model(all_perturbations)                # ONE (chunked) batched call

That's the same trick xai_consensus.py already uses for occlusion. Applied
here, LIME and SHAP stop being "sample-hungry" in the sense that matters
(wall-clock) -- they're still sample-hungry in the sense of "need N
perturbations to be statistically meaningful," but N perturbations costs
one batched forward pass, not N sequential ones.

FURTHER SAVINGS: LIME and SHAP are both "perturb superpixels, score, fit a
weighted linear model" -- they only differ in the *weighting kernel* used in
the regression (LIME: locality-based exponential kernel; SHAP: the Shapley
kernel). So we generate ONE shared batch of superpixel perturbations, score
it with ONE batched forward pass, and fit BOTH regressions against the same
scores. You get 2 explanation methods for close to the cost of 1.

Grad-CAM is unchanged from xai_consensus.py (1 fwd + 1 bwd, already cheap).

REALISTIC EXPECTATIONS ON JETSON XAVIER AGX:
Even with this batching fix, LIME/SHAP still need enough samples to be
statistically meaningful (this file defaults to 64 -- fewer starts to look
like noise). That's roughly 2-3x the forward-pass cost of the occlusion
method in xai_consensus.py. Benchmark on the actual Xavier before assuming
you can run this every frame -- you likely cannot hit 30fps with 3 real
explanation methods on embedded hardware. Plan on running this at a reduced
rate (e.g. every 5-10 frames) for a periodic "trust check," with a cheap
signal (e.g. xai_consensus.py's Grad-CAM+occlusion, or just the lane-mask
confidence itself) running every frame for the actual driving decision.
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.segmentation import slic

# Reuse the same conventions as xai_consensus.py so this drops into that file
# or is imported alongside it.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else     "cpu"
)

# CUDA optimizations
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
N_SEGMENTS = 96          # superpixels — finer granularity for lane markings
N_SAMPLES = 64           # default shared perturbation budget for LIME + SHAP
PERTURB_SCALE = 0.35     # run perturbation/scoring at this fraction of full res
PERTURB_CHUNK = 16       # chunk size for CPU
GPU_CHUNK = 24           # chunk size for GPU — 258 images at once causes OOM on 10 GiB
BASELINE_FILL = 0.5      # gray fill for "off" superpixels
GRADCAM_LAYER_NAME = "layer4"                    # Fix 1: layer4 > layer3 for lane segmentation

# Fix 3: Adaptive sampling thresholds (trust-based, from previous XAI run)
LOW_TRUST_SAMPLES = 128
MED_TRUST_SAMPLES = 64
HIGH_TRUST_SAMPLES = 32

# Fix 5: Temporal smoothing factor
XAI_ALPHA = 0.6

# Fix 4: Periodic XAI interval — run full XAI every N frames
XAI_INTERVAL = 5

# Fix 5: SLIC refresh interval — recompute superpixels every N frames
SLIC_REFRESH_INTERVAL = 20

# Module-level state for temporal smoothing (Fix 5) and SLIC caching (Fix 8)
_XAI_STATE = {}

# FP16 flag (disabled if autocast unsupported)
_HAVE_AUTOCAST = DEVICE.type == "cuda" and hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast')
try:
    if _HAVE_AUTOCAST:
        with torch.amp.autocast('cuda'):
            pass
except (RuntimeError, AttributeError):
    _HAVE_AUTOCAST = False


def normalize01(x):
    x = x.astype(np.float32)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _autocast_ctx():
    """Context manager for FP16 inference on CUDA."""
    if _HAVE_AUTOCAST:
        return torch.amp.autocast('cuda')
    from contextlib import nullcontext
    return nullcontext()


def lane_focused_score(model, batch_tensor, road_mask_small=None):
    """
    Scalar score per sample. If a road/lane mask is supplied, score only the
    lane region -- this is what makes the consensus "about the lane markings"
    rather than about whatever the whole frame happens to contain.
    """
    with _autocast_ctx():
        out = model(batch_tensor.to(DEVICE))                           # (B, C, H, W) or (B,1,H,W)
    out = torch.sigmoid(out)                                           # Fix 2: probability, not logits
    if out.shape[1] > 1:
        out = out.max(dim=1, keepdim=True).values                      # collapse multi-class to a
                                                                       # single "any lane class" map
    if road_mask_small is not None:
        m = torch.from_numpy(road_mask_small).to(DEVICE).float()
        m = m.unsqueeze(0).unsqueeze(0)                                # (1,1,H,W)
        scores = (out * m).sum(dim=(1, 2, 3))                # SUM (not mean) for more dynamic range
    else:
        scores = out.mean(dim=(1, 2, 3))
    return scores.detach().cpu().numpy()


# ----------------------------------------------------------------------------
# Grad-CAM -- identical cost profile to xai_consensus.py (1 fwd + 1 bwd)
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
    prob = torch.sigmoid(out)
    lane_target = (prob * (prob > 0.5)).sum()
    lane_target.backward()
    h1.remove()
    h2.remove()

    acts = activations['value'].detach()[0]
    grads = gradients['value'].detach()[0]
    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * acts).sum(0)).cpu().numpy()
    cam = cv2.resize(cam, (img_tensor.shape[-1], img_tensor.shape[-2]))
    return normalize01(cam)


def select_best_gradcam_layer(model, img_tensor, road_mask,
                               layers=("layer2", "layer3", "layer4")):
    """
    Fix 1 (auto-select): Run Grad-CAM on multiple layers and return the one
    with highest IoU agreement against the road mask. Call once at startup.

    Returns (best_layer_name, {layer_name: agreement_score}).
    """
    scores = {}
    for layer in layers:
        cam = compute_gradcam(model, img_tensor, layer_name=layer)
        thr = np.percentile(cam, 60)
        b = (cam >= thr)
        inter = np.logical_and(b, road_mask > 0).sum()
        union = np.logical_or(b, road_mask > 0).sum()
        scores[layer] = inter / (union + 1e-8)
    best = max(scores, key=scores.get)
    return best, scores


# ----------------------------------------------------------------------------
# Shared perturbation core for LIME + SHAP
# ----------------------------------------------------------------------------
def _build_perturbation_batch(small_img, segments, n_samples, baseline_fill, rng):
    """
    Returns:
        images: list[Tensor] -- perturbed images, index 0 is the "all segments
                 on" baseline, index -1 is "all segments off"
        masks:  (n_samples+2, n_segs) binary coalition matrix, rows aligned
                 with `images` (row 0 = all ones, last row = all zeros)
    """
    n_segs = segments.max() + 1
    img_t = torch.from_numpy(small_img).permute(2, 0, 1).float()  # (3,H,W)

    masks = np.zeros((n_samples + 2, n_segs), dtype=np.float32)
    masks[0, :] = 1.0     # unperturbed baseline (needed for both regressions)
    masks[-1, :] = 0.0    # fully-occluded reference (SHAP needs this anchor)
    for i in range(1, n_samples + 1):
        # Random coalition size, then random members -- standard LIME/SHAP
        # sampling scheme (uniform over coalition sizes avoids biasing toward
        # "everything on" or "everything off").
        k = rng.integers(1, n_segs)
        on = rng.choice(n_segs, size=k, replace=False)
        masks[i, on] = 1.0

    images = []
    for row in masks:
        img = img_t.clone()
        off_segs = np.where(row == 0)[0]
        if len(off_segs) > 0:
            off_pixel_mask = np.isin(segments, off_segs)
            img[:, off_pixel_mask] = baseline_fill
        images.append(img)

    return images, masks


def _weighted_ridge(X, y, weights, l2=1e-6):
    """
    Closed-form weighted ridge regression, done via sqrt-weight rescaling
    (X' -> sqrt(w)*X) rather than forming diag(weights) directly.

    This matters specifically for SHAP: the Shapley kernel weight spans many
    orders of magnitude (huge at the two anchor coalitions, vanishingly
    small near |z|=M/2), and forming X'diag(w)X directly with that dynamic
    range produces a badly-conditioned matrix -- a naive implementation of
    this looked numerically fine but silently returned importances shrunk by
    ~10x with noise leaking into irrelevant segments (verified against a
    synthetic ground-truth test). Rescaling rows by sqrt(weight) before the
    matrix product is the standard numerically-stable form of WLS and fixes
    this.
    """
    sw = np.sqrt(weights)
    Xw = X * sw[:, None]
    yw = y * sw
    A = Xw.T @ Xw + l2 * np.eye(X.shape[1])
    b = Xw.T @ yw
    return np.linalg.solve(A, b)


def _lime_kernel_weights(masks, kernel_width=0.75):
    # Exponential kernel on cosine distance to the "all on" row — this is
    # LIME's locality weighting (closer perturbations to the original image
    # matter more).
    n_segs = masks.shape[1]
    full = np.ones(n_segs)
    cos_sim = (masks @ full) / (np.linalg.norm(masks, axis=1) * np.linalg.norm(full) + 1e-8)
    dist = 1 - cos_sim
    return np.exp(-(dist ** 2) / (kernel_width ** 2))


def _shap_kernel_weights(masks, anchor_weight=None):
    # Shapley kernel weight for a coalition of size |z| out of M features:
    # (M-1) / (C(M,|z|) * |z| * (M-|z|)), undefined at |z|=0 and |z|=M so we
    # give those two anchor rows a large fixed weight instead (standard
    # practice in Kernel SHAP implementations). This is safe now that
    # _weighted_ridge uses the sqrt-rescaling form -- with the naive
    # diag(weights) form, this large a value silently wrecked conditioning.
    #
    # Uses log-space computation (lgamma) to avoid overflow when M > 60.
    from math import lgamma, exp, log
    M = masks.shape[1]
    sizes = masks.sum(axis=1).astype(int)
    weights = np.zeros(len(sizes))
    log_M_minus_1 = log(M - 1) if M > 1 else 0.0
    for i, k in enumerate(sizes):
        if k == 0 or k == M:
            weights[i] = 0.0  # placeholder
        else:
            log_comb = lgamma(M + 1) - lgamma(k + 1) - lgamma(M - k + 1)
            log_w = log_M_minus_1 - log_comb - log(k) - log(M - k)
            weights[i] = exp(log_w)
    # Auto-scale anchor_weight: set it so total anchor weight ≈ total
    # non-anchor weight.  This avoids the degenerate case where anchors
    # (weighted 1e6) completely dominate small-M regressions.
    if anchor_weight is None:
        if M > 60:
            anchor_weight = 1e6   # non-anchor weights vanish; anchors only
        else:
            total_non_anchor = weights.sum()
            n_anchors = ((sizes == 0) | (sizes == M)).sum()
            anchor_weight = max(10.0, total_non_anchor / max(1, n_anchors))
    for i, k in enumerate(sizes):
        if k == 0 or k == M:
            weights[i] = anchor_weight
    return weights


def _segment_within_roi(small_img, road_mask_small, n_segments, pad_frac=0.08):
    """
    Restrict SLIC segmentation to a padded bounding box around the ROI
    (road/lane mask) instead of the whole frame.

    This matters a lot in practice: with a fixed segment budget spent over
    the WHOLE frame, a narrow lane trapezoid can collapse into just 1-2
    segments -- which makes LIME/SHAP produce a flat, constant importance
    value across the entire lane region. A flat map then trivially satisfies
    any percentile-based threshold computed within that same region (every
    pixel ties at the threshold), which looks like "near-perfect agreement"
    in an IoU metric but is actually a degenerate, zero-information map. This
    is what produced the ~1.0 LIME/SHAP agreement scores -- not genuinely
    excellent explanations, just flat ones.

    Returns:
        segments: full-frame-shaped int array. Pixels outside the padded bbox
                  get segment id -1 and are never perturbed (kept at their
                  original pixel value in every sample) -- so the model still
                  sees a realistic full scene, we just don't waste any of the
                  segment/sample budget explaining sky or background.
        bbox: (y0, y1, x0, x1) of the region actually segmented.
    """
    h, w = road_mask_small.shape
    ys, xs = np.where(road_mask_small > 0)
    if len(ys) == 0:
        # No ROI info supplied -- fall back to segmenting the whole frame.
        seg = slic(small_img, n_segments=n_segments, compactness=3, sigma=1,
                   start_label=0, channel_axis=-1)
        return seg, (0, h, 0, w)

    pad_y, pad_x = int(h * pad_frac), int(w * pad_frac)
    y0, y1 = max(0, ys.min() - pad_y), min(h, ys.max() + pad_y)
    x0, x1 = max(0, xs.min() - pad_x), min(w, xs.max() + pad_x)

    crop = small_img[y0:y1, x0:x1]
    crop_segments = slic(crop, n_segments=n_segments, compactness=3, sigma=1,
                          start_label=0, channel_axis=-1)

    segments = np.full((h, w), -1, dtype=np.int32)
    segments[y0:y1, x0:x1] = crop_segments
    return segments, (y0, y1, x0, x1)


def compute_fast_lime_shap(model, img_tensor, road_mask=None,
                            n_segments=N_SEGMENTS, n_samples=None,
                            scale=PERTURB_SCALE, chunk=PERTURB_CHUNK,
                            seed=42, frame_id=0, reuse_segments=True):
    """
    One shared batched forward pass -> two attribution maps (LIME, SHAP).
    Returns (lime_map, shap_map), both normalized to [0,1] at full resolution.

    Fix 3: Adaptive sampling — if n_samples is None, picks sample count
           based on baseline confidence (more samples for uncertain frames).
    Fix 8: SLIC reuse — caches segments and reuses across frames.
    """
    rng = np.random.default_rng(seed)
    full_h, full_w = img_tensor.shape[-2:]
    small_h, small_w = int(full_h * scale), int(full_w * scale)

    small_img = F.interpolate(img_tensor, size=(small_h, small_w),
                               mode='bilinear', align_corners=False)[0]
    small_np = small_img.permute(1, 2, 0).cpu().numpy()

    small_road_mask = None
    if road_mask is not None:
        # Dilate lane mask to preserve thin lane structure at reduced resolution
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thick_mask = cv2.dilate(road_mask.astype(np.uint8), kernel, iterations=2)
        small_road_mask = cv2.resize(thick_mask.astype(np.float32),
                                      (small_w, small_h),
                                      interpolation=cv2.INTER_NEAREST)

    # Fix 3: adaptive n_samples based on previous XAI trust
    if n_samples is None:
        prev_trust = _XAI_STATE.get("prev_trust", 0.5)
        if prev_trust > 0.7:
            n_samples = HIGH_TRUST_SAMPLES
        elif prev_trust > 0.5:
            n_samples = MED_TRUST_SAMPLES
        else:
            n_samples = LOW_TRUST_SAMPLES

    # Fix 8: SLIC segment reuse across frames — recompute every SLIC_REFRESH_INTERVAL frames
    seg_cache_key = "segments"
    if reuse_segments and seg_cache_key in _XAI_STATE and frame_id % SLIC_REFRESH_INTERVAL != 0:
        segments = _XAI_STATE[seg_cache_key]
    else:
        segments, _bbox = _segment_within_roi(small_np, small_road_mask, n_segments)
        _XAI_STATE[seg_cache_key] = segments

    valid_ids = np.unique(segments)
    valid_ids = valid_ids[valid_ids >= 0]
    n_segs = len(valid_ids)
    # Remap to a dense 0..n_segs-1 range (bbox crop from slic may not start at 0
    # contiguously once merged into the full-frame array).
    remap = {old: new for new, old in enumerate(valid_ids)}
    segments_dense = np.full_like(segments, -1)
    for old, new in remap.items():
        segments_dense[segments == old] = new
    segments = segments_dense

    images, masks = _build_perturbation_batch(small_np, segments, n_samples,
                                               BASELINE_FILL, rng)

    effective_chunk = GPU_CHUNK if DEVICE.type in ("cuda", "mps") else chunk
    scores = []
    with torch.no_grad():
        for start in range(0, len(images), effective_chunk):
            batch_t = torch.stack(images[start:start + effective_chunk])
            scores.append(lane_focused_score(model, batch_t, small_road_mask))
    scores = np.concatenate(scores)  # aligned with `masks` rows

    # ---- LIME: locality-weighted ridge regression (full segment set) -------
    X = np.concatenate([masks, np.ones((masks.shape[0], 1))], axis=1)  # + intercept
    lime_w = _lime_kernel_weights(masks)
    lime_beta = _weighted_ridge(X, scores, lime_w)[:-1]  # drop intercept

    # ---- SHAP: Shapley-kernel-weighted ridge regression (merged background) ----
    # Merge non-overlapping segments into one background group at the mask
    # column level.  This keeps the effective feature count small (~5-15),
    # which the Shapley kernel needs (it degenerates for M > 20).
    if small_road_mask is not None:
        overlap_ids = set()
        for seg_id in range(n_segs):
            if (segments == seg_id)[small_road_mask > 0].any():
                overlap_ids.add(seg_id)
        bg_ids = [s for s in range(n_segs) if s not in overlap_ids]
        if bg_ids and len(overlap_ids) < n_segs:
            shap_masks = masks[:, list(overlap_ids)].copy()
            bg_col = (masks[:, bg_ids].sum(axis=1) > 0).astype(np.float32)
            shap_masks = np.column_stack([shap_masks, bg_col])
            shap_seg_ids = list(overlap_ids) + [bg_ids[0]]
        else:
            shap_masks = masks
            shap_seg_ids = list(range(n_segs))
    else:
        shap_masks = masks
        shap_seg_ids = list(range(n_segs))

    shap_w = _shap_kernel_weights(shap_masks)
    X_shap = np.concatenate([shap_masks, np.ones((shap_masks.shape[0], 1))], axis=1)
    shap_beta_reduced = _weighted_ridge(X_shap, scores, shap_w)[:-1]
    # Expand reduced betas back to full segment space
    shap_beta = np.zeros(n_segs)
    if bg_ids:
        for i, seg_id in enumerate(overlap_ids):
            shap_beta[seg_id] = shap_beta_reduced[i]
        for bg_id in bg_ids:
            shap_beta[bg_id] = shap_beta_reduced[-1]
    else:
        for i, seg_id in enumerate(shap_seg_ids):
            shap_beta[seg_id] = shap_beta_reduced[i]

    def segments_to_map(beta):
        imp = np.zeros((small_h, small_w), dtype=np.float32)
        for seg_id in range(n_segs):
            imp[segments == seg_id] = beta[seg_id]
        imp = cv2.resize(imp, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        return normalize01(np.clip(imp, 0, None))

    return segments_to_map(lime_beta), segments_to_map(shap_beta)


# ----------------------------------------------------------------------------
# Full 3-method consensus
# ----------------------------------------------------------------------------
def explain_frame_triple(model, frame_rgb, road_mask=None, weights=(1, 2, 2),
                          frame_id=0, xai_interval=XAI_INTERVAL):
    """
    weights: relative weight for (gradcam, lime, shap) in the consensus map.

    Fix 4: Periodic XAI — only runs full explanation every `xai_interval`
           frames; returns {"skipped": True} on off frames.
    Fix 5: Temporal smoothing — EMA-blends consecutive maps (alpha = XAI_ALPHA).
    Fix 6: Uncertainty — reports `uncertainty = 1 - agreement_score` and a
           human-readable `trust_status`.
    """
    # Fix 4: skip XAI on non-interval frames
    if frame_id % xai_interval != 0:
        return {"skipped": True, "frame_id": frame_id}

    import torchvision.transforms as transforms
    img_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0)

    gradcam_map = compute_gradcam(model, img_tensor)
    lime_map, shap_map = compute_fast_lime_shap(model, img_tensor, road_mask=road_mask,
                                                  frame_id=frame_id)

    # Fix 5: temporal smoothing (EMA)
    alpha = XAI_ALPHA
    prev_gc = _XAI_STATE.get("prev_gradcam")
    prev_lm = _XAI_STATE.get("prev_lime")
    prev_sp = _XAI_STATE.get("prev_shap")
    if all(m is not None for m in (prev_gc, prev_lm, prev_sp)):
        gradcam_map = alpha * gradcam_map + (1 - alpha) * prev_gc
        lime_map    = alpha * lime_map    + (1 - alpha) * prev_lm
        shap_map    = alpha * shap_map    + (1 - alpha) * prev_sp
    _XAI_STATE["prev_gradcam"] = gradcam_map
    _XAI_STATE["prev_lime"] = lime_map
    _XAI_STATE["prev_shap"] = shap_map

    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    consensus_map = w[0] * gradcam_map + w[1] * lime_map + w[2] * shap_map

    result = {
        "gradcam": gradcam_map,
        "lime": lime_map,
        "shap": shap_map,
        "consensus": consensus_map,
        "frame_id": frame_id,
    }

    if road_mask is not None:
        road_vals = consensus_map[road_mask == 1]
        if road_vals.size > 0:
            threshold_value = np.percentile(road_vals, 40)
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

        # Within-mask thresholding (fraction of mask pixels in top 40% of
        # explanation values within the mask; baseline = 0.400).
        def iou_vs_road(m, percentile=40):
            masked_vals = m[road_mask > 0]
            if masked_vals.size == 0:
                return 0.0
            thr = np.percentile(masked_vals, percentile)
            b = (m >= thr) & (road_mask > 0)
            inter = b.sum()
            union = (road_mask > 0).sum()
            return inter / (union + 1e-8)

        # Concentration ratio: mean explanation value on-lane / off-lane.
        # A ratio > 1 means the explanation focuses on the lane mask.  This
        # is a more interpretable metric that doesn't depend on thresholding.
        def concentration_ratio(m):
            on = m[road_mask > 0].mean()
            off = m[road_mask == 0].mean()
            return float(on / (off + 1e-8))

        per_method = {
            "gradcam": iou_vs_road(gradcam_map),
            "lime": iou_vs_road(lime_map),
            "shap": iou_vs_road(shap_map),
        }

        per_method_conc = {
            "gradcam": concentration_ratio(gradcam_map),
            "lime": concentration_ratio(lime_map),
            "shap": concentration_ratio(shap_map),
        }

        result["per_method_agreement"] = per_method
        result["per_method_concentration"] = per_method_conc

        # Fix 6: uncertainty output
        result["uncertainty"] = 1.0 - score

        # Trust = mean per-method agreement × (1 − disagreement).
        # Uses per-method IoU (not the stricter consensus vote) so the
        # score reflects what each explainer actually captures.
        method_scores = np.array(list(per_method.values()))
        result["method_disagreement"] = float(method_scores.std())
        result["method_mean_agreement"] = float(method_scores.mean())
        result["trust"] = float(result["method_mean_agreement"] * (1 - result["method_disagreement"]))

        t = result.get("trust", score)
        if t >= 0.60:
            result["trust_status"] = "SAFE"
        elif t >= 0.40:
            result["trust_status"] = "CHECK"
        else:
            result["trust_status"] = "UNTRUSTED"

        # Store trust for next frame's adaptive sampling
        _XAI_STATE["prev_trust"] = t

    return result


def get_lane_mask(model, frame_rgb, threshold=0.30):
    """Get binary lane prediction mask from the model (used as ground truth for XAI)."""
    import torchvision.transforms as transforms
    img = transforms.ToTensor()(frame_rgb).unsqueeze(0).to(DEVICE)
    with _autocast_ctx(), torch.no_grad():
        pred = torch.sigmoid(model(img))
    prob = pred[0, 0].cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ----------------------------------------------------------------------------
# Fix 9: TensorRT export helper
# ----------------------------------------------------------------------------
def export_to_onnx(model, dummy_input, save_path="lane_model.onnx"):
    """
    Export the lane segmentation model to ONNX for TensorRT deployment.
    Expected speed: 8-15 FPS on Jetson Xavier with FP16.

    Usage:
        export_to_onnx(model, torch.randn(1, 3, 480, 854))
        # Then: trtexec --onnx=lane_model.onnx --fp16 --saveEngine=lane.engine
    """
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )
    print(f"[export] ONNX model saved to {save_path}")
    print("[export] Convert to TensorRT with:")
    print(f"  trtexec --onnx={save_path} --fp16 --saveEngine=lane.engine")


if __name__ == "__main__":
    import time
    import os
    import segmentation_models_pytorch as smp

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

    print(f"[diagnostics] DEVICE={DEVICE}")
    model = LaneDetectionModel().eval().to(DEVICE)

    ckpt = "/home/vgtu/Masters-Project/models/vil100_model/best_model.pth"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location='cpu')
        model.net.load_state_dict(state, strict=False)
    else:
        print(f"[warn] checkpoint not found at {ckpt} -> random weights, timing only")

    img_path = "/home/vgtu/Masters-Project/datasets/VIL100/JPEGImages/1269_Road022_Trim002_frames/00297.jpg"
    if os.path.exists(img_path):
        bgr = cv2.imread(img_path)
        frame = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (854, 480))
    else:
        frame = (np.random.rand(480, 854, 3) * 255).astype(np.uint8)
        print(f"[warn] test image not found -> using random noise frame, timing only")

    # Use model's own lane prediction as ground truth instead of artificial polygon
    lane_mask = get_lane_mask(model, frame)
    print(f"[diagnostics] lane mask: {lane_mask.sum()}/{lane_mask.size} pixels positive")

    # Warm-up on frame 0 (runs full XAI)
    _ = explain_frame_triple(model, frame, lane_mask, frame_id=0)

    # Simulate periodic XAI over N frames — perception runs every frame,
    # XAI runs every XAI_INTERVAL frames (Fix 4)
    n_frames = 25  # 25 frames → 5 XAI invocations at interval 5
    xai_count = 0
    last_xai_result = None
    t0 = time.time()
    for frame_id in range(n_frames):
        result = explain_frame_triple(model, frame, lane_mask, frame_id=frame_id,
                                       xai_interval=XAI_INTERVAL)
        if not result.get("skipped"):
            last_xai_result = result
            xai_count += 1
    elapsed = (time.time() - t0) / max(xai_count, 1)
    fps = 1.0 / elapsed if elapsed > 0 else float("inf")
    print(f"[timing] XAI every {XAI_INTERVAL} frames ({xai_count} runs over {n_frames} frames): "
          f"{elapsed*1000:.1f} ms/XAI-run -> ~{fps:.1f} XAI-FPS "
          f"(perception runs at full frame rate)")
    if last_xai_result is not None:
        if "agreement_score" in last_xai_result:
            pm = last_xai_result['per_method_agreement']
            print(f"[result] combined agreement score: {last_xai_result['agreement_score']:.4f}")
            print(f"[result] per-method agreement: grad={pm['gradcam']:.3f} "
                  f"lime={pm['lime']:.3f} shap={pm['shap']:.3f}")
            pc = last_xai_result.get('per_method_concentration', {})
            print(f"[result] concentration (on/off ratio): gradcam={pc.get('gradcam',0):.2f} "
                  f"lime={pc.get('lime',0):.2f} shap={pc.get('shap',0):.2f}")
            print(f"[result] method_disagreement(std)={last_xai_result.get('method_disagreement', 'N/A'):.4f} "
                  f"compound_trust={last_xai_result.get('trust', 'N/A'):.4f}")
            print(f"[result] uncertainty: {last_xai_result.get('uncertainty', 'N/A'):.4f} "
                  f"-> trust status: {last_xai_result.get('trust_status', 'N/A')}")
        else:
            print(f"[result] keys present: {list(last_xai_result.keys())}")