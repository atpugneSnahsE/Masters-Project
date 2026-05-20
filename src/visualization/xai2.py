import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import carla
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import os
import gc
import shap

# Create output directory
os.makedirs("xai_output", exist_ok=True)

# ====================== MODEL WRAPPER FOR SHAP ======================
class ShapModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        probs = torch.sigmoid(out)
        return probs.sum(dim=(1, 2, 3)).unsqueeze(1) 

# ====================== MODELS & NORMALIZATION ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization remains for SHAP clarity
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 512)),
    T.ToTensor(),
    T.Normalize(mean=norm_mean, std=norm_std)
])

model_gpu = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
model_gpu.load_state_dict(torch.load("lane_model.pth", map_location=device))
model_gpu.eval()

model_cpu_base = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to("cpu")
model_cpu_base.load_state_dict(torch.load("lane_model.pth", map_location="cpu"))
model_cpu_base.eval()

model_shap = ShapModelWrapper(model_cpu_base)
background_cpu = torch.zeros((1, 3, 256, 512)).to("cpu")
explainer = shap.GradientExplainer(model_shap, background_cpu)

# ====================== DYNAMIC BOUNDARY LOGIC ======================
def extract_strong_ego_lane(prob_vis, h, w):
    pH, pW = prob_vis.shape
    roi_top = int(pH * 0.45) # Lower ROI to capture more horizon
    mid = pW // 2

    # DYNAMIC THRESHOLD:
    # Near the car (bottom), we want high confidence (0.35)
    # At the horizon (top), we accept lower confidence (0.12) to see further
    y_range = pH - roi_top
    thresholds = np.linspace(0.12, 0.35, y_range)
    threshold_map = np.tile(thresholds.reshape(-1, 1), (1, pW))

    binary = np.zeros_like(prob_vis, dtype=np.uint8)
    roi_probs = prob_vis[roi_top:]
    binary[roi_top:] = (roi_probs > threshold_map).astype(np.uint8) * 255

    # Larger closing kernel to bridge gaps in far-away detection
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
    
    left_xs, left_ys, right_xs, right_ys = [], [], [], []
    n_bands = 25 # More bands for smoother lines
    band_h = max(4, (pH - roi_top) // n_bands)

    for b in range(n_bands):
        y0 = roi_top + b * band_h
        y1 = min(pH, y0 + band_h + 4)
        band = binary[y0:y1, :]
        if band.sum() < 15: continue

        profile = band.sum(axis=0).astype(np.float32)
        profile_s = cv2.GaussianBlur(profile.reshape(1,-1), (1,15), 0).reshape(-1)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(profile_s, height=profile_s.max()*0.25, distance=12)

        y_c = (y0 + y1) / 2
        l_c = peaks[peaks < mid - 15]
        r_c = peaks[peaks > mid + 15]

        if len(l_c) > 0:
            left_xs.append(float(l_c[-1])); left_ys.append(y_c)
        if len(r_c) > 0:
            right_xs.append(float(r_c[0])); right_ys.append(y_c)

    ego_small = np.zeros((pH, pW), dtype=np.uint8)
    ys_full = np.arange(roi_top, pH, dtype=np.float32)
    
    try:
        if len(left_ys) >= 5 and len(right_ys) >= 5:
            # Fit curves and draw a filled polygon for the lane
            lx = np.polyval(np.polyfit(left_ys, left_xs, 2), ys_full)
            rx = np.polyval(np.polyfit(right_ys, right_xs, 2), ys_full)
            for i, y in enumerate(ys_full.astype(int)):
                cv2.line(ego_small, (int(lx[i]), y), (int(rx[i]), y), 255, thickness=3)
        else:
            ego_small = binary.copy()
    except:
        ego_small = binary.copy()

    return cv2.resize(ego_small, (w, h), cv2.INTER_LINEAR)

# ====================== XAI FIGURE GENERATION ======================
def generate_xai_figure(rgb_image, frame_id):
    save_path = f"xai_output/xai_frame_{frame_id:06d}.png"
    h, w = rgb_image.shape[:2]

    # Pre-process for SHAP
    inp_cpu = transform(rgb_image).unsqueeze(0).to("cpu")
    
    # IMPROVED FOCUS: Dim the sky instead of blacking it out
    # This helps SHAP understand context without being distracted by it
    inp_cpu_focused = inp_cpu.clone()
    inp_cpu_focused[:, :, :int(256 * 0.45), :] *= 0.1 
    
    shap_values = explainer.shap_values(inp_cpu_focused, nsamples=20) 
    attr = shap_values[0] if isinstance(shap_values, list) else shap_values

    # Clean SHAP Map
    shap_map = np.abs(attr).mean(axis=1).squeeze()
    shap_map[shap_map < (shap_map.max() * 0.18)] = 0 
    shap_map = cv2.resize(shap_map, (w, h), cv2.INTER_LINEAR)

    # Visualization
    inp_gpu = transform(rgb_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_vis = cv2.resize(torch.sigmoid(model_gpu(inp_gpu)).cpu().squeeze().numpy(), (w, h), cv2.INTER_LINEAR)
    
    ego_lane = extract_strong_ego_lane(prob_vis, h, w)
    
    fig, axs = plt.subplots(1, 4, figsize=(26, 6.5))
    axs[0].imshow(rgb_image); axs[0].set_title("Input RGB"); axs[0].axis('off')
    
    # Overlay Lane
    overlay = rgb_image.copy()
    overlay[ego_lane > 0] = [0, 220, 80]
    axs[1].imshow(cv2.addWeighted(rgb_image, 0.7, overlay, 0.3, 0))
    axs[1].set_title("Balanced Drivable Area"); axs[1].axis('off')
    
    # SHAP
    axs[2].imshow(rgb_image)
    axs[2].imshow(shap_map, cmap='jet', alpha=0.6)
    axs[2].set_title("SHAP (Road Focused)"); axs[2].axis('off')

    # Trust Score
    trust_map = cv2.GaussianBlur((ego_lane > 0).astype(np.float32), (41, 41), 0)
    im = axs[3].imshow(trust_map, cmap='RdYlGn', vmin=0, vmax=1)
    axs[3].set_title("Model Confidence"); axs[3].axis('off')
    plt.colorbar(im, ax=axs[3])

    plt.suptitle(f"Frame {frame_id} - Dynamic Threshold Analysis", fontsize=16)
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close('all')
    gc.collect()
    torch.cuda.empty_cache()

# ====================== CARLA EXECUTION ======================
client = carla.Client("localhost", 2000)
world = client.load_world("Town04")
vehicle = world.try_spawn_actor(world.get_blueprint_library().filter("vehicle.audi.a2")[0], world.get_map().get_spawn_points()[0])
vehicle.set_autopilot(True)

cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1024"); cam_bp.set_attribute("image_size_y", "512")
camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.35, z=1.25), carla.Rotation(pitch=-5)), attach_to=vehicle)

frame_count = 0
latest = None

def process(image):
    global frame_count, latest
    frame_count += 1
    raw_rgb = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]

    inp = transform(raw_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = cv2.resize(torch.sigmoid(model_gpu(inp)).cpu().squeeze().numpy(), (1024, 512), cv2.INTER_LINEAR)
    
    ego_lane = extract_strong_ego_lane(prob, 512, 1024)

    if frame_count % 150 == 0:
        generate_xai_figure(raw_rgb.copy(), frame_count)

    latest = (raw_rgb, ego_lane)

camera.listen(process)

try:
    while True:
        if latest:
            rgb, mask = latest
            overlay = rgb.copy(); overlay[mask > 0] = [0, 255, 120]
            cv2.imshow("CARLA: Balanced XAI View", overlay)
        if cv2.waitKey(1) == 27: break
finally:
    camera.stop(); vehicle.destroy(); cv2.destroyAllWindows()