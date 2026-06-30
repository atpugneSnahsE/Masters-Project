import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
import shap
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from tqdm import tqdm

# -------------------------------------------------------------
# 1. SETUP & MODEL LOADING
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIL100_ROOT = "/home/vgtu/VIL100"
MODEL_WEIGHTS_PATH = "/home/vgtu/vil100_model/best_model.pth"

def load_real_vil100_frame():
    image_dir = os.path.join(VIL100_ROOT, "JPEGImages")
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(root, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return cv2.resize(img, (640, 368))
    raise FileNotFoundError("Could not find any frames.")

image_np = load_real_vil100_frame()

# Load Model
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, decoder_attention="scse", in_channels=3, classes=1)
if os.path.exists(MODEL_WEIGHTS_PATH):
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model.to(device).eval()

def preprocess_frame(img_np):
    tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0).to(device)

# Get base target mask
with torch.no_grad():
    base_out = torch.sigmoid(model(preprocess_frame(image_np)).squeeze(1)).squeeze().cpu().numpy()
base_mask = (base_out > 0.5).astype(np.float32)

# -------------------------------------------------------------
# 2. SHAP & LIME UNIFIED PREDICTION LOGIC
# -------------------------------------------------------------
# Create superpixels once to ensure both explainers evaluate identical regions
d_fixed = 50
superpixels = slic(image_np, n_segments=d_fixed, compactness=10, sigma=1, start_label=0)
num_sp = len(np.unique(superpixels))

def segment_predict_fn(masks):
    """
    Expects binary mask array of shape (M, num_superpixels) indicating 
    which superpixels are active. Returns target concept probabilities.
    """
    model.eval()
    scores = []
    background = np.zeros_like(image_np)
    
    with torch.no_grad():
        for mask in masks:
            # Map superpixel binary toggles back to full image space
            temp_img = background.copy()
            active_sp_indices = np.where(mask == 1)[0]
            img_mask = np.isin(superpixels, active_sp_indices)
            temp_img[img_mask] = image_np[img_mask]
            
            # Pass to model
            out = torch.sigmoid(model(preprocess_frame(temp_img)).squeeze(1)).squeeze().cpu().numpy()
            if np.sum(base_mask > 0.5) > 0:
                score = np.mean(out[base_mask > 0.5])
            else:
                score = np.mean(out)
            scores.append(score)
            
    return np.array(scores)

# -------------------------------------------------------------
# 3. RUN KERNEL SHAP EXPERIMENT
# -------------------------------------------------------------
print("\n--- Running KernelSHAP Optimization ---")
# Background reference dataset: all superpixels deactivated (masked out)
background_ref = np.zeros((1, num_sp))
explainer = shap.KernelExplainer(segment_predict_fn, background_ref)

# Compute Shapley values (using N=500 permutations matching LIME benchmark)
N_samples = 500
shap_values = explainer.shap_values(np.ones((1, num_sp)), nsamples=N_samples)[0]
base_value = explainer.expected_value

# --- VERIFY LOCAL ACCURACY PROPERTY ---
actual_f_x = segment_predict_fn(np.ones((1, num_sp)))[0]
sum_shapley = np.sum(shap_values) + base_value
discrepancy = abs(sum_shapley - actual_f_x)

print("\n" + "="*45)
print("  SHAP LOCAL ACCURACY VERIFICATION")
print("="*45)
print(f"  Base Value (phi_0)        : {base_value:.5f}")
print(f"  Sum of Shapley (Sum phi_k): {np.sum(shap_values):.5f}")
print(f"  Theoretical Total Output  : {sum_shapley:.5f}")
print(f"  Actual Model Output f(x)  : {actual_f_x:.5f}")
print(f"  Absolute Discrepancy     : {discrepancy:.2e}")
print("="*45)

# -------------------------------------------------------------
# 4. RUN COMPARATIVE LIME PIPELINE
# -------------------------------------------------------------
print("\n--- Running LIME for Comparative Benchmark ---")
from lime import lime_image
lime_explainer = lime_image.LimeImageExplainer()

def lime_predict_wrapper(images):
    # Map LIME's perturbed images back to our targeted function
    scores = []
    with torch.no_grad():
        for img in images:
            out = torch.sigmoid(model(preprocess_frame(img)).squeeze(1)).squeeze().cpu().numpy()
            if np.sum(base_mask > 0.5) > 0:
                score = np.mean(out[base_mask > 0.5])
            else:
                score = np.mean(out)
            scores.append([1.0 - score, score])
    return np.array(scores)

lime_exp = lime_explainer.explain_instance(
    image_np, lime_predict_wrapper, top_labels=1, num_samples=N_samples,
    segmentation_fn=lambda x: superpixels
)
lime_weights = np.zeros(num_sp)
for sp_id, w in lime_exp.local_exp[1]:
    lime_weights[sp_id] = w

# -------------------------------------------------------------
# 5. COMPUTE INSERTION/DELETION FAITHFULNESS (FIXED)
# -------------------------------------------------------------
def get_faithfulness_curves(weights, steps=10):
    sorted_sp = np.argsort(weights)[::-1]
    sp_per_step = max(1, len(sorted_sp) // steps)
    
    del_scores = []
    ins_scores = []
    
    # Wrap entire execution loop in no_grad to prevent graph accumulation
    with torch.no_grad():
        # --- Deletion ---
        img_del = image_np.copy()
        for i in range(0, len(sorted_sp), sp_per_step):
            img_del[np.isin(superpixels, sorted_sp[:i])] = 0
            out = torch.sigmoid(model(preprocess_frame(img_del)).squeeze(1)).squeeze().cpu().numpy()
            del_scores.append(np.mean(out[base_mask > 0.5]) if np.sum(base_mask > 0.5) > 0 else np.mean(out))
            
        # --- Insertion ---
        img_ins = np.zeros_like(image_np)
        for i in range(0, len(sorted_sp), sp_per_step):
            img_ins[np.isin(superpixels, sorted_sp[:i])] = image_np[np.isin(superpixels, sorted_sp[:i])]
            out = torch.sigmoid(model(preprocess_frame(img_ins)).squeeze(1)).squeeze().cpu().numpy()
            ins_scores.append(np.mean(out[base_mask > 0.5]) if np.sum(base_mask > 0.5) > 0 else np.mean(out))
        
    return del_scores, ins_scores

shap_del, shap_ins = get_faithfulness_curves(shap_values)
lime_del, lime_ins = get_faithfulness_curves(lime_weights)

# -------------------------------------------------------------
# 6. VISUALIZE COMPARATIVE ATTRIBUTIONS (Figures)
# -------------------------------------------------------------
# Figure 1: Side-by-Side Attribution Overlays
fig_vis, axes_vis = plt.subplots(1, 3, figsize=(18, 5))
axes_vis[0].imshow(image_np)
axes_vis[0].set_title("Input Frame & Base Mask")
axes_vis[0].contour(base_mask, colors='cyan', linewidths=1.5)
axes_vis[0].axis('off')

# LIME Overlay
_, l_mask = lime_exp.get_image_and_mask(1, positive_only=False, num_features=8, hide_rest=False)
axes_vis[1].imshow(mark_boundaries(image_np, l_mask, color=(1, 0, 0)))
axes_vis[1].set_title("LIME Attribution Overlay")
axes_vis[1].axis('off')

# SHAP Overlay (top features)
shap_mask = np.isin(superpixels, np.argsort(shap_values)[::-1][:8])
axes_vis[2].imshow(mark_boundaries(image_np, shap_mask, color=(0, 1, 0)))
axes_vis[2].set_title("KernelSHAP Attribution Overlay")
axes_vis[2].axis('off')
plt.tight_layout()
plt.show()

# Figure 2: Shared Axis Faithfulness Curves
plt.figure(figsize=(10, 6))
steps_axis = np.arange(len(shap_del))
plt.plot(steps_axis, lime_del, label="LIME Deletion (Faithfulness ↓)", linestyle="--", marker="x", color="blue")
plt.plot(steps_axis, lime_ins, label="LIME Insertion (Faithfulness ↑)", linestyle="-", marker="x", color="blue")
plt.plot(steps_axis, shap_del, label="SHAP Deletion (Faithfulness ↓)", linestyle="--", marker="o", color="green")
plt.plot(steps_axis, shap_ins, label="SHAP Insertion (Faithfulness ↑)", linestyle="-", marker="o", color="green")

plt.title("Faithfulness Benchmarking: LIME vs KernelSHAP")
plt.xlabel("Features Modulated (Steps)")
plt.ylabel("Target Performance Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()