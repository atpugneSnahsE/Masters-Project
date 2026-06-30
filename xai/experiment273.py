import os
import json
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_image
import shap
import segmentation_models_pytorch as smp
from skimage.segmentation import slic
from tqdm import tqdm

# -------------------------------------------------------------
# 1. INITIALIZATION & CONFIGURATION
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIL100_ROOT = "/home/vgtu/VIL100"
MODEL_WEIGHTS_PATH = "/home/vgtu/vil100_model/best_model.pth"

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

# -------------------------------------------------------------
# 2. DATASET INFERENCE & CONDITION CLASSIFICATION
# -------------------------------------------------------------
def collect_scenarios(dataset_root, max_samples_per_cond=15):
    """
    Parses VIL100 Json annotations safely to sort frames into 'Normal/Clear' 
    vs 'Challenging' (faded, occluded, shadow, or night scene conditions).
    """
    json_root = os.path.join(dataset_root, "Json")
    img_root = os.path.join(dataset_root, "JPEGImages")
    
    clear_pool = []
    difficult_pool = []
    
    for root, _, files in os.walk(json_root):
        for file in files:
            if not file.endswith('.json'):
                continue
            json_path = os.path.join(root, file)
            
            with open(json_path, 'r') as f:
                meta = json.load(f)
            
            condition_tags = meta.get("info", {}).get("attribute", ["normal"])
            
            # Type-safe parsing of string vs integer metadata attribute tags
            is_difficult = False
            for tag in condition_tags:
                if isinstance(tag, str):
                    if tag.lower() in ["faded", "occlusion", "shadow", "night", "crowd"]:
                        is_difficult = True
                        break
                    elif tag.isdigit() and int(tag) > 1:
                        is_difficult = True
                        break
                elif isinstance(tag, (int, float)) and tag > 1:
                    is_difficult = True
                    break
            
            # Reconstruct pairing image path
            sub_path = os.path.relpath(json_path, json_root).replace(".json", "")
            img_path = os.path.join(img_root, sub_path)
            
            # Handle possible alternate subfolder layouts or extensions (.jpg / .png)
            if not os.path.exists(img_path):
                for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
                    if os.path.exists(img_path + ext):
                        img_path = img_path + ext
                        break
            
            if os.path.exists(img_path):
                record = {"img_path": img_path, "json_path": json_path}
                if is_difficult:
                    difficult_pool.append(record)
                else:
                    clear_pool.append(record)
                    
            if len(clear_pool) >= max_samples_per_cond and len(difficult_pool) >= max_samples_per_cond:
                break
                
    return clear_pool[:max_samples_per_cond], difficult_pool[:max_samples_per_cond]

print("Parsing VIL100 metadata attributes...")
clear_batch, difficult_batch = collect_scenarios(VIL100_ROOT)
print(f"Collected {len(clear_batch)} Clear frames and {len(difficult_batch)} Difficult (faded/occluded) frames.")

# -------------------------------------------------------------
# 3. DUAL EXPLAINER EXECUTION PIPELINE
# -------------------------------------------------------------
def compute_explainers_attribution(image_np, d=50, N=300):
    superpixels = slic(image_np, n_segments=d, compactness=10, sigma=1, start_label=0)
    num_sp = len(np.unique(superpixels))
    
    # Get baseline model prediction mask
    with torch.no_grad():
        base_out = torch.sigmoid(model(preprocess_frame(image_np)).squeeze(1)).squeeze().cpu().numpy()
    base_mask = (base_out > 0.5).astype(np.float32)
    
    # Define Concept Predictor Function
    def concept_predict_fn(masks):
        scores = []
        background = np.zeros_like(image_np)
        with torch.no_grad():
            for m in masks:
                temp_img = background.copy()
                temp_img[np.isin(superpixels, np.where(m == 1)[0])] = image_np[np.isin(superpixels, np.where(m == 1)[0])]
                out = torch.sigmoid(model(preprocess_frame(temp_img)).squeeze(1)).squeeze().cpu().numpy()
                scores.append(np.mean(out[base_mask > 0.5]) if np.sum(base_mask > 0.5) > 0 else np.mean(out))
        return np.array(scores)

    # 1. Calculate KernelSHAP Values
    explainer_shap = shap.KernelExplainer(concept_predict_fn, np.zeros((1, num_sp)))
    shap_vals = explainer_shap.shap_values(np.ones((1, num_sp)), nsamples=N, silent=True)[0]
    
    # 2. Calculate LIME Values
    lime_explainer = lime_image.LimeImageExplainer()
    def lime_wrapper(images):
        scores = []
        with torch.no_grad():
            for img in images:
                out = torch.sigmoid(model(preprocess_frame(img)).squeeze(1)).squeeze().cpu().numpy()
                score = np.mean(out[base_mask > 0.5]) if np.sum(base_mask > 0.5) > 0 else np.mean(out)
                scores.append([1.0 - score, score])
        return np.array(scores)
        
    lime_exp = lime_explainer.explain_instance(image_np, lime_wrapper, top_labels=1, num_samples=N, segmentation_fn=lambda x: superpixels)
    lime_vals = np.zeros(num_sp)
    for sp_id, w in lime_exp.local_exp[1]:
        lime_vals[sp_id] = w
        
    # Map back to full pixel space and normalize values to [0, 1] for scale alignment
    shap_map = np.zeros(image_np.shape[:2])
    lime_map = np.zeros(image_np.shape[:2])
    for i in range(num_sp):
        shap_map[superpixels == i] = shap_vals[i]
        lime_map[superpixels == i] = lime_vals[i]
        
    def normalize_map(m):
        denom = (np.max(m) - np.min(m) + 1e-8)
        return (m - np.min(m)) / denom

    return normalize_map(lime_map), normalize_map(shap_map)

# -------------------------------------------------------------
# 4. METRIC COMPUTATION ENGINE (c(i,j), e(i,j), A_R)
# -------------------------------------------------------------
def evaluate_batch_consistency(batch_list):
    a_r_scores = []
    sample_maps = None
    
    for idx, rec in enumerate(tqdm(batch_list, desc="Processing Consistency")):
        img = cv2.imread(rec["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 368))
        
        # Extract normalized maps
        w_lime, w_shap = compute_explainers_attribution(img)
        
        # Calculate pixel metrics
        c_ij = 1.0 - np.abs(w_lime - w_shap)
        e_ij = np.abs(w_lime - w_shap)
        
        # Global aggregate consistency A_R per frame
        a_r = np.mean(c_ij)
        a_r_scores.append(a_r)
        
        if idx == 0:
            sample_maps = (img, c_ij, e_ij)
            
    return a_r_scores, sample_maps

# Execute
print("\n--- Computing Consistency Over Clear Driving Scenes ---")
clear_ar, clear_sample = evaluate_batch_consistency(clear_batch)

print("\n--- Computing Consistency Over Faded / Occluded Scenes ---")
diff_ar, diff_sample = evaluate_batch_consistency(difficult_batch)

# -------------------------------------------------------------
# 5. GENERATE TARGET GRAPH VISUALS
# -------------------------------------------------------------
# FIGURE A: Pixel Maps c(i,j) and e(i,j)
if clear_sample is not None:
    img_f, c_map, e_map = clear_sample
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(img_f)
    axes[0].set_title("Input Sample Evaluation Frame")
    axes[0].axis('off')

    im1 = axes[1].imshow(c_map, cmap='viridis')
    axes[1].set_title("Consistency Map $c(i,j)$")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(e_map, cmap='magma')
    axes[2].set_title("Inconsistency Map $e(i,j)$")
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# FIGURE B: Boxplot distribution of A_R grouped by Scene Condition
plt.figure(figsize=(8, 6))
data_to_plot = [clear_ar, diff_ar]
sns.boxplot(data=data_to_plot, palette=["#2ecc71", "#e74c3c"])
plt.xticks([0, 1], ['Clear Conditions', 'Challenging (Faded / Occluded)'])
plt.ylabel("Aggregate Robustness Score $\mathcal{A}_R$")
plt.title("Explanation Robustness Dissipation Across Scene Complexity")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()