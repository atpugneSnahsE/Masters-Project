import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp  # Added for your scSE architecture
from lime import lime_image
from skimage.segmentation import slic
from tqdm import tqdm

# -------------------------------------------------------------
# 1. CONFIGURATION & REAL DATA LOADING
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIL100_ROOT = "/home/vgtu/VIL100"
MODEL_WEIGHTS_PATH = "/home/vgtu/vil100_model/best_model.pth"

def load_real_vil100_frame():
    """
    Finds and loads the first available real frame from the VIL100 dataset.
    """
    image_dir = os.path.join(VIL100_ROOT, "JPEGImages")
    if not os.path.exists(image_dir):
        image_dir = VIL100_ROOT
        
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                print(f"Successfully loaded frame: {img_path}")
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resizing to your exact training dimensions
                img = cv2.resize(img, (640, 368)) 
                return img
                
    raise FileNotFoundError(f"Could not find any images in {VIL100_ROOT}.")

try:
    image_np = load_real_vil100_frame()
except Exception as e:
    print(f"Error loading image: {e}")
    print("Falling back to a synthesized dummy frame...")
    image_np = np.zeros((368, 640, 3), dtype=np.uint8)
    image_np[200:368, 300:340] = [255, 255, 255] 

# -------------------------------------------------------------
# 2. MODEL INITIALIZATION & LOADING CHECKPOINT
# -------------------------------------------------------------
# Instantiate your exact training architecture setup
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,       # Not loading imagenet weights here since we load your checkpoint
    decoder_attention="scse",   # Spatial and Channel Squeeze & Excitation Attention
    in_channels=3,
    classes=1,
)

# Load your custom trained model weights safely
if os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"-> Loading trained weights from {MODEL_WEIGHTS_PATH}")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
else:
    print(f"[WARNING] Weights checkpoint not found at {MODEL_WEIGHTS_PATH}! Running with randomly initialized weights.")

model.to(device)
model.eval()

def preprocess_frame(img_np):
    """Preprocesses a numpy image (H, W, C) to matching model input standards."""
    # Convert range to [0.0, 1.0] and rearrange dimensions to (1, C, H, W)
    tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    
    # Apply standard ImageNet normalization matching your training transform pipeline
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    
    return tensor.unsqueeze(0).to(device)

# -------------------------------------------------------------
# 3. FIXED CUSTOM LIME WRAPPER
# -------------------------------------------------------------
def get_lime_prediction_fn(model, base_prediction_mask, device):
    def predict_fn(images):
        model.eval()
        scores = []
        
        with torch.no_grad():
            for img in images:
                tensor = preprocess_frame(img)
                output = model(tensor).squeeze(1) # Match logits extraction
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                
                if np.sum(base_prediction_mask > 0.5) > 0:
                    target_score = np.mean(probs[base_prediction_mask > 0.5])
                else:
                    target_score = np.mean(probs) 
                
                scores.append([1.0 - target_score, target_score])
                
        return np.array(scores)
    return predict_fn

# -------------------------------------------------------------
# 4. FIXED PERTURBATION METRICS (DELETION / INSERTION AUC)
# -------------------------------------------------------------
def calculate_auc_metrics(model, image_np, superpixels, weights, base_mask, steps=10):
    sorted_sp_ids = np.argsort(weights)[::-1] 
    deletion_scores = []
    insertion_scores = []
    sp_per_step = max(1, len(sorted_sp_ids) // steps)
    background = np.zeros_like(image_np)
    
    with torch.no_grad():
        # --- Deletion Loop ---
        img_del = image_np.copy()
        for i in range(0, len(sorted_sp_ids), sp_per_step):
            to_mask = sorted_sp_ids[:i]
            mask = np.isin(superpixels, to_mask)
            img_del[mask] = background[mask]
            
            out = torch.sigmoid(model(preprocess_frame(img_del)).squeeze(1)).squeeze().cpu().numpy()
            if np.sum(base_mask > 0.5) > 0:
                deletion_scores.append(np.mean(out[base_mask > 0.5]))
            else:
                deletion_scores.append(np.mean(out))
            
        # --- Insertion Loop ---
        img_ins = background.copy()
        for i in range(0, len(sorted_sp_ids), sp_per_step):
            to_reveal = sorted_sp_ids[:i]
            mask = np.isin(superpixels, to_reveal)
            img_ins[mask] = image_np[mask]
            
            out = torch.sigmoid(model(preprocess_frame(img_ins)).squeeze(1)).squeeze().cpu().numpy()
            if np.sum(base_mask > 0.5) > 0:
                insertion_scores.append(np.mean(out[base_mask > 0.5]))
            else:
                insertion_scores.append(np.mean(out))

    del_auc = np.trapz(deletion_scores) / (len(deletion_scores) - 1 + 1e-8)
    ins_auc = np.trapz(insertion_scores) / (len(insertion_scores) - 1 + 1e-8)
    
    return del_auc, ins_auc

# -------------------------------------------------------------
# 5. EXPERIMENT RUNNER
# -------------------------------------------------------------
def run_lime_experiment(image_np, d_sweep=[30, 50, 100], N_sweep=[100, 250, 500, 1000]):
    explainer = lime_image.LimeImageExplainer()
    
    with torch.no_grad():
        base_out = torch.sigmoid(model(preprocess_frame(image_np)).squeeze(1)).squeeze().cpu().numpy()
    base_mask = (base_out > 0.5).astype(np.float32)
    
    results = {d: {'N': [], 'R2': [], 'Del_AUC': [], 'Ins_AUC': []} for d in d_sweep}
    
    for d in d_sweep:
        print(f"\n--- Running Sweep for Superpixel Count d={d} ---")
        superpixels = slic(image_np, n_segments=d, compactness=10, sigma=1, start_label=0)
        predict_fn = get_lime_prediction_fn(model, base_mask, device)
        
        for N in tqdm(N_sweep, desc="Sweeping Sample Sizes N"):
            explanation = explainer.explain_instance(
                image_np, 
                predict_fn, 
                top_labels=1, 
                hide_color=0, 
                num_samples=N,
                segmentation_fn=lambda x: superpixels
            )
            
            r2_score = explanation.score
            local_exp = explanation.local_exp[1]
            weights = np.zeros(len(np.unique(superpixels)))
            for sp_id, weight in local_exp:
                weights[sp_id] = weight
                
            del_auc, ins_auc = calculate_auc_metrics(model, image_np, superpixels, weights, base_mask)
            
            results[d]['N'].append(N)
            results[d]['R2'].append(r2_score)
            results[d]['Del_AUC'].append(del_auc)
            results[d]['Ins_AUC'].append(ins_auc)
            
    # --- FIGURE B: METRICS GENERATION ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for d in d_sweep:
        axes[0].plot(results[d]['N'], results[d]['R2'], label=f'd={d}', marker='o')
        axes[1].plot(results[d]['N'], results[d]['Del_AUC'], label=f'd={d} (Del)', linestyle='--', marker='x')
        axes[1].plot(results[d]['N'], results[d]['Ins_AUC'], label=f'd={d} (Ins)', linestyle='-', marker='o')

    axes[0].set_title("Surrogate Fidelity ($R^2$) vs Sample Count ($N$)")
    axes[0].set_xlabel("Sample Count ($N$)")
    axes[0].set_ylabel("$R^2$ Score")
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].set_title("Fidelity Metrics (AUC) vs Sample Count ($N$)")
    axes[1].set_xlabel("Sample Count ($N$)")
    axes[1].set_ylabel("AUC Score")
    axes[1].grid(True)
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    
    # --- FIGURE A: VISUAL MATPLOTLIB OVERLAY ---
    temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=5, hide_rest=False)
    from skimage.segmentation import mark_boundaries
    
    fig_overlay, ax_over = plt.subplots(1, 3, figsize=(15, 5))
    ax_over[0].imshow(image_np)
    ax_over[0].set_title("Input Frame")
    ax_over[0].axis('off')
    
    ax_over[1].imshow(base_mask, cmap='gray')
    ax_over[1].set_title("U-Net Mask Prediction")
    ax_over[1].axis('off')
    
    ax_over[2].imshow(mark_boundaries(temp, mask))
    ax_over[2].set_title(f"LIME Attribution Overlay\n(d={d}, N={N})")
    ax_over[2].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_lime_experiment(image_np)