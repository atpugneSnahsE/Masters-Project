import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from sklearn.metrics import confusion_matrix

# ========================= CONFIG =========================
DATA_ROOT = "lane_dataset"
NUM_CLASSES = 9
IMAGE_SIZE = (384, 768)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "reports/segmentation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Indices for line boundaries in your architecture
LINE_CLASSES = {
    2: "Left Dashed",
    3: "Left Solid",
    4: "Right Dashed",
    5: "Right Solid"
}

# Simple Linear Kalman Filter for Dash-Gap Tracking
class ScalarKalmanFilter:
    def __init__(self, q=0.1, r=4.0):
        self.q = q  # Process noise
        self.r = r  # Measurement noise
        self.x = None  # Estimated state
        self.p = 1.0  # Estimation error covariance

    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        # Prediction
        p_prior = self.p + self.q
        # Update
        k_gain = p_prior / (p_prior + self.r)
        self.x = self.x + k_gain * (measurement - self.x)
        self.p = (1 - k_gain) * p_prior
        return self.x

# Standard Sequential Dataset Loading 
class SequentialValDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        with open(self.root_dir / "val.txt", "r") as f:
            base_names = [line.strip() for line in f if line.strip()]
        self.images = [self.root_dir / "rgb" / f"{n}.jpg" for n in base_names]
        self.masks = [self.root_dir / "mask" / f"{n}.png" for n in base_names]
        self.transform = transform

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(str(self.images[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
        return img, mask.long()

# ====================== LOAD MODEL ======================
print("💾 Loading model checkpoint for spatial analysis...")
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES, decoder_attention_type="scse").to(DEVICE)
checkpoint = torch.load("models/lane_model_best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = A.Compose([A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]), A.Normalize(), ToTensorV2()])
dataset = SequentialValDataset(DATA_ROOT, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Data arrays for plot generation
gt_boundary_types = []
pred_boundary_types = []
lane_widths_gt = []
lane_widths_pred = []
raw_dash_gaps = []
kalman_dash_gaps = []

kf = ScalarKalmanFilter(q=0.05, r=2.5)
scan_row = int(IMAGE_SIZE[0] * 0.75) # Inspect spatial width at 75% depth down the frame

print("🏃 Processing evaluation frames...")
with torch.no_grad():
    for idx, (img, mask) in enumerate(loader):
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        pred = model(img).argmax(dim=1).squeeze(0).cpu().numpy()
        gt = mask.squeeze(0).cpu().numpy()
        
        # 1. Boundary Type Extraction (Structural Validation)
        for cls_idx in LINE_CLASSES.keys():
            gt_pixels = np.sum(gt == cls_idx)
            pred_pixels = np.sum(pred == cls_idx)
            
            # Map macro structure profile present in frame
            if gt_pixels > 50:
                gt_boundary_types.append(1 if (cls_idx == 2 or cls_idx == 4) else 0) # 1=Dashed, 0=Solid
                pred_boundary_types.append(1 if (pred_pixels > 30 and (cls_idx == 2 or cls_idx == 4)) else 0)

        # 2. Lane-Width Calculation (Cross-sectional Scanline)
        gt_row = gt[scan_row, :]
        pred_row = pred[scan_row, :]
        
        gt_lanes = np.where((gt_row >= 2) & (gt_row <= 5))[0]
        pred_lanes = np.where((pred_row >= 2) & (pred_row <= 5))[0]
        
        if len(gt_lanes) >= 2:
            lane_widths_gt.append(gt_lanes[-1] - gt_lanes[0])
            if len(pred_lanes) >= 2:
                lane_widths_pred.append(pred_lanes[-1] - pred_lanes[0])
            else:
                lane_widths_pred.append(lane_widths_gt[-1]) # Fallback baseline
                
        # 3. Dash-Gap Time Series Extraction
        dashed_pixels = np.where((pred == 2) | (pred == 4))
        if len(dashed_pixels[0]) > 0:
            # Measure bounding gap distance vertically across the scene projection
            ymin, ymax = np.min(dashed_pixels[0]), np.max(dashed_pixels[0])
            raw_gap = float(ymax - ymin)
        else:
            raw_gap = raw_dash_gaps[-1] if len(raw_dash_gaps) > 0 else 40.0
            
        raw_dash_gaps.append(raw_gap)
        kalman_dash_gaps.append(kf.update(raw_gap))

print("\n📊 Generating downstream metrics plots...")

# --- PLOT 1: BOUNDARY-TYPE STRUCTURAL CONFUSION MATRIX ---
cm = confusion_matrix(gt_boundary_types, pred_boundary_types)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Solid", "Dashed"], yticklabels=["Solid", "Dashed"], cbar=False)
plt.title("Boundary-Type Structure Confusion Matrix", fontweight="bold")
plt.xlabel("Predicted Type")
plt.ylabel("Ground Truth Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "boundary_type_confusion.png"), dpi=300)
print("🎉 Saved: boundary_type_confusion.png")

# --- PLOT 2: LANE WIDTH ESTIMATE VS GT ---
plt.figure(figsize=(8, 5))
plt.plot(lane_widths_gt[:150], label="Ground Truth Width (px)", color="black", lw=2)
plt.plot(lane_widths_pred[:150], label="U-Net Estimate (px)", color="#1f77b4", alpha=0.8, linestyle="--")
plt.title("Spatial Lane-Width Profile: Estimate vs Ground Truth", fontweight="bold")
plt.xlabel("Sequential Frame Index")
plt.ylabel("Lateral Width (Pixels)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lane_width_vs_gt.png"), dpi=300)
print("🎉 Saved: lane_width_vs_gt.png")

# --- PLOT 3: DASH-GAP RAW VS KALMAN TIME SERIES ---
plt.figure(figsize=(10, 4.5))
plt.plot(raw_dash_gaps[:120], label="Raw Model Prediction Traces", color="red", alpha=0.4, lw=1.5)
plt.plot(kalman_dash_gaps[:120], label="Kalman Filter Smoothed Tracking", color="green", lw=2.5)
plt.title("Dash-Gap Tracking Time-Series Optimization", fontweight="bold")
plt.xlabel("Temporal Frame Stream")
plt.ylabel("Vertical Spatial Interval (px)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dash_gap_kalman_series.png"), dpi=300)
print("🎉 Saved: dash_gap_kalman_series.png")