import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchmetrics import JaccardIndex, Precision, Recall
from pathlib import Path
import cv2

# ========================= CONFIG =========================
DATA_ROOT = "lane_dataset"
NUM_CLASSES = 9
IMAGE_SIZE = (384, 768)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PLOT_DIR = "reports/segmentation_plots"

CLASS_LABELS = [
    "Background", "Ego Road", "Left Dashed", "Left Solid", 
    "Right Dashed", "Right Solid", "Obstacle", "Stop Line", "Crosswalk"
]

class CarlaMultiClassDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="val"):
        self.root_dir = Path(root_dir)
        with open(self.root_dir / f"{split}.txt", "r") as f:
            base_names = [line.strip() for line in f if line.strip()]
        self.images = [self.root_dir / "rgb" / f"{name}.jpg" for name in base_names]
        self.masks = [self.root_dir / "mask" / f"{name}.png" for name in base_names]
        self.transform = transform

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(str(self.images[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        return image, mask.long()

# ====================== EVALUATION ======================
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES, decoder_attention_type="scse").to(DEVICE)
checkpoint = torch.load("models/lane_model_best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

val_loader = DataLoader(CarlaMultiClassDataset(DATA_ROOT, split="val", transform=A.Compose([A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]), A.Normalize(), ToTensorV2()])), batch_size=4, shuffle=False, num_workers=2)

jaccard = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="none").to(DEVICE)
precision_metric = Precision(task="multiclass", num_classes=NUM_CLASSES, average="none").to(DEVICE)
recall_metric = Recall(task="multiclass", num_classes=NUM_CLASSES, average="none").to(DEVICE)

with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        preds = model(images).argmax(dim=1)
        jaccard.update(preds, masks)
        precision_metric.update(preds, masks)
        recall_metric.update(preds, masks)

# Extract scores
iou_scores = jaccard.compute().cpu().numpy()
prec_scores = precision_metric.compute().cpu().numpy()
rec_scores = recall_metric.compute().cpu().numpy()

# ====================== PLOTTING ======================
x = np.arange(len(CLASS_LABELS))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(x - width, iou_scores, width, label='IoU / Jaccard', color='#2ca02c', alpha=0.85)
ax.bar(x, prec_scores, width, label='Precision', color='#1f77b4', alpha=0.85)
ax.bar(x + width, rec_scores, width, label='Recall', color='#ff7f0e', alpha=0.85)

ax.set_ylabel('Score Value (0.0 - 1.0)', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Segmentation Evaluation Profile', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_LABELS, rotation=25, ha='right', fontsize=11)
ax.legend(loc='lower left', frameon=True, shadow=False)
ax.set_ylim(0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "segmentation_metrics_bars.png"), dpi=300)
print("🎉 Bar chart successfully saved to reports/segmentation_plots/segmentation_metrics_bars.png")