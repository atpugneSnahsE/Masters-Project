import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
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
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

CLASS_LABELS = [
    "Background", "Ego Road Surface", "Left Dashed Line", "Left Solid Line", 
    "Right Dashed Line", "Right Solid Line", "Other-Veh/Obstacle", "Stop Line Marking", "Crosswalk Marking"
]

class CarlaMultiClassDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="val"):
        self.root_dir = Path(root_dir)
        self.rgb_dir = self.root_dir / "rgb"
        self.mask_dir = self.root_dir / "mask"
        self.transform = transform
        with open(self.root_dir / f"{split}.txt", "r") as f:
            base_names = [line.strip() for line in f if line.strip()]
        self.images = [self.rgb_dir / f"{name}.jpg" for name in base_names]

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name.replace(".jpg", ".png")
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        return image, mask.long()

# ====================== LOAD CHECKPOINT (THE FIX) ======================
print("🔧 Initializing U-Net Architecture...")
model = smp.Unet(
    encoder_name="resnet34", encoder_weights=None,
    in_channels=3, classes=NUM_CLASSES, decoder_attention_type="scse"
).to(DEVICE)

print("💾 Loading saved checkpoint file safely...")
# CRITICAL FIX: explicitly setting weights_only=False to allow numpy scalars
checkpoint = torch.load("models/lane_model_best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Success! Loaded checkpoint with best Validation mIoU: {checkpoint['iou']:.4f}")

# ====================== EVALUATE & PLOT ======================
val_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
val_dataset = CarlaMultiClassDataset(DATA_ROOT, transform=val_transform, split="val")
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

jaccard = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="none").to(DEVICE)
precision_metric = Precision(task="multiclass", num_classes=NUM_CLASSES, average="none").to(DEVICE)
recall_metric = Recall(task="multiclass", num_classes=NUM_CLASSES, average="none").to(DEVICE)

print("📊 Computing comprehensive accuracy metrics across validation dataset...")
with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        preds = model(images).argmax(dim=1)
        jaccard.update(preds, masks)
        precision_metric.update(preds, masks)
        recall_metric.update(preds, masks)

# Generate Matrix Plot
plt.style.use("seaborn-v0_8-white")
metric_matrix = np.vstack([jaccard.compute().cpu().numpy(), precision_metric.compute().cpu().numpy(), recall_metric.compute().cpu().numpy()]).T

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(metric_matrix, annot=True, fmt=".4f", cmap="magma_r", 
            xticklabels=["IoU", "Precision", "Recall"], yticklabels=CLASS_LABELS,
            cbar=True, annot_kws={"size": 11, "weight": "bold"}, ax=ax, linewidths=.5)
ax.set_title("U-Net Lane Segmentation Efficiency Matrix", fontsize=13, fontweight='bold', pad=15)
plt.subplots_adjust(left=0.28, bottom=0.15)
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "segmentation_efficiency_matrix.png"), dpi=300)
print("🎉 Saved: reports/segmentation_plots/segmentation_efficiency_matrix.png")

# Generate Image Overlay Grid
raw_dataset = CarlaMultiClassDataset(DATA_ROOT, transform=A.Compose([A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]), A.Normalize(mean=(0,0,0), std=(1,1,1)), ToTensorV2()]), split="val")
indices = [min(15, len(raw_dataset)-1), min(45, len(raw_dataset)-1), min(90, len(raw_dataset)-1)]
cmap_vals = np.array([[0,0,0],[150,150,150],[255,255,0],[255,255,255],[0,0,255],[0,255,255],[255,0,0],[255,100,0],[0,255,0]], dtype=np.uint8)

fig, axes = plt.subplots(len(indices), 3, figsize=(12, 3.5 * len(indices)))
for i, idx in enumerate(indices):
    img_tensor, mask_tensor = raw_dataset[idx]
    axes[i, 0].imshow(img_tensor.cpu().numpy().transpose(1, 2, 0).clip(0, 1))
    axes[i, 0].axis('off')
    if i == 0: axes[i, 0].set_title("Input RGB Frame", fontweight='bold')

    axes[i, 1].imshow(cmap_vals[mask_tensor.cpu().numpy()])
    axes[i, 1].axis('off')
    if i == 0: axes[i, 1].set_title("Ground Truth Mask", fontweight='bold')

    with torch.no_grad():
        pred_batch = model(img_tensor.unsqueeze(0).to(DEVICE)).argmax(dim=1).squeeze(0)
    axes[i, 2].imshow(cmap_vals[pred_batch.cpu().numpy()])
    axes[i, 2].axis('off')
    if i == 0: axes[i, 2].set_title("U-Net Model Prediction", fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "segmentation_prediction_overlays.png"), dpi=300)
print("🎉 Saved: reports/segmentation_plots/segmentation_prediction_overlays.png")