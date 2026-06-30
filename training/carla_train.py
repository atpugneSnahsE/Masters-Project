import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import os
import gc
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics import JaccardIndex, Precision, Recall

# Environment optimization parameters
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ========================= CONFIG =========================
DATA_ROOT = "lane_dataset"
NUM_CLASSES = 9
IMAGE_SIZE = (384, 768)
BATCH_SIZE = 2          
GRAD_ACCUM_STEPS = 2    
NUM_WORKERS = 2         
NUM_EPOCHS = 50
LEARNING_RATE = 1.2e-4  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VALID_METRIC_CLASSES = [0, 1, 2, 3, 4, 5, 8]
CLASS_LABELS = [
    "Background", "Ego Road Surface", "Left Dashed Line", "Left Solid Line", 
    "Right Dashed Line", "Right Solid Line", "Other-Veh/Obstacle", "Stop Line Marking", "Crosswalk Marking"
]

OUTPUT_PLOT_DIR = "reports/segmentation_plots"
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

print(f"Using device: {DEVICE}")

# ====================== AUGMENTATIONS ======================
train_transform = A.ReplayCompose([
    A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.CLAHE(clip_limit=3.0, p=0.4),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=8,
                       border_mode=cv2.BORDER_CONSTANT, p=0.6),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ====================== DATASET ======================
def _was_flipped(replay: dict) -> bool:
    for t in replay.get("transforms", []):
        if "HorizontalFlip" in t.get("__class_fullname__", "") and t.get("applied", False):
            return True
    return False

class CarlaMultiClassDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train"):
        self.root_dir = Path(root_dir)
        self.rgb_dir = self.root_dir / "rgb"
        self.mask_dir = self.root_dir / "mask"
        self.transform = transform
        self.split = split

        split_file = self.root_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing required split file: {split_file}")
        
        with open(split_file, "r") as f:
            base_names = [line.strip() for line in f if line.strip()]
        
        self.images = [self.rgb_dir / f"{name}.jpg" for name in base_names]

        if split == "train":
            print("Indexing minority instances for balanced sampling...")
            minority_images = []
            for img_path in tqdm(self.images, desc="Scanning for minority masks"):
                mask_path = self.mask_dir / img_path.name.replace(".jpg", ".png")
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None and (np.any(mask == 3) or np.any(mask == 4)):
                    minority_images.append(img_path)

            if len(minority_images) > 0:
                print(f"Found {len(minority_images)} minority images. Applying 3x oversampling boost.")
                self.images = self.images + (minority_images * 2)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name.replace(".jpg", ".png")
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        flipped = False
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            if self.split == "train":
                flipped = _was_flipped(augmented["replay"])

        if flipped:
            np_mask = mask.cpu().numpy() if torch.is_tensor(mask) else mask.copy()
            swapped = np_mask.copy()
            swapped[np_mask == 2] = 4
            swapped[np_mask == 4] = 2
            swapped[np_mask == 3] = 5
            swapped[np_mask == 5] = 3
            mask = torch.tensor(swapped, dtype=mask.dtype) if torch.is_tensor(mask) else torch.from_numpy(swapped)

        return image, mask.long()

# ====================== COMPUTE CLASS WEIGHTS ======================
def compute_weights(mask_dir):
    print("Computing class weights from dataset...")
    mask_paths = list(Path(mask_dir).glob("*.png"))
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for p in tqdm(mask_paths[:5000], desc="Computing weights"):
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        unique, val_counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, val_counts):
            if u < NUM_CLASSES: counts[u] += c

    total = counts.sum()
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for c in range(NUM_CLASSES):
        weights[c] = total / (NUM_CLASSES * counts[c] + 1e-5) if counts[c] > 0 else 1.0

    weights = torch.tensor(weights, dtype=torch.float32)
    return torch.clamp(torch.log1p(weights), min=0.6, max=25.0)

# ====================== LOSS FUNCTION ======================
class HybridSegmentationLoss(nn.Module):
    def __init__(self, ce_weights):
        super(HybridSegmentationLoss, self).__init__()
        self.focal = smp.losses.FocalLoss(mode="multiclass", gamma=2.0)
        self.dice = smp.losses.DiceLoss(mode="multiclass", classes=NUM_CLASSES, smooth=1.0)
        self.register_buffer("ce_weights", ce_weights)

    def forward(self, preds, targets):
        return (1.5 * self.focal(preds, targets)) + \
               (0.5 * nn.functional.cross_entropy(preds, targets, weight=self.ce_weights)) + \
               (2.0 * self.dice(preds, targets))

# ====================== INITIALIZATION ======================
gc.collect()
torch.cuda.empty_cache()

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
    decoder_attention_type="scse",
).to(DEVICE)

class_weights = compute_weights(Path(DATA_ROOT) / "mask")
criterion = HybridSegmentationLoss(ce_weights=class_weights.to(DEVICE))

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

jaccard = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="none").to(DEVICE)
precision_metric = Precision(task="multiclass", num_classes=NUM_CLASSES, average="none").to(DEVICE)
recall_metric = Recall(task="multiclass", num_classes=NUM_CLASSES, average="none").to(DEVICE)

train_dataset = CarlaMultiClassDataset(DATA_ROOT, transform=train_transform, split="train")
val_dataset = CarlaMultiClassDataset(DATA_ROOT, transform=val_transform, split="val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# History dictionaries to generate plots later
history = {"train_loss": [], "val_loss": [], "val_miou": []}

# ====================== TRAINING LOOP ======================
best_val_iou = 0.0
patience = 12
patience_counter = 0

print("\n🚀 Commencing Model Training Pass...")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks) in enumerate(pbar):
        images, masks = images.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks) / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_train_loss += loss.item() * GRAD_ACCUM_STEPS
        pbar.set_postfix(loss=f"{(loss.item() * GRAD_ACCUM_STEPS):.4f}")

    # Validation Pass
    model.eval()
    epoch_val_loss = 0.0
    jaccard.reset()
    precision_metric.reset()
    recall_metric.reset()

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
            outputs = model(images)
            epoch_val_loss += criterion(outputs, masks).item()
            
            preds = outputs.argmax(dim=1)
            jaccard.update(preds, masks)
            precision_metric.update(preds, masks)
            recall_metric.update(preds, masks)

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_val_loss = epoch_val_loss / len(val_loader)
    
    per_class_iou = jaccard.compute().cpu().numpy()
    avg_mean_iou = np.mean([per_class_iou[idx] for idx in VALID_METRIC_CLASSES])

    scheduler.step()

    # Append metrics to history logs for graphing
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_miou"].append(avg_mean_iou)

    print(f"\n--- Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Target mIoU: {avg_mean_iou:.4f} ---")

    if avg_mean_iou > best_val_iou:
        best_val_iou = avg_mean_iou
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "iou": best_val_iou}, "models/lane_model_best.pth")
        print(f"✅ Saved Best Model Checkpoint!")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"🛑 Early stopping triggered at Epoch {epoch+1}")
        break

# ====================== AUTOMATIC PLOTTING PHASE ======================
print("\n==================== TRAINING COMPLETE. GENERATING PLOTS ====================")

# Load best checkpoint weights for metrics plotting
best_checkpoint = torch.load("models/lane_model_best.pth", map_location=DEVICE)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

plt.style.use("seaborn-v0_8-whitegrid")

# 1. Plot Learning Curves from real history data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
actual_epochs = len(history["train_loss"])
ax1.plot(range(actual_epochs), history["train_loss"], label='Training Loss', color='#1f77b4', lw=2)
ax1.plot(range(actual_epochs), history["val_loss"], label='Validation Loss', color='#ff7f0e', lw=2, linestyle='--')
ax1.set_title("Hybrid Loss Curves", fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss Value")
ax1.legend()

ax2.plot(range(actual_epochs), history["val_miou"], label='Val mIoU', color='#2ca02c', lw=2.5)
ax2.axhline(y=best_val_iou, color='red', linestyle=':', label=f"Best mIoU: {best_val_iou:.4f}")
ax2.set_title("Validation Macro mIoU Curve", fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mean Intersection-over-Union")
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "segmentation_learning_curves.png"), dpi=300)
print("🎉 Saved: segmentation_learning_curves.png")

# 2. Compute and Plot Efficiency Matrix Heatmap
print("Generating final efficiency heatmap matrix...")
metric_matrix = np.vstack([jaccard.compute().cpu().numpy(), precision_metric.compute().cpu().numpy(), recall_metric.compute().cpu().numpy()]).T

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(metric_matrix, annot=True, fmt=".4f", cmap="magma_r", 
            xticklabels=["IoU", "Precision", "Recall"], yticklabels=CLASS_LABELS,
            cbar=True, annot_kws={"size": 11, "weight": "bold"}, ax=ax, linewidths=.5)
ax.set_title("U-Net Lane Segmentation Efficiency Matrix", fontsize=13, fontweight='bold', pad=15)
plt.subplots_adjust(left=0.28, bottom=0.15)
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "segmentation_efficiency_matrix.png"), dpi=300)
print("🎉 Saved: segmentation_efficiency_matrix.png")

# 3. Plot Qualitative Image Comparison Overlays
print("Creating prediction overlay comparative grids...")
raw_dataset = CarlaMultiClassDataset(DATA_ROOT, transform=A.Compose([A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]), A.Normalize(mean=(0,0,0), std=(1,1,1)), ToTensorV2()]), split="val")
indices = [min(15, len(raw_dataset)-1), min(45, len(raw_dataset)-1), min(90, len(raw_dataset)-1)]

cmap_vals = np.array([[0,0,0],[150,150,150],[255,255,0],[255,255,255],[0,0,255],[0,255,255],[255,0,0],[255,100,0],[0,255,0]], dtype=np.uint8)
fig, axes = plt.subplots(len(indices), 3, figsize=(12, 3.5 * len(indices)))

for i, idx in enumerate(indices):
    img_tensor, mask_tensor = raw_dataset[idx]
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    axes[i, 0].imshow(img_np.clip(0, 1))
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
print("🎉 Saved: segmentation_prediction_overlays.png\nAll experimental pipeline execution plots finalized!")