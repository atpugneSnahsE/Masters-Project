import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from transformers import SegformerForSemanticSegmentation
import warnings
warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = "/home/vgtu/lane_dataset"
IMAGE_SIZE = (256, 512)
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 4

MODELS_TO_TEST = [
    # 1. Vision Transformer — loaded via HuggingFace transformers
    {"name": "SegFormer_B0",        "arch": "segformer",     "encoder": "mit_b0",         "weights": "imagenet"},

    # 2. Atrous Spatial Pyramid Pooling (great for lane lines stretching to horizon)
    {"name": "DeepLabV3Plus_RN34",  "arch": "deeplabv3plus", "encoder": "resnet34",       "weights": "imagenet"},

    # 3. Dense/Nested Skip Connections variant
    {"name": "UnetPlusPlus_RN34",   "arch": "unetplusplus",  "encoder": "resnet34",       "weights": "imagenet"},

    # 4. Standard UNet baselines (varying complexity)
    {"name": "Unet_ResNet18",       "arch": "unet",          "encoder": "resnet18",       "weights": "imagenet"},
    {"name": "Unet_ResNet34",       "arch": "unet",          "encoder": "resnet34",       "weights": "imagenet"},
    {"name": "Unet_MobileNetV2",    "arch": "unet",          "encoder": "mobilenet_v2",   "weights": "imagenet"},
    {"name": "Unet_EfficientNetB0", "arch": "unet",          "encoder": "efficientnet-b0","weights": "imagenet"},
]


# ===================== SEGFORMER WRAPPER =====================
# smp does not include SegFormer. We load it from HuggingFace transformers
# and wrap it so it outputs a (B, 1, H, W) logit tensor — same interface as smp.

class SegFormerWrapper(nn.Module):
    """
    Wraps HuggingFace SegformerForSemanticSegmentation for binary segmentation.
    Upsamples the model's 4x-downsampled output back to the input resolution.
    """
    CHECKPOINT_MAP = {
        "mit_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "mit_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
        "mit_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "mit_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
        "mit_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
        "mit_b5": "nvidia/segformer-b5-finetuned-ade-640-640",
    }

    def __init__(self, encoder="mit_b0"):
        super().__init__()
        checkpoint = self.CHECKPOINT_MAP.get(encoder)
        if checkpoint is None:
            raise ValueError(f"Unknown SegFormer encoder '{encoder}'. "
                             f"Choose from: {list(self.CHECKPOINT_MAP.keys())}")

        # Load pretrained backbone; replace the 150-class ADE head with a 1-class head
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        out = self.model(pixel_values=x)
        # out.logits shape: (B, 1, H/4, W/4) — upsample to full resolution
        logits = torch.nn.functional.interpolate(
            out.logits, size=(h, w), mode='bilinear', align_corners=False
        )
        return logits  # (B, 1, H, W)


# ===================== CARLA SIMULATION DATASET =====================
class CarlaLaneDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_dir = self.root_dir / "rgb"
        self.mask_dir  = self.root_dir / "mask"
        self.samples   = self._load_samples()

    def _load_samples(self):
        samples = []
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            mask_path = self.mask_dir / img_path.name.replace(".jpg", ".png")
            if mask_path.exists():
                samples.append((img_path, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']

        mask = (mask > 0).float().unsqueeze(0)
        return image, mask


# ===================== METRICS =====================
def compute_metrics(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # Squeeze channel dim: (B, 1, H, W) -> (B, H, W)
    pred   = pred.squeeze(1)
    target = target.squeeze(1)

    intersection = (pred * target).sum(dim=(1, 2))
    pred_sum     = pred.sum(dim=(1, 2))
    target_sum   = target.sum(dim=(1, 2))
    union        = pred_sum + target_sum - intersection

    # Epsilon only in denominator. When both pred & target are empty
    # (union == 0) treat as a correct prediction (iou = 1).
    iou       = torch.where(union > 0,
                    intersection / (union + 1e-6),
                    torch.ones_like(intersection))
    precision = torch.where(pred_sum > 0,
                    intersection / (pred_sum + 1e-6),
                    torch.ones_like(intersection))
    recall    = torch.where(target_sum > 0,
                    intersection / (target_sum + 1e-6),
                    torch.ones_like(intersection))

    return iou.mean().item(), precision.mean().item(), recall.mean().item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_fps(model, device, runs=100):
    model.eval()
    dummy = torch.randn(1, 3, *IMAGE_SIZE).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy)
    return runs / (time.time() - start)


# ===================== MODEL FACTORY =====================
def build_model(config):
    """Returns a SegFormerWrapper for segformer arch, smp model for everything else."""
    if config['arch'] == 'segformer':
        return SegFormerWrapper(encoder=config['encoder'])

    return smp.create_model(
        arch=config['arch'],
        encoder_name=config['encoder'],
        encoder_weights=config['weights'],
        classes=1,
    )


# ===================== TRAINING & EVAL =====================
def train_and_evaluate(config, train_loader, val_loader):
    print(f"\n{'='*70}\nTraining {config['name']}\n{'='*70}")

    model = build_model(config).to(DEVICE)

    optimizer    = torch.optim.Adam(model.parameters(), lr=3e-4)
    dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    bce_loss_fn  = nn.BCEWithLogitsLoss()

    best_miou         = 0.0
    val_prec, val_rec = 0.0, 0.0
    start_time        = time.time()

    for epoch in range(NUM_EPOCHS):
        # ---- Training ----
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)

            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )

            loss = dice_loss_fn(outputs, masks) + bce_loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

        # ---- Validation ----
        model.eval()
        val_iou, val_prec, val_rec = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)

                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )

                iou, prec, rec = compute_metrics(outputs, masks)
                val_iou  += iou
                val_prec += prec
                val_rec  += rec

        val_iou  /= len(val_loader)
        val_prec /= len(val_loader)
        val_rec  /= len(val_loader)

        if val_iou > best_miou:
            best_miou = val_iou
            torch.save(model.state_dict(), f"sim_{config['name']}_best.pth")

        print(f"Epoch {epoch+1:2d} | Val mIoU: {val_iou:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

    total_time = (time.time() - start_time) / 60
    fps    = benchmark_fps(model, DEVICE)
    params = count_parameters(model) / 1e6

    results = {
        "Model":          config['name'],
        "mIoU":           round(best_miou, 4),
        "Precision":      round(val_prec, 4),
        "Recall":         round(val_rec, 4),
        "FPS":            round(fps, 2),
        "Params_M":       round(params, 2),
        "Train_Time_min": round(total_time, 2),
        "Epochs":         NUM_EPOCHS,
        "Date":           datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    print(f"✅ {config['name']} completed → mIoU: {best_miou:.4f} | FPS: {fps:.1f}")
    return results


# ===================== MAIN =====================
if __name__ == "__main__":
    # Check transformers is installed before doing anything else
    try:
        import transformers
    except ImportError:
        raise SystemExit("❌ Please install HuggingFace transformers:\n"
                         "   pip install transformers")

    transform = A.Compose([
        A.Resize(*IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    dataset = CarlaLaneDataset(ROOT_DIR, transform=transform)

    if len(dataset) == 0:
        raise SystemExit(f"❌ Found 0 matching pairs. Ensure RGB .jpg files and mask .png files "
                         f"exist inside {ROOT_DIR}/rgb and {ROOT_DIR}/mask")

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Total Simulation Samples: {len(dataset)} | Train: {train_size} | Val: {val_size}\n")

    all_results = []
    for cfg in MODELS_TO_TEST:
        try:
            result = train_and_evaluate(cfg, train_loader, val_loader)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Failed to run benchmark for {cfg['name']}. Error: {e}")

    if all_results:
        df       = pd.DataFrame(all_results)
        csv_path = f"carla_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(csv_path, index=False)

        print("\n" + "="*80)
        print("FINAL CARLA SIMULATION BENCHMARK TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print(f"\n📊 Results saved to: {csv_path}")
    else:
        print("\n❌ No models were evaluated successfully.")