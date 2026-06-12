import os
# ── Set before importing torch to reduce CUDA memory fragmentation ──────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import gc
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
import json
from pathlib import Path
from datetime import datetime
from transformers import SegformerForSemanticSegmentation
import warnings
warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR    = "/home/vgtu/VIL100"   # PATH
IMAGE_SIZE  = (256, 512)
BATCH_SIZE  = 8   # Starting batch size — will be halved automatically on OOM
NUM_EPOCHS  = 10
NUM_WORKERS = 4

# Use mixed-precision when a CUDA GPU is available (halves activation memory)
USE_AMP = torch.cuda.is_available()

MODELS_TO_TEST = [
    {"name": "UnetPlusPlus_RN34",   "arch": "unetplusplus",  "encoder": "resnet34",       "weights": "imagenet"},
    {"name": "SegFormer_B0",        "arch": "segformer",     "encoder": "mit_b0",         "weights": "imagenet"},
    {"name": "DeepLabV3Plus_RN34",  "arch": "deeplabv3plus", "encoder": "resnet34",       "weights": "imagenet"},
    {"name": "Unet_ResNet18",       "arch": "unet",          "encoder": "resnet18",       "weights": "imagenet"},
    {"name": "Unet_ResNet34",       "arch": "unet",          "encoder": "resnet34",       "weights": "imagenet"},
    {"name": "Unet_MobileNetV2",    "arch": "unet",          "encoder": "mobilenet_v2",   "weights": "imagenet"},
    {"name": "Unet_EfficientNetB0", "arch": "unet",          "encoder": "efficientnet-b0","weights": "imagenet"},
]


# ===================== MEMORY HELPERS =====================
def free_gpu_memory(*objects):
    """Delete objects, run gc, and empty the CUDA cache."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_mem_free_mb():
    if not torch.cuda.is_available():
        return float("inf")
    free, _ = torch.cuda.mem_get_info()
    return free / 1024 ** 2


# ===================== SEGFORMER WRAPPER =====================
class SegFormerWrapper(nn.Module):
    """
    Wraps HuggingFace SegformerForSemanticSegmentation for binary segmentation.
    Upsamples the model's 4x-downsampled output back to the input resolution.
    Supports gradient checkpointing to reduce activation memory.
    """
    CHECKPOINT_MAP = {
        "mit_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "mit_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
        "mit_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "mit_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
        "mit_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
        "mit_b5": "nvidia/segformer-b5-finetuned-ade-640-640",
    }

    def __init__(self, encoder="mit_b0", gradient_checkpointing=False):
        super().__init__()
        checkpoint = self.CHECKPOINT_MAP.get(encoder)
        if checkpoint is None:
            raise ValueError(f"Unknown SegFormer encoder '{encoder}'. "
                             f"Choose from: {list(self.CHECKPOINT_MAP.keys())}")

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        out = self.model(pixel_values=x)
        logits = torch.nn.functional.interpolate(
            out.logits, size=(h, w), mode='bilinear', align_corners=False
        )
        return logits  # (B, 1, H, W)


# ===================== VIL-100 DATASET =====================
class VIL100Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.image_dir = self.root_dir / "JPEGImages"
        self.json_dir  = self.root_dir / "Json"
        self.samples   = self._load_samples()

    def _load_samples(self):
        samples = []
        for clip_folder in sorted(self.image_dir.glob("*_frames")):
            json_clip = self.json_dir / clip_folder.name
            if not json_clip.exists():
                continue
            for json_file in sorted(json_clip.glob("*.jpg.json")):
                img_file = clip_folder / json_file.name.replace(".json", "")
                if img_file.exists():
                    samples.append((img_file, json_file))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w  = image.shape[:2]
        mask  = self._create_mask(json_path, (h, w))

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']

        mask = (mask > 0).float().unsqueeze(0)
        return image, mask

    def _create_mask(self, json_path, img_shape):
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            lanes = data.get('annotations', {}).get('lane', [])
            for lane in lanes:
                pts = lane.get('points', [])
                if len(pts) < 2:
                    continue
                points = np.array(pts, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(mask, [points], isClosed=False, color=1, thickness=12)
        except Exception:
            pass
        return mask


# ===================== METRICS =====================
def compute_metrics(pred, target):
    pred   = (torch.sigmoid(pred) > 0.5).float().squeeze(1)
    target = target.squeeze(1)

    intersection = (pred * target).sum(dim=(1, 2))
    pred_sum     = pred.sum(dim=(1, 2))
    target_sum   = target.sum(dim=(1, 2))
    union        = pred_sum + target_sum - intersection

    iou       = torch.where(union > 0,       intersection / (union + 1e-6),       torch.ones_like(intersection))
    precision = torch.where(pred_sum > 0,    intersection / (pred_sum + 1e-6),    torch.ones_like(intersection))
    recall    = torch.where(target_sum > 0,  intersection / (target_sum + 1e-6),  torch.ones_like(intersection))

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
def build_model(config, gradient_checkpointing=False):
    """Returns a SegFormerWrapper or smp model. Enables gradient checkpointing when requested."""
    if config['arch'] == 'segformer':
        return SegFormerWrapper(encoder=config['encoder'],
                                gradient_checkpointing=gradient_checkpointing)

    model = smp.create_model(
        arch=config['arch'],
        encoder_name=config['encoder'],
        encoder_weights=config['weights'],
        classes=1,
    )

    # smp encoders expose set_grad_checkpointing on recent versions
    if gradient_checkpointing and hasattr(model.encoder, 'set_grad_checkpointing'):
        model.encoder.set_grad_checkpointing(enable=True)

    return model


# ===================== TRAINING & EVAL =====================
def make_loaders(train_ds, val_ds, batch_size):
    """(Re-)create DataLoaders with the given batch_size."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader


def train_and_evaluate(config, train_ds, val_ds, initial_batch_size=BATCH_SIZE):
    """
    Train a model and evaluate it.  If a CUDA OOM is encountered during training
    the batch size is halved (up to MAX_RETRIES times) and training restarts.
    Gradient checkpointing is also enabled on the first OOM to further reduce
    activation memory.
    """
    MAX_RETRIES = 3
    batch_size  = initial_batch_size
    grad_ckpt   = False  # escalated to True after first OOM

    for attempt in range(MAX_RETRIES + 1):
        free_gpu_memory()  # always start clean

        print(f"\n{'='*70}")
        print(f"Training {config['name']}  "
              f"(attempt {attempt+1}, batch_size={batch_size}, grad_ckpt={grad_ckpt})")
        print(f"  GPU free before build: {gpu_mem_free_mb():.0f} MB")
        print('='*70)

        try:
            model        = build_model(config, gradient_checkpointing=grad_ckpt).to(DEVICE)
            optimizer    = torch.optim.Adam(model.parameters(), lr=3e-4)
            dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
            bce_loss_fn  = nn.BCEWithLogitsLoss()
            scaler       = torch.cuda.amp.GradScaler(enabled=USE_AMP)

            train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size)

            best_miou         = 0.0
            val_prec, val_rec = 0.0, 0.0
            start_time        = time.time()

            for epoch in range(NUM_EPOCHS):
                # ---- Training ----
                model.train()
                for images, masks in train_loader:
                    images, masks = images.to(DEVICE), masks.to(DEVICE)
                    optimizer.zero_grad(set_to_none=True)  # slightly less memory than zero_grad()

                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        outputs = model(images)
                        if outputs.shape[-2:] != masks.shape[-2:]:
                            outputs = torch.nn.functional.interpolate(
                                outputs, size=masks.shape[-2:],
                                mode='bilinear', align_corners=False
                            )
                        loss = dice_loss_fn(outputs, masks) + bce_loss_fn(outputs, masks)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # ---- Validation ----
                model.eval()
                val_iou, val_prec, val_rec = 0.0, 0.0, 0.0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(DEVICE), masks.to(DEVICE)
                        with torch.cuda.amp.autocast(enabled=USE_AMP):
                            outputs = model(images)
                            if outputs.shape[-2:] != masks.shape[-2:]:
                                outputs = torch.nn.functional.interpolate(
                                    outputs, size=masks.shape[-2:],
                                    mode='bilinear', align_corners=False
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
                    prefix    = "sim_" if "rgb" in str(ROOT_DIR) else ""
                    torch.save(model.state_dict(), f"{prefix}{config['name']}_best.pth")

                print(f"  Epoch {epoch+1:2d} | Val mIoU: {val_iou:.4f} | "
                      f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

            # ---- Success ----
            total_time = (time.time() - start_time) / 60
            fps        = benchmark_fps(model, DEVICE)
            params     = count_parameters(model) / 1e6

            results = {
                "Model":          config['name'],
                "mIoU":           round(best_miou, 4),
                "Precision":      round(val_prec, 4),
                "Recall":         round(val_rec, 4),
                "FPS":            round(fps, 2),
                "Params_M":       round(params, 2),
                "Train_Time_min": round(total_time, 2),
                "Batch_Size":     batch_size,
                "Grad_Ckpt":      grad_ckpt,
                "AMP":            USE_AMP,
                "Epochs":         NUM_EPOCHS,
                "Date":           datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

            print(f"✅ {config['name']} completed → mIoU: {best_miou:.4f} | FPS: {fps:.1f}")
            free_gpu_memory(model, optimizer, scaler)
            return results

        except torch.cuda.OutOfMemoryError as oom:
            print(f"⚠️  OOM on attempt {attempt+1}: {oom}")
            free_gpu_memory(model, optimizer, scaler)

            if attempt >= MAX_RETRIES:
                print(f"❌ {config['name']} failed after {MAX_RETRIES+1} attempts.")
                raise

            # Escalation strategy:
            #   1st OOM  → enable gradient checkpointing, keep batch size
            #   2nd OOM  → also halve batch size
            #   3rd OOM  → halve batch size again
            if not grad_ckpt:
                grad_ckpt = True
                print(f"   → Retrying with gradient checkpointing enabled "
                      f"(batch_size={batch_size})")
            else:
                batch_size = max(1, batch_size // 2)
                print(f"   → Retrying with batch_size={batch_size}")


# ===================== MAIN =====================
if __name__ == "__main__":
    try:
        import transformers
    except ImportError:
        raise SystemExit("❌ Please install HuggingFace transformers:\n"
                         "   pip install transformers")

    transform = A.Compose([
        A.Resize(*IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    dataset    = VIL100Dataset(ROOT_DIR, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Total samples: {len(dataset)} | Train: {train_size} | Val: {val_size}")
    print(f"Device: {DEVICE} | AMP: {USE_AMP} | Starting batch size: {BATCH_SIZE}\n")

    # ---- DIAGNOSTIC ----
    import random as _random
    print("[DIAGNOSTIC] Checking mask coverage and JSON structure...")
    _nonzero = 0
    for _i in _random.sample(range(len(dataset)), min(50, len(dataset))):
        _, _mask = dataset[_i]
        if _mask.sum() > 0:
            _nonzero += 1
    with open(str(dataset.samples[0][1])) as _f:
        _sj = json.load(_f)
    print(f"  JSON top-level keys : {list(_sj.keys())}")
    if "lanes" in _sj and len(_sj["lanes"]) > 0:
        print(f"  First lane keys     : {list(_sj['lanes'][0].keys())}")
        print(f"  First lane sample   : {str(_sj['lanes'][0])[:300]}")
    else:
        print(f"  Full JSON (first 600 chars):\n{json.dumps(_sj, indent=2)[:600]}")
    print(f"  Non-zero masks      : {_nonzero}/50")
    print("[DIAGNOSTIC END]\n")
    # ---- END DIAGNOSTIC ----

    all_results = []
    for cfg in MODELS_TO_TEST:
        try:
            result = train_and_evaluate(cfg, train_ds, val_ds,
                                        initial_batch_size=BATCH_SIZE)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Failed to run benchmark for {cfg['name']}. Error: {e}")

    if all_results:
        df       = pd.DataFrame(all_results)
        csv_path = f"vil100_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(csv_path, index=False)

        print("\n" + "="*80)
        print("FINAL COMPARISON TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print(f"\n📊 Results saved to: {csv_path}")
    else:
        print("\n❌ No models were evaluated successfully.")