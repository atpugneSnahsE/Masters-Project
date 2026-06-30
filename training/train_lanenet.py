import os, json, cv2, glob, torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 1. EXACT PATHS FOR YOUR MACBOOK
# ─────────────────────────────────────────────────────────────────────────────
TUSIMPLE_TRAIN = "/Users/mac/Downloads/archive/TUSimple/train_set"
MASK_OUTPUT    = "/Users/mac/Downloads/archive/TUSimple/generated_masks"
MODEL_SAVE     = "/Users/mac/Downloads/lane_model_final.pth"
DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu" # Use Mac GPU

os.makedirs(MASK_OUTPUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVERTER (Run this first to create the images)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_data():
    # TuSimple JSONs are inside the train_set folder
    label_files = glob.glob(os.path.join(TUSIMPLE_TRAIN, "*.json"))
    if not label_files:
        print(f"Error: No JSON files found in {TUSIMPLE_TRAIN}")
        return

    print(f"Converting {len(label_files)} label files...")
    for j_file in label_files:
        with open(j_file, 'r') as f:
            for line in tqdm(f.readlines(), desc=f"Masking {os.path.basename(j_file)}"):
                data = json.loads(line)
                raw_path = data['raw_file'] # e.g. "clips/0313-1/..."
                lanes = data['lanes']
                y_samples = data['h_samples']
                
                mask = np.zeros((720, 1280), dtype=np.uint8)
                for i, lane in enumerate(lanes):
                    cls_id = i + 1
                    if cls_id > 5: break
                    pts = [(x, y) for x, y in zip(lane, y_samples) if x != -2]
                    for k in range(len(pts)-1):
                        cv2.line(mask, pts[k], pts[k+1], cls_id, thickness=10)
                
                # Create a flat filename for the mask
                mask_name = raw_path.replace("/", "_").replace(".jpg", ".png")
                cv2.imwrite(os.path.join(MASK_OUTPUT, mask_name), mask)

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET & TRAINING
# ─────────────────────────────────────────────────────────────────────────────
class LaneDataset(Dataset):
    def __init__(self, root, mask_dir):
        self.root = root
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.tf = A.Compose([
            A.Resize(368, 640),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self): return len(self.mask_paths)

    def __getitem__(self, idx):
        m_path = self.mask_paths[idx]
        # Convert flat filename back to folder structure
        # clips_0313-1_...png -> clips/0313-1/...jpg
        rel_path = os.path.basename(m_path).replace("_", "/", 3).replace(".png", ".jpg")
        img_path = os.path.join(self.root, rel_path)
        
        img = cv2.imread(img_path)
        if img is None: # Fallback if path replace logic is too aggressive
            parts = os.path.basename(m_path).replace(".png", ".jpg").split("_")
            img_path = os.path.join(self.root, *parts)
            img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        aug = self.tf(image=img, mask=mask)
        return aug['image'], aug['mask'].long()

def train():
    ds = LaneDataset(TUSIMPLE_TRAIN, MASK_OUTPUT)
    if len(ds) == 0:
        print("Still 0 samples. Ensure prepare_data() ran and MASK_OUTPUT has files.")
        return
        
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    model = smp.Unet("resnet34", classes=6, decoder_attention_type='scse').to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = smp.losses.DiceLoss(mode='multiclass')

    print(f"Training started on {DEVICE}...")
    for epoch in range(10):
        model.train()
        for imgs, msks in loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), msks)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), MODEL_SAVE)

if __name__ == "__main__":
    # 1. Run this first. If it finishes, comment it out and run Step 2.
    prepare_data() 
    
    # 2. Run this to start training
    train()