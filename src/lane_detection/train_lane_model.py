import sys

# ---- FIX timm / smp COMPATIBILITY ----
import timm.models.regnet
if not hasattr(timm.models.regnet, "RegNetCfg"):
    class RegNetCfg:
        pass
    timm.models.regnet.RegNetCfg = RegNetCfg
# -------------------------------------

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# -------- DATASET --------
class LaneDataset(Dataset):
    def __init__(self, root):
        self.rgb_paths = sorted(os.listdir(f"{root}/rgb"))
        self.mask_paths = sorted(os.listdir(f"{root}/mask"))
        self.root = root

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 512)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = cv2.imread(f"{self.root}/rgb/{self.rgb_paths[idx]}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb,(512,256))
        mask = cv2.imread(f"{self.root}/mask/{self.mask_paths[idx]}", 0)
        mask= cv2.resize(mask,(512,256), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        rgb = self.transform(rgb)
        mask = torch.tensor(mask).unsqueeze(0)
        return rgb, mask

# -------- LOAD DATA --------
dataset = LaneDataset("data")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# -------- MODEL --------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------- LOSS --------
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------- TRAIN --------
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss/len(loader)
    print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")

# -------- SAVE --------
torch.save(model.state_dict(), "lane_model.pth")
