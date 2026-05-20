# save as ~/Downloads/convert_masks.py
import json, os, cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# ── CONFIG ──────────────────────────────────────────────────────────────────
EXPORT_JSON   = os.path.expanduser("~/Downloads/label_studio_export.json")
FRAMES_DIR    = os.path.expanduser("~/Downloads/label_frames/")
OUTPUT_DIR    = os.path.expanduser("~/Downloads/unet_dataset/")
IMG_W, IMG_H  = 640, 368   # must match your actual frame size

# Class mapping  (0 = background)
# Replace LABEL_MAP in convert_masks.py
LABEL_MAP = {
    "line_1":   1,
    "line_2":   2,
    "line_3":   3,
    "line_4":   4,
    "line_5":   5,
    "unknown":  0,   # intersection → background
}
# ────────────────────────────────────────────────────────────────────────────

def make_dirs():
    for split in ("train", "val"):
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/masks/{split}",  exist_ok=True)

def points_to_px(points, img_w, img_h):
    """Label Studio points are in % of image size → convert to pixels."""
    return np.array([[p[0]/100*img_w, p[1]/100*img_h] for p in points],
                    dtype=np.int32)

def export_to_masks(data):
    records = []   # (image_path, mask_path)

    for item in data:
        filename = os.path.basename(item["file_upload"])
        # Label Studio sometimes prepends a hash — strip it
        # e.g.  "a1b2c3d4-frame_0001.jpg" → "frame_0001.jpg"
        if "-" in filename:
            filename = filename.split("-", 1)[1]

        img_path = os.path.join(FRAMES_DIR, filename)
        if not os.path.exists(img_path):
            print(f"  [SKIP] image not found: {filename}")
            continue

        # Read actual image size (safer than assuming 640x368)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)

        annotations = item.get("annotations", [])
        if not annotations:
            print(f"  [SKIP] no annotation: {filename}")
            continue

        results = annotations[0].get("result", [])
        for r in results:
            if r.get("type") != "polygonlabels":
                continue
            val    = r["value"]
            label  = val["polygonlabels"][0]
            cls_id = LABEL_MAP.get(label, 0)
            if cls_id == 0:
                continue   # skip unknown/background polygons

            points_pct = val["points"]          # [[x%, y%], ...]
            poly_px    = points_to_px(points_pct, w, h)
            cv2.fillPoly(mask, [poly_px], color=cls_id)

        records.append((img_path, mask, filename))

    return records

def save_split(records):
    stems      = [r[2] for r in records]
    train_idx, val_idx = train_test_split(
        range(len(records)), test_size=0.15, random_state=42)

    for idx_list, split in [(train_idx, "train"), (val_idx, "val")]:
        for i in idx_list:
            img_path, mask, fname = records[i]
            stem = os.path.splitext(fname)[0]

            # copy image
            shutil.copy(img_path, f"{OUTPUT_DIR}/images/{split}/{fname}")

            # save mask as PNG (single-channel, class IDs)
            cv2.imwrite(f"{OUTPUT_DIR}/masks/{split}/{stem}.png", mask)

        print(f"  {split}: {len(idx_list)} samples")

def write_yaml():
    yaml = f"""path: {OUTPUT_DIR}
train: images/train
val:   images/val

nc: 6
names: [background, line_1, line_2, line_3, line_4, line_5]
"""
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        f.write(yaml)
    print(f"  data.yaml written → {OUTPUT_DIR}/data.yaml")

if __name__ == "__main__":
    make_dirs()
    print("Loading export JSON...")
    data = json.load(open(EXPORT_JSON))
    print(f"  {len(data)} items found")

    print("Converting polygons → masks...")
    records = export_to_masks(data)
    print(f"  {len(records)} valid frames converted")

    print("Splitting train/val...")
    save_split(records)

    write_yaml()

    print("\n✅ Done! Dataset ready at:", OUTPUT_DIR)
    print("   Verify a few masks with:")
    print("   python3 ~/Downloads/verify_masks.py")