import sys

# ---- FIX timm ----
import timm.models.regnet
if not hasattr(timm.models.regnet, "RegNetCfg"):
    class RegNetCfg:
        pass
    timm.models.regnet.RegNetCfg = RegNetCfg

import carla
import torch
import numpy as np
import cv2
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# ---------- MODEL ----------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(torch.load("lane_model.pth", map_location="cpu"))
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 512)),
    T.ToTensor()
])

# ---------- CARLA ----------
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
bp = world.get_blueprint_library()

vehicle_bp = bp.filter("vehicle.tesla.model3")[0]
spawn_points = world.get_map().get_spawn_points()

vehicle = None
for sp in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, sp)
    if vehicle:
        break

if vehicle is None:
    raise RuntimeError("Vehicle spawn failed")

vehicle.set_autopilot(True)

camera_bp = bp.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "1024")
camera_bp.set_attribute("image_size_y", "512")

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=0.3, z=1.3)),
    attach_to=vehicle
)

latest_frame = None

# ---------- UTILS ----------
def carla_to_rgb(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3][:, :, ::-1]

# ---------- CALLBACK ----------
def process(image):
    global latest_frame

    rgb = carla_to_rgb(image)

    inp = transform(rgb).unsqueeze(0)

    with torch.no_grad():
        pred = model(inp)

    pred = torch.sigmoid(pred).squeeze().numpy()
    pred = (pred > 0.5).astype(np.uint8) * 255

    pred = cv2.resize(pred, (rgb.shape[1], rgb.shape[0]))

    latest_frame = (rgb, pred)

camera.listen(process)

# ---------- MAIN LOOP ----------
try:
    while True:
        if latest_frame is not None:
            rgb, mask = latest_frame

            overlay = rgb.copy()
            overlay[mask == 255] = [0, 255, 0]

            # small mask preview
            small = cv2.resize(mask, (200, 100))
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            overlay[10:110, 10:210] = small

            cv2.imshow("Lane Detection", overlay)

        if cv2.waitKey(1) == 27:
            break

finally:
    camera.stop()
    vehicle.destroy()
    cv2.destroyAllWindows()
