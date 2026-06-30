import sys

# ---- FIX timm / smp COMPATIBILITY ----
import timm.models.regnet
if not hasattr(timm.models.regnet, "RegNetCfg"):
    class RegNetCfg:
        pass
    timm.models.regnet.RegNetCfg = RegNetCfg
# -------------------------------------

import carla
import torch
import numpy as np
import cv2
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# ---------------- CARLA SETUP ----------------
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
bp_lib = world.get_blueprint_library()

vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
spawn_points = world.get_map().get_spawn_points()

vehicle = None
for sp in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, sp)
    if vehicle is not None:
        break

if vehicle is None:
    raise RuntimeError("Vehicle spawn failed")

vehicle.set_autopilot(True)

camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1024')
camera_bp.set_attribute('image_size_y', '512')

transform = carla.Transform(
    carla.Location(x=0.3, z=1.3),
    carla.Rotation(pitch=-5.0)
)

camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)

# ---------------- MODEL ----------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=6
)

weights = torch.load('/home/vgtu/best_lane_model.pth', map_location='cpu')
model.load_state_dict(weights)
model.eval()

# ---------------- TRANSFORM ----------------
transform_fn = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 512)),
    T.ToTensor()
])

# ---------------- SHARED BUFFER ----------------
latest_frame = None

# ---------------- UTIL ----------------
def carla_to_numpy(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    return rgb

# -------- MULTI-CLASS VISUALIZATION --------
colors = [
    [0, 0, 0],        # class 0
    [255, 0, 0],      # class 1
    [0, 255, 0],      # class 2
    [0, 0, 255],      # class 3
    [255, 255, 0],    # class 4
    [0, 255, 255]     # class 5
]

def visualize(rgb, pred):
    pred = np.argmax(pred, axis=0)
    pred = cv2.resize(pred.astype(np.uint8), (rgb.shape[1], rgb.shape[0]))

    color_mask = np.zeros_like(rgb)

    for c in range(6):
        color_mask[pred == c] = colors[c]

    overlay = cv2.addWeighted(rgb, 0.7, color_mask, 0.3, 0)

    cv2.imshow("Lane Classes", overlay)
    cv2.waitKey(1)

# ---------------- CALLBACK ----------------
def process_image(image):
    global latest_frame

    rgb = carla_to_numpy(image)
    inp = transform_fn(rgb).unsqueeze(0)

    with torch.no_grad():
        output = model(inp)

    pred = output.squeeze(0).cpu().numpy()
    latest_frame = (rgb, pred)

camera.listen(process_image)

# ---------------- MAIN LOOP ----------------
try:
    while True:
        if latest_frame is not None:
            rgb, pred = latest_frame
            visualize(rgb, pred)

        if cv2.waitKey(1) == 27:
            break

finally:
    camera.stop()
    vehicle.destroy()
    cv2.destroyAllWindows()
