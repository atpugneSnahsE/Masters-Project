import carla
import numpy as np
import os
import json
import cv2

# ---------- OUTPUT ----------
os.makedirs("data/rgb", exist_ok=True)
os.makedirs("data/mask", exist_ok=True)
os.makedirs("data/meta", exist_ok=True)

frame_id = 0

# ---------- CONNECT ----------
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
bp_lib = world.get_blueprint_library()

# ---------- SPAWN VEHICLE ----------
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

# ---------- CAMERA SETUP ----------
transform = carla.Transform(
    carla.Location(x=0.3, z=1.3),
    carla.Rotation(pitch=-5.0)
)

# RGB camera
rgb_bp = bp_lib.find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', '1024')
rgb_bp.set_attribute('image_size_y', '512')
rgb_cam = world.spawn_actor(rgb_bp, transform, attach_to=vehicle)

# Segmentation camera
seg_bp = bp_lib.find('sensor.camera.semantic_segmentation')
seg_bp.set_attribute('image_size_x', '1024')
seg_bp.set_attribute('image_size_y', '512')
seg_cam = world.spawn_actor(seg_bp, transform, attach_to=vehicle)

latest_rgb = None
latest_seg = None

# ---------- CALLBACKS ----------
def rgb_callback(image):
    global latest_rgb
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    latest_rgb = array[:, :, :3]

def seg_callback(image):
    global latest_seg
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    # Correct channel for class IDs
    latest_seg = array[:, :, 2]

rgb_cam.listen(rgb_callback)
seg_cam.listen(seg_callback)

# ---------- MAIN LOOP ----------
try:
    while True:
        if latest_rgb is not None and latest_seg is not None:

            # Save RGB
            rgb_path = f"data/rgb/{frame_id:06d}.png"
            cv2.imwrite(rgb_path, latest_rgb)

            # Lane marking class (FOUND = 24)
            LANE_MARKING = 24
            mask = (latest_seg == LANE_MARKING).astype(np.uint8) * 255

            mask_path = f"data/mask/{frame_id:06d}.png"
            cv2.imwrite(mask_path, mask)

            # Metadata
            wp = world.get_map().get_waypoint(vehicle.get_location())

            meta = {
                "lane_id": wp.lane_id,
                "road_id": wp.road_id,
                "lane_width": wp.lane_width,
                "left_marking": str(wp.left_lane_marking.type),
                "right_marking": str(wp.right_lane_marking.type)
            }

            with open(f"data/meta/{frame_id:06d}.json", "w") as f:
                json.dump(meta, f, indent=2)

            print("Saved frame:", frame_id)
            frame_id += 1

except KeyboardInterrupt:
    pass

finally:
    rgb_cam.stop()
    seg_cam.stop()
    vehicle.destroy()
