import carla
import numpy as np
import cv2
import os
import random
import json
import time
from queue import Queue, Empty
from datetime import datetime

# ========================= CONFIG =========================
DATA_ROOT = "lane_dataset"
RGB_DIR = os.path.join(DATA_ROOT, "rgb")
MASK_DIR = os.path.join(DATA_ROOT, "mask")
META_DIR = os.path.join(DATA_ROOT, "metadata")

for d in [RGB_DIR, MASK_DIR, META_DIR]:
    os.makedirs(d, exist_ok=True)

IMAGE_W = 640
IMAGE_H = 360
FOV = 100                    
SAVE_EVERY = 3
TRAFFIC = 25

# Single-town target execution configuration
CURRENT_TOWN = "Town05"  # Options: Town01, Town02, Town03, Town04, Town05
SAMPLES_PER_CONDITION = 600   

# Classes
CLASS_BG = 0
CLASS_EGO = 1
CLASS_LEFT_DASHED = 2
CLASS_LEFT_SOLID = 3
CLASS_RIGHT_DASHED = 4
CLASS_RIGHT_SOLID = 5
CLASS_OTHER = 6
CLASS_STOP = 7
CLASS_CROSS = 8

WEATHERS = {
    "clear_day":    carla.WeatherParameters(cloudiness=10,  precipitation=0,   sun_altitude_angle=70),
    "overcast_day": carla.WeatherParameters(cloudiness=75,  precipitation=0,   sun_altitude_angle=45),
    "night_clear":  carla.WeatherParameters(cloudiness=15,  precipitation=0,   fog_density=0,    sun_altitude_angle=-80),
    "night_fog":    carla.WeatherParameters(cloudiness=35,  precipitation=0,   fog_density=50,   fog_distance=40, sun_altitude_angle=-72),
    "night_rain":   carla.WeatherParameters(cloudiness=60,  precipitation=35,  fog_density=35,   wetness=55, sun_altitude_angle=-75),
}

# Tracking dict to make sure every condition hits exactly the target quota
condition_counts = {k: 0 for k in WEATHERS.keys()}

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(np.radians(fov / 2.0)))
    K = np.identity(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

K = build_projection_matrix(IMAGE_W, IMAGE_H, FOV)

def world_to_image(location, camera_transform):
    try:
        world_2_camera = np.array(camera_transform.get_inverse_matrix())
        point = np.array([location.x, location.y, location.z, 1.0])
        point_camera = np.dot(world_2_camera, point)[:3]
        point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])
        
        if point_camera[2] <= 0:
            return None
        
        point_img = np.dot(K, point_camera)
        point_img /= point_img[2]
        
        x = int(point_img[0])
        y = int(point_img[1])
        if x < 0 or x >= IMAGE_W or y < 0 or y >= IMAGE_H:
            return None
        return (x, y)
    except:
        return None

def lane_class(marking_type, side):
    if marking_type is None:
        return None
    dashed = {carla.LaneMarkingType.Broken, carla.LaneMarkingType.BrokenBroken, carla.LaneMarkingType.BrokenSolid}
    solid = {carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidBroken, carla.LaneMarkingType.SolidSolid}
    
    if marking_type in dashed:
        return CLASS_LEFT_DASHED if side == "left" else CLASS_RIGHT_DASHED
    if marking_type in solid:
        return CLASS_LEFT_SOLID if side == "left" else CLASS_RIGHT_SOLID
    return None

def simulate_headlights(rgb, is_night):
    if not is_night:
        return rgb
    overlay = rgb.copy()
    h, w = rgb.shape[:2]
    center = (w//2, int(h * 0.78))
    cv2.circle(overlay, center, 90, (255, 255, 240), -1)
    rgb = cv2.addWeighted(rgb, 0.75, overlay, 0.25, 0)
    rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
    return rgb

def sensor_callback(data, queue):
    queue.put(data)

# ====================== MAIN SCRIPT ======================
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)

print(f"\n{'='*60}\nLoading Target: {CURRENT_TOWN}...\n{'='*60}")
world = client.load_world(CURRENT_TOWN)

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)
tm.set_global_distance_to_leading_vehicle(2.5)

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# Ego vehicle
ego_bp = bp_lib.find("vehicle.tesla.model3")
ego = world.try_spawn_actor(ego_bp, random.choice(spawn_points))
if ego is None:
    raise RuntimeError(f"Critical error: Failed to spawn ego vehicle in {CURRENT_TOWN}.")

ego.set_autopilot(True, tm.get_port())

# Traffic
traffic_actors = []
veh_bps = [bp for bp in bp_lib.filter("vehicle.*") 
           if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4]

for _ in range(TRAFFIC):
    sp = random.choice(spawn_points)
    v = world.try_spawn_actor(random.choice(veh_bps), sp)
    if v:
        v.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(v, random.uniform(-25, 20))
        traffic_actors.append(v)

# Cameras Setup
cam_tf = carla.Transform(carla.Location(x=0.25, y=0.0, z=1.45), carla.Rotation(pitch=-5))
rgb_bp = bp_lib.find("sensor.camera.rgb")
seg_bp = bp_lib.find("sensor.camera.semantic_segmentation")

for bp in [rgb_bp, seg_bp]:
    bp.set_attribute("image_size_x", str(IMAGE_W))
    bp.set_attribute("image_size_y", str(IMAGE_H))
    bp.set_attribute("fov", str(FOV))

rgb_bp.set_attribute("bloom_intensity", "0.75")
rgb_bp.set_attribute("exposure_compensation", "-0.4")

rgb_cam = world.spawn_actor(rgb_bp, cam_tf, attach_to=ego)
seg_cam = world.spawn_actor(seg_bp, cam_tf, attach_to=ego)

rgb_queue = Queue()
seg_queue = Queue()

rgb_cam.listen(lambda data: sensor_callback(data, rgb_queue))
seg_cam.listen(lambda data: sensor_callback(data, seg_queue))

frame = 0
current_weather_key = "clear_day"
world.set_weather(WEATHERS[current_weather_key])

print(f"Starting balance-targeted collection in {CURRENT_TOWN}...")

try:
    # Stay in loop until all condition sets have hit SAMPLES_PER_CONDITION frames
    while any(count < SAMPLES_PER_CONDITION for count in condition_counts.values()):
        world_frame = world.tick()
        frame += 1

        try:
            rgb_data = rgb_queue.get(timeout=2.0)
            seg_data = seg_queue.get(timeout=2.0)
            
            while rgb_data.frame < world_frame:
                rgb_data = rgb_queue.get(timeout=2.0)
            while seg_data.frame < world_frame:
                seg_data = seg_queue.get(timeout=2.0)
        except Empty:
            print("Skipping frame: Sensor lag detected.")
            continue

        # Dynamic balanced weather rotation strategy
        if frame % 400 == 0:
            incomplete_weathers = [k for k, v in condition_counts.items() if v < SAMPLES_PER_CONDITION]
            if incomplete_weathers:
                current_weather_key = random.choice(incomplete_weathers)
                world.set_weather(WEATHERS[current_weather_key])
                print(f"\n→ Rotated weather to: [{current_weather_key.upper()}]")
                print(f"Current Quotas Status: {condition_counts}\n")

        # Skip recording if the active condition is already full
        if condition_counts[current_weather_key] >= SAMPLES_PER_CONDITION:
            continue

        if frame % SAVE_EVERY != 0:
            continue

        is_night = "night" in current_weather_key

        # Parse frame matrices
        arr_rgb = np.frombuffer(rgb_data.raw_data, dtype=np.uint8).reshape((IMAGE_H, IMAGE_W, 4))
        rgb_frame = arr_rgb[:, :, :3].copy()
        
        arr_seg = np.frombuffer(seg_data.raw_data, dtype=np.uint8).reshape((IMAGE_H, IMAGE_W, 4))
        seg_raw = arr_seg[:, :, 2] 

        wp = world.get_map().get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        mask = np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8)

        # Ego lane calculation with look-ahead spatial validation
        left_pts, right_pts = [], []
        cur_wp = wp
        
        for _ in range(45):
            tf = cur_wp.transform
            loc = tf.location
            right_vector = tf.get_right_vector()
            width = cur_wp.lane_width * 0.92

            left = loc - carla.Location(right_vector * (width / 2.0))
            right = loc + carla.Location(right_vector * (width / 2.0))

            left_pts.append(left)
            right_pts.append(right)

            nxt = cur_wp.next(1.0)
            if not nxt: 
                break
            cur_wp = nxt[0]

        cam_current_transform = rgb_cam.get_transform()

        poly = []
        for p in left_pts:
            pt = world_to_image(p, cam_current_transform)
            if pt: poly.append(pt)
        for p in reversed(right_pts):
            pt = world_to_image(p, cam_current_transform)
            if pt: poly.append(pt)

        # Draw the base road layer if valid points exist
        if len(poly) > 6:
            cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], CLASS_EGO)

        # --- FIX 1: Static Line Thickness ---
        # Constant thickness avoids day/night spatial dataset bias
        line_thickness = 5 
        left_cls = lane_class(wp.left_lane_marking.type if wp.left_lane_marking else None, "left")
        right_cls = lane_class(wp.right_lane_marking.type if wp.right_lane_marking else None, "right")

        for pts, cls in [(left_pts, left_cls), (right_pts, right_cls)]:
            if cls is None: continue
            proj = [world_to_image(p, cam_current_transform) for p in pts]
            proj = [p for p in proj if p]
            if len(proj) > 3:
                cv2.polylines(mask, [np.array(proj, dtype=np.int32)], False, cls, thickness=line_thickness)

        # Fast Crosswalk injection
        mask[seg_raw == 24] = CLASS_CROSS

        # --- FIX 2: Dynamic Occlusion Masking ---
        # Clear out labels where other vehicles or pedestrians hide the road geometry
        # CARLA Semantic tags: 10 = Vehicles, 12 = Pedestrians
        dynamic_objects_mask = (seg_raw == 10) | (seg_raw == 12)
        mask[dynamic_objects_mask] = CLASS_OTHER

        # Night processing injection
        rgb_to_save = simulate_headlights(rgb_frame, is_night)

        # File I/O structure with explicit condition logging
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{CURRENT_TOWN}_{current_weather_key}_{condition_counts[current_weather_key]:04d}_{ts}"

        cv2.imwrite(os.path.join(RGB_DIR, name + ".jpg"), cv2.cvtColor(rgb_to_save, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(MASK_DIR, name + ".png"), mask)

        meta = {
            "town": CURRENT_TOWN,
            "weather": current_weather_key,
            "lane_id": wp.lane_id,
            "road_id": wp.road_id,
            "junction": wp.is_junction,
            "speed_kmh": round(ego.get_velocity().length() * 3.6, 1),
            "is_night": is_night
        }
        with open(os.path.join(META_DIR, name + ".json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Increment specific condition count tracker
        condition_counts[current_weather_key] += 1
        
        if sum(condition_counts.values()) % 50 == 0:
            print(f"Progress Update -> Tracked Totals: {condition_counts}")

except KeyboardInterrupt:
    print("\nStopped early by user request.")
except Exception as e:
    print(f"Runtime error: {e}")
finally:
    print(f"\nCleaning up actors for {CURRENT_TOWN}...")
    for sensor in [rgb_cam, seg_cam]:
        if sensor and sensor.is_alive:
            sensor.stop()
            sensor.destroy()
    
    for actor in [ego] + traffic_actors:
        if actor and actor.is_alive:
            actor.destroy()
            
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    except:
        pass

print(f"\n🎉 Finished execution for {CURRENT_TOWN}! Run Summary: {condition_counts}")