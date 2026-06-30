import carla
import random
import numpy as np
import cv2
import time

# ==========================================================
# CONNECT TO CARLA
# ==========================================================
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

print("Loading Town04...")
world = client.load_world("Town04")

# ==========================================================
# SYNCHRONOUS MODE
# ==========================================================
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# ==========================================================
# NIGHT WEATHER
# ==========================================================
weather = carla.WeatherParameters(
    cloudiness=20.0,
    precipitation=0.0,
    fog_density=0.0,
    wetness=0.0,
    sun_altitude_angle=-35.0
)

world.set_weather(weather)

blueprints = world.get_blueprint_library()

# ==========================================================
# VEHICLE
# Tesla has good windshield geometry
# ==========================================================
vehicle_bp = blueprints.find("vehicle.tesla.model3")

spawn_points = world.get_map().get_spawn_points()

# Urban spawn in Town04
spawn_transform = spawn_points[20]

vehicle = world.try_spawn_actor(
    vehicle_bp,
    spawn_transform
)

if vehicle is None:
    raise RuntimeError("Vehicle spawn failed")

print("Vehicle spawned")

# ==========================================================
# STRONG HEADLIGHTS
# KEEPING SAME LOGIC
# ==========================================================
vehicle.set_light_state(
    carla.VehicleLightState(
        carla.VehicleLightState.Position
        | carla.VehicleLightState.LowBeam
        | carla.VehicleLightState.HighBeam
        | carla.VehicleLightState.Fog
    )
)

print("Headlights ON")

# ==========================================================
# AUTOPILOT
# ==========================================================
vehicle.set_autopilot(True)

traffic_manager.vehicle_percentage_speed_difference(
    vehicle,
    0
)

# ==========================================================
# CAMERA
# Rear view mirror location
# Top centre windshield
# ==========================================================
camera_bp = blueprints.find("sensor.camera.rgb")

camera_bp.set_attribute("image_size_x", "1280")
camera_bp.set_attribute("image_size_y", "720")
camera_bp.set_attribute("fov", "100")

camera_transform = carla.Transform(
    carla.Location(
        x=0.25,
        y=0.0,
        z=1.45
    ),
    carla.Rotation(
        pitch=-5
    )
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle
)

latest_frame = None


# ==========================================================
# CAMERA CALLBACK
# ==========================================================
def process_image(image):
    global latest_frame

    img = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    img = img.reshape(
        (image.height, image.width, 4)
    )

    rgb = img[:, :, :3]
    rgb = rgb[:, :, ::-1]

    # slight brightness boost
    rgb = cv2.convertScaleAbs(
        rgb,
        alpha=1.25,
        beta=8
    )

    latest_frame = rgb


camera.listen(process_image)

# ==========================================================
# LET VEHICLE START MOVING
# ==========================================================
print("Waiting for vehicle movement...")

for _ in range(80):
    world.tick()

time.sleep(2)

# ==========================================================
# MAIN LOOP
# ==========================================================
try:

    while True:

        world.tick()

        if latest_frame is not None:

            frame = latest_frame.copy()

            velocity = vehicle.get_velocity()

            speed = 3.6 * np.sqrt(
                velocity.x**2 +
                velocity.y**2 +
                velocity.z**2
            )

            cv2.putText(
                frame,
                f"Speed: {speed:.1f} km/h",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "CARLA Night Driving - Town04",
                frame
            )

        key = cv2.waitKey(1)

        if key == 27:
            break

finally:

    print("Cleaning actors...")

    camera.stop()

    if camera.is_alive:
        camera.destroy()

    if vehicle.is_alive:
        vehicle.destroy()

    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(False)

    cv2.destroyAllWindows()

    print("Done.")