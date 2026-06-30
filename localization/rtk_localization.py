import carla
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

HOST = "localhost"
PORT = 2000

TOWN = "Town03"

TOTAL_STEPS = 10000
SIM_DT = 0.05

TARGET_SPEED = 35.0 / 3.6  # km/h → m/s

WARMUP_STEPS = 100  # warmup steps before starting logging

NUM_VEHICLES = 35
NUM_PEDESTRIANS = 35

OPEN_RTK_STD = 0.03
URBAN_RTK_STD = 0.8
MULTIPATH_PROB = 0.03
MULTIPATH_STD = 1.2

CSV_NAME = "localization_log.csv"

# ==========================================================
# TUNNEL REGION (YOUR VERIFIED COORDINATES)
# ==========================================================

TUNNEL_X_MIN = 145.0
TUNNEL_X_MAX = 255.0

TUNNEL_Y_MIN = -220.0
TUNNEL_Y_MAX = -35.0

# ==========================================================
# CONNECT
# ==========================================================

client = carla.Client(HOST, PORT)
client.set_timeout(30.0)

print("Loading world...")

world = client.load_world(TOWN)
world.wait_for_tick()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = SIM_DT
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)
traffic_manager.global_percentage_speed_difference(-10)

amap = world.get_map()
bp_lib = world.get_blueprint_library()

print(f"{TOWN} loaded")

# ==========================================================
# EKF
# STATE = [x, y, velocity, yaw]
# ==========================================================

class EKF:
    def __init__(self):
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 0.2
        self.Q = np.diag([0.05, 0.05, 0.10, 0.005])
        self.R_open = np.diag([OPEN_RTK_STD**2, OPEN_RTK_STD**2])
        self.R_urban = np.diag([URBAN_RTK_STD**2, URBAN_RTK_STD**2])
        self.initialized = False

    def initialize(self, x, y, v, yaw):
        self.x[0, 0] = x
        self.x[1, 0] = y
        self.x[2, 0] = v
        self.x[3, 0] = yaw
        self.initialized = True

    def predict(self, velocity, yaw_rate, dt):
        if not self.initialized:
            return

        x = self.x[0, 0]
        y = self.x[1, 0]
        yaw = self.x[3, 0]

        # stabilize yaw propagation
        yaw = yaw + yaw_rate * dt
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        x = x + velocity * math.cos(yaw) * dt
        y = y + velocity * math.sin(yaw) * dt

        self.x = np.array([[x], [y], [velocity], [yaw]])

        F = np.eye(4)
        F[0, 2] = math.cos(yaw) * dt
        F[1, 2] = math.sin(yaw) * dt
        F[0, 3] = -velocity * math.sin(yaw) * dt
        F[1, 3] = velocity * math.cos(yaw) * dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, mx, my, urban=False, force_reset=False):
        if force_reset:
            self.x[0, 0] = mx
            self.x[1, 0] = my
            self.P = np.eye(4) * 0.2
            return

        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        z = np.array([[mx], [my]])

        R = self.R_urban if urban else self.R_open
        innovation = z - H @ self.x
        innovation_norm = np.linalg.norm(innovation)

        # robust multipath rejection
        threshold = 12.0 if urban else 5.0
        if innovation_norm > threshold:
            return

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P


# ==========================================================
# ENVIRONMENT DETECTION
# ==========================================================

def in_tunnel(x, y):
    return (TUNNEL_X_MIN <= x <= TUNNEL_X_MAX and TUNNEL_Y_MIN <= y <= TUNNEL_Y_MAX)

def urban_canyon(x, y):
    return (x < 100) or (abs(y) < 100)


# ==========================================================
# SPAWN EGO VEHICLE
# ==========================================================

ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]

spawn = carla.Transform(
    carla.Location(x=250.0, y=-30.0, z=3.0),
    carla.Rotation(yaw=-100.0)
)

ego = world.spawn_actor(ego_bp, spawn)

# Track actors cleanly in separate lists for robust destruction
vehicles_list = [ego]
walkers_list = []
controllers_list = []

ego.set_autopilot(True, traffic_manager.get_port())
traffic_manager.ignore_lights_percentage(ego, 100)
traffic_manager.vehicle_percentage_speed_difference(ego, -15)

print("Ego vehicle spawned")

# ==========================================================
# FORCE LIGHTS GREEN
# ==========================================================

def force_green():
    for actor in world.get_actors():
        if "traffic_light" in actor.type_id:
            actor.set_state(carla.TrafficLightState.Green)
            actor.set_green_time(999999.0)


# ==========================================================
# SPAWN NPC VEHICLES
# ==========================================================

spawn_points = amap.get_spawn_points()
vehicle_bps = bp_lib.filter("vehicle.*")
random.shuffle(spawn_points)

for sp in spawn_points[:NUM_VEHICLES]:
    try:
        bp = random.choice(vehicle_bps)
        npc = world.try_spawn_actor(bp, sp)
        if npc:
            npc.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.ignore_lights_percentage(npc, 100)
            vehicles_list.append(npc)
    except Exception:
        pass

print(f"Spawned {len(vehicles_list) - 1} vehicles")

# ==========================================================
# PEDESTRIANS SAFE SPAWN
# ==========================================================

walker_bps = bp_lib.filter(
    "walker.pedestrian.*"
)

walker_controller_bp = bp_lib.find(
    "controller.ai.walker"
)

walkers = []

spawned_walkers = 0

for _ in range(NUM_PEDESTRIANS):

    try:

        loc = world.get_random_location_from_navigation()

        if loc is None:
            continue

        walker_bp = random.choice(
            walker_bps
        )

        walker = world.try_spawn_actor(
            walker_bp,
            carla.Transform(loc)
        )

        if walker is None:
            continue

        walkers_list.append(walker)

        # tick before controller
        world.tick()

        controller = world.try_spawn_actor(
            walker_controller_bp,
            carla.Transform(),
            walker
        )

        if controller is None:

            walker.destroy()
            continue

        controllers_list.append(controller)

        world.tick()

        controller.start()

        destination = (
            world
            .get_random_location_from_navigation()
        )

        if destination:

            controller.go_to_location(
                destination
            )

        controller.set_max_speed(
            random.uniform(
                0.8,
                1.4
            )
        )

        walkers.append(controller)

        spawned_walkers += 1

        if spawned_walkers % 10 == 0:

            print(
                f"Spawned walkers:"
                f" {spawned_walkers}"
            )

    except Exception as e:

        print(
            "Walker spawn error:",
            e
        )

print(
    f"Spawned {spawned_walkers}"
    f" pedestrians"
)

# ==========================================================
# CAMERA
# ==========================================================

spectator = world.get_spectator()

# ==========================================================
# EKF INIT
# ==========================================================

loc = ego.get_location()
vel = ego.get_velocity()
speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
yaw = math.radians(ego.get_transform().rotation.yaw)

ekf = EKF()
ekf.initialize(loc.x, loc.y, speed, yaw)

# ==========================================================
# DATA LOGGING
# ==========================================================

logs = []

gt_xs = []
gt_ys = []
ekf_xs = []
ekf_ys = []
rtk_xs = []
rtk_ys = []
errors = []

# ==========================================================
# MAIN LOOP
# ==========================================================

print("Running simulation...")

previous_rtk = True

try:
    for step in range(TOTAL_STEPS):
        force_green()
        try:
            world.tick()
        except RuntimeError:
            break

        transform = ego.get_transform()
        loc = transform.location
        gt_x = loc.x
        gt_y = loc.y

        vel = ego.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        ang_vel = ego.get_angular_velocity()
        yaw_rate = math.radians(ang_vel.z)

        # ======================================
        # ENVIRONMENT
        # ======================================

        if in_tunnel(gt_x, gt_y):
            env = "TUNNEL"
            rtk_available = False
        elif urban_canyon(gt_x, gt_y):
            env = "URBAN"
            rtk_available = True
            noise_std = URBAN_RTK_STD
        else:
            env = "OPEN"
            rtk_available = True
            noise_std = OPEN_RTK_STD

        # ======================================
        # RTK MODEL
        # ======================================

        rtk_x, rtk_y = np.nan, np.nan
        if rtk_available:
            rtk_x = gt_x + np.random.normal(0, noise_std)
            rtk_y = gt_y + np.random.normal(0, noise_std)

            # multipath spikes
            if env == "URBAN" and random.random() < MULTIPATH_PROB:
                rtk_x += np.random.normal(0, MULTIPATH_STD)
                rtk_y += np.random.normal(0, MULTIPATH_STD)

        # ======================================
        # EKF
        # ======================================

        ekf.predict(speed, yaw_rate, SIM_DT)

        # tunnel exit recovery
        force_reset = (not previous_rtk and rtk_available)

        if rtk_available and step > WARMUP_STEPS:
            ekf.update(rtk_x, rtk_y, urban=(env == "URBAN"), force_reset=force_reset)

        previous_rtk = rtk_available

        # Secure extraction safeguards against empty matrices
        ekf_x = float(ekf.x[0, 0]) if ekf.x is not None else gt_x
        ekf_y = float(ekf.x[1, 0]) if ekf.x is not None else gt_y
        error = math.sqrt((ekf_x - gt_x)**2 + (ekf_y - gt_y)**2)

        # ======================================
        # CAMERA CONTROL
        # ======================================
        spectator.set_transform(
            carla.Transform(
                loc + carla.Location(x=-10, z=5),
                carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
            )
        )

        # ======================================
        # LOGGING
        # ======================================

        gt_xs.append(gt_x)
        gt_ys.append(gt_y)
        ekf_xs.append(ekf_x)
        ekf_ys.append(ekf_y)
        errors.append(error)

        if rtk_available:
            rtk_xs.append(rtk_x)
            rtk_ys.append(rtk_y)
        else:
            rtk_xs.append(np.nan)
            rtk_ys.append(np.nan)

        logs.append([step, env, gt_x, gt_y, ekf_x, ekf_y, error])

        if step % 50 == 0:
            print("\n----------------")
            print(f"Step {step}")
            print(f"Environment: {env}")
            print(f"Ground Truth: ({gt_x:.2f}, {gt_y:.2f})")
            if rtk_available:
                print(f"RTK: ({rtk_x:.2f}, {rtk_y:.2f})")
            else:
                print("RTK: LOST")
            print(f"EKF: ({ekf_x:.2f}, {ekf_y:.2f})")
            print(f"Error: {error:.3f} m")

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    print("Saving results...")
    if logs:
        df = pd.DataFrame(
            logs,
            columns=["step", "environment", "gt_x", "gt_y", "ekf_x", "ekf_y", "error_m"]
        )
        df.to_csv(CSV_NAME, index=False)

    # ==========================================
    # PLOTS
    # ==========================================
    if errors:
        plt.figure(figsize=(10, 5))
        plt.plot(errors)
        plt.title("Localization Error")
        plt.xlabel("Step")
        plt.ylabel("Error (m)")
        plt.grid(True)
        plt.savefig("error_vs_time.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(errors, bins=50)
        plt.title("Error Distribution")
        plt.xlabel("Error (m)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.savefig("error_histogram.png")
        plt.close()

    if gt_xs:
        plt.figure(figsize=(8, 8))
        plt.plot(gt_xs, gt_ys, label="Ground Truth")
        plt.plot(ekf_xs, ekf_ys, label="EKF")
        plt.scatter(rtk_xs, rtk_ys, s=2, alpha=0.4, label="RTK")
        plt.legend()
        plt.title("Trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.savefig("trajectory.png")
        plt.close()

    print("Cleaning up...")

    # Disable Synchronous mode safely
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(False)
    except Exception:
        pass

    # Orderly & Protected Destruction of Actors
    for controller in controllers_list:
        try:
            if controller.is_alive:
                controller.stop()
                controller.destroy()
        except Exception:
            pass

    for walker in walkers_list:
        try:
            if walker.is_alive:
                walker.destroy()
        except Exception:
            pass

    for vehicle in vehicles_list:
        try:
            if vehicle.is_alive:
                vehicle.destroy()
        except Exception:
            pass

    print("Done")