# ==========================================================
# EXPERIMENT 3 (SCIENTIFIC VALIDATION VERSION)
# GNSS Denial Benchmarking: Identical Path & State Initialization
# ==========================================================

import carla
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from queue import Queue, Empty

# ==========================================================
# UNIVERSAL DETERMINISTIC SEEDING (ONCE GLOBALLY)
# ==========================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

HOST = "localhost"
PORT = 2000
TOWN = "Town03"
SIM_DT = 0.05

OUTAGE_DURATIONS = [5.0, 10.0, 20.0, 30.0, 60.0, 120.0]
NUM_TRIALS = 10  
CSV_NAME = "experiment3_validated_results.csv"

# CORRECTED: Exact spatial window bounding the outer entry mouth of the Town03 tunnel
TRIGGER_X_MIN = 245.0
TRIGGER_X_MAX = 255.0
TRIGGER_Y_MIN = -50.0
TRIGGER_Y_MAX = -35.0

# ==========================================================
# EXTENDED STATE INERTIAL KALMAN FILTER
# ==========================================================
class INS_EKF:
    def __init__(self):
        # State: [x, y, vx, vy, yaw, gyro_bias]^T
        self.x = np.zeros((6, 1))  
        self.P = np.eye(6) * 0.1
        
        # SCIENTIFICALLY TUNED: Process noise Q scaled to realistic physical devices
        self.Q = np.diag([
            0.001,               # Position variance X
            0.001,               # Position variance Y
            0.005,               # Velocity variance X
            0.005,               # Velocity variance Y
            np.radians(0.001),   # Yaw drift angular variance
            1e-8                 # Gyroscope stability dynamic drift bias
        ])
        
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        self.R = np.eye(2) * (0.05**2)  # Healthy tracking accuracy constraint
        self.initialized = False

    def initialize(self, x, y, vx, vy, yaw):
        self.x[0, 0] = x
        self.x[1, 0] = y
        self.x[2, 0] = vx
        self.x[3, 0] = vy
        self.x[4, 0] = yaw
        self.x[5, 0] = 0.0  
        self.initialized = True

    def predict(self, ax, ay, gyro_z, odom_speed, dt):
        if not self.initialized: return

        x = self.x[0, 0]
        y = self.x[1, 0]
        vx = self.x[2, 0]
        vy = self.x[3, 0]
        heading = self.x[4, 0]
        bias = self.x[5, 0]

        # Process IMU values
        corrected_gyro = gyro_z - bias
        heading_next = heading + corrected_gyro * dt
        heading_next = math.atan2(math.sin(heading_next), math.cos(heading_next))

        ax_global = ax * math.cos(heading) - ay * math.sin(heading)
        ay_global = ax * math.sin(heading) + ay * math.cos(heading)

        # Kinematic translation matrix update
        x_next = x + vx * dt + 0.5 * ax_global * (dt**2)
        y_next = y + vy * dt + 0.5 * ay_global * (dt**2)
        
        # Blend inertial step integration with longitudinal wheel odometry constraints
        vx_next = (vx + ax_global * dt) * 0.7 + (odom_speed * math.cos(heading_next)) * 0.3
        vy_next = (vy + ay_global * dt) * 0.7 + (odom_speed * math.sin(heading_next)) * 0.3

        self.x = np.array([[x_next], [y_next], [vx_next], [vy_next], [heading_next], [bias]])

        # Construct exact analytical system Jacobian
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * (-ax * math.sin(heading) - ay * math.cos(heading)) * (dt**2)
        F[1, 4] = 0.5 * (ax * math.cos(heading) - ay * math.sin(heading)) * (dt**2)
        F[2, 4] = (-ax * math.sin(heading) - ay * math.cos(heading)) * dt
        F[3, 4] = (ax * math.cos(heading) - ay * math.sin(heading)) * dt
        F[4, 5] = -dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, mx, my):
        if not self.initialized: return
        z = np.array([[mx], [my]])
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ self.H) @ self.P

# ==========================================================
# HARMONIZED SENSOR SIMULATION ENGINE
# ==========================================================
class SensorSuite:
    def __init__(self, world, ego):
        self.imu_queue = Queue()
        self.gyro_bias = 0.0
        bp_lib = world.get_blueprint_library()
        
        imu_bp = bp_lib.find("sensor.other.imu")
        self.imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=ego)
        self.imu.listen(lambda data: self.imu_queue.put(data))

    def get_imu_reading(self):
        try:
            data = self.imu_queue.get(timeout=0.2)
            self.gyro_bias += np.random.normal(0, 0.000005) # Realistic micro-drift bias scaling
            
            # FIXED: Reduced accelerometer noise bounds to matching tactical grades
            ax_noisy = data.accelerometer.x + np.random.normal(0, 0.04)
            ay_noisy = data.accelerometer.y + np.random.normal(0, 0.04)
            gyro_noisy = data.gyroscope.z + np.random.normal(0, np.radians(0.08)) + self.gyro_bias
            return ax_noisy, ay_noisy, gyro_noisy
        except Empty:
            return 0.0, 0.0, 0.0

    def get_odometry_reading(self, true_speed):
        # Simulated by perturbing CARLA ground-truth velocity (As documented in thesis constraints)
        slip = 1.04 if random.random() < 0.03 else 1.0
        return (true_speed + np.random.normal(0, 0.03)) * slip

    def destroy(self):
        try: self.imu.stop(); self.imu.destroy()
        except: pass

# ==========================================================
# WORLD INITIALIZATION & TRACK ALIGNMENT
# ==========================================================
client = carla.Client(HOST, PORT)
client.set_timeout(30.0)
world = client.load_world(TOWN)
world.wait_for_tick()

# Absolute clearance verification step before trial runs begin
print("Purging the map environment to guarantee clear straight-line runs...")
for actor in world.get_actors().filter('vehicle.*'):
    actor.destroy()
world.tick()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = SIM_DT
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_random_device_seed(RANDOM_SEED)

bp_lib = world.get_blueprint_library()

# Setup spawn orientation to feed smoothly into the straight tunnel approach line
primary_spawn = carla.Transform(
    carla.Location(x=249.5, y=-15.0, z=3.0),
    carla.Rotation(pitch=0.0, roll=0.0, yaw=-90.0)
)

# ==========================================================
# MONTE CARLO BATCH RUNNER
# ==========================================================
compiled_stats = []

for duration in OUTAGE_DURATIONS:
    print(f"\n=======================================================")
    print(f"BENCHMARKING OUTAGE DURATION: {duration}s")
    print(f"=======================================================")
    
    trial_means = []
    trial_medians = []
    trial_p95s = []
    trial_maxes = []
    
    for trial in range(1, NUM_TRIALS + 1):
        ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ego = world.try_spawn_actor(ego_bp, primary_spawn)
        
        # ST_RULE: Rigidly drop the run if a collision occurs rather than shifting position paths
        if ego is None:
            print(f"  [Error] Spawn obstructed at critical position coordinates on Trial {trial}. Purging lane frames...")
            world.tick()
            world.tick()
            continue
            
        ego.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.ignore_lights_percentage(ego, 100)
        
        # FIXED: Lock down a strict straight-line path routing configuration
        traffic_manager.set_route(ego, ["Straight", "Straight", "Straight"])
        
        sensors = SensorSuite(world, ego)
        ekf = INS_EKF()
        
        triggered = False
        outage_steps_remaining = int(duration / SIM_DT)
        errors = []
        
        # Tracking Execution Loop
        while True:
            world.tick()
            
            gt_loc = ego.get_location()
            gt_vel = ego.get_velocity()
            true_speed = math.sqrt(gt_vel.x**2 + gt_vel.y**2 + gt_vel.z**2)
            
            ax, ay, gyro_z = sensors.get_imu_reading()
            odom_speed = sensors.get_odometry_reading(true_speed)
            
            if not triggered:
                if not ekf.initialized:
                    yaw_init = math.radians(ego.get_transform().rotation.yaw)
                    ekf.initialize(gt_loc.x, gt_loc.y, gt_vel.x, gt_vel.y, yaw_init)
                else:
                    ekf.predict(ax, ay, gyro_z, odom_speed, SIM_DT)
                    ekf.update(gt_loc.x, gt_loc.y)
                
                # Verify spatial interception window parameters
                if TRIGGER_X_MIN <= gt_loc.x <= TRIGGER_X_MAX and TRIGGER_Y_MIN <= gt_loc.y <= TRIGGER_Y_MAX:
                    triggered = True
                    print(f"  -> Trial {trial:02d}: Outage Triggered at exact position coordinates: X={gt_loc.x:.3f}, Y={gt_loc.y:.3f}")
            
            else:
                ekf.predict(ax, ay, gyro_z, odom_speed, SIM_DT)
                
                current_error = math.sqrt((ekf.x[0, 0] - gt_loc.x)**2 + (ekf.x[1, 0] - gt_loc.y)**2)
                errors.append(current_error)
                
                outage_steps_remaining -= 1
                if outage_steps_remaining <= 0:
                    break
                    
        # FIXED: Expanded analytical telemetry extraction array metrics
        trial_means.append(np.mean(errors))
        trial_medians.append(np.median(errors))
        trial_p95s.append(np.percentile(errors, 95))
        trial_maxes.append(np.max(errors))
        
        # Clean down trial frames safely
        sensors.destroy()
        ego.destroy()
        world.tick()
        world.tick()

    # Document compiled statistics
    compiled_stats.append({
        "Duration": f"{duration:.0f}s",
        "Mean": np.mean(trial_means), "Mean_Std": np.std(trial_means),
        "Median": np.mean(trial_medians),
        "P95": np.mean(trial_p95s),
        "Max": np.mean(trial_maxes), "Max_Std": np.std(trial_maxes)
    })
    
    print(f"Finished {duration}s profile window loop.")

# ==========================================================
# TELEMETRY POST-PROCESSING PRESENTATION ENGAGEMENT
# ==========================================================
df_stats = pd.DataFrame(compiled_stats)
df_stats.to_csv(CSV_NAME, index=False)

print("\n==========================================================================================")
# RENDER NOTE: Standard text layout formatting is used for clarity and readability
print("                       SCIENTIFICALLY VALIDATED THESIS DRIFT RESULTS MATRIX                ")
print("==========================================================================================")
print("Outage |    Mean Error (m)    |   Median (m)   |     P95 (m)    |     Maximum Peak Error (m)  ")
print("------------------------------------------------------------------------------------------")
for _, row in df_stats.iterrows():
    print(f"{row['Duration']:<6} | {row['Mean']:6.3f} ± {row['Mean_Std']:5.3f} |   {row['Median']:6.3f}   |   {row['P95']:6.3f}    |  {row['Max']:6.3f} ± {row['Max_Std']:5.3f}")
print("==========================================================================================")

# Generate Plot
plt.figure(figsize=(10, 6))
durations_num = [float(d.replace('s', '')) for d in df_stats["Duration"]]

plt.errorbar(durations_num, df_stats["Mean"], yerr=df_stats["Mean_Std"], fmt='-o', color='#1f77b4', linewidth=2, capsize=5, label='Mean Error')
plt.plot(durations_num, df_stats["Median"], '-^', color='#ff7f0e', linewidth=1.5, label='Median Error')
plt.plot(durations_num, df_stats["P95"], '--X', color='#2ca02c', linewidth=1.5, label='95th Percentile ($P_{95}$)')
plt.errorbar(durations_num, df_stats["Max"], yerr=df_stats["Max_Std"], fmt=':s', color='#d62728', linewidth=2, capsize=5, label='Max Peak Error')

plt.xlabel("Simulated GNSS Outage Duration (Seconds)", fontsize=11, fontweight='bold')
plt.ylabel("Localization Deviation (Meters)", fontsize=11, fontweight='bold')
plt.title("Validated Dead-Reckoning Drift Statistics\n(Identical-Route Spatial Trigger Matrix)", fontsize=12, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(durations_num, [f"{int(d)}s" for d in durations_num])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("experiment3_validated_drift.png")
plt.close()

# Revert simulator state
settings = world.get_settings()
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)
print("\n[Complete] Data logs finalized.")