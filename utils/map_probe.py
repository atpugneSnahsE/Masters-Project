#!/usr/bin/env python

import carla
import time

def main():
    client = None
    try:
        # 1. Connect to the CARLA simulator
        # Default port is 2000. Change 'localhost' if running on a server.
        print("Connecting to CARLA simulator...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # 10 second timeout threshold
        
        # 2. Get the current world
        world = client.get_world()
        print("Connected successfully!")
        
        # 3. Grab the spectator object from the world
        spectator = world.get_spectator()
        
        # 4. Get its exact transform data
        spectator_transform = spectator.get_transform()
        
        # Extract location and rotation components
        loc = spectator_transform.location
        rot = spectator_transform.rotation
        
        # 5. Print the precise global world coordinates cleanly
        print("\n" + "="*40)
        print("CURRENT SPECTATOR CAMERA POSE")
        print("="*40)
        print(f"Location: X={loc.x:.3f}, Y={loc.y:.3f}, Z={loc.z:.3f}")
        print(f"Rotation: Pitch={rot.pitch:.3f}, Yaw={rot.yaw:.3f}, Roll={rot.roll:.3f}")
        print("="*40 + "\n")

    except RuntimeError as e:
        print(f"ERROR: Could not connect to CARLA. Is the simulator running? \nDetails: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Script execution finished.")

if __name__ == '__main__':
    main()