"""CARLA-based data collection for training."""
import carla, cv2, numpy as np, os
from pathlib import Path

class CARLADataCollector:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.vehicle = None
        self.camera = None
    
    def spawn_vehicle(self):
        """Spawn vehicle in CARLA."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point}")
    
    def attach_camera(self):
        """Attach camera to vehicle."""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '368')
        
        spawn_point = carla.Transform(carla.Location(x=0.5, z=1.2))
        self.camera = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)
        print("Camera attached")
    
    def collect_frames(self, num_frames=100, output_dir='data/raw'):
        """Collect frames from CARLA."""
        os.makedirs(output_dir, exist_ok=True)
        frames = []
        
        def process_image(image):
            frames.append(image)
        
        self.camera.listen(process_image)
        
        for i in range(num_frames):
            self.world.tick()
            if i % 10 == 0:
                print(f"Collected {i}/{num_frames} frames")
        
        self.camera.stop()
        
        # Save frames
        for i, frame in enumerate(frames):
            frame.save_to_disk(f'{output_dir}/frame_{i:06d}.png')
        
        print(f"Saved {len(frames)} frames to {output_dir}")
    
    def cleanup(self):
        """Clean up actors."""
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        print("Cleanup complete")

if __name__ == '__main__':
    collector = CARLADataCollector()
    try:
        collector.spawn_vehicle()
        collector.attach_camera()
        collector.collect_frames(num_frames=500)
    finally:
        collector.cleanup()