"""BEV transformation for video streams."""
import cv2, numpy as np
from bev import BEVTransform

class BEVVideoProcessor:
    def __init__(self, src_points, dst_points, dst_size=(800, 400)):
        self.bev = BEVTransform(src_points, dst_points, dst_size)
        self.dst_size = dst_size
    
    def process_video(self, input_path, output_path, skip_frames=1):
        """Process video and save BEV output."""
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, self.dst_size)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                bev_frame = self.bev.transform(frame)
                out.write(bev_frame)
                
                # Display
                cv2.imshow('BEV Transform', bev_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    src = np.array([[100, 50], [300, 50], [350, 300], [50, 300]], dtype=np.float32)
    dst = np.array([[50, 0], [350, 0], [350, 800], [50, 800]], dtype=np.float32)
    
    processor = BEVVideoProcessor(src, dst, (800, 400))
    processor.process_video('input.mp4', 'output_bev.mp4')