"""Traffic detection with multi-model inference."""
import cv2, numpy as np
from ultralytics import YOLO
import time

class MultiModelDetector:
    def __init__(self):
        self.yolo = YOLO('yolov8n.pt')
        self.conf_threshold = 0.5
        self.class_names = self.yolo.model.names
    
    def detect(self, frame):
        """Run YOLO inference."""
        results = self.yolo(frame, conf=self.conf_threshold, verbose=False)
        return results
    
    def draw_detections(self, frame, results):
        """Draw bounding boxes on frame."""
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                
                color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{self.class_names[cls_id]}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    def process_video(self, video_path, output_path=None):
        """Process video file."""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.detect(frame)
            frame = self.draw_detections(frame, results)
            
            if out:
                out.write(frame)
            
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = MultiModelDetector()
    detector.process_video('test_video.mp4', 'output.mp4')