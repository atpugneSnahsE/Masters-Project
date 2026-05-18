"""YOLO model wrapper for traffic detection."""
import cv2, numpy as np
from ultralytics import YOLO
from typing import List, Tuple

class TrafficYOLO:
    def __init__(self, model_name='yolov8n.pt', conf=0.5):
        self.model = YOLO(model_name)
        self.conf = conf
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """Run YOLO detection.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            List of detections with keys: box, class, conf
        """
        results = self.model(image, conf=self.conf, verbose=False)
        detections = []
        
        for r in results:
            for box in r.boxes:
                detection = {
                    'box': box.xyxy[0].cpu().numpy(),
                    'class': int(box.cls[0]),
                    'conf': float(box.conf[0]),
                    'class_name': self.model.names[int(box.cls[0])]
                }
                detections.append(detection)
        
        return detections
    
    def draw(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw detections on image."""
        for det in detections:
            x1, y1, x2, y2 = det['box'].astype(int)
            conf = det['conf']
            class_name = det['class_name']
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image
    
    def process_video(self, video_path: str, output_path: str = None) -> None:
        """Process video file."""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detect(frame)
            frame = self.draw(frame, detections)
            
            if out:
                out.write(frame)
            
            cv2.imshow('YOLO Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    yolo = TrafficYOLO()
    yolo.process_video('input.mp4', 'output.mp4')