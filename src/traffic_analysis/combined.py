import cv2, numpy as np, os, sys
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
conf = 0.5  # confidence
color_dict = {0: (255,0,0), 1: (0,255,0), 2: (0,0,255)}  # BGR

def run():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, conf=conf)
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                
                color = color_dict.get(cls, (255,255,255))
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f'{conf_score:.2f}', (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()