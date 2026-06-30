from ultralytics import YOLO
import cv2, os

model = YOLO(os.path.expanduser('~/Downloads/best.pt'))
cap = cv2.VideoCapture(os.path.expanduser('~/Downloads/road_clip1.mp4'))

# Scan through video every 300 frames looking for detections
found = False
for frame_no in range(0, 10000, 300):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.25, verbose=False)
    if len(results[0].boxes) > 0:
        detections = [(model.names[int(b.cls)], float(b.conf)) 
                      for b in results[0].boxes]
        print(f"Frame {frame_no}: {detections}")
        results[0].save(filename=os.path.expanduser(f'~/Downloads/sign_frame_{frame_no}.png'))
        found = True
        if len([d for d in detections if 'Speed' in d[0] or 'Sign' in d[0] or 'Stop' in d[0]]) > 0:
            break  # found a real sign

cap.release()
if not found:
    print("No detections in first 10000 frames — try road_clip1.mp4")