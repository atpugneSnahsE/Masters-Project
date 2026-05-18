"""Test YOLO model on single image."""
import cv2
from ultralytics import YOLO

def test_single_image(img_path, model_path='yolov8n.pt'):
    """Load model and run inference on image."""
    model = YOLO(model_path)
    results = model(img_path, conf=0.5)
    
    img = cv2.imread(img_path)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'{conf:.2f}', (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_single_image('test_image.jpg')