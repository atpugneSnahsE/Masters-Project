"""Extract frames from video files."""
import cv2, os
from pathlib import Path

def extract_frames(video_path, output_dir, interval=1, max_frames=None):
    """Extract frames from video.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        interval: Extract every Nth frame
        max_frames: Maximum frames to extract (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames/fps:.1f}s")
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            filename = f"{output_dir}/frame_{extracted_count:06d}.jpg"
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted_count += 1
            
            if extracted_count % 100 == 0:
                print(f"  Extracted {extracted_count} frames...")
            
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Done! Extracted {extracted_count} frames to {output_dir}")

if __name__ == '__main__':
    extract_frames('video.mp4', 'frames/', interval=1, max_frames=500)