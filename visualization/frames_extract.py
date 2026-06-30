import cv2
import numpy as np
import random
import os

def extract_random_frames(video_path, num_frames=4, seed=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    random.seed(seed)
    indices = sorted(random.sample(range(total_frames), num_frames))

    frames = []
    timestamps = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
        timestamps.append(idx / fps)

    cap.release()
    return frames, timestamps


def stack_frames_horizontally(frames, timestamps):
    h = min(f.shape[0] for f in frames)
    resized = []

    for f, t in zip(frames, timestamps):
        scale = h / f.shape[0]
        new_w = int(f.shape[1] * scale)
        f_resized = cv2.resize(f, (new_w, h))

        label = f"{t:.2f}s"
        cv2.putText(f_resized, label, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        resized.append(f_resized)

    return np.hstack(resized)


if __name__ == "__main__":
    video_path = "/Users/mac/Downloads/road_clip1_annotated_v5.mp4"
    output_path = os.path.abspath("output_strip.jpg")

    frames, timestamps = extract_random_frames(video_path, 4)

    if len(frames) == 0:
        raise ValueError("No frames extracted")

    combined = stack_frames_horizontally(frames, timestamps)

    success = cv2.imwrite(output_path, combined)

    if not success:
        raise IOError("Failed to save image")

    print(f"Saved to: {output_path}")