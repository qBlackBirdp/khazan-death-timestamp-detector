# video_loader.py

import cv2


def extract_frames_from_video(video_path: str, fps: int = 1):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[❌] Failed to open video: {video_path}")
        return [], []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[ℹ️] Detected video FPS: {video_fps}")
    print(f"[🧪] Total frames: {frame_count}")

    if video_fps == 0 or frame_count == 0:
        print(f"[⚠️] Video FPS or Frame Count is 0. Cannot proceed.")
        return [], []

    frame_interval = max(1, int(video_fps // fps))
    print(f"[ℹ️] Using frame interval: {frame_interval}")

    frames = []
    timestamps = []

    frame_index = 0
    while frame_index < frame_count:
        # print(f"======================= while =========================")
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"[🛑] cap.read() failed at frame_index={frame_index}")
            break

        if frame_index % frame_interval == 0:
            time_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frames.append(frame)
            timestamps.append(time_seconds)

        # print(f"[🔄] frame_index={frame_index}, ret={ret}")
        frame_index += 1

    cap.release()
    print(f"[✅] Extracted {len(frames)} frames")
    return frames, timestamps


def extract_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[❌] Failed to open video: {video_path}")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[🛑] Failed to read first frame.")
        return None

    print("[📸] First frame extracted for resolution check.")
    return frame
