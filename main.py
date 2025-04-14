# main.py

import os
from core.video_loader import extract_frames_from_video
from core.death_detector import detect_death_text
from core.timestamp_writer import save_timestamps

VIDEO_DIR = "videos"
OUTPUT_DIR = "outputs"
FPS = 1  # 초당 프레임 수 (너무 높이면 느려짐)


def process_video(video_path):
    print(f"[▶] Processing {video_path}...")
    frames, times = extract_frames_from_video(video_path, fps=FPS)

    death_timestamps = []
    for frame, second in zip(frames, times):
        if detect_death_text(frame):
            print(f"    💀 Death detected at {int(second)} sec")
            death_timestamps.append(second)

    save_timestamps(video_path, death_timestamps, OUTPUT_DIR)


if __name__ == "__main__":
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.mkv', '.avi'))]
    for video in video_files:
        full_path = os.path.join(VIDEO_DIR, video)
        process_video(full_path)
