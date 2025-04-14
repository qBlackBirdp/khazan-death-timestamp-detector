# video_loader.py

import cv2
import os


def extract_frames_from_video(video_path: str, fps: int = 1):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps // fps)

    frames = []
    timestamps = []

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            time_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frames.append(frame)
            timestamps.append(time_seconds)
        frame_index += 1
    cap.release()
    return frames, timestamps
