# main.py

# import os
# from core.video_loader import extract_first_frame, extract_frames_from_video
# from core.timestamp_writer import save_timestamps
# from core.death_detector import load_resized_templates, resize_templates_to_frame_ratio_safe, detect_death_by_template, \
#     load_death_templates, pad_template_to_uniform_size
#
# VIDEO_DIR = "videos"
# OUTPUT_DIR = "outputs"
# SAMPLES_DIR = "death_samples_images"
# FPS = 1  # 초당 프레임 수
#
#
# def process_video(video_path, templates):
#     print(f"[▶] Processing {video_path}...")
#     frames, times = extract_frames_from_video(video_path, fps=FPS)
#
#     death_timestamps = []
#     for frame, second in zip(frames, times):
#         if detect_death_by_template(frame, templates):
#             print(f"    💀 Death detected at {int(second)} sec")
#             death_timestamps.append(second)
#
#     save_timestamps(video_path, death_timestamps, OUTPUT_DIR)
#
#
# if __name__ == "__main__":
#     video_path = os.path.join(VIDEO_DIR, "khazan3.mp4")
#
#     # 1️⃣ 프레임 한 장 추출 → 템플릿 리사이즈 + 저장
#     first_frame = extract_first_frame(video_path)
#     if first_frame is None:
#         exit()
#
#     frame_shape = first_frame.shape
#     raw_templates = load_death_templates(SAMPLES_DIR)
#     resize_templates_to_frame_ratio_safe(
#         raw_templates, frame_shape, target_width_ratio=0.3, save_dir="resized_templates"
#     )
#     # 2️⃣ 저장된 리사이즈 템플릿 불러오기
#     resized_templates = load_resized_templates("resized_templates")
#
#     # 2-2️⃣ 패딩으로 크기 통일
#     resized_templates = pad_template_to_uniform_size(resized_templates)
#
#     # 3️⃣ 분석
#     process_video(video_path, resized_templates)

# 전체 처리하려면 아래 주석 해제
# if __name__ == "__main__":

# video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.mkv', '.avi'))]
#
# for video_filename in video_files:
#     video_path = os.path.join(VIDEO_DIR, video_filename)
#
#     print(f"[🔍] Extracting first frame from {video_filename}")
#     first_frame = extract_first_frame(video_path)
#     if first_frame is None:
#         continue
#
#     frame_shape = first_frame.shape
#     resized_templates = resize_templates_to_frame_ratio_safe(raw_templates, frame_shape)
#
#     process_video(video_path, resized_templates)


import cv2
import os
import datetime


def extract_summary_frames(video_path, output_dir, fps=1):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[❌] Cannot open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps // fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)

    print(f"[ℹ️] Total frames: {frame_count}, Interval: {frame_interval}")

    frame_index = 0
    saved_count = 0

    while cap.isOpened() and frame_index < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = str(datetime.timedelta(milliseconds=time_msec)).split(".")[0]
            h, m, s = timestamp.split(":")
            filename = f"frame_{h}h{m}m{s}s.png"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    print(f"[✅] Saved {saved_count} summary frames to {output_dir}")


# 예시 사용
if __name__ == "__main__":
    extract_summary_frames(
        video_path="videos/khazan3.mp4",
        output_dir="extracted_summary_frames",
        fps=1
    )
