# main.py

import os
from core.video_loader import extract_first_frame, extract_frames_from_video
from core.timestamp_writer import save_timestamps
from core.death_detector import load_resized_templates, resize_templates_to_frame_ratio_safe, detect_death_by_template, \
    load_death_templates, pad_template_to_uniform_size, preprocess_gray

VIDEO_DIR = "videos"
OUTPUT_DIR = "outputs"
SAMPLES_DIR = "death_samples_images"
FPS = 1  # 초당 프레임 수


def process_video(video_path, templates):
    print(f"[▶] Processing {video_path}...")
    frames, times = extract_frames_from_video(video_path, fps=FPS)

    death_timestamps = []
    for frame, second in zip(frames, times):
        if detect_death_by_template(frame, templates, current_time=second):
            print(f"    💀 Death detected at {int(second)} sec")
            death_timestamps.append(second)

    save_timestamps(video_path, death_timestamps, OUTPUT_DIR)


if __name__ == "__main__":
    video_path = os.path.join(VIDEO_DIR, "khazan3.mp4")

    # 1️⃣ 프레임 한 장 추출 → 템플릿 리사이즈 + 저장
    first_frame = extract_first_frame(video_path)
    if first_frame is None:
        exit()

    frame_shape = first_frame.shape
    raw_templates = load_death_templates(SAMPLES_DIR)
    resize_templates_to_frame_ratio_safe(
        raw_templates, frame_shape, target_width_ratio=0.3, save_dir="resized_templates"
    )
    # 2️⃣ 저장된 리사이즈 템플릿 불러오기
    resized_templates, filenames = load_resized_templates("resized_templates")

    # 2-2️⃣ 패딩으로 크기 통일
    resized_templates = pad_template_to_uniform_size(resized_templates)

    # 2-3️⃣ 템플릿 선택 (7, 9, 13번만 사용) → 0-based 인덱스
    selected_ids = [6, 8, 12]
    selected_templates = [resized_templates[i] for i in selected_ids]
    selected_filenames = [filenames[i] for i in selected_ids]

    print("[🎯] Selected templates:")
    for idx, name in zip(selected_ids, selected_filenames):
        print(f"  └─ Template {idx+1}: {name}")
    resized_templates = [resized_templates[i] for i in selected_ids]

    # 대비 보정 + 노이즈 제거 + 엣지 강조
    resized_templates = [preprocess_gray(t) for t in resized_templates]

    # 3️⃣ 분석
    process_video(video_path, resized_templates)

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
