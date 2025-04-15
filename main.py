# main.py

import os
from core.video_loader import extract_first_frame, extract_frames_from_video
from core.timestamp_writer import save_timestamps
from core.death_detector import load_resized_templates, resize_templates_to_frame_ratio_safe, detect_death_by_template, \
    load_death_templates, pad_template_to_uniform_size,preprocess_template

selected_filenames = [
    "template_1_resized.png",
    "template_2_resized.png",
    "template_3_resized.png",
    "template_4_resized.png",
    "template_5_resized.png",
    "template_6_resized.png",
    "template_7_resized.png",
    "template_8_resized.png",
    "template_9_resized.png",
    "template_10_resized.png",
    "template_11_resized.png",
    "template_12_resized.png",
    "template_13_resized.png",
    "template_14_resized.png",
    "template_15_resized.png",
    "template_16_resized.png",
]

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

    # 2-3️⃣ 템플릿 선택
    all_templates, all_filenames = load_resized_templates("resized_templates")
    selected_templates = []
    for fname in selected_filenames:
        if fname in all_filenames:
            idx = all_filenames.index(fname)
            selected_templates.append(all_templates[idx])
        else:
            print(f"[⚠️] Template file not found: {fname}")

    print("[🎯] Using templates:")
    for i, fname in enumerate(selected_filenames):
        print(f"  └─ Template {i+1}: {fname}")

    # 2-4️⃣ 패딩 적용
    selected_templates = pad_template_to_uniform_size(selected_templates)

    # 2-5️⃣ 전처리 (equalizeHist, blur, canny)
    selected_templates = [preprocess_template(t) for t in selected_templates]

    # 3️⃣ 분석
    process_video(video_path, selected_templates)

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
