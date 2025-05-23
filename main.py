# main.py

import os
import cv2
from core.video_loader import extract_first_frame, extract_frames_from_video
from core.timestamp_writer import save_timestamps
from core.death_detector import load_resized_templates, resize_templates_to_frame_ratio_safe, detect_death_by_template, \
    load_death_templates, pad_template_to_uniform_size, preprocess_template, remove_padding

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


# def process_video(video_path, templates, masks=None):
#     print(f"[▶] Processing {video_path}...")
#     frames, times = extract_frames_from_video(video_path, fps=FPS)
#
#     death_timestamps = []
#     for frame, second in zip(frames, times):
#         if detect_death_by_template(frame, templates, masks=masks, current_time=second):
#             print(f"    💀 Death detected at {int(second)} sec")
#             death_timestamps.append(second)
#
#     save_timestamps(video_path, death_timestamps, OUTPUT_DIR)


def process_video(video_path, templates, masks=None, start_time=0):
    print(f"[▶] Processing {video_path} from {start_time} sec...")
    frames, times = extract_frames_from_video(video_path, fps=FPS, start_time=start_time)

    death_timestamps = []
    for frame, second in zip(frames, times):
        if detect_death_by_template(frame, templates, masks=masks, current_time=second):
            print(f"    💀 Death detected at {int(second)} sec")
            death_timestamps.append(second)

    save_timestamps(video_path, death_timestamps, OUTPUT_DIR)


def load_template_masks(mask_dir):
    masks = []
    filenames = []

    for filename in sorted(os.listdir(mask_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(mask_dir, filename)
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            masks.append(mask)
            filenames.append(filename)

    print(f"[📥] Loaded {len(masks)} template masks from {mask_dir}")
    return masks, filenames


if __name__ == "__main__":
    video_path = os.path.join(VIDEO_DIR, "khazan3.mp4")

    # 1️⃣ 프레임 한 장 추출 → 템플릿 리사이즈 + 저장
    first_frame = extract_first_frame(video_path)
    if first_frame is None:
        exit()

    # 마스크 로드
    all_masks, mask_filenames = load_template_masks("resized_template_masks")

    # 템플릿 로드
    all_templates, all_filenames = load_resized_templates("resized_templates")

    selected_templates = []
    selected_masks = []

    for fname in selected_filenames:
        if fname in all_filenames:
            idx = all_filenames.index(fname)
            selected_templates.append(all_templates[idx])

            # 🔁 마스크 이름 구성: template_1_resized → template_1_resized_mask.png
            expected_mask_name = fname.replace(".png", "_mask.png")

            if expected_mask_name in mask_filenames:
                mask_idx = mask_filenames.index(expected_mask_name)
                selected_masks.append(all_masks[mask_idx])
            else:
                print(f"[⚠️] Corresponding mask not found for {expected_mask_name}")
        else:
            print(f"[⚠️] Template file not found: {fname}")

    print("[🎯] Using templates:")
    for i, fname in enumerate(selected_filenames):
        print(f"  └─ Template {i + 1}: {fname}")

    # 🔪 1. 여백 제거 (패딩 전에!)
    selected_templates = [remove_padding(t) for t in selected_templates]

    # 🐞 Debug 저장용 — 첫 번째 템플릿만
    template_trimmed = selected_templates[0]
    cv2.imwrite("debug/trimmed_template.png", template_trimmed)

    # 📐 2. 패딩으로 사이즈 통일
    selected_templates = pad_template_to_uniform_size(selected_templates)
    selected_masks = pad_template_to_uniform_size(selected_masks)

    # 🧼 3. 전처리
    selected_templates = [preprocess_template(t) for t in selected_templates]
    selected_masks = [preprocess_template(m) for m in selected_masks]

    for i, (t, m) in enumerate(zip(selected_templates, selected_masks)):
        if t.shape != m.shape:
            print(f"[❌] Shape mismatch at index {i}: template {t.shape}, mask {m.shape}")

    process_video(video_path, selected_templates, masks=selected_masks, start_time=1200)

    # 🧠 분석 (마스크 포함)
    # process_video(video_path, selected_templates, masks=selected_masks)

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
