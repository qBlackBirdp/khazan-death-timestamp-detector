# death_detector.py
from datetime import timedelta

import cv2
import os


def load_death_templates(samples_dir: str) -> list:
    templates = []
    for filename in os.listdir(samples_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(samples_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            templates.append(img)
    print(f"[📥] Loaded {len(templates)} death sample templates")
    return templates


def preprocess_template(img):
    img = cv2.equalizeHist(img)  # 명암 대비 향상
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 노이즈 제거
    # img = cv2.Canny(img, 50, 150)  # 엣지 강조
    return img


def preprocess_frame(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이 스케일
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.equalizeHist(img)  # 명암 대비만 적용
    return img


def load_resized_templates(resized_dir="resized_templates"):
    templates = []
    filenames = []

    for filename in sorted(os.listdir(resized_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(resized_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            templates.append(img)
            filenames.append(filename)

    print(f"[📥] Loaded {len(templates)} resized templates from {resized_dir}")
    return templates, filenames


def detect_death_by_template(frame, templates, threshold=0.65, debug_threshold=0.4, current_time=None) -> bool:
    gray_frame = preprocess_frame(frame)  # 👈 프레임 전처리 (명암 대비만)
    gray_frame = crop_center(gray_frame, cropx=800, cropy=400)  # 👈 중앙만 잘라서 비교

    for i, template in enumerate(templates):
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val >= debug_threshold:
            timestamp_str = str(timedelta(seconds=int(current_time))) if current_time is not None else "unknown_time"
            print(f"    🔎 [{timestamp_str}] Template {i + 1} match max_val: {max_val:.4f}")

        if max_val >= threshold:
            print(f"    ✅ Match passed threshold ({threshold}) with Template {i + 1}")
            return True

    return False


# def resize_templates_to_frame(templates, frame_shape, target_ratio=(0.3, 0.1)):
#     """프레임 크기에 맞춰 템플릿 resize (w*h 비율 기준)"""
#     h, w = frame_shape[:2]
#     tw, th = int(w * target_ratio[0]), int(h * target_ratio[1])
#     print(f"[📐] Resizing templates to: {tw} x {th}")
#
#     resized = []
#     for i, template in enumerate(templates):
#         resized_template = cv2.resize(template, (tw, th))
#         resized.append(resized_template)
#         print(f"  └─ Template {i+1} resized to: {resized_template.shape[1]} x {resized_template.shape[0]}")
#
#     return resized


def resize_templates_to_frame_ratio_safe(
        templates, frame_shape, target_width_ratio=0.3, save_dir="resized_templates"
):
    os.makedirs(save_dir, exist_ok=True)
    w = frame_shape[1]
    target_width = int(w * target_width_ratio)

    resized = []
    for i, template in enumerate(templates):
        h_t, w_t = template.shape[:2]
        scale = target_width / w_t
        target_height = int(h_t * scale)

        resized_template = cv2.resize(template, (target_width, target_height))
        resized.append(resized_template)

        save_path = os.path.join(save_dir, f"template_{i + 1}_resized.png")
        cv2.imwrite(save_path, resized_template)

        print(f"  └─ Template {i + 1} resized to: {target_width} x {target_height} (saved to {save_path})")

    return resized


def pad_template_to_uniform_size(templates):
    max_h = max(t.shape[0] for t in templates)
    max_w = max(t.shape[1] for t in templates)

    padded = []
    for i, t in enumerate(templates):
        h, w = t.shape[:2]
        pad_top = (max_h - h) // 2
        pad_bottom = max_h - h - pad_top
        pad_left = (max_w - w) // 2
        pad_right = max_w - w - pad_left

        padded_template = cv2.copyMakeBorder(t, pad_top, pad_bottom, pad_left, pad_right,
                                             borderType=cv2.BORDER_CONSTANT, value=0)
        padded.append(padded_template)

        print(f"  └─ Template {i + 1} padded to: {max_w} x {max_h}")
    return padded


def crop_center(img, cropx, cropy):
    h, w = img.shape
    startx = w // 2 - cropx // 2
    starty = h // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx]
