# generate_template_masks.py

import cv2
import os
import numpy as np

from death_detector import crop_center

CROP_X = 1000
CROP_Y = 400

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "death_samples_images")
MASK_SAVE_DIR = os.path.join(BASE_DIR, "resized_template_masks")
DEBUG_DIR = os.path.join(BASE_DIR, "resized_template_masks_debug")

os.makedirs(MASK_SAVE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


def generate_mask_focused_on_red(template_bgr, cropx, cropy):
    # 1️⃣ 중앙 크롭
    cropped = crop_center(template_bgr, cropx, cropy)

    # 2️⃣ 붉은색 강조 필터링 (BGR 기준)
    b, g, r = cv2.split(cropped)
    red_mask = (r > 100) & (g < 130) & (b < 130)
    red_only = np.zeros_like(cropped)
    red_only[red_mask] = cropped[red_mask]

    # 3️⃣ Grayscale → 이진화 → Contour
    gray = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 10000:  # 텍스트 크기 범위
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    return mask, cropped


def generate_all_masks(resized_dir=TEMPLATE_DIR, output_dir=MASK_SAVE_DIR):
    templates = []
    filenames = []

    for filename in sorted(os.listdir(resized_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(resized_dir, filename)
            img = cv2.imread(path)  # BGR로 읽기 (색상 추출을 위해)
            templates.append(img)
            filenames.append(filename)

    print(f"[📥] Loaded {len(templates)} templates from {resized_dir}")

    for i, (template, fname) in enumerate(zip(templates, filenames)):
        mask, cropped = generate_mask_focused_on_red(template, cropx=CROP_X, cropy=CROP_Y)

        save_name = os.path.splitext(fname)[0] + "_mask.png"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, mask)

        # 디버깅 저장
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{i + 1:02d}_cropped.png"), cropped)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{i + 1:02d}_mask.png"), mask)

        print(f"  └─ Mask saved: {save_name}")

    print(f"[✅] All {len(templates)} masks generated and saved.")


if __name__ == "__main__":
    generate_all_masks()
