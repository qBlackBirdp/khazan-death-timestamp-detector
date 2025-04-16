# generate_template_masks.py

import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "death_samples_images")
MASK_SAVE_DIR = os.path.join(BASE_DIR, "resized_template_masks")
DEBUG_DIR = os.path.join(BASE_DIR, "resized_template_masks_debug")

os.makedirs(MASK_SAVE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


def generate_mask_focused_on_red(template_bgr, target_size=(576, 324)):
    resized = cv2.resize(template_bgr, target_size)

    b, g, r = cv2.split(resized)
    red_mask = (r > 70) & (g < 170) & (b < 170)
    red_only = np.zeros_like(resized)
    red_only[red_mask] = resized[red_mask]

    gray = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 20000:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    return mask, resized


def generate_all_masks(resized_dir=TEMPLATE_DIR, output_dir=MASK_SAVE_DIR):
    templates = []
    filenames = []

    for filename in sorted(os.listdir(resized_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(resized_dir, filename)
            img = cv2.imread(path)  # BGR
            templates.append(img)
            filenames.append(filename)

    print(f"[📥] Loaded {len(templates)} templates from {resized_dir}")

    target_size = (576, 324)

    for i, (template, fname) in enumerate(zip(templates, filenames)):
        mask, resized = generate_mask_focused_on_red(template, target_size=target_size)

        save_name = f"template_{i + 1}_resized_mask.png"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, mask)

        # 디버깅 저장
        cv2.imwrite(os.path.join(DEBUG_DIR, f"template_{i + 1}_resized_debug.png"), resized)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"template_{i + 1}_resized_mask.png"), mask)

        print(f"  └─ Mask saved: {save_name}")

    print(f"[✅] All {len(templates)} masks generated and saved.")


if __name__ == "__main__":
    generate_all_masks()
