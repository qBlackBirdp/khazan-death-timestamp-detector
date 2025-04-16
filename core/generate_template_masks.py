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

    # ✅ HSV 기반 붉은색 영역 추출
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # ✅ dilation으로 범위 확장
    kernel = np.ones((3, 3), np.uint8)
    red_mask_dilated = cv2.dilate(red_mask, kernel, iterations=1)

    return red_mask_dilated, resized


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

        # 디버그 이미지 저장
        cv2.imwrite(os.path.join(DEBUG_DIR, f"template_{i + 1}_resized_debug.png"), resized)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"template_{i + 1}_resized_mask.png"), mask)

        print(f"  └─ Mask saved: {save_name}")

    print(f"[✅] All {len(templates)} masks generated and saved.")


if __name__ == "__main__":
    generate_all_masks()
