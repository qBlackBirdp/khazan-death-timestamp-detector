# death_detector.py

from datetime import timedelta

import cv2
import os

CROP_X = 1000
CROP_Y = 400


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


def detect_death_by_template(frame, templates, masks=None, threshold=0.85, current_time=None) -> bool:
    from datetime import timedelta
    gray_frame = preprocess_frame(frame)
    cropped = crop_center(gray_frame)

    detected = False

    for i, template in enumerate(templates):
        mask = masks[i] if masks else None

        # 중앙 자름
        template_cropped = crop_center(template)
        mask_cropped = crop_center(mask) if mask is not None else None

        method = cv2.TM_CCORR_NORMED if mask_cropped is not None else cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(cropped, template_cropped, method, mask=mask_cropped)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        if mask_cropped is not None:
            x, y = max_loc
            h, w = template_cropped.shape[:2]
            matched_region = cropped[y:y + h, x:x + w]
            matched_region = preprocess_template(matched_region)

            _, template_bin = cv2.threshold(template_cropped, 30, 255, cv2.THRESH_BINARY)
            _, matched_bin = cv2.threshold(matched_region, 30, 255, cv2.THRESH_BINARY)

            # template_bin이 1인 영역만 비교
            masked_area = cv2.countNonZero(cv2.bitwise_and(template_bin, matched_bin, mask=template_bin))
            valid_area = cv2.countNonZero(template_bin)
            coverage_ratio = masked_area / valid_area if valid_area > 0 else 0
        else:
            coverage_ratio = 0.0  # 마스크 없는 경우는 무조건 False 처리

        if current_time is not None:
            timestamp_str = str(timedelta(seconds=int(current_time)))
            print(
                f"    🔎 [{timestamp_str}] Template {i + 1} → coverage: {coverage_ratio:.4f}"
            )

        if coverage_ratio >= threshold:
            print(f"    ✅ Match passed threshold ({threshold}) with Template {i + 1}")
            detected = True

    return detected


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


def crop_center(img, cropx=1000, cropy=400):
    # print(f"========================== crop_center 실행 ==============================")
    y, x = img.shape[:2]
    startx = max(x // 2 - CROP_X // 2, 0)
    starty = max(y // 2 - CROP_Y // 2, 0)
    endx = startx + CROP_X
    endy = starty + CROP_Y
    return img[starty:endy, startx:endx]


def remove_padding(template):
    # template은 그레이스케일 또는 바이너리 이미지라고 가정
    if len(template.shape) != 2:
        raise ValueError("remove_padding() expects a single-channel (grayscale) image")

    # non-zero 영역 좌표 가져오기
    coords = cv2.findNonZero(template)
    if coords is None:
        return template  # 내용이 없는 경우 원본 그대로 반환

    x, y, w, h = cv2.boundingRect(coords)
    return template[y:y + h, x:x + w]
