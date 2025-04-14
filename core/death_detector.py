# death_detector.py

import pytesseract
import cv2

# 기본 텍스트 기반 감지 함수
def detect_death_text(frame) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 선택적으로 threshold를 적용해 글자 선명하게
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # OCR 인식
    text = pytesseract.image_to_string(thresh).upper()

    # 키워드 감지
    keywords = ["KHAZAN", "HAS", "FALLEN"]
    match_count = sum(kw in text for kw in keywords)

    return match_count >= 2  # 3개 중 2개 이상이면 사망 판정