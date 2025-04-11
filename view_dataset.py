import os
import cv2

# 데이터 경로 설정
image_dir = "./images"
label_dir = "./labels"

# 이미지 파일 목록
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

# 각 이미지에 대해 반복
for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

    # 이미지 읽기
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지 로드 실패: {img_path}")
        continue

    h, w, _ = img.shape

    # 라벨 읽기 및 박스 표시
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, cx, cy, bw, bh = map(float, parts)
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                # 박스 그리기
                color = (0, 0, 255) if class_id == 0 else (0, 255, 0)
                label = "Fall" if class_id == 0 else "Other"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 이미지 보기
    # cv2.startWindowThread()
    cv2.imshow("Fall Detection Dataset Viewer", img)
    key = cv2.waitKey(0)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
