# find_camera.py
import cv2

for i in range(5):  # 0~4까지 시도
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ 카메라 {i} 사용 가능")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(1000)
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"❌ 카메라 {i} 열 수 없음")
