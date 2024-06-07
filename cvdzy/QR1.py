import cv2
from pyzbar.pyzbar import decode
import numpy as np


def run(video_source=0):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 对帧进行灰度化处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 二值化处理
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # 查找二维码
        decoded_objects = decode(frame)

        # 在帧上绘制识别结果
        for obj in decoded_objects:
            points = obj.polygon
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                hull = list(map(tuple, np.squeeze(hull)))
            else:
                hull = points
            n = len(hull)
            for j in range(n):
                cv2.line(frame, hull[j], hull[(j + 1) % n], (0, 255, 0), 3)

            x = obj.rect.left
            y = obj.rect.top
            barcode_info = obj.data.decode("utf-8")
            barcode_type = obj.type
            print("二维码类型:", barcode_type)
            print("二维码数据:", barcode_info)
            cv2.putText(frame, barcode_info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('QR Code Scanner', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 测试代码
if __name__ == '__main__':
    run()  # 打开摄像头实时识别二维码
    # run('your_video_file.mp4')  # 打开本地视频文件实时识别二维码
