import os
import cv2
from pyzbar.pyzbar import decode

def run(file_path):
    # 判断文件类型
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:  # 图像文件
        img = cv2.imread(file_path)
        decoded_objects = decode(img)  # 进行二维码识别
        if decoded_objects:
            # 在图像上绘制识别结果
            for obj in decoded_objects:
                cv2.rectangle(img, obj.rect, (0, 255, 0), 2)
                text = "{}".format(obj.data.decode())
                # 调整文字大小
                font_scale = 1.0
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.putText(img, text, (obj.rect[0], obj.rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, min(1, font_scale * min(img.shape[0], img.shape[1]) / max(text_width, text_height)), (0, 255, 0), thickness)
                print(f"二维码类型: {obj.type}, 数据: {obj.data.decode()}")
        else:
            print("没有检测到二维码。")
        return img
    elif file_extension.lower() in ['.mp4', '.avi']:  # 视频文件
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('QR/result.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            decoded_objects = decode(frame)  # 进行二维码识别
            text_display = "no code。"  # 初始化显示信息
            if decoded_objects:
                text_display = "data: "
                for obj in decoded_objects:
                    cv2.rectangle(frame, (obj.rect.left, obj.rect.top, obj.rect.width, obj.rect.height), (0, 255, 0), 2)
                    text = "{}".format(obj.data.decode())
                    cv2.putText(frame, text, (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    text_display += f"{obj.data.decode()} "
            cv2.putText(frame, text_display, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 显示处理信息
            cv2.namedWindow("Processing", cv2.WINDOW_NORMAL)
            cv2.imshow('Processing', frame,)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            out.write(frame)
        print("处理完成！")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return cv2.VideoCapture('QR/result.mp4')
    else:
        print("Unsupported file format.")
        return None
