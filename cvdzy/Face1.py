import cv2
def run():
    # 导入级联分类器引擎
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    # 调用摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 获取摄像头拍摄到的画面
        ret, frame = cap.read()
        if not ret:
            break  # 如果无法获取画面，跳出循环

        # 用人脸级联分类器引擎进行人脸识别
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))

        for (x, y, w, h) in faces:
            # 画出人脸框，蓝色，画笔宽度为2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 框选出人脸区域，在人脸区域而不是全图中进行人眼和微笑检测
            face_area = frame[y:y + h, x:x + w]

            # 人眼检测
            eyes = eye_cascade.detectMultiScale(face_area, scaleFactor=1.3, minNeighbors=10)
            # 限制最多两个眼睛
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # 微笑检测
            smiles = smile_cascade.detectMultiScale(face_area, scaleFactor=1.16, minNeighbors=65, minSize=(25, 25))
            # 限制最多一个微笑
            if len(smiles) > 0:
                ex, ey, ew, eh = smiles[0]
                cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
                # 把"smile"文字放在蓝色方框的上方
                cv2.putText(frame, "smile", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 实时展示效果画面
        cv2.imshow("my_window", frame)

        # 每5毫秒监听一次键盘动作，按q键结束
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
