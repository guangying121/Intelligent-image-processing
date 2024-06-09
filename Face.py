import os
import cv2


def run(file_path):
    if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        image = process_image(file_path)
        return image
    elif file_path.endswith(('.mp4', '.avi', '.mov')):
        result=process_video(file_path)
        return result

def process_image(file_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    img = cv2.imread(file_path)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_area = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_area, scaleFactor=1.3, minNeighbors=10)
        smiles = smile_cascade.detectMultiScale(face_area, scaleFactor=1.16, minNeighbors=65, minSize=(25, 25))

        # 限制最多检测到的眼睛数为2
        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
            cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # 限制最多检测到的微笑数为1
        if len(smiles) > 0:
            ex, ey, ew, eh = smiles[0]
            cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            cv2.putText(img, "smile", (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return img

def process_video(file_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('Face/result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    print('Processing video, please wait...')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_area = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_area, scaleFactor=1.3, minNeighbors=10)
            smiles = smile_cascade.detectMultiScale(face_area, scaleFactor=1.16, minNeighbors=65, minSize=(25, 25))

            # Limit to max 2 eyes
            for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Limit to max 1 smile
            if len(smiles) > 0:
                ex, ey, ew, eh = smiles[0]
                cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
                cv2.putText(frame, "smile", (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.namedWindow("processing", cv2.WINDOW_NORMAL)
        cv2.imshow("processing", frame)
        cv2.waitKey(1)
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return cv2.VideoCapture('Face/result.mp4')