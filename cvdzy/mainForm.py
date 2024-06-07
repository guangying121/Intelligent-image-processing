import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import Face
import Face1
import Fix
from UI import Ui_Dialog
from PyQt5.QtCore import QTimer
import cv2
import os
import QR,QR1,Car,Car1
import functools

class VideoProcessingThread(QThread):
    finished = pyqtSignal(object)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        result = QR.run(self.file_path)
        self.finished.emit(result)

class CarRecognitionThread(QThread):
    finished = pyqtSignal(object)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        result = Car.run(self.file_path)
        self.finished.emit(result)

class FaceThread(QThread):
    finished = pyqtSignal(object)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        result = Face.run(self.file_path)
        self.finished.emit(result)

class MyWindow(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.video_capture = None
        self.timer = QTimer()
        self.timer_video = QTimer()
        self.currentImage = None  # 用于存储当前打开的图像或视频帧
        self.processing_thread = None

    def inclear(self):
        if self.timer_video.isActive():
            self.timer_video.stop()  # 如果活动则停止计时器

    def outclear(self):
        if self.timer.isActive():
            self.timer.stop()  # 如果活动则停止计时器

    # 打开文件函数
    def openFile(self):
        if self.timer_video.isActive():
            self.timer_video.stop()  # 如果活动则停止计时器
        if self.video_capture is not None:
            self.video_capture.release()  # 释放任何先前的捕获
        fileName, _ = QFileDialog.getOpenFileName(self, "选择文件", "",
                                                  "Image Files (*.png *.jpg *.jpeg *.bmp);;Video Files (*.mp4 *.avi)")
        if fileName:
            self.currentImage = fileName
            if fileName.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image = cv2.imread(fileName)
                self.indisplayImage(image)
            elif fileName.lower().endswith(('.mp4', '.avi')):
                self.inplayVideo(fileName)
    def indisplayImage(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色格式
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.inlabel.setPixmap(pixmap)
        self.inlabel.setScaledContents(True)  # 自适应缩放图像到label大小

    def updateInFrame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            self.timer_video.stop()
            self.video_capture.release()
            return
        self.indisplayImage(frame)

    def outdisplayImage(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色格式
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.outlabel.setPixmap(pixmap)
        self.outlabel.setScaledContents(True)  # 自适应缩放图像到label大小

    def inplayVideo(self, fileName):
        self.video_capture = cv2.VideoCapture(fileName)
        if not self.video_capture.isOpened():
            QMessageBox.warning(self, "Error", "Cannot open video file.")
            return
        fps = round(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.timer_video = QTimer()
        self.timer_video.timeout.connect(self.updateInFrame)
        self.timer_video.start(1000 / (2.0*fps))  # start timer based on video fps并2倍速播放

    def outplayVideo(self, fileName):
        self.video = cv2.VideoCapture(fileName)
        fps = round(self.video.get(cv2.CAP_PROP_FPS))  # 获取视频帧率并四舍五入
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrame)
        self.timer.start(1000 // fps)  # 设置定时器的超时时间为每帧的时间间隔

    def nextFrame(self):
        ret, frame = self.video.read()
        if not ret:
            self.timer.stop()
            self.video.release()
            cv2.destroyAllWindows()
            return
        self.outdisplayImage(frame)

    def handle_processing_finished(self, result, path):
        if isinstance(result, cv2.VideoCapture):  # 如果返回的是视频对象
            self.outplayVideo(path)
        elif isinstance(result, np.ndarray):  # 如果返回的是图像对象
            self.outdisplayImage(result)
        # 关闭处理中的消息框
        if hasattr(self, 'processing_message_box'):
            self.processing_message_box.close()
        self.processing_thread = None  # 清除处理线程

    # 实现 QR 识别按钮响应函数
    def QRrecognize(self):
        if self.currentImage is not None and self.processing_thread is None:
            # 显示正在处理中的消息框
            _, file_extension = os.path.splitext(self.currentImage)
            if file_extension.lower() in ['.mp4', '.avi']:  # 如果文件是视频
                self.processing_message_box = QMessageBox(self)
                self.processing_message_box.setText("正在处理中，请稍候...")
                self.processing_message_box.show()

            self.processing_thread = VideoProcessingThread(self.currentImage)
            # 使用 functools.partial 部分应用函数参数
            handler = functools.partial(self.handle_processing_finished, path='QR/result.mp4')
            self.processing_thread.finished.connect(handler)
            self.processing_thread.start()

    # 实现车牌识别按钮响应函数
    def carrecognize(self):
        if self.currentImage is not None and self.processing_thread is None:
            # 显示正在处理中的消息框
            _, file_extension = os.path.splitext(self.currentImage)
            if file_extension.lower() in ['.mp4', '.avi']:  # 如果文件是视频
                self.processing_message_box = QMessageBox(self)
                self.processing_message_box.setText("正在处理中，请稍候...")
                self.processing_message_box.show()

            self.processing_thread = CarRecognitionThread(self.currentImage)
            # 使用 functools.partial 部分应用函数参数
            handler = functools.partial(self.handle_processing_finished, path='Car/result.mp4')
            self.processing_thread.finished.connect(handler)
            self.processing_thread.start()

    # 实现人脸检测按钮响应函数
    def facedetect(self):
        if self.currentImage is not None and self.processing_thread is None:
            # 显示正在处理中的消息框
            _, file_extension = os.path.splitext(self.currentImage)
            if file_extension.lower() in ['.mp4', '.avi']:  # 如果文件是视频
                self.processing_message_box = QMessageBox(self)
                self.processing_message_box.setText("正在处理中，请稍候...")
                self.processing_message_box.show()

            self.processing_thread = FaceThread(self.currentImage)
            # 使用 functools.partial 部分应用函数参数
            handler = functools.partial(self.handle_processing_finished, path='Face/result.mp4')
            self.processing_thread.finished.connect(handler)
            self.processing_thread.start()

    # 实现图像拼接按钮响应函数
    def imgfix(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", "")
        if folder_path:
            Fix.run(folder_path)

    def QRrecognize1(self):
        QR1.run()

    def carrecognize1(self):
        Car1.run()

    def facedetect1(self):
        Face1.run()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())