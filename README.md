```markdown
# Intelligent Image Processing Software (智能图像处理软件)

![GitHub License](https://img.shields.io/badge/license-MIT-blue)
![Python Version](https://img.shields.io/badge/python-3.7%2B-green)

基于PyQt5的多功能图像处理工具，集成**YOLOv5车牌识别**、**Haar级联人脸检测**、**OpenCV图像拼接**和**二维码识别**功能，支持图片/视频输入与实时摄像头处理。

---

## 核心功能

### 1. 二维码识别
- **多场景支持**：静态图片、本地视频、实时摄像头
- **高效解码**：`pyzbar`库实现多二维码同步解析
- **动态标注**：实时框选二维码区域并显示解码内容

### 2. 车牌识别
- **精准定位**：YOLOv5模型检测车牌位置
- **透视矫正**：四点变换消除倾斜干扰
- **OCR识别**：深度学习模型提取车牌字符

### 3. 人脸检测
- **多特征检测**：人脸、眼睛、微笑同步识别
- **实时跟踪**：视频流动态标注检测结果
- **区域优化**：限制单脸最大2眼1嘴检测

### 4. 图像拼接
- **智能对齐**：柱面/球面投影自动匹配特征点
- **无缝融合**：黑边裁剪与拼接痕迹优化
- **容错处理**：自动过滤无效输入图像

---

## 快速部署

### 环境要求
```bash
git clone https://github.com/guangying121/Intelligent-image-processing.git
cd Intelligent-image-processing
pip install -r requirements.txt
```

### 关键依赖
- **PyTorch**：需匹配CUDA版本（推荐11.3+）
- **OpenCV**：4.5+（包含contrib模块）
- **PyQt5**：5.15+ 图形界面支持

---

## 使用指南

### 启动程序
```bash
python mainform.py
```

### 操作流程
1. **输入选择**：
   - 文件模式：支持`.jpg/.png/.mp4`格式
   - 摄像头模式：实时视频流处理
2. **功能调用**：
   - 文件模式：通过左侧面板选择功能
   - 摄像头模式：右侧按钮触发实时检测
3. **结果查看**：
   - 图片结果：右侧面板直接显示
   - 视频结果：保存至`results/`目录

### 界面示例
| 主操作界面 | 车牌识别效果 |
|------------|--------------|
| ![主界面](https://github.com/guangying121/Intelligent-image-processing/assets/126480485/ce315875-d7ed-43cd-bd5d-6b826f7f97fc) | ![车牌识别](https://github.com/guangying121/Intelligent-image-processing/assets/126480485/27fc3cb7-ed23-472e-9629-c42493aa57f6) |

---

## 技术细节

### 系统架构
- **前端**：PyQt5实现多线程界面
- **模块化设计**：
  - QR.py/QR1.py：二维码处理
  - Car.py/Car1.py：车牌识别
  - Face.py/Face1.py：人脸检测
  - Fix.py：图像拼接

### 关键算法
| 功能         | 技术栈                          | 性能优化                     |
|--------------|---------------------------------|------------------------------|
| 二维码识别   | pyzbar解码 + 动态ROI裁剪       | 多线程视频处理               |
| 车牌识别     | YOLOv5定位 + OCR字符分割       | 透视变换消除畸变             |
| 人脸检测     | Haar级联分类器 + 区域约束      | 限制检测区域提升速度         |
| 图像拼接     | OpenCV Stitcher + 最大轮廓裁剪 | 自动排除无效输入             |

---

## 参考文献
1. [Image Stitching with OpenCV and Python](https://pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/)
2. [YOLOv5车牌识别实现](https://github.com/we0091234/Chinese_license_plate_detection_recognition)

---


**欢迎提交Issue和PR！**  
🚀 项目持续更新中，点亮Star获取最新动态 →
```
