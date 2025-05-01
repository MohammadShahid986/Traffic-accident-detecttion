# Traffic Accident Detection Project

This project demonstrates real-time traffic object detection using the YOLOv4 model and OpenCV. It captures video from your webcam, detects traffic-related objects such as cars, buses, trucks, motorbikes, and bicycles, and highlights them with bounding boxes and labels. Additionally, it provides a simple alert if multiple vehicles are detected simultaneously, indicating a potential traffic accident.

## Files

- `traffic_accident_detection.py`: The main Python script that runs real-time object detection on webcam video.
- `yolov4.weights`: Pre-trained YOLOv4 weights file (download separately).
- `yolov4.cfg`: YOLOv4 configuration file (download separately).
- `coco.names`: List of class names from the COCO dataset (download separately).

## Requirements

- Python 3.x
- OpenCV with DNN module (`opencv-python` and `opencv-python-headless`)
- Numpy

## Setup Instructions

1. Download the required YOLOv4 files and place them in the same directory as the script:

   - [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights)
   - [yolov4.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg)
   - [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

2. Install Python dependencies:

   ```bash
   pip install opencv-python numpy
   ```

3. Run the script:

   ```bash
   python traffic_accident_detection.py
   ```

4. A window will open showing the webcam video with detected traffic objects highlighted. Press `ESC` to exit.

## How It Works

- The script loads the YOLOv4 model with pre-trained weights.
- It captures video frames from the webcam.
- Each frame is processed by the model to detect objects.
- Only traffic-related objects (car, bus, truck, motorbike, bicycle) are considered.
- Bounding boxes and confidence labels are drawn around detected objects.
- If more than 3 vehicles are detected in a frame, an alert message is displayed indicating a potential traffic accident.

## Notes

- For best performance, use a system with a CUDA-enabled GPU.
- The detection threshold and alert criteria can be adjusted in the script as needed.

## Demonstration

This project can be used to demonstrate real-time traffic monitoring and accident detection capabilities using AI and computer vision.

---

Feel free to ask if you need any help running or modifying the project.
