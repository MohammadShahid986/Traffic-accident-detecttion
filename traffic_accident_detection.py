import cv2
import numpy as np

# Traffic Accident Detection using YOLOv4 and OpenCV
# This script captures video from webcam, detects traffic-related objects in real-time,
# highlights them with bounding boxes and labels, and alerts if potential accident detected.

weights_path = "yolov4.weights"
config_path = "yolov4.cfg"
names_path = "coco.names"

# Load class names from COCO dataset
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define traffic-related classes to detect
traffic_classes = ['car', 'bus', 'truck', 'motorbike', 'bicycle']

# Load the YOLOv4 network
net = cv2.dnn.readNet(weights_path, config_path)

# User-configurable flag to enable CUDA (set to False if CUDA not supported)
USE_CUDA = False

if USE_CUDA:
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using CUDA backend and target for GPU acceleration.")
    except:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Failed to set CUDA backend, using CPU instead.")
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU for computation.")

# Get the output layer names of the network
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_out_layers[0], (list, tuple, np.ndarray)):
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
else:
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Initialize webcam video capture
cap = cv2.VideoCapture(0)

print("Starting traffic accident detection. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    height, width, channels = frame.shape

    # Prepare the frame as input to the network
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass to get detections
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = classes[class_id]
                if label in traffic_classes:
                    # Object detected with confidence > 50% and is traffic-related
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate top-left corner of bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply Non-Maximum Suppression to reduce overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Draw bounding boxes and labels on detected traffic objects
    vehicle_count = 0
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), font, 1, color, 2)
            vehicle_count += 1

    # Simple alert if multiple vehicles detected (potential accident)
    if vehicle_count > 3:
        cv2.putText(frame, "ALERT: Potential Traffic Accident Detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display the output frame
    cv2.imshow("Traffic Accident Detection - YOLOv4", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
