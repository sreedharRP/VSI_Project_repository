import cv2
from ultralytics import YOLO  # Assuming YOLOv8 or compatible

def detect_with_three_models(model_path_1, model_path_2, model_path_3, source=0, resolution="1280x720"):
    model1 = YOLO(model_path_1)
    model2 = YOLO(model_path_2)
    model3 = YOLO(model_path_3)

    width, height = map(int, resolution.split("x"))
    cap = cv2.VideoCapture(0 if source == "usb0" else source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    class_names_1 = model1.names
    class_names_2 = model2.names
    class_names_3 = model3.names

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference with all three models
        results1 = model1.predict(source=rgb_frame, verbose=False)[0]
        results2 = model2.predict(source=rgb_frame, verbose=False)[0]
        results3 = model3.predict(source=rgb_frame, verbose=False)[0]

        # Draw model 1 detections (Red)
        for box in results1.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"M1: {class_names_1.get(cls_id, 'Unknown')} {conf*100:.1f}%"
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw model 2 detections (Green)
        for box in results2.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"M2: {class_names_2.get(cls_id, 'Unknown')} {conf*100:.1f}%"
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb_frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw model 3 detections (Blue)
        for box in results3.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"M3: {class_names_3.get(cls_id, 'Unknown')} {conf*100:.1f}%"
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(rgb_frame, label, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        yield rgb_frame

    cap.release()
