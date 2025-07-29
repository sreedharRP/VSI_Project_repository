# main.py

import sys
import cv2
import threading
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.uic import loadUi
from yolo_stream import detect_with_three_models  # ✅ Use updated 3-model function


class MainApp(QtWidgets.QDialog):
    def __init__(self):
        super(MainApp, self).__init__()
        loadUi("test.ui", self)

        self.pushButton.clicked.connect(self.start_detection)
        self.pushButton_2.clicked.connect(self.stop_detection)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.running = False
        self.frame = None

    def start_detection(self):
        if self.running:
            return
        self.running = True
        self.label.setText("🧠 Running 3 YOLO models on the same stream...")

        self.thread = threading.Thread(target=self.run_detection, daemon=True)
        self.thread.start()
        self.timer.start(30)

    def run_detection(self):
        try:
            for frame in detect_with_three_models(
                model_path_1=r"C:\Users\Lightning_sree\Documents\Design\college_files\Visual_&_Spoken_Interface\Yolo_finger_model\faces_data\my_model_1\my_model.pt",
                model_path_2=r"C:\Users\Lightning_sree\Documents\Design\college_files\Visual_&_Spoken_Interface\Yolo_finger_model\my_model_2\my_model.pt",
                model_path_3=r"C:\Users\Lightning_sree\Documents\Design\college_files\Visual_&_Spoken_Interface\Yolo_finger_model\Expression\my_model_3\my_model.pt",
                source=0,  # webcam index, not "usb0" unless it's symbolic in your system
                resolution="1280x720"
            ):
                if not self.running:
                    break
                self.frame = frame
        except Exception as e:
            self.label.setText(f"❌ Error: {str(e)}")
            self.running = False

    def update_frame(self):
        if self.frame is not None:
            # Convert RGB (from OpenCV) to BGR for QImage display
            bgr_frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            h, w, ch = bgr_frame.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(bgr_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888)
            pixmap = QtGui.QPixmap.fromImage(qimg)

            scene = QtWidgets.QGraphicsScene()
            scene.addPixmap(pixmap)
            self.graphicsView.setScene(scene)
            self.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def stop_detection(self):
        self.running = False
        self.timer.stop()
        if self.graphicsView.scene():
            self.graphicsView.scene().clear()
        self.label.setText("🛑 Detection stopped")

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.setWindowTitle("YOLO - 3 Models on One Stream")
    window.show()
    sys.exit(app.exec_())
