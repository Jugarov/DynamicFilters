import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:

    def __init__(self, model_path="./vision_models/blaze_face_short_range.tflite"):

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )

        self.detector = vision.FaceDetector.create_from_options(options)
        self.timestamp = 0

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.timestamp = int(time.time() * 1000)

        result = self.detector.detect_for_video(
            mp_image,
            self.timestamp
        )

        return result.detections