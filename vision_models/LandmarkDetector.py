import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class LandmarkDetector:

    def __init__(self, model_path="./vision_models/face_landmarker.task"):

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.timestamp = 0

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.timestamp = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp
        )

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]