import cv2
from vision_models.FaceDetector import FaceDetector
from vision_models.LandmarkDetector import LandmarkDetector
from train_face_model.FaceRollNet import FaceRollNet

class FramePipeline:
    def __init__(self, effect):
        model = FaceRollNet()  # tu clase del modelo
        self.face_detector = FaceDetector(
            model,
            weights_path="./train_face_model/face_roll_model_300WLP.pth",
            device="cuda"
        )
        self.landmark_detector = LandmarkDetector()
        self.effect = effect

    def process(self, frame):

        detections = self.face_detector.detect(frame)
        landmarks = self.landmark_detector.get_landmarks(frame)

        bbox = None
        if detections is None:
            return frame

        bbox = detections["bbox"]
        roll = detections["roll"]

        frame = self.effect.apply(frame, landmarks, bbox)

        return frame