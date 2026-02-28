import cv2
from vision_models.FaceDetector import FaceDetector
from vision_models.LandmarkDetector import LandmarkDetector

class FramePipeline:
    def __init__(self, effect):
        self.face_detector = FaceDetector()
        self.landmark_detector = LandmarkDetector()
        self.effect = effect

    def process(self, frame):

        detections = self.face_detector.detect(frame)
        landmarks = self.landmark_detector.get_landmarks(frame)

        bbox = None
        if detections:
            bbox = detections[0].bounding_box

        frame = self.effect.apply(frame, landmarks, bbox)

        return frame