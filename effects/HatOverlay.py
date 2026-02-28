import cv2
import numpy as np
from effects.BaseEffect import BaseEffect
from utils.blending import overlay_image


class HatOverlayEffect(BaseEffect):

    def __init__(self, hat_image_path: str):
        self.hat = cv2.imread(hat_image_path, cv2.IMREAD_UNCHANGED)
        if self.hat is None:
            raise ValueError("No se pudo cargar la imagen del sombrero")

    def apply(self, frame, landmarks, bbox):

        if landmarks is None or bbox is None:
            return frame

        h, w, _ = frame.shape

        # 📏 1. Escalado usando bounding box
        face_width = int(bbox.width)
        face_height = int(bbox.height)

        hat_width = int(face_width * 1.4)
        scale = hat_width / self.hat.shape[1]
        hat_height = int(self.hat.shape[0] * scale)

        resized_hat = cv2.resize(self.hat, (hat_width, hat_height))

        # 🔄 2. Calcular ángulo usando ojos
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        x1 = int(left_eye.x * w)
        y1 = int(left_eye.y * h)
        x2 = int(right_eye.x * w)
        y2 = int(right_eye.y * h)

        angle = -np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # 🔁 Rotar sombrero
        center = (hat_width // 2, hat_height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_hat = cv2.warpAffine(
            resized_hat,
            rotation_matrix,
            (hat_width, hat_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # 🎯 3. Posicionamiento

        top_head = landmarks[10]

        head_x = int(top_head.x * w)
        head_y = int(top_head.y * h)

        x = head_x - hat_width // 2
        y = head_y - int(hat_height * 0.9)

        return overlay_image(frame, rotated_hat, x, y)