import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

class FaceDetector:

    def __init__(self, model, weights_path, device=None):
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        # IMPORTANTE: mismo transform que en entrenamiento
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def detect(self, frame):
        h, w = frame.shape[:2]

        # OpenCV BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_bbox, pred_roll = self.model(input_tensor)

        # Convertir UNA SOLA VEZ
        pred_bbox = pred_bbox.detach().cpu().numpy().reshape(-1)
        pred_roll = pred_roll.detach().cpu().item()

        cx, cy, bw, bh = pred_bbox

        # Convertir de centro a esquinas (normalizado)
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Escalar a tamaño real
        x1 = int(x1 * w)
        x2 = int(x2 * w)
        y1 = int(y1 * h)
        y2 = int(y2 * h)

        # Orden seguro
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        return {
            "bbox": (x1, y1, x2, y2),
            "roll": pred_roll
        }