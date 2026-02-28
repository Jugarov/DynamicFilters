import cv2

class Camera:
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("No se pudo leer frame de la cámara")
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()