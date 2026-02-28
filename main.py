import cv2
from camera.Camera import Camera
from camera.FramePipeline import FramePipeline
from effects.HatOverlay import HatOverlayEffect

def main():
    camera = Camera()
    camera.start()

    hat_effect = HatOverlayEffect("assets/wizard_hat.png")
    pipeline = FramePipeline(hat_effect)

    try:
        while True:
            frame = camera.read()
            frame = pipeline.process(frame)

            cv2.imshow("Hat Filter", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        camera.release()


if __name__ == "__main__":
    main()