import cv2
import numpy as np

def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]

    if x < 0 or y < 0:
        return background

    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    roi = background[y:y+h, x:x+w]

    blended = roi * (1 - mask) + overlay_img * mask
    background[y:y+h, x:x+w] = blended.astype(np.uint8)

    return background