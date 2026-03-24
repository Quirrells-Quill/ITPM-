import numpy as np
import cv2

heatmap = None

def update_heatmap(frame, cx, cy):
    global heatmap

    if heatmap is None:
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    heatmap[cy, cx] += 2
    return heatmap


def overlay_heatmap(frame, heatmap):
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    return cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)