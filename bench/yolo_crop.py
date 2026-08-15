"""YOLO: bbox человека и вычисление кропа."""

from . import config


def get_person_bbox(frame_bgr, yolo_model):
    """Возвращает [x1, y1, x2, y2] самого большого человека или None."""
    results = yolo_model.predict(frame_bgr, classes=[0],
                                 conf=config.YOLO_CONF, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    boxes = r.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(areas.argmax())]


def get_region(bbox, frame_w, frame_h, padding=0.0):
    """Возвращает (ox, oy, rw, rh) кропа с учётом padding, либо None."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if padding > 0:
        x1 -= bw * padding
        y1 -= bh * padding
        x2 += bw * padding
        y2 += bh * padding
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(frame_w, int(x2)); y2 = min(frame_h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)
