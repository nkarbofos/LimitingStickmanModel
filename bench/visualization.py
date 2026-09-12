"""Отрисовка: маска сегментации, скелет, овал лица."""

import cv2
import numpy as np

from . import config
from .stickman_model import torso_vertex


def region_or_full(region, frame):
    if region is not None:
        return region
    return (0, 0, frame.shape[1], frame.shape[0])


def category_mask_to_colored(category_mask_uint8):
    h, w = category_mask_uint8.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in config.CATEGORY_COLORS.items():
        colored[category_mask_uint8 == label] = color
    return colored

def confidence_masks_to_heatmap(confidence_masks, colormap=cv2.COLORMAP_JET):
    """Строит тепловую карту уверенности человека из confidence masks.

    Берёт максимум уверенности по категориям человека (индексы 1..N),
    ИСКЛЮЧАЯ background (индекс 0).

    Возвращает:
        heatmap   - BGR-изображение (H, W, 3), раскрашенная тепловая карта
        conf_u8   - grayscale (H, W), uint8, сама уверенность (для альфа-наложения)
    """
    conf_list = []
    for m in confidence_masks:
        try:
            arr = m.numpy_view()
        except AttributeError:
            arr = m.numpy()
        arr = np.squeeze(np.array(arr, dtype=np.float32))
        conf_list.append(arr)

    # Стек категорий: (num_categories, H, W)
    conf_stack = np.stack(conf_list, axis=0)

    # # Максимум по категориям человека (1..N), исключая background (0)
    # if conf_stack.shape[0] > 1:
    #     # person_conf = np.max(conf_stack[1:], axis=0)
    #     # person_conf = np.ones_like(conf_stack[0]) - conf_stack[0]
    # else:
    #     person_conf = conf_stack[0]
    person_conf = conf_stack[0]

    person_conf = np.ones_like(conf_stack[0]) - np.clip(person_conf, 0.0, 1.0)
    conf_u8 = (person_conf * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(conf_u8, colormap)
    return heatmap, conf_u8


def draw_pose_landmarks(frame, pose_landmarks, region):
    ox, oy, rw, rh = region_or_full(region, frame)
    for landmarks in pose_landmarks:
        pts = {}
        for idx, lm in enumerate(landmarks):
            if lm.visibility < 0.5:
                continue
            pts[idx] = (int(lm.x * rw) + ox, int(lm.y * rh) + oy)
        for a, b in config.POSE_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2)
        for pt in pts.values():
            cv2.circle(frame, pt, 3, (0, 0, 255), -1)
    return frame


def draw_face_oval(frame, face_landmarks, region):
    ox, oy, rw, rh = region_or_full(region, frame)
    for landmarks in face_landmarks:
        pts = [(int(landmarks[i].x * rw) + ox, int(landmarks[i].y * rh) + oy)
               for i in config.FACE_OVAL]
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=2)
    return frame


def compute_face_oval_size(face_landmarks, rw, rh):
    if not face_landmarks:
        return None
    landmarks = face_landmarks[0]
    xs = [landmarks[i].x * rw for i in config.FACE_OVAL]
    ys = [landmarks[i].y * rh for i in config.FACE_OVAL]
    return (max(xs) - min(xs)), (max(ys) - min(ys))


def draw_neck_points(frame, neck_quad, head_corners, torso_quad,
                     sh_l=None, sh_r=None):
    """Вершины фигуры шеи и опорные точки, по которым она строится.

    Включается config.DRAW_NECK_POINTS. Контур шеи собран как левая сторона
    от головы к торсу, затем правая от торса к голове, поэтому уровень j --
    это neck_quad[j] слева и neck_quad[2k-1-j] справа, где k -- половина
    длины контура.
    """
    def dot(p, color, label):
        if p is None:
            return
        q = np.asarray(p, dtype=np.float64)
        if q.shape != (2,) or not np.all(np.isfinite(q)):
            return
        c = (int(round(q[0])), int(round(q[1])))
        cv2.circle(frame, c, config.NECK_POINT_RADIUS, color, -1)
        cv2.circle(frame, c, config.NECK_POINT_RADIUS + 2, (0, 0, 0), 1)
        if config.NECK_POINT_LABELS and label:
            cv2.putText(frame, label, (c[0] + 7, c[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if neck_quad is not None:
        q = np.asarray(neck_quad, dtype=np.float64)
        k = len(q) // 2
        for j in range(k):
            dot(q[j], config.NECK_POINT_COLOR, 'L%d' % j)
            dot(q[len(q) - 1 - j], config.NECK_POINT_COLOR, 'R%d' % j)

    if head_corners is not None:
        h = np.asarray(head_corners, dtype=np.float64)
        if len(h) >= 4:
            dot(h[3], config.NECK_REF_COLOR, 'H3')
            dot(h[2], config.NECK_REF_COLOR, 'H2')
    if torso_quad is not None:
        for name in ('TL', 'TR'):
            dot(torso_vertex(torso_quad, name), config.NECK_REF_COLOR, name)
    if sh_l is not None and sh_r is not None:
        a = np.asarray(sh_l, dtype=np.float64)
        b = np.asarray(sh_r, dtype=np.float64)
        dot(a, config.NECK_REF_COLOR, '11')
        dot(b, config.NECK_REF_COLOR, '12')
        dot((a + b) / 2.0, config.NECK_REF_COLOR, 'C')
    return frame


def fill_poly_with_alpha(frame, polygon, color, alpha):
    """Заполняет полигон цветом с прозрачностью.

    frame   - кадр (H, W, 3), BGR.
    polygon - полигон (N, 2) как np.array (координаты вершин).
    color   - цвет заполнения (B, G, R).
    alpha   - прозрачность заполнения (0.0 = не видно, 1.0 = полностью).

    Возвращает кадр с заполненным полигоном (модифицированная копия).
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    colored = np.zeros_like(frame)
    colored[mask > 0] = color
    mask_bool = mask > 0
    blended = cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)
    result = frame.copy()
    result[mask_bool] = blended[mask_bool]
    return result
