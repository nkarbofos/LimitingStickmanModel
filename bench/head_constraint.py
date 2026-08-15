"""Ограничение головы эллипсом (верхний полуэллипс)."""

import cv2
import numpy as np

from . import config


def compute_head_ellipse_from_pose(pose_landmarks, region, frame_w, frame_h,
                                   radius_u_coef=0.35, radius_n_coef=0.5):
    """Вычисляет эллипс головы в координатах ИСХОДНОГО кадра.

    Возвращает (cx, cy, a, b, u, n) или None.
    cx, cy - центр (нос);
    a - полуось вдоль плеч (полуширина);
    b - полуось к голове (полувысота);
    u - единичный вектор вдоль плеч;
    n - перпендикуляр к плечам в сторону головы.
    """
    if not pose_landmarks:
        return None
    lm = pose_landmarks[0]
    if len(lm) < 13:
        return None

    ox, oy, rw, rh = region if region else (0, 0, frame_w, frame_h)

    def lm_pt(idx):
        return (lm[idx].x * rw + ox, lm[idx].y * rh + oy)

    N = np.array(lm_pt(config.POSE_NOSE), dtype=np.float64)
    A = np.array(lm_pt(config.POSE_LEFT_SHOULDER), dtype=np.float64)
    B = np.array(lm_pt(config.POSE_RIGHT_SHOULDER), dtype=np.float64)

    AB = B - A
    S = float(np.linalg.norm(AB))
    if S < 1e-6:
        return None
    u = AB / S

    # Нормаль к плечам в сторону головы
    perp = np.array([-u[1], u[0]])
    to_nose = N - (A + B) / 2.0
    n = perp if float(np.dot(perp, to_nose)) > 0 else -perp

    a = radius_u_coef * S   # полуось вдоль плеч
    b = radius_n_coef * S   # полуось к голове
    return (N[0], N[1], a, b, u, n)


def apply_head_ellipse_constraint(mask_rgb, params):
    """Обрезает RGB-маску: всё ВЫШЕ диаметра и ВНЕ эллипса -> чёрный.

    Ниже диаметра (подбородок/шея/торс) не ограничивается.
    """
    cx, cy, a, b, u, n = params
    h, w = mask_rgb.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = xs - cx
    dy = ys - cy

    # Проекции на оси эллипса
    proj_u = dx * u[0] + dy * u[1]
    proj_n = dx * n[0] + dy * n[1]

    inside_ellipse = (proj_u * proj_u) / (a * a) + (proj_n * proj_n) / (b * b) <= 1.0
    below_diameter = proj_n <= 0          # ниже диаметра (в сторону торса)

    allowed = inside_ellipse | below_diameter
    mask_rgb[~allowed] = 0
    return mask_rgb


def draw_head_ellipse_arc(frame, params, color=(0, 200, 255),
                          thickness=2, num_pts=48):
    """Рисует верхний полуэллипс (дугу) и диаметр на кадре."""
    cx, cy, a, b, u, n = params
    pts = []
    for i in range(num_pts + 1):
        theta = np.pi * i / num_pts
        # P = C + a*cos(theta)*u + b*sin(theta)*n
        px = cx + a * np.cos(theta) * u[0] + b * np.sin(theta) * n[0]
        py = cy + a * np.cos(theta) * u[1] + b * np.sin(theta) * n[1]
        pts.append((int(px), int(py)))
    pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts_arr], isClosed=False, color=color, thickness=thickness)

    # Диаметр (вдоль плеч, через нос), полуось a
    p1 = (int(cx + a * u[0]), int(cy + a * u[1]))
    p2 = (int(cx - a * u[0]), int(cy - a * u[1]))
    cv2.line(frame, p1, p2, color, 1)
    return frame
