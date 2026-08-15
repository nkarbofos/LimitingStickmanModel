"""Построение 2D-модели тела человека (stickman + rectangles).

Модель собирается из примитивов, все размеры масштабируются от S = |11 - 12|
(ширина плеч):
  - Голова: прямоугольник у носа, короткие стороны параллельны линии ушей 7-8
    (фолбэк ориентации - линия плеч 11-12).
  - Шея: четырёхугольник E-LE-RE-G (из generate-track).
  - Торс: четырёхугольник 11-12-24-23, иначе фолбэк-трапеция 75 градусов.
  - 8 конечностей: прямоугольники (длина = |A-B|, ширина = coef * S).
  - Ладони (15, 16): квадраты, ориентированы по кадру (axis-aligned).
"""

import cv2
import numpy as np

from . import config

# Индексы точек MediaPipe Pose
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_WRIST = 15
RIGHT_WRIST = 16


# ------------------------------------------------------------------
# Вспомогательные
# ------------------------------------------------------------------
def _get_point_px(landmarks, idx, region, frame_w, frame_h, min_vis=0.5):
    """Возвращает точку idx в пикселях полного кадра или None (низкая видимость)."""
    lm = landmarks[idx]
    if lm.visibility < min_vis:
        return None
    if region is not None:
        ox, oy, rw, rh = region
        x = lm.x * rw + ox
        y = lm.y * rh + oy
    else:
        x = lm.x * frame_w
        y = lm.y * frame_h
    return np.array([x, y], dtype=np.float64)


def _rotate90(v):
    """Перпендикуляр к вектору v (поворот на 90 градусов)."""
    return np.array([-v[1], v[0]], dtype=np.float64)

def _compute_head_geometry(landmarks, region, frame_w, frame_h, S):
    """Возвращает геометрию прямоугольника головы или None.

    Возврат: (nose, e1, e2, hw, hh), где
      nose - центр (точка 0);
      e1   - вдоль коротких сторон (линия ушей 7-8, фолбэк - плечи 11-12);
      e2   - перпендикуляр к e1 (вдоль длинных сторон);
      hw   - полуширина головы (короткая сторона / 2);
      hh   - полувсота головы (длинная сторона / 2).
    """
    nose = _get_point_px(landmarks, NOSE, region, frame_w, frame_h)
    if nose is None:
        return None
    W = config.STICKMAN_HEAD_W_COEF * S
    H = config.STICKMAN_HEAD_H_COEF * S

    # Ориентация: по ушам 7-8, фолбэк по плечам 11-12
    ear_l = _get_point_px(landmarks, LEFT_EAR, region, frame_w, frame_h)
    ear_r = _get_point_px(landmarks, RIGHT_EAR, region, frame_w, frame_h)
    if ear_l is not None and ear_r is not None:
        dir_vec = ear_r - ear_l
    else:
        sh_l = _get_point_px(landmarks, LEFT_SHOULDER, region, frame_w, frame_h)
        sh_r = _get_point_px(landmarks, RIGHT_SHOULDER, region, frame_w, frame_h)
        if sh_l is None or sh_r is None:
            return None
        dir_vec = sh_r - sh_l

    dlen = np.linalg.norm(dir_vec)
    if dlen < 1e-6:
        return None
    e1 = dir_vec / dlen
    e2 = _rotate90(e1)
    hw, hh = W / 2.0, H / 2.0
    return (nose, e1, e2, hw, hh)


def _scale_poly_around_center(poly, scale):
    """Масштабирует многоугольник относительно его центра (среднего вершин).

    poly  - массив вершин (N, 2), float.
    scale - коэффициент масштабирования (1.0 = без изменений).
    Возвращает масштабированный массив вершин (N, 2), float.

    Каждая вершина отодвигается от центра в scale раз:
        P_new = center + (P - center) * scale
    Порядок обхода вершин не меняется.
    """
    if abs(scale - 1.0) < 1e-9:
        return poly
    center = np.mean(poly, axis=0)
    return center + (poly - center) * scale


# ------------------------------------------------------------------
# Примитивы
# ------------------------------------------------------------------

def _build_head_rect(landmarks, region, frame_w, frame_h, S):
    """Прямоугольник головы у носа. Короткие стороны параллельны линии ушей 7-8.

    Ширина W = k_hw * S (короткая сторона, вдоль ушей/плеч).
    Высота H = k_hh * S (длинная сторона, перпендикулярно).
    Фолбэк ориентации (если уши не видны) - линия плеч 11-12.
    """
    geom = _compute_head_geometry(landmarks, region, frame_w, frame_h, S)
    if geom is None:
        return None
    nose, e1, e2, hw, hh = geom
    corners = np.array([
        nose + hw * e1 + hh * e2,
        nose - hw * e1 + hh * e2,
        nose - hw * e1 - hh * e2,
        nose + hw * e1 - hh * e2,
    ], dtype=np.int32)
    return corners


def _build_neck_quad(landmarks, region, frame_w, frame_h, S):
    """Шея: от плеч до точек на нижней стороне прямоугольника головы,
    делящих её на 3 равные части.

    Низ шеи:  E на 1/4 и G на 3/4 линии плеч 11-12.
    Верх шеи: P1, P2 - точки на НИЖНЕЙ (ближней к плечам) стороне
              прямоугольника головы, делящие её на 3 равные части
              (P1 на 1/3, P2 на 2/3). Ширина верха шеи = средняя треть
              ширины головы.
    Обход: E -> P1 -> P2 -> G.
    """
    # Плечи
    A = _get_point_px(landmarks, LEFT_SHOULDER, region, frame_w, frame_h)
    B = _get_point_px(landmarks, RIGHT_SHOULDER, region, frame_w, frame_h)
    if A is None or B is None:
        return None

    # Геометрия головы
    geom = _compute_head_geometry(landmarks, region, frame_w, frame_h, S)
    if geom is None:
        return None
    nose, e1, e2, hw, hh = geom

    # Низ шеи на линии плеч
    # E = A + 0.25 * (B - A)
    # G = A + 0.75 * (B - A)
    E = A + 0 * (B - A)
    G = A + 1 * (B - A)

    # Определяем нижнюю сторону прямоугольника головы (ближнюю к плечам):
    # это сторона, лежащая с той же стороны от носа, что и середина плеч
    M = (A + B) / 2.0
    v = M - nose
    bottom_sign = 1.0 if np.dot(v, e2) > 0 else -1.0

    # Нижняя сторона головы: от bottom_left до bottom_right вдоль e1
    bottom_offset = bottom_sign * hh * e2
    bottom_left = nose - hw * e1 + bottom_offset
    bottom_right = nose + hw * e1 + bottom_offset

    # Точки, делящие нижнюю сторону головы на 3 равные части
    P1 = bottom_left + (1.0 / 3.0) * (bottom_right - bottom_left)
    P2 = bottom_left + (2.0 / 3.0) * (bottom_right - bottom_left)

    return np.array([E, P1, P2, G], dtype=np.int32)


def _build_torso_poly(landmarks, region, frame_w, frame_h):
    """Торс из generate-track: четырёхугольник 11-12-24-23, иначе трапеция 75°.

    Вариант 1: все 4 точки (11,12,23,24) видимы -> четырёхугольник.
    Вариант 2 (бёдра не видны): фолбэк-трапеция от плеч вниз, угол 75° у основания.

    После построения торс масштабируется относительно своего центра в
    config.STICKMAN_TORSO_SCALE раз. Точки крепления рук/ног/шеи при этом
    НЕ меняются - масштабируется только сам примитив торса.
    """
    scale = config.STICKMAN_TORSO_SCALE

    # Вариант 1: четырёхугольник
    pts = {}
    for idx in (LEFT_SHOULDER, RIGHT_SHOULDER, 23, 24):
        pts[idx] = _get_point_px(landmarks, idx, region, frame_w, frame_h)
    if all(p is not None for p in pts.values()):
        poly = np.array([pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER],
                         pts[24], pts[23]], dtype=np.float64)
        poly = _scale_poly_around_center(poly, scale)
        return poly.astype(np.int32)

    # Вариант 2: фолбэк-трапеция (нужны плечи и нос)
    A = _get_point_px(landmarks, LEFT_SHOULDER, region, frame_w, frame_h)
    B = _get_point_px(landmarks, RIGHT_SHOULDER, region, frame_w, frame_h)
    N = _get_point_px(landmarks, NOSE, region, frame_w, frame_h)
    if A is None or B is None or N is None:
        return None

    AB = B - A
    ab_len = np.linalg.norm(AB)
    if ab_len < 1e-6:
        return None
    u = AB / ab_len                       # вдоль плеч (A -> B)

    # H - проекция носа на линию AB; NH направлен от носа к плечам (вниз)
    H = A + float(np.dot(N - A, u)) * u
    NH = H - N
    h = np.linalg.norm(NH)
    if h < 1e-3:
        return None
    e2 = NH / h                           # направление вниз (от головы к ногам)

    depth = (config.STICKMAN_FALLBACK_LENGTH_FACTOR - 1.0) * h
    alpha = np.radians(config.STICKMAN_FALLBACK_TORSO_ANGLE)
    inset = depth / np.tan(alpha)

    C = A + inset * u + depth * e2
    D = B - inset * u + depth * e2

    poly = np.array([A, B, D, C], dtype=np.float64)
    poly = _scale_poly_around_center(poly, scale)
    return poly.astype(np.int32)


def _build_limb_rect(A, B, width):
    """Прямоугольник конечности от A до B.

    Длинные стороны параллельны A-B, длина = |A-B|, ширина (короткая) = width.
    """
    AB = B - A
    length = np.linalg.norm(AB)
    if length < 1e-6:
        return None
    u = AB / length
    n = _rotate90(u)
    mid = (A + B) / 2.0
    hl, hw = length / 2.0, width / 2.0
    corners = np.array([
        mid + hl * u + hw * n,
        mid - hl * u + hw * n,
        mid - hl * u - hw * n,
        mid + hl * u - hw * n,
    ], dtype=np.int32)
    return corners


def _build_palm_square(wrist, side):
    """Квадрат ладони, центрированный в запястье, axis-aligned (по кадру)."""
    half = side / 2.0
    x, y = wrist
    return np.array([
        [x - half, y - half],
        [x + half, y - half],
        [x + half, y + half],
        [x - half, y + half],
    ], dtype=np.int32)


# ------------------------------------------------------------------
# Сборка модели
# ------------------------------------------------------------------
def build_stickman_mask(pose_landmarks_list, region, frame_w, frame_h):
    """Строит бинарную маску модели тела (uint8, 0/255) в координатах полного кадра.

    Возвращает маску (frame_h, frame_w) или None.
    """
    if not pose_landmarks_list:
        return None
    landmarks = pose_landmarks_list[0]
    if len(landmarks) < 29:
        return None

    # Ширина плеч S
    sh_l = _get_point_px(landmarks, LEFT_SHOULDER, region, frame_w, frame_h)
    sh_r = _get_point_px(landmarks, RIGHT_SHOULDER, region, frame_w, frame_h)
    if sh_l is None or sh_r is None:
        return None
    S = np.linalg.norm(sh_r - sh_l)
    if S < 1e-6:
        return None

    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

    # 1. Торс
    torso = _build_torso_poly(landmarks, region, frame_w, frame_h)
    if torso is not None:
        cv2.fillPoly(mask, [torso], 255)

    # neck = _build_neck_quad(landmarks, region, frame_w, frame_h, S)
    # if neck is not None:
    #     cv2.fillPoly(mask, [neck], 255)

    # 3. Голова
    head = _build_head_rect(landmarks, region, frame_w, frame_h, S)
    if head is not None:
        cv2.fillPoly(mask, [head], 255)

    # 4. Конечности (8 прямоугольников)
    for (a, b), coef in config.STICKMAN_LIMB_COEFS.items():
        A = _get_point_px(landmarks, a, region, frame_w, frame_h)
        B = _get_point_px(landmarks, b, region, frame_w, frame_h)
        if A is None or B is None:
            continue
        rect = _build_limb_rect(A, B, coef * S)
        if rect is not None:
            cv2.fillPoly(mask, [rect], 255)

    # 5. Ладони (квадраты в запястьях)
    palm_side = config.STICKMAN_PALM_COEF * S
    for wrist_idx in (LEFT_WRIST, RIGHT_WRIST):
        wrist = _get_point_px(landmarks, wrist_idx, region, frame_w, frame_h)
        if wrist is None:
            continue
        square = _build_palm_square(wrist, palm_side)
        cv2.fillPoly(mask, [square], 255)

    return mask


# ------------------------------------------------------------------
# Визуализация
# ------------------------------------------------------------------
def overlay_stickman(frame, mask, color=(255, 100, 0), alpha=0.4):
    """Накладывает маску модели полупрозрачно + рисует контуры."""
    if mask is None:
        return frame

    colored = np.zeros_like(frame)
    colored[mask > 0] = color

    mask_bool = mask > 0
    blended = cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)
    result = frame.copy()
    result[mask_bool] = blended[mask_bool]

    # Контуры для наглядности
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result
