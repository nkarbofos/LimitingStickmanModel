"""Построение 2D-модели тела человека (stickman + rectangles).

Модель собирается из примитивов, все размеры масштабируются от S = |11 - 12|
(ширина плеч):
  - Голова: прямоугольник у носа, короткие стороны параллельны линии ушей 7-8
    (фолбэк ориентации - линия плеч 11-12).
  - Шея: четырёхугольник E-LE-RE-G (из generate-track).
  - Торс: четырёхугольник 11-12-24-23, иначе фолбэк-трапеция 75 градусов.
  - 8 конечностей: прямоугольники (длина = |A-B|, ширина = coef * S).
  - Ладони (15, 16): квадраты, ориентированы по кадру (axis-aligned).
  - Верх ног: четырёхугольники BL/BR + вершины у колена (нужен торс).
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
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


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


# Пары точек: руки, ноги, ладони, ступни
ARM_PAIRS = ((11, 13), (13, 15), (12, 14), (14, 16))
# Бёдра вынесены отдельно: вместо прямоугольников для них строятся
# четырёхугольники (см. build_thigh_quads). LEG_PAIRS оставлен целиком --
# по нему определяется, что масштабировать шириной таза.
THIGH_PAIRS = ((23, 25), (24, 26))
SHIN_PAIRS = ((25, 27), (26, 28))
# Верхняя сторона четырёхугольника бедра: (индекс вершины торса, точка бедра).
# Пары перекрёстные: ноге (24, 26) -- BL и точка 23, ноге (23, 25) -- BR и 24.
# BL как раз и найден вытягиванием от точки 23, а BR -- от точки 24, поэтому
# верхняя сторона совпадает с этим вытягиванием.
# Индексы в четырёхугольнике торса [TL, TR, BR, BL]: BR = 2, BL = 3.
# Имена вершин торса. Контур бывает двух размеров: четырёхугольник
# [TL, TR, BR, BL] (старые калибровки) и шестиугольник плечи-живот-торс
# [TL, TR, MR, BR, BL, ML]. Обращаться по имени, а не по числу: при переходе
# к шестиугольнику индексы BR и BL сдвинулись.
_TORSO_IDX = {4: {'TL': 0, 'TR': 1, 'BR': 2, 'BL': 3},
              6: {'TL': 0, 'TR': 1, 'MR': 2, 'BR': 3, 'BL': 4, 'ML': 5}}


def polygon_self_intersects(poly):
    """True, если контур самопересекается (проверяются несмежные рёбра)."""
    p = np.asarray(poly, dtype=np.float64)
    n = len(p)
    if n < 4:
        return False

    def side(a, b, c):
        return np.sign(np.cross(b - a, c - a))

    for i in range(n):
        for j in range(i + 1, n):
            if (j + 1) % n == i or (i + 1) % n == j:
                continue                      # смежные рёбра
            a, b = p[i], p[(i + 1) % n]
            c, d = p[j], p[(j + 1) % n]
            if side(a, b, c) != side(a, b, d) and side(c, d, a) != side(c, d, b):
                return True
    return False


def torso_vertex(torso_quad, name):
    """Вершина контура торса по имени ('TL', 'TR', 'BR', 'BL', 'ML', 'MR').

    Возвращает None, если контур не того размера или вершины в нём нет
    (например 'ML' у старого четырёхугольника).
    """
    if torso_quad is None:
        return None
    q = np.asarray(torso_quad, dtype=np.float64)
    idx = _TORSO_IDX.get(len(q), {}).get(name)
    return None if idx is None else q[idx]


THIGH_TOP_SIDES = {(24, 26): ('BL', LEFT_HIP),    # BL, точка 23
                   (23, 25): ('BR', RIGHT_HIP)}   # BR, точка 24
LEG_PAIRS = THIGH_PAIRS + SHIN_PAIRS
# Ладони и ступни: (точка сустава, точки дальнего конца).
# Если дальних точек несколько -- берётся их середина. Для ладони это середина
# между указательным (19/20) и мизинцем (17/18), т.е. центр линии костяшек.
PALM_SPECS = ((LEFT_WRIST, (LEFT_INDEX, LEFT_PINKY)),
              (RIGHT_WRIST, (RIGHT_INDEX, RIGHT_PINKY)))
FOOT_SPECS = ((LEFT_ANKLE, (LEFT_FOOT_INDEX,)),
              (RIGHT_ANKLE, (RIGHT_FOOT_INDEX,)))

# Треугольники таза: (правое бедро, левое бедро, колено). Закрывают область
# между бёдрами и коленями, которую не покрывают ни торс, ни прямоугольники ног.
HIP_TRIANGLES = ((24, 23, 26), (24, 23, 25))



# Треугольники "бедро - колено - нижняя вершина торса".
# Вершина торса берётся ПРОТИВОПОЛОЖНАЯ ноге: правой ноге (24, 26) -- BL,
# левой (23, 25) -- BR. Индекс в четырёхугольнике [TL, TR, BR, BL]: BR=2, BL=3.
TORSO_LEG_TRIANGLES = ((RIGHT_HIP, RIGHT_KNEE, 3),   # 24, 26, BL
                       (LEFT_HIP,  LEFT_KNEE,  2))   # 23, 25, BR


def build_torso_leg_triangles(landmarks, region, frame_w, frame_h, torso_quad):
    """Треугольники (24, 26, BL) и (23, 25, BR).

    torso_quad -- четырёхугольник торса [TL, TR, BR, BL] из calibrate_torso;
    None -- торс не откалиброван, треугольники не строятся.
    Треугольники, у которых бедро или колено не видны, пропускаются.
    Возвращает список np.array (3, 2) int32.
    """
    if torso_quad is None:
        return []
    quad = np.asarray(torso_quad, dtype=np.float64)
    if quad.shape[0] < 4:
        return []
    tris = []
    for hip_idx, knee_idx, quad_idx in TORSO_LEG_TRIANGLES:
        hip = _get_point_px(landmarks, hip_idx, region, frame_w, frame_h)
        knee = _get_point_px(landmarks, knee_idx, region, frame_w, frame_h)
        if hip is None or knee is None:
            continue
        tris.append(np.array([hip, knee, quad[quad_idx]], dtype=np.int32))
    return tris



def build_thigh_quads(landmarks, region, frame_w, frame_h, torso_quad):
    """Четырёхугольники верхней части ног вместо прямоугольников бедра.

    Для отрезка 24-26 (и 23-25) берутся две вершины БЫВШЕГО прямоугольника
    бедра со стороны колена и верхняя сторона из THIGH_TOP_SIDES: для ноги
    24-26 это BL и точка 23, для ноги 23-25 -- BR и точка 24. Ширина бывшего
    прямоугольника считается как раньше: STICKMAN_LIMB_COEFS * ширина таза.

    torso_quad -- четырёхугольник торса [TL, TR, BR, BL]; None -- не строим.
    """
    if torso_quad is None:
        return []
    quad = np.asarray(torso_quad, dtype=np.float64)
    if quad.shape[0] < 4:
        return []
    def point(idx):
        return _get_point_px(landmarks, idx, region, frame_w, frame_h)

    sh_l, sh_r = point(LEFT_SHOULDER), point(RIGHT_SHOULDER)
    hip_l, hip_r = point(LEFT_HIP), point(RIGHT_HIP)
    scale = None
    if hip_l is not None and hip_r is not None:
        d = float(np.linalg.norm(hip_r - hip_l))
        if d > 1e-6:
            scale = d
    if scale is None and sh_l is not None and sh_r is not None:
        d = float(np.linalg.norm(sh_r - sh_l))
        if d > 1e-6:
            scale = d
    if scale is None:
        return []

    quads = []
    for hip_idx, knee_idx in THIGH_PAIRS:
        A, B = point(hip_idx), point(knee_idx)
        if A is None or B is None:
            continue
        coef = config.STICKMAN_LIMB_COEFS.get((hip_idx, knee_idx))
        if coef is None:
            continue
        rect = _build_limb_rect(A, B, coef * scale)
        if rect is None:
            continue
        AB = B - A
        u = AB / np.linalg.norm(AB)
        mid = (A + B) / 2.0
        # вершины со стороны колена -- те, чья проекция на u положительна
        knee_pts = [p for p in rect.astype(np.float64)
                    if float(np.dot(p - mid, u)) > 0.0]
        if len(knee_pts) != 2:
            continue
        side = THIGH_TOP_SIDES.get((hip_idx, knee_idx))
        if side is None:
            continue
        corner_name, top_hip_idx = side
        corner = torso_vertex(quad, corner_name)   # BL или BR
        if corner is None:
            continue
        top_hip = point(top_hip_idx)            # точка 23 или 24
        if top_hip is None:
            continue
        # Обход: вершина торса -> точка бедра -> вершина у колена ближе к точке
        # бедра -> вершина ближе к вершине торса. Порядок задаётся проекцией на
        # саму верхнюю сторону, поэтому не самопересекается при любом наклоне.
        w_axis = top_hip - corner
        knee_pts.sort(key=lambda p: float(np.dot(p - corner, w_axis)))
        near_corner, near_hip = knee_pts[0], knee_pts[1]
        quads.append(np.array([corner, top_hip, near_hip, near_corner],
                              dtype=np.int32))
    return quads



# Верхние сегменты рук: (плечо, локоть). Для них строятся области до головы.
UPPER_ARM_PAIRS = ((12, 14), (11, 13))


def _ray_polygon_hit(origin, direction, poly):
    """Первое пересечение луча origin+t*direction (t>0) с контуром poly.

    Возвращает точку (2,) или None.
    """
    d = np.asarray(direction, dtype=np.float64)
    O = np.asarray(origin, dtype=np.float64)
    if np.linalg.norm(d) < 1e-9:
        return None
    best_t, best_p = None, None
    n = len(poly)
    for i in range(n):
        A = np.asarray(poly[i], dtype=np.float64)
        B = np.asarray(poly[(i + 1) % n], dtype=np.float64)
        e = B - A
        den = d[0] * (-e[1]) - d[1] * (-e[0])
        if abs(den) < 1e-9:
            continue                      # луч параллелен ребру
        w = A - O
        t = (w[0] * (-e[1]) - w[1] * (-e[0])) / den    # вдоль луча
        u = (d[0] * w[1] - d[1] * w[0]) / den          # вдоль ребра
        if t > 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
            if best_t is None or t < best_t:
                best_t, best_p = t, O + t * d
    return best_p



def _build_triangle(landmarks, idxs, region, frame_w, frame_h):
    """Треугольник по трём индексам точек позы или None, если точка не видна."""
    pts = []
    for idx in idxs:
        p = _get_point_px(landmarks, idx, region, frame_w, frame_h)
        if p is None:
            return None
        pts.append(p)
    return np.array(pts, dtype=np.int32)


def _build_extended_rect(A, B, width, extend_coef):
    """Прямоугольник вокруг отрезка A-B, удлинённого НАРУЖУ от A.

    Точка A (сустав: запястье или лодыжка) остаётся на месте, дальняя точка B
    отодвигается: B' = A + (B - A) * extend_coef. При extend_coef = 1.0
    удлинения нет. Ширина (короткая сторона) = width.
    """
    AB = B - A
    if np.linalg.norm(AB) < 1e-6:
        return None
    return _build_limb_rect(A, A + AB * extend_coef, width)


def build_body_rects(landmarks, region, frame_w, frame_h):
    """Строит прямоугольники частей тела в координатах полного кадра.

    Возвращает dict со списками прямоугольников (каждый -- np.array (4, 2)):
        'arms'  -- 4 прямоугольника рук (ширина из STICKMAN_LIMB_COEFS)
        'legs'  -- 4 прямоугольника ног (ширина из STICKMAN_LIMB_COEFS)
        'palms' -- 2 ладони: отрезок от запястья (15/16) к середине между
                   указательным и мизинцем (19+17 / 20+18), удлинённый наружу
        'feet'  -- 2 ступни: отрезок 27-31 / 28-32, удлинённый наружу
        'arm_tops_quad' -- четырёхугольник ABCD по верхним сторонам рук
    Ширина ног и ступней масштабируется тазом |23-24| ('S_hip'),
    рук и ладоней -- плечами ('S').
    Части, для которых точки не видны, просто отсутствуют в списках.
    Возвращает None, если не видны плечи (не от чего масштабировать).
    """
    sh_l = _get_point_px(landmarks, LEFT_SHOULDER, region, frame_w, frame_h)
    sh_r = _get_point_px(landmarks, RIGHT_SHOULDER, region, frame_w, frame_h)
    if sh_l is None or sh_r is None:
        return None
    S = float(np.linalg.norm(sh_r - sh_l))
    if S < 1e-6:
        return None

    def point(idx):
        return _get_point_px(landmarks, idx, region, frame_w, frame_h)

    # Ноги и ступни масштабируются шириной таза |23-24|, а не плеч:
    # у бедра и голени размер связан с тазом, а не с плечевым поясом.
    # Если бёдра не видны -- откатываемся на ширину плеч.
    hip_l = point(LEFT_HIP)
    hip_r = point(RIGHT_HIP)
    S_hip = S
    if hip_l is not None and hip_r is not None:
        d_hip = float(np.linalg.norm(hip_r - hip_l))
        if d_hip > 1e-6:
            S_hip = d_hip

    rects = {'arms': [], 'legs': [], 'palms': [], 'feet': [],
             'arm_tops_quad': [], 'S': S, 'S_hip': S_hip}

    for pairs, key, scale in ((ARM_PAIRS, 'arms', S), (SHIN_PAIRS, 'legs', S_hip)):
        for pair in pairs:
            coef = config.STICKMAN_LIMB_COEFS.get(pair)
            if coef is None:
                continue
            A, B = point(pair[0]), point(pair[1])
            if A is None or B is None:
                continue
            rect = _build_limb_rect(A, B, coef * scale)
            if rect is not None:
                rects[key].append(rect)

    for specs, key, width_coef, ext_coef, scale in (
            (PALM_SPECS, 'palms', config.STICKMAN_PALM_COEF,
             config.STICKMAN_PALM_EXTEND_COEF, S),
            (FOOT_SPECS, 'feet', config.STICKMAN_FOOT_COEF,
             config.STICKMAN_FOOT_EXTEND_COEF, S_hip)):
        for joint_idx, distal_ids in specs:
            A = point(joint_idx)
            if A is None:
                continue
            # Дальний конец -- середина видимых точек (одна точка -- она сама)
            distal = [q for q in (point(i) for i in distal_ids) if q is not None]
            if not distal:
                continue
            B = np.mean(distal, axis=0)
            rect = _build_extended_rect(A, B, width_coef * scale, ext_coef)
            if rect is not None:
                rects[key].append(rect)

    rects['arm_tops_quad'] = build_arm_tops_quad(
        landmarks, region, frame_w, frame_h)

    return rects


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
def build_upper_body_hull(landmarks, region, frame_w, frame_h,
                          head_corners, torso_quad):
    """Выпуклый многоугольник XABCDY между верхом рук и головой.

    A, B -- верхняя (плечевая) сторона прямоугольника отрезка 14-12;
    C, D -- то же для отрезка 13-11;
    X, Y -- пересечения с прямоугольником головы лучей, выпущенных из ВНЕШНИХ
            верхних вершин этих прямоугольников параллельно соответствующей
            стороне четырёхугольника шеи (стороне, что соединяет нижний угол
            головы с откалиброванной вершиной торса TL/TR).

    Берётся именно выпуклая оболочка, а не обход X->A->B->C->D->Y: на реальных
    кадрах все шесть точек и так выпуклы, но при резком наклоне порядок может
    нарушиться, и оболочка гарантирует несамопересекающийся многоугольник.

    Нужны голова и торс из калибровки; без них возвращается пустой список.
    Если луч не дошёл до головы, соответствующая точка снапится на угол головы,
    задающий ту же сторону шеи, -- фигура строится всегда.
    """
    if head_corners is None or torso_quad is None:
        return []
    head = np.asarray(head_corners, dtype=np.float64)
    quad = np.asarray(torso_quad, dtype=np.float64)
    if head.shape[0] < 4 or quad.shape[0] < 4:
        return []

    head_poly = [head[0], head[1], head[2], head[3]]
    head_center = head.mean(axis=0)
    # стороны шеи: (вершина торса, нижний угол головы)
    neck_sides = ((torso_vertex(quad, 'TL'), head[3]),
                  (torso_vertex(quad, 'TR'), head[2]))

    def point(idx):
        return _get_point_px(landmarks, idx, region, frame_w, frame_h)

    sh_l, sh_r = point(LEFT_SHOULDER), point(RIGHT_SHOULDER)
    if sh_l is None or sh_r is None:
        return []
    S = float(np.linalg.norm(sh_r - sh_l))
    if S < 1e-6:
        return []

    pts = []
    for pair in UPPER_ARM_PAIRS:
        A, B = point(pair[0]), point(pair[1])
        coef = config.STICKMAN_LIMB_COEFS.get(pair)
        if A is None or B is None or coef is None:
            return []
        rect = _build_limb_rect(A, B, coef * S)
        if rect is None:
            return []
        u = (B - A) / np.linalg.norm(B - A)
        mid = (A + B) / 2.0
        top = [q for q in rect.astype(np.float64)
               if float(np.dot(q - mid, u)) < 0.0]     # вершины со стороны плеча
        if len(top) != 2:
            return []
        top.sort(key=lambda q: float(np.linalg.norm(q - head_center)))
        inner, outer = top[0], top[1]

        # сторона шеи с той же стороны тела, что и это плечо
        torso_pt, head_pt = min(neck_sides,
                                key=lambda sd: float(np.linalg.norm(sd[0] - A)))
        hit = _ray_polygon_hit(outer, head_pt - torso_pt, head_poly)
        if hit is None:
            # Луч прошёл мимо головы: так бывает, когда внешняя вершина руки
            # смещена от вершины торса наружу настолько, что параллельная
            # прямая проходит вне головы (например, корпус развёрнут и линия
            # плеч почти вертикальна). Тогда снапим на угол головы, который и
            # задаёт эту сторону шеи, -- фигура остаётся привязанной к голове.
            hit = head_pt
        pts.extend([hit, outer, inner])

    hull = cv2.convexHull(np.asarray(pts, dtype=np.float32))
    return [hull.reshape(-1, 2).astype(np.int32)]


def build_arm_tops_quad(landmarks, region, frame_w, frame_h):
    """Четырёхугольник ABCD по верхним (плечевым) сторонам прямоугольников рук.

    A, B -- верхняя сторона прямоугольника отрезка 14-12;
    C, D -- верхняя сторона прямоугольника отрезка 13-11.

    В отличие от build_upper_body_hull, ни голова, ни торс не нужны -- нужны
    только точки позы. Порядок вершин задаётся выпуклой оболочкой, чтобы
    многоугольник не самопересекался при любом положении рук.

    Возвращает список из одного np.array (N, 2) int32 либо пустой список.
    """
    def point(idx):
        return _get_point_px(landmarks, idx, region, frame_w, frame_h)

    sh_l, sh_r = point(LEFT_SHOULDER), point(RIGHT_SHOULDER)
    if sh_l is None or sh_r is None:
        return []
    S = float(np.linalg.norm(sh_r - sh_l))
    if S < 1e-6:
        return []

    pts = []
    for pair in UPPER_ARM_PAIRS:
        A, B = point(pair[0]), point(pair[1])
        coef = config.STICKMAN_LIMB_COEFS.get(pair)
        if A is None or B is None or coef is None:
            return []
        rect = _build_limb_rect(A, B, coef * S)
        if rect is None:
            return []
        u = (B - A) / np.linalg.norm(B - A)
        mid = (A + B) / 2.0
        top = [q for q in rect.astype(np.float64)
               if float(np.dot(q - mid, u)) < 0.0]     # вершины со стороны плеча
        if len(top) != 2:
            return []
        pts.extend(top)

    hull = cv2.convexHull(np.asarray(pts, dtype=np.float32))
    return [hull.reshape(-1, 2).astype(np.int32)]


def build_stickman_mask(pose_landmarks_list, region, frame_w, frame_h,
                        torso_quad=None, head_corners=None):
    """Строит бинарную маску модели тела (uint8, 0/255) в координатах полного кадра.

    torso_quad -- четырёхугольник торса [TL, TR, BR, BL] из калибровки.
    Нужен только для треугольников торс-нога (24-26-BL, 23-25-BR): свой торс
    этой функции опирается нижними вершинами прямо на бёдра 23/24, и такие
    треугольники выродились бы в дубли треугольников таза. None -- треугольники
    торс-нога не строятся.

    head_corners -- прямоугольник головы из калибровки. Нужен для
    многоугольника XABCDY; свой прямоугольник головы этой функции вырожден при нулевых
    STICKMAN_HEAD_*_COEF. None -- области не строятся.

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

    # 4. Конечности (8 прямоугольников).
    # Ноги масштабируются шириной таза |23-24|, руки -- шириной плеч.
    hip_l = _get_point_px(landmarks, LEFT_HIP, region, frame_w, frame_h)
    hip_r = _get_point_px(landmarks, RIGHT_HIP, region, frame_w, frame_h)
    S_hip = S
    if hip_l is not None and hip_r is not None:
        d_hip = float(np.linalg.norm(hip_r - hip_l))
        if d_hip > 1e-6:
            S_hip = d_hip

    for (a, b), coef in config.STICKMAN_LIMB_COEFS.items():
        A = _get_point_px(landmarks, a, region, frame_w, frame_h)
        B = _get_point_px(landmarks, b, region, frame_w, frame_h)
        if A is None or B is None:
            continue
        if (a, b) in THIGH_PAIRS:
            continue          # верх ноги закрывается четырёхугольником, п.7
        scale = S_hip if (a, b) in LEG_PAIRS else S
        rect = _build_limb_rect(A, B, coef * scale)
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

    # 6. Верх ног: четырёхугольники BL/BR + вершины у колена (нужен торс)
    for q in build_thigh_quads(landmarks, region, frame_w, frame_h, torso_quad):
        cv2.fillPoly(mask, [q], 255)

    # 7. Четырёхугольник ABCD по верхним сторонам прямоугольников рук.
    # ВРЕМЕННО вместо многоугольника XABCDY (build_upper_body_hull): чтобы
    # вернуть его, замените вызов ниже -- сама функция сохранена.
    for q in build_arm_tops_quad(landmarks, region, frame_w, frame_h):
        cv2.fillPoly(mask, [q], 255)

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
