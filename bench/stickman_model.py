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
LEFT_HEEL = 29
RIGHT_HEEL = 30
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


def _build_limb_quad(A, B, width_a, width_b):
    """Четырёхугольник конечности от A до B с разной шириной на концах.

    В каждой из точек A и B строится отрезок своей длины (width_a / width_b),
    серединой на самой точке и перпендикулярный A-B; концы отрезков
    соединяются. При width_a == width_b это обычный прямоугольник.

    Порядок вершин: [B+n, A+n, A-n, B-n]. Он зафиксирован -- по нему
    LIMB_END_IDX достаёт сторону нужного конца.
    """
    AB = B - A
    length = np.linalg.norm(AB)
    if length < 1e-6:
        return None
    n = _rotate90(AB / length)
    ha, hb = width_a / 2.0, width_b / 2.0
    corners = np.array([
        B + hb * n,
        A + ha * n,
        A - ha * n,
        B - hb * n,
    ], dtype=np.int32)
    return corners


# Вершины четырёхугольника конечности со стороны каждого из концов.
# Порядок задан в _build_limb_quad: 'A' -- первая точка пары, 'B' -- вторая.
# Раньше эти вершины отбирались по знаку проекции на ось конечности, но
# координаты целочисленные, и у почти перпендикулярной кадру конечности
# округление давало не ровно две вершины -- фигура молча выбрасывалась.
LIMB_END_IDX = {'A': (1, 2), 'B': (0, 3)}


def limb_end_points(quad, end):
    """Пара вершин четырёхугольника конечности у конца 'A' или 'B'."""
    q = np.asarray(quad, dtype=np.float64)
    return [q[i] for i in LIMB_END_IDX[end]]


def _build_limb_rect(A, B, width):
    """Прямоугольник конечности от A до B (одинаковая ширина на концах)."""
    return _build_limb_quad(A, B, width, width)


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


def clamp_head_down_to_shoulders(nose, e1, e2, right, left, down, sh_l, sh_r):
    """Урезает down так, чтобы низ головы не ушёл ниже линии плеч 11-12.

    Прямоугольник головы стоит на своих осях (e1 -- вдоль ушей, e2 -- вверх),
    поэтому проверяются оба нижних угла: nose + right*e1 - down*e2 и
    nose - left*e1 - down*e2. Оба должны остаться по ту же сторону прямой
    11-12, что и нос.

    Возвращает исходный down, если плечи не видны, если шаг вниз от плеч не
    удаляет (вырожденный наклон) или если нос уже сам ниже линии плеч --
    в последнем случае ограничение невыполнимо, и резать голову в ноль хуже,
    чем оставить как есть.
    """
    if sh_l is None or sh_r is None:
        return down
    sh_l = np.asarray(sh_l, dtype=np.float64)
    sh_r = np.asarray(sh_r, dtype=np.float64)
    S = float(np.linalg.norm(sh_r - sh_l))
    if S < 1e-6:
        return down
    n = _rotate90((sh_r - sh_l) / S)
    a = float(np.dot(np.asarray(nose, dtype=np.float64) - (sh_l + sh_r) / 2.0, n))
    if a < 0:
        n, a = -n, -a                 # нормаль смотрит от плеч к голове
    c = float(np.dot(e2, n))          # насколько шаг вниз приближает к плечам
    if c <= 1e-9:
        return down
    d1 = float(np.dot(e1, n))
    limit = min(a + d1 * right, a - d1 * left) / c
    if limit <= 0.0:
        return down
    return min(down, limit)


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


def limb_scale(pair, S, S_hip):
    """Чем масштабируется отрезок: ноги -- тазом, остальное -- плечами."""
    return S_hip if pair in LEG_PAIRS else S


def limb_grow_px(pair, S, S_hip, limb_grow=None):
    """Раздвижение фигуры пары по маске, в пикселях НА СТОРОНУ (0 -- нет).

    Коэффициент замерен при калибровке в долях того же масштаба, что и сама
    ширина (плечи для рук, таз для ног), поэтому переносится на любой кадр.
    """
    if not limb_grow:
        return 0.0
    coef = limb_grow.get('%d_%d' % pair)
    if coef is None:
        return 0.0
    return float(coef) * limb_scale(pair, S, S_hip)


def limb_widths_px(pair, S, S_hip, limb_widths=None, limb_grow=None):
    """Ширина конечности у каждого конца отрезка pair, в пикселях.

    limb_widths -- калиброванные коэффициенты по точкам {индекс: K}. Для точек,
    которых там нет (или если словаря нет вовсе), берётся постоянная ширина
    STICKMAN_LIMB_COEFS[pair] -- то есть прежний прямоугольник. Так модель
    работает и без калибровки, ровно как до появления коэффициентов.

    Возвращает (width_a, width_b) или None, если для отрезка нет коэффициента.
    """
    coef = config.STICKMAN_LIMB_COEFS.get(pair)
    if coef is None:
        return None
    scale = limb_scale(pair, S, S_hip)
    grow = 2.0 * limb_grow_px(pair, S, S_hip, limb_grow)   # на обе стороны
    widths = []
    for idx in pair:
        k = None if limb_widths is None else limb_widths.get(idx)
        widths.append((coef if k is None else float(k)) * scale + grow)
    return widths[0], widths[1]
# Ладони и ступни: (точка сустава, точки дальнего конца).
# Если дальних точек несколько -- берётся их середина. Для ладони это середина
# между указательным (19/20) и мизинцем (17/18), т.е. центр линии костяшек.
PALM_SPECS = ((LEFT_WRIST, (LEFT_INDEX, LEFT_PINKY)),
              (RIGHT_WRIST, (RIGHT_INDEX, RIGHT_PINKY)))
FOOT_SPECS = ((LEFT_ANKLE, (LEFT_FOOT_INDEX,)),
              (RIGHT_ANKLE, (RIGHT_FOOT_INDEX,)))

# Ступня: (пара голени, (лодыжка, пятка, носок)). Голень нужна целиком --
# её нижняя сторона служит общим основанием обеих фигур ступни.
FOOT_PARTS = (((LEFT_KNEE, LEFT_ANKLE), (LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX)),
              ((RIGHT_KNEE, RIGHT_ANKLE), (RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX)))

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



def build_thigh_quads(landmarks, region, frame_w, frame_h, torso_quad,
                      limb_widths=None, mode=None):
    """Четырёхугольники верхней части ног вместо прямоугольников бедра.

    Для отрезка 24-26 (и 23-25) берутся две вершины фигуры бедра со стороны
    колена и верхняя сторона из THIGH_TOP_SIDES: для ноги 24-26 это BL и
    точка 23, для ноги 23-25 -- BR и точка 24.

    Схему задаёт config.STICKMAN_THIGH_QUAD_MODE:
      0 -- как описано выше;
      1 -- дальний конец верхней стороны не противоположная точка таза, а
           точка A посередине отрезка 23-24: (26-24-A) и (25-23-A) сходятся
           в ней и не залезают на чужую половину;
      2 -- одна фигура на обе ноги, шестиугольник: нижние вершины торса BL и
           BR и обе вершины у каждого колена (26 и 25).

    Ширина у бедра -- прежняя (STICKMAN_LIMB_COEFS * ширина таза), а у колена
    берётся калиброванная K колена, та же, что у верха голени. Иначе на колене
    оставался бы разрыв контура: бедро 0.676, голень 0.6084.

    torso_quad -- четырёхугольник торса [TL, TR, BR, BL]; None -- не строим.
    """
    if torso_quad is None:
        return []
    quad = np.asarray(torso_quad, dtype=np.float64)
    if quad.shape[0] < 4:
        return []
    def point(idx):
        if idx in _OTHER_KNEE:
            return knee_point(landmarks, region, frame_w, frame_h, idx, quad)
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

    def knee_pair(hip_idx, knee_idx):
        """Две вершины фигуры бедра у колена -- ровно те, что даёт модель."""
        A, B = point(hip_idx), point(knee_idx)
        if A is None or B is None:
            return None
        coef = config.STICKMAN_LIMB_COEFS.get((hip_idx, knee_idx))
        if coef is None:
            return None
        k_knee = None if limb_widths is None else limb_widths.get(knee_idx)
        w_knee = coef if k_knee is None else float(k_knee)
        rect = _build_limb_quad(A, B, coef * scale, w_knee * scale)
        return None if rect is None else limb_end_points(rect, 'B')

    mode = (int(getattr(config, 'STICKMAN_THIGH_QUAD_MODE', 0))
            if mode is None else int(mode))
    if mode == 2:
        # Одна фигура на обе ноги: низ торса и обе вершины у каждого колена.
        BL = torso_vertex(quad, 'BL')
        BR = torso_vertex(quad, 'BR')
        k26 = knee_pair(24, 26)
        k25 = knee_pair(23, 25)
        if BL is None or BR is None or k26 is None or k25 is None:
            return []
        axis = np.asarray(BR, dtype=np.float64) - np.asarray(BL, dtype=np.float64)
        if float(np.linalg.norm(axis)) < 1e-6:
            return []
        # Четыре вершины колен по порядку слева направо -- так обход остаётся
        # простым и при сведённых, и при скрещенных ногах.
        knees = sorted(list(k26) + list(k25),
                       key=lambda p: float(np.dot(p - BL, axis)))
        return [np.array([BL] + knees + [BR], dtype=np.int32)]

    quads = []
    for hip_idx, knee_idx in THIGH_PAIRS:
        A, B = point(hip_idx), point(knee_idx)
        if A is None or B is None:
            continue
        coef = config.STICKMAN_LIMB_COEFS.get((hip_idx, knee_idx))
        if coef is None:
            continue
        # У бедра ширина прежняя, у колена -- калиброванная (общая с голенью).
        k_knee = None if limb_widths is None else limb_widths.get(knee_idx)
        w_knee = coef if k_knee is None else float(k_knee)
        rect = _build_limb_quad(A, B, coef * scale, w_knee * scale)
        if rect is None:
            continue
        knee_pts = limb_end_points(rect, 'B')   # 'B' -- конец у колена
        side = THIGH_TOP_SIDES.get((hip_idx, knee_idx))
        if side is None:
            continue
        corner_name, top_hip_idx = side
        corner = torso_vertex(quad, corner_name)   # BL или BR
        if corner is None:
            continue
        if mode == 1:
            # Точка A -- середина отрезка таза, общая для обеих ног.
            if hip_l is None or hip_r is None:
                continue
            top_hip = (hip_l + hip_r) / 2.0
        else:
            top_hip = point(top_hip_idx)        # точка 23 или 24
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


def build_foot_poly(ankle, heel, toe, S_hip, shin_quad):
    """Фигура ступни: выпуклый пятиугольник, опирающийся на голень.

    ankle, heel, toe -- точки 27/29/31 (или 28/30/32) в координатах кадра.
    shin_quad -- четырёхугольник голени 25-27 (26-28). Из него берётся сторона
    у лодыжки (limb_end_points(..., 'B')), поэтому шов с голенью сходится без
    зазора.

    Основа -- четырёхугольник: отрезок 27-31 продлевается за носок в
    STICKMAN_FOOT_EXTEND_COEF раз, в конце ставится перпендикулярный ему
    отрезок с серединой в этой точке, концы отрезка соединяются со стороной
    голени. Длина отрезка обратно зависит от |27-29|: чем сильнее стопа
    развёрнута на камеру, тем короче проекция лодыжка-пятка и тем шире видна
    стопа. Потолок нужен, потому что при |27-29| -> 0 формула расходится.

    Пятая вершина -- пятка. Контур берётся выпуклой оболочкой пяти точек. Если
    пятка попала внутрь четырёхугольника, оболочка выходит меньше пяти вершин
    -- выпуклого пятиугольника не существует, и остаётся сам четырёхугольник.

    Возвращает контур (int32) либо None, если данных не хватает -- тогда
    вызывающий строит прежний прямоугольник.
    """
    if shin_quad is None or ankle is None or heel is None or toe is None:
        return None
    ankle = np.asarray(ankle, dtype=np.float64)
    heel = np.asarray(heel, dtype=np.float64)
    toe = np.asarray(toe, dtype=np.float64)
    AT = toe - ankle
    length = float(np.linalg.norm(AT))
    if length < 1e-6:
        return None
    n = _rotate90(AT / length)
    P = ankle + AT * config.STICKMAN_FOOT_EXTEND_COEF

    cap = config.STICKMAN_FOOT_WIDTH_CAP_COEF * S_hip
    d_heel = float(np.linalg.norm(heel - ankle))
    width = cap if d_heel < 1e-6 else min(
        config.STICKMAN_FOOT_WIDTH_COEF * S_hip / d_heel, cap)
    half = width / 2.0

    # Сортировка по проекции на нормаль ставит вершины голени и концы отрезка
    # по одну сторону оси -- обход [e+, P+, P-, e-] не самопересекается, если
    # пятка внутри и оболочка не пригодилась.
    ends = sorted(limb_end_points(shin_quad, 'B'),
                  key=lambda q: float(np.dot(q - ankle, n)))
    quad = [ends[1], P + half * n, P - half * n, ends[0]]
    hull = cv2.convexHull(np.array(quad + [heel],
                                   dtype=np.float32)).reshape(-1, 2)
    poly = hull if len(hull) == 5 else np.array(quad)
    # Округление, а не усечение: пятка лежит ровно в вершине контура, и
    # отбрасывание дробной части сдвигало бы ребро внутрь на целый пиксель --
    # вершина оказывалась снаружи собственного контура.
    return np.round(poly).astype(np.int32)


# Сустав и две фигуры, которые в нём сходятся: (точка, ближняя пара, дальняя).
JOINT_WEDGE_SPECS = ((13, (11, 13), (13, 15)),
                     (14, (12, 14), (14, 16)),
                     (25, (23, 25), (25, 27)),
                     (26, (24, 26), (26, 28)))


def _pair_limb_quad(pair, points, S, S_hip, limb_widths=None,
                    limb_grow=None):
    """Фигура конечности для пары точек -- ровно та, что строит модель.

    У бедра ширина в тазобедренном конце всегда номинальная, а в колене --
    калиброванная (так же, как в build_thigh_quads), у остальных пар обе
    ширины берутся из limb_widths_px.
    """
    A, B = points(pair[0]), points(pair[1])
    if A is None or B is None:
        return None
    if pair in THIGH_PAIRS:
        coef = config.STICKMAN_LIMB_COEFS.get(pair)
        if coef is None:
            return None
        scale = limb_scale(pair, S, S_hip)
        k_knee = None if limb_widths is None else limb_widths.get(pair[1])
        widths = (coef * scale,
                  (coef if k_knee is None else float(k_knee)) * scale)
    else:
        widths = limb_widths_px(pair, S, S_hip, limb_widths, limb_grow)
        if widths is None:
            return None
    return _build_limb_quad(A, B, widths[0], widths[1])


# Бедро -> голень, чья ширина в колене задаёт ширину прямоугольника бедра.
THIGH_WIDTH_SOURCE = {(23, 25): (25, 27), (24, 26): (26, 28)}


def build_thigh_rects(landmarks, region, frame_w, frame_h, S, S_hip,
                      limb_widths=None, limb_grow=None, torso_quad=None):
    """Прямоугольники бёдер 23-25 и 24-26 -- постоянной ширины.

    Ширина берётся замеренная в колене: ровно та, с которой из колена выходит
    фигура голени (25-27 или 26-28), вместе с её раздвижением. Так бедро и
    голень стыкуются без ступеньки, а ширина остаётся мерой этого человека,
    а не номиналом.

    Отдельно от build_thigh_quads: тот строит верх ноги от вершин торса и
    закрывает пах, а здесь -- прямая кость от таза к колену.

    Возвращает список контуров (может быть пустым).
    """
    out = []
    for pair, shin in THIGH_WIDTH_SOURCE.items():
        A = _get_point_px(landmarks, pair[0], region, frame_w, frame_h)
        B = knee_point(landmarks, region, frame_w, frame_h, pair[1], torso_quad)
        if A is None or B is None:
            continue
        widths = limb_widths_px(shin, S, S_hip, limb_widths, limb_grow)
        if widths is None:
            continue
        w = widths[0]                       # ширина в колене
        rect = _build_limb_quad(A, B, w, w)
        if rect is not None:
            out.append(rect)
    return out


def torso_axis(torso_quad):
    """Осевая линия торса: (точка на ней, единичное направление) либо None.

    Идёт от середины верхнего ребра TL-TR к середине нижнего BL-BR.
    """
    if torso_quad is None:
        return None
    q = np.asarray(torso_quad, dtype=np.float64)
    pts = [torso_vertex(q, name) for name in ('TL', 'TR', 'BL', 'BR')]
    if any(p is None for p in pts):
        return None
    top = (pts[0] + pts[1]) / 2.0
    bottom = (pts[2] + pts[3]) / 2.0
    d = bottom - top
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return None
    return top, d / n


def mirror_across_torso(point, torso_quad):
    """Отражение точки относительно осевой линии торса."""
    axis = torso_axis(torso_quad)
    if axis is None:
        return None
    o, u = axis
    v = np.asarray(point, dtype=np.float64) - o
    return o + (2.0 * float(np.dot(v, u))) * u - v


_OTHER_KNEE = {LEFT_KNEE: RIGHT_KNEE, RIGHT_KNEE: LEFT_KNEE}


def knee_point(landmarks, region, frame_w, frame_h, idx, torso_quad=None):
    """Точка колена: своя, а если её не видно -- зеркало второго колена.

    Зеркало берётся относительно осевой линии торса (см. torso_axis). Нужно,
    когда поза потеряла одно колено: на half010 кадр 75 у точки 25
    видимость 0.49, у точки 26 -- 0.58, и без зеркала нога слева просто
    исчезает из модели.
    """
    p = _get_point_px(landmarks, idx, region, frame_w, frame_h)
    if p is not None:
        return p
    if not getattr(config, 'STICKMAN_MIRROR_MISSING_KNEE', False):
        return None
    other = _OTHER_KNEE.get(idx)
    if other is None or torso_quad is None:
        return None
    q = _get_point_px(landmarks, other, region, frame_w, frame_h)
    if q is None:
        return None
    return mirror_across_torso(q, torso_quad)


def build_legs_below_frame_quad(landmarks, region, frame_w, frame_h,
                                torso_quad=None):
    """Прямоугольник вниз за кадр -- когда ног не видно.

    Строится, только если ни одно колено не попало в кадр, а обе точки таза
    (23 и 24) видны. Верхнее ребро -- нижнее ребро откалиброванного торса
    (BL-BR), оно подогнано по маске и шире отрезка самих точек таза; без
    торса берётся отрезок 23-24. Нижнее ребро опущено по нормали до полного
    выхода за низ кадра, как у торса без бёдер.

    Возвращает контур (4, 2) либо None.
    """
    if not getattr(config, 'STICKMAN_LEGS_BELOW_FRAME', False):
        return None
    P = lambda i: _get_point_px(landmarks, i, region, frame_w, frame_h)

    def knee_in_frame(idx):
        """Колено и видно, и лежит в кадре.

        Одной видимости мало: поза дорисовывает точки за кадром (на half010
        кадр 30 колени "видны" с 0.74-0.76, но стоят на y=1462 при высоте
        кадра 1280). Для нас нога не видна, пока её нет в кадре.
        """
        p = P(idx)
        return p is not None and _point_in_frame(p, frame_w, frame_h)

    if knee_in_frame(LEFT_KNEE) or knee_in_frame(RIGHT_KNEE):
        return None                      # колено в кадре -- ноги строятся как обычно
    hip_l, hip_r = P(LEFT_HIP), P(RIGHT_HIP)
    if hip_l is None or hip_r is None:
        return None
    # Верхнее ребро: нижняя сторона торса, если она есть.
    top_a, top_b, n_down = hip_r, hip_l, None
    if torso_quad is not None:
        bl = torso_vertex(torso_quad, 'BL')
        br = torso_vertex(torso_quad, 'BR')
        axis = torso_axis(torso_quad)
        if bl is not None and br is not None and axis is not None:
            top_a, top_b, n_down = np.asarray(bl, dtype=np.float64), \
                np.asarray(br, dtype=np.float64), axis[1]

    if n_down is None:
        d = top_b - top_a
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return None
        n_down = _rotate90(d / n)
        sh_l, sh_r = P(LEFT_SHOULDER), P(RIGHT_SHOULDER)
        if sh_l is not None and sh_r is not None:
            # Нормаль смотрит прочь от плеч, то есть вниз по телу.
            if float(np.dot((hip_l + hip_r) / 2.0
                            - (sh_l + sh_r) / 2.0, n_down)) < 0:
                n_down = -n_down
        elif n_down[1] < 0:
            n_down = -n_down

    low_a, low_b = drop_top_edge_below_frame(top_a, top_b, n_down, frame_h)
    return np.array([top_a, top_b, low_b, low_a], dtype=np.float64)


def choose_thigh_quad_mode(mask, landmarks, region, frame_w, frame_h,
                           torso_quad, limb_widths=None, limb_grow=None):
    """Режим верха ног, выбранный по маске: см. config.STICKMAN_THIGH_QUAD_AUTO.

    Областью сравнения служит шестиугольник режима 2. Внутри него у каждого
    варианта считается IoU его фигур с ground truth маской, берётся лучший.

    Проверка на руки идёт по другой фигуре -- четырёхугольнику 23-24-26-25
    (таз и колени). Он описывает саму зону ног по точкам позы, без ширин и
    калибровки, и не зависит от того, какой вариант сейчас строится.

    Возвращает (режим, причина). Причина -- строка для печати.
    """
    default = int(getattr(config, 'STICKMAN_THIGH_QUAD_MODE', 0))
    if not getattr(config, 'STICKMAN_THIGH_QUAD_AUTO', False):
        return default, 'выбор выключен'
    if mask is None or torso_quad is None:
        return 0, 'нет маски или торса'

    hexes = build_thigh_quads(landmarks, region, frame_w, frame_h, torso_quad,
                              limb_widths=limb_widths, mode=2)
    if not hexes:
        return 0, 'шестиугольник не построен'
    shape = mask.shape[:2]

    def fill(polys):
        buf = np.zeros(shape, dtype=np.uint8)
        for p in polys:
            cv2.fillPoly(buf, [np.asarray(p, dtype=np.int32)], 255)
        return buf

    area = fill(hexes)

    # Руки в зоне ног: маска там не про ноги. Зона -- четырёхугольник по
    # точкам 23-24-26-25 (таз и колени).
    zone_pts = [_get_point_px(landmarks, i, region, frame_w, frame_h)
                for i in (LEFT_HIP, RIGHT_HIP, 26, 25)]
    if any(p is None for p in zone_pts):
        return 0, 'таз или колени не видны'
    zone = fill([np.array(zone_pts, dtype=np.float64)])
    arms = build_body_rects(landmarks, region, frame_w, frame_h,
                            limb_widths=limb_widths, limb_grow=limb_grow)
    if arms:
        hands = list(arms.get('arms', [])) + list(arms.get('palms', []))
        if hands and int(cv2.countNonZero(cv2.bitwise_and(fill(hands), zone))):
            return 0, 'в зону 23-24-26-25 попали фигуры рук'

    # Фигуры бёдер касаются друг друга: просвета между ног нет.
    def dist(i, j):
        a = _get_point_px(landmarks, i, region, frame_w, frame_h)
        b = _get_point_px(landmarks, j, region, frame_w, frame_h)
        if a is None or b is None:
            return None
        d = float(np.linalg.norm(b - a))
        return d if d > 1e-6 else None

    S = dist(LEFT_SHOULDER, RIGHT_SHOULDER)
    if S is None:
        return 0, 'плечи не видны'
    S_hip = dist(LEFT_HIP, RIGHT_HIP) or S
    rects = build_thigh_rects(landmarks, region, frame_w, frame_h,
                              S, S_hip, limb_widths, limb_grow,
                              torso_quad=torso_quad)
    if len(rects) == 2:
        if int(cv2.countNonZero(cv2.bitwise_and(fill([rects[0]]),
                                                fill([rects[1]])))):
            return 0, 'фигуры бёдер касаются'

    gt = cv2.bitwise_and(mask, area)
    best, best_iou = 0, -1.0
    scores = []
    for m in (0, 1, 2):
        polys = build_thigh_quads(landmarks, region, frame_w, frame_h,
                                  torso_quad, limb_widths=limb_widths, mode=m)
        pred = cv2.bitwise_and(fill(polys), area) if polys else np.zeros(shape, np.uint8)
        inter = int(cv2.countNonZero(cv2.bitwise_and(pred, gt)))
        union = int(cv2.countNonZero(cv2.bitwise_or(pred, gt)))
        iou = inter / union if union else 0.0
        scores.append(iou)
        if iou > best_iou:
            best, best_iou = m, iou
    return best, 'IoU 0:%.3f 1:%.3f 2:%.3f' % tuple(scores)


def build_joint_wedges(landmarks, region, frame_w, frame_h, limb_widths=None,
                       limb_grow=None):
    """Фигуры, закрывающие излом в локтях и коленях.

    Возвращает {'joint_tris': [...], 'forearm_squares': [...]}.

    joint_tris -- по два треугольника на сустав: точка сустава и две вершины
    сторон, проходящих через него, -- ближней фигуры (плечо/бедро) и дальней
    (предплечье/голень), взятые с одной стороны конечности. На согнутой руке
    между двумя четырёхугольниками с внешней стороны остаётся клин, эти
    треугольники его и закрывают.

    forearm_squares -- квадрат на локтевой стороне плечевой кости, когда
    предплечье короче её в STICKMAN_FOREARM_SHORT_RATIO раз и более: рука
    смотрит в камеру, её собственная фигура вырождается в полоску, а кисть
    занимает на кадре примерно квадрат со стороной локтевого ребра. Сторона
    квадрата -- само это ребро, растёт он от локтя дальше по оси плеча.
    """
    out = {'joint_tris': [], 'forearm_squares': []}

    def points(idx):
        return _get_point_px(landmarks, idx, region, frame_w, frame_h)

    sh_l, sh_r = points(LEFT_SHOULDER), points(RIGHT_SHOULDER)
    if sh_l is None or sh_r is None:
        return out
    S = float(np.linalg.norm(sh_r - sh_l))
    if S < 1e-6:
        return out
    hip_l, hip_r = points(LEFT_HIP), points(RIGHT_HIP)
    S_hip = S
    if hip_l is not None and hip_r is not None:
        d_hip = float(np.linalg.norm(hip_r - hip_l))
        if d_hip > 1e-6:
            S_hip = d_hip

    for joint, prox_pair, dist_pair in JOINT_WEDGE_SPECS:
        prox = _pair_limb_quad(prox_pair, points, S, S_hip, limb_widths,
                               limb_grow)
        if prox is None:
            continue
        J = points(joint)
        dist = _pair_limb_quad(dist_pair, points, S, S_hip, limb_widths,
                               limb_grow)

        # Квадрат: только для рук и только при вырожденном предплечье.
        if joint in (13, 14):
            far = points(dist_pair[1])
            near = points(prox_pair[0])
            if forearm_degenerate(near, J, far):
                square = _square_on_end_side(prox, J, near)
                if square is not None:
                    out['forearm_squares'].append(square)

        if J is None or dist is None:
            continue
        # Порядок вершин задан в _build_limb_quad: [B+n, A+n, A-n, B-n].
        # Значит сторона +n ближней фигуры -- вершина 0, дальней -- вершина 1,
        # сторона -n -- вершины 3 и 2. Нормаль у обеих повёрнута от своей оси
        # одинаково, поэтому пары вершин лежат по одну сторону конечности.
        for i_prox, i_dist in ((0, 1), (3, 2)):
            out['joint_tris'].append(np.round(np.array(
                [J, prox[i_prox], dist[i_dist]], dtype=np.float64)).astype(np.int32))
    return out


def forearm_degenerate(near, joint, far):
    """Предплечье вырождено: короче плечевой кости в STICKMAN_FOREARM_SHORT_RATIO
    раз и более (рука смотрит в камеру). near/joint/far -- плечо, локоть, кисть.
    """
    if near is None or joint is None or far is None:
        return False
    len_prox = float(np.linalg.norm(np.asarray(joint) - np.asarray(near)))
    len_dist = float(np.linalg.norm(np.asarray(far) - np.asarray(joint)))
    if len_prox < 1e-6:
        return False
    return len_dist * config.STICKMAN_FOREARM_SHORT_RATIO <= len_prox


def _square_on_end_side(quad, joint, near):
    """Квадрат на стороне фигуры, проходящей через сустав (конец 'B').

    Растёт от этой стороны прочь от ближнего сустава near, на длину самой
    стороны. Возвращает контур (4, 2) int32 или None.
    """
    P1, P2 = [np.asarray(v, dtype=np.float64) for v in limb_end_points(quad, 'B')]
    side = float(np.linalg.norm(P2 - P1))
    if side < 1e-6:
        return None
    d = np.asarray(joint, dtype=np.float64) - np.asarray(near, dtype=np.float64)
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return None
    d = d / length
    return np.round(np.array([P1, P2, P2 + side * d, P1 + side * d],
                             dtype=np.float64)).astype(np.int32)


def drop_top_edge_below_frame(TL, TR, n_down, frame_h, margin=2.0):
    """Опускает верхнее ребро TL-TR по нормали n_down до выхода за низ кадра.

    Возвращает пару нижних вершин прямоугольника: обе строго ниже последней
    строки кадра, ребро остаётся параллельным TL-TR и перпендикулярным
    боковым сторонам. Если нормаль почти горизонтальна (человек лежит),
    опускание вырождено -- ребро уносится на две высоты кадра, чтобы фигура
    заведомо покрыла низ.
    """
    TL = np.asarray(TL, dtype=np.float64)
    TR = np.asarray(TR, dtype=np.float64)
    n_down = np.asarray(n_down, dtype=np.float64)
    ny = float(n_down[1])
    if ny <= 1e-6:
        h = 2.0 * float(frame_h)
    else:
        h = max((frame_h - 1 - float(TL[1])) / ny,
                (frame_h - 1 - float(TR[1])) / ny) + margin
        h = max(h, margin)
    return TL + h * n_down, TR + h * n_down


# Плечевая кость и её предплечье: (пара плеча, пара предплечья, ключ JSON).
ARM_FRAME_SPECS = (((11, 13), (13, 15), '11_13'),
                   ((12, 14), (14, 16), '12_14'))


def _point_in_frame(point, frame_w, frame_h):
    """Точка внутри кадра (кромка считается кадром)."""
    x, y = float(point[0]), float(point[1])
    return 0 <= x <= frame_w - 1 and 0 <= y <= frame_h - 1


def _frame_exit_distance(point, u, frame_w, frame_h):
    """Сколько пройти из point вдоль u, чтобы выйти за кадр (0 -- уже вне)."""
    x, y = float(point[0]), float(point[1])
    if not _point_in_frame(point, frame_w, frame_h):
        return 0.0
    ts = []
    if u[0] > 1e-9:
        ts.append((frame_w - 1 - x) / u[0])
    elif u[0] < -1e-9:
        ts.append(x / -u[0])
    if u[1] > 1e-9:
        ts.append((frame_h - 1 - y) / u[1])
    elif u[1] < -1e-9:
        ts.append(y / -u[1])
    return min(ts) if ts else 0.0


def extend_limb_quad_out_of_frame(quad, A, B, frame_w, frame_h, margin=2.0):
    """Продлевает фигуру конечности от A к B, пока её дальняя сторона целиком
    не выйдет за кадр.

    Двигаются только две вершины конца 'B', поэтому ближний конец (у сустава)
    и ширина дальней стороны остаются прежними. Возвращает новый контур
    int32; если продлевать некуда (направление вырождено), -- исходный.
    """
    d = np.asarray(B, dtype=np.float64) - np.asarray(A, dtype=np.float64)
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return quad
    u = d / length
    q = np.asarray(quad, dtype=np.float64).copy()
    # Вершина на самой кромке кадра ещё в нём: продлевать надо, пока обе
    # вершины не окажутся строго снаружи. Сдвиг для обеих одинаков, иначе
    # дальняя сторона перекосится.
    inside = [i for i in LIMB_END_IDX['B']
              if _point_in_frame(q[i], frame_w, frame_h)]
    if not inside:
        return quad
    t = max(_frame_exit_distance(q[i], u, frame_w, frame_h) for i in inside)
    for i in LIMB_END_IDX['B']:
        q[i] = q[i] + (t + margin) * u
    return np.round(q).astype(np.int32)


def arm_frame_extend_flags(landmarks, region, frame_w, frame_h,
                           limb_widths=None, limb_grow=None):
    """Каким плечевым костям нужно продление за кадр.

    Условие: сама кость видна, а её предплечье -- нет. Продолжать руку нечем,
    поэтому её фигура тянется по своей оси, пока обе вершины дальнего конца
    не уйдут за кадр (см. extend_limb_quad_out_of_frame). Условие проверяется
    на КАЖДОМ кадре по текущим точкам позы, а не берётся из калибровки: рука
    выходит из кадра и возвращается по ходу видео.

    Возвращает {'11_13': bool, '12_14': bool}.
    """
    flags = {key: False for _, _, key in ARM_FRAME_SPECS}
    sh_l = _get_point_px(landmarks, LEFT_SHOULDER, region, frame_w, frame_h)
    sh_r = _get_point_px(landmarks, RIGHT_SHOULDER, region, frame_w, frame_h)
    if sh_l is None or sh_r is None:
        return flags
    S = float(np.linalg.norm(sh_r - sh_l))
    if S < 1e-6:
        return flags

    def point(idx):
        return _get_point_px(landmarks, idx, region, frame_w, frame_h)

    for pair, forearm, key in ARM_FRAME_SPECS:
        A, B = point(pair[0]), point(pair[1])
        if A is None or B is None:
            continue
        if point(forearm[0]) is not None and point(forearm[1]) is not None:
            continue                     # предплечье видно -- продлевать нечего
        flags[key] = True
    return flags


def arm_extend_wanted(arm_extend, pair):
    """Стоит ли флаг продления для пары плечевой кости (ключи JSON -- строки)."""
    if not arm_extend:
        return False
    key = '%d_%d' % pair
    return bool(arm_extend.get(key, False))


def build_body_rects(landmarks, region, frame_w, frame_h, limb_widths=None,
                     limb_grow=None):
    """Строит фигуры частей тела в координатах полного кадра.

    limb_widths -- калиброванные коэффициенты ширины по точкам {индекс: K}.
    Руки и голени тогда строятся четырёхугольниками, сужающимися к запястью и
    лодыжке. None -- прежние прямоугольники постоянной ширины.

    Возвращает dict со списками фигур (каждая -- np.array (4, 2)):
        'arms'  -- 4 фигуры рук (ширина из limb_widths / STICKMAN_LIMB_COEFS)
        'legs'  -- 2 фигуры голеней (то же)
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
    rects['thigh_rects'] = build_thigh_rects(
        landmarks, region, frame_w, frame_h, S, S_hip, limb_widths, limb_grow)
    below = build_legs_below_frame_quad(landmarks, region, frame_w, frame_h)
    rects['legs_below'] = [] if below is None else [below.astype(np.int32)]

    # Рука уходит из кадра: предплечья не видно, а фигура плечевой кости
    # упирается в границу. Тогда она продлевается до полного выхода дальней
    # стороны за кадр -- и здесь, и в маске модели при отслеживании.
    arm_flags = arm_frame_extend_flags(landmarks, region, frame_w, frame_h,
                                       limb_widths, limb_grow)
    rects['arm_extend'] = arm_flags

    # Голени запоминаются по паре: на их нижнюю сторону опираются фигуры
    # ступней, а брать голень по порядку в списке нельзя -- при невидимых
    # точках нога в список не попадает.
    shin_quads = {}
    for pairs, key in ((ARM_PAIRS, 'arms'), (SHIN_PAIRS, 'legs')):
        for pair in pairs:
            widths = limb_widths_px(pair, S, S_hip, limb_widths, limb_grow)
            if widths is None:
                continue
            A, B = point(pair[0]), point(pair[1])
            if A is None or B is None:
                continue
            rect = _build_limb_quad(A, B, widths[0], widths[1])
            if rect is not None:
                if arm_extend_wanted(arm_flags, pair):
                    rect = extend_limb_quad_out_of_frame(rect, A, B,
                                                         frame_w, frame_h)
                rects[key].append(rect)
                if pair in SHIN_PAIRS:
                    shin_quads[pair] = rect

    for joint_idx, distal_ids in PALM_SPECS:
        A = point(joint_idx)
        if A is None:
            continue
        # Дальний конец -- середина видимых точек (одна точка -- она сама)
        distal = [q for q in (point(i) for i in distal_ids) if q is not None]
        if not distal:
            continue
        B = np.mean(distal, axis=0)
        rect = _build_extended_rect(A, B, config.STICKMAN_PALM_COEF * S,
                                    config.STICKMAN_PALM_EXTEND_COEF)
        if rect is not None:
            rects['palms'].append(rect)

    for shin_pair, (ankle_idx, heel_idx, toe_idx) in FOOT_PARTS:
        ankle, toe = point(ankle_idx), point(toe_idx)
        if ankle is None or toe is None:
            continue
        poly = build_foot_poly(ankle, point(heel_idx), toe, S_hip,
                               shin_quads.get(shin_pair))
        if poly is not None:
            rects['feet'].append(poly)
            continue
        # Нет пятки или голени -- прежний прямоугольник вдоль 27-31.
        rect = _build_extended_rect(ankle, toe,
                                    config.STICKMAN_FOOT_COEF * S_hip,
                                    config.STICKMAN_FOOT_EXTEND_COEF)
        if rect is not None:
            rects['feet'].append(rect)

    rects.update(build_joint_wedges(landmarks, region, frame_w, frame_h,
                                    limb_widths, limb_grow))
    rects['arm_tops_quad'] = build_arm_tops_quad(
        landmarks, region, frame_w, frame_h, limb_widths=limb_widths)

    return rects


# ------------------------------------------------------------------
# Сборка модели
# ------------------------------------------------------------------
def build_upper_body_hull(landmarks, region, frame_w, frame_h,
                          head_corners, torso_quad, limb_widths=None,
                          limb_grow=None):
    """Выпуклый многоугольник XABCDY между верхом рук и головой.

    A, B -- плечевая сторона фигуры отрезка 14-12;
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
        widths = limb_widths_px(pair, S, S, limb_widths, limb_grow)
        if A is None or B is None or widths is None:
            return []
        rect = _build_limb_quad(A, B, widths[0], widths[1])
        if rect is None:
            return []
        top = limb_end_points(rect, 'A')        # 'A' -- конец у плеча
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


def build_arm_tops_quad(landmarks, region, frame_w, frame_h, limb_widths=None,
                        limb_grow=None):
    """Четырёхугольник ABCD по плечевым сторонам фигур рук.

    A, B -- плечевая сторона фигуры отрезка 14-12;
    C, D -- плечевая сторона фигуры отрезка 13-11.

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
        widths = limb_widths_px(pair, S, S, limb_widths, limb_grow)
        if A is None or B is None or widths is None:
            return []
        rect = _build_limb_quad(A, B, widths[0], widths[1])
        if rect is None:
            return []
        pts.extend(limb_end_points(rect, 'A'))   # 'A' -- конец у плеча

    hull = cv2.convexHull(np.asarray(pts, dtype=np.float32))
    return [hull.reshape(-1, 2).astype(np.int32)]


# Треугольник плеча: (пара плечевой кости, вершина торса своей стороны).
# TR стоит со стороны точки 11, TL -- со стороны точки 12: обе вершины
# строятся лучом из своего плеча (см. calibrate_torso).
ARM_TRI_SPECS = (((11, 13), 'TR', RIGHT_SHOULDER),
                 ((12, 14), 'TL', LEFT_SHOULDER))


def build_arm_shoulder_tris(landmarks, region, frame_w, frame_h, torso_quad,
                            limb_widths=None, limb_grow=None):
    """Треугольники плеча: откалиброванная вершина -- точка плеча -- рука.

    Третья вершина -- НАРУЖНАЯ вершина прямоугольника плечевой кости у самого
    плеча (из двух вершин этого конца берётся дальняя от второго плеча).
    Треугольник закрывает клин между скатом плеча, который даёт луч под
    CALIBRATION_SHOULDER_RAY_DEG, и прямоугольником руки.

    torso_quad -- контур торса из калибровки; None -- фигуры не строятся.
    Возвращает список контуров (3, 2) int32.
    """
    if torso_quad is None:
        return []

    def point(idx):
        return _get_point_px(landmarks, idx, region, frame_w, frame_h)

    sh_l, sh_r = point(LEFT_SHOULDER), point(RIGHT_SHOULDER)
    if sh_l is None or sh_r is None:
        return []
    S = float(np.linalg.norm(sh_r - sh_l))
    if S < 1e-6:
        return []

    tris = []
    for pair, vertex_name, other_idx in ARM_TRI_SPECS:
        A, B = point(pair[0]), point(pair[1])
        other = point(other_idx)
        if A is None or B is None or other is None:
            continue
        T = torso_vertex(torso_quad, vertex_name)
        if T is None:
            continue
        widths = limb_widths_px(pair, S, S, limb_widths, limb_grow)
        if widths is None:
            continue
        rect = _build_limb_quad(A, B, widths[0], widths[1])
        if rect is None:
            continue
        # 'A' -- конец прямоугольника у плеча; наружная из двух его вершин
        outer = max(limb_end_points(rect, 'A'),
                    key=lambda p: float(np.linalg.norm(p - other)))
        tris.append(np.round(np.array([T, A, outer],
                                      dtype=np.float64)).astype(np.int32))
    return tris


def build_stickman_mask(pose_landmarks_list, region, frame_w, frame_h,
                        torso_quad=None, head_corners=None, limb_widths=None,
                        neck_quad=None, shoulders_bottom_quads=None,
                        lower_neck_quad=None, limb_grow=None,
                        neck_base_quad=None, neck_top_quad=None,
                        thigh_mode=None):
    """Строит бинарную маску модели тела (uint8, 0/255) в координатах полного кадра.

    limb_widths -- калиброванные коэффициенты ширины по точкам {индекс: K}.
    None -- конечности строятся прямоугольниками постоянной ширины, как раньше.

    torso_quad -- контур торса из калибровки (четырёхугольник плечи-торс или
    шестиугольник плечи-живот-торс). Заливается в маску как есть и служит
    опорой четырёхугольникам бёдер. None -- торс строится по точкам позы
    (_build_torso_poly), но при STICKMAN_TORSO_SCALE = 0 он вырожден.

    head_corners -- прямоугольник головы из калибровки. Заливается в маску.
    None -- голова строится по точкам позы (_build_head_rect), но при нулевых
    STICKMAN_HEAD_*_COEF она вырождена.

    neck_quad -- трапеция шеи (build_neck_quad_from_torso_and_head). Своей шеи
    у маски нет: она строится от откалиброванных головы и торса, поэтому
    приходит готовой. None -- шея в маску не входит.

    shoulders_bottom_quads -- список пятиугольников "плечи-низ"
    (build_shoulders_bottom_quads): низ силуэта на половинном кадре без бёдер,
    по одному на каждую сторону, чья рука не видна. Приходят готовыми.

    lower_neck_quad -- шея во весь низ кадра (build_lower_neck_quad): плечи не
    видны, торса нет, всё ниже головы считается шеей. Если плечи не видны, то
    из фигур в маску входят только голова и эта шея -- остальное строится от
    ширины плеч, которой нет.

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
    S = (np.linalg.norm(sh_r - sh_l)
         if sh_l is not None and sh_r is not None else 0.0)

    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

    # Плечи не видны: ширины плеч нет, а от неё считаются все размеры модели.
    # Тогда в маску входят только фигуры, построенные без неё, -- голова и
    # шея во весь низ кадра (см. lower_neck_quad).
    if S < 1e-6:
        empty = True
        for poly in (head_corners, lower_neck_quad):
            if poly is None:
                continue
            cv2.fillPoly(mask, [np.round(np.asarray(
                poly, dtype=np.float64)).astype(np.int32)], 255)
            empty = False
        return None if empty else mask

    # 1. Торс: откалиброванный контур, если он передан. Свой торс по точкам
    # позы -- фолбэк для вызовов без калибровки (evaluate_stickman_accuracy).
    torso = (np.asarray(torso_quad, dtype=np.float64)
             if torso_quad is not None
             else _build_torso_poly(landmarks, region, frame_w, frame_h))
    if torso is not None:
        cv2.fillPoly(mask, [np.round(torso).astype(np.int32)], 255)

    # 1b. "Плечи-низ": продолжение торса до нижней кромки кадра. Строится от
    # откалиброванных вершин плеч, поэтому приходит готовым.
    for quad in (shoulders_bottom_quads or []):
        cv2.fillPoly(mask, [np.round(np.asarray(
            quad, dtype=np.float64)).astype(np.int32)], 255)

    # 2. Шея: строится от откалиброванных головы и торса, поэтому приходит
    # готовой трапецией.
    if neck_quad is not None:
        cv2.fillPoly(mask, [np.round(np.asarray(neck_quad,
                                                dtype=np.float64)).astype(np.int32)], 255)

    # 2b. Основание шеи: полоса между уровнями нижних точек её сторон.
    if neck_base_quad is not None:
        cv2.fillPoly(mask, [np.round(np.asarray(
            neck_base_quad, dtype=np.float64)).astype(np.int32)], 255)

    # 2c. Верх шеи: полоса до нижнего ребра головы.
    if neck_top_quad is not None:
        cv2.fillPoly(mask, [np.round(np.asarray(
            neck_top_quad, dtype=np.float64)).astype(np.int32)], 255)

    # 3. Голова: откалиброванный прямоугольник, если он передан. Свой -- тот же
    # фолбэк, что и у торса.
    head = (np.asarray(head_corners, dtype=np.float64)
            if head_corners is not None
            else _build_head_rect(landmarks, region, frame_w, frame_h, S))
    if head is not None:
        cv2.fillPoly(mask, [np.round(head).astype(np.int32)], 255)

    # 4. Конечности (8 прямоугольников).
    # Ноги масштабируются шириной таза |23-24|, руки -- шириной плеч.
    hip_l = _get_point_px(landmarks, LEFT_HIP, region, frame_w, frame_h)
    hip_r = _get_point_px(landmarks, RIGHT_HIP, region, frame_w, frame_h)
    S_hip = S
    if hip_l is not None and hip_r is not None:
        d_hip = float(np.linalg.norm(hip_r - hip_l))
        if d_hip > 1e-6:
            S_hip = d_hip

    # Рука без видимого предплечья тянется за кадр -- условие проверяется по
    # текущим точкам позы, поэтому флаги считаются здесь же.
    arm_extend = arm_frame_extend_flags(landmarks, region, frame_w, frame_h,
                                        limb_widths, limb_grow)

    shin_quads = {}
    for pair in config.STICKMAN_LIMB_COEFS:
        a, b = pair
        A = _get_point_px(landmarks, a, region, frame_w, frame_h)
        B = _get_point_px(landmarks, b, region, frame_w, frame_h)
        if A is None or B is None:
            continue
        if pair in THIGH_PAIRS:
            continue          # верх ноги закрывается четырёхугольником, п.7
        widths = limb_widths_px(pair, S, S_hip, limb_widths, limb_grow)
        if widths is None:
            continue
        rect = _build_limb_quad(A, B, widths[0], widths[1])
        if rect is not None:
            if arm_extend_wanted(arm_extend, pair):
                rect = extend_limb_quad_out_of_frame(rect, A, B,
                                                     frame_w, frame_h)
            cv2.fillPoly(mask, [rect], 255)
            if pair in SHIN_PAIRS:
                shin_quads[pair] = rect

    # 4a. Прямоугольники бёдер: постоянная ширина, равная замеренной в колене.
    for rect in build_thigh_rects(landmarks, region, frame_w, frame_h,
                                  S, S_hip, limb_widths, limb_grow,
                                  torso_quad=torso_quad):
        cv2.fillPoly(mask, [rect], 255)

    # 4b. Треугольники плеча: клин между скатом плеча (луч под углом) и
    # прямоугольником плечевой кости. Нужен откалиброванный торс.
    for tri in build_arm_shoulder_tris(landmarks, region, frame_w, frame_h,
                                       torso_quad, limb_widths, limb_grow):
        cv2.fillPoly(mask, [tri], 255)

    # 4c. Клинья в локтях и коленях: треугольники между фигурами соседних
    # костей и квадрат вместо вырожденного предплечья.
    wedges = build_joint_wedges(landmarks, region, frame_w, frame_h, limb_widths,
                                limb_grow)
    for poly in wedges['joint_tris'] + wedges['forearm_squares']:
        cv2.fillPoly(mask, [poly], 255)

    # 5. Ладони: те же прямоугольники, что строит build_body_rects и что
    # рисуются в calibration_result.png -- от запястья к середине видимых
    # точек кисти. Прежние осевые квадраты в запястьях убраны.
    for joint_idx, distal_ids in PALM_SPECS:
        A = _get_point_px(landmarks, joint_idx, region, frame_w, frame_h)
        if A is None:
            continue
        distal = [q for q in (_get_point_px(landmarks, i, region, frame_w, frame_h)
                              for i in distal_ids) if q is not None]
        if not distal:
            continue
        rect = _build_extended_rect(A, np.mean(distal, axis=0),
                                    config.STICKMAN_PALM_COEF * S,
                                    config.STICKMAN_PALM_EXTEND_COEF)
        if rect is not None:
            cv2.fillPoly(mask, [rect], 255)

    # 5b. Ног не видно, таз виден: прямоугольник от таза вниз за кадр.
    legs_below = build_legs_below_frame_quad(landmarks, region,
                                             frame_w, frame_h, torso_quad)
    if legs_below is not None:
        cv2.fillPoly(mask, [np.round(legs_below).astype(np.int32)], 255)

    # 6. Верх ног: четырёхугольники BL/BR + вершины у колена (нужен торс).
    # Режим приходит из калибровки, если он там выбирался.
    for q in build_thigh_quads(landmarks, region, frame_w, frame_h, torso_quad,
                               limb_widths=limb_widths, mode=thigh_mode):
        cv2.fillPoly(mask, [q], 255)

    # 7. Четырёхугольник ABCD по плечевым сторонам фигур рук.
    # ВРЕМЕННО вместо многоугольника XABCDY (build_upper_body_hull): чтобы
    # вернуть его, замените вызов ниже -- сама функция сохранена.
    for q in build_arm_tops_quad(landmarks, region, frame_w, frame_h,
                                 limb_widths=limb_widths):
        cv2.fillPoly(mask, [q], 255)

    # 8. Ступни: та же фигура, что рисуется при калибровке и трекинге.
    # Пятка -- индекс 29, носок -- 31, поэтому нужен полный список точек:
    # проверки в начале функции (len < 29) для них недостаточно.
    if len(landmarks) > RIGHT_FOOT_INDEX:
        for shin_pair, (ankle_idx, heel_idx, toe_idx) in FOOT_PARTS:
            ankle = _get_point_px(landmarks, ankle_idx, region, frame_w, frame_h)
            toe = _get_point_px(landmarks, toe_idx, region, frame_w, frame_h)
            if ankle is None or toe is None:
                continue
            heel = _get_point_px(landmarks, heel_idx, region, frame_w, frame_h)
            poly = build_foot_poly(ankle, heel, toe, S_hip,
                                   shin_quads.get(shin_pair))
            if poly is None:
                poly = _build_extended_rect(
                    ankle, toe, config.STICKMAN_FOOT_COEF * S_hip,
                    config.STICKMAN_FOOT_EXTEND_COEF)
            if poly is not None:
                cv2.fillPoly(mask, [poly], 255)

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
