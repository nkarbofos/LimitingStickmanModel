"""Калибровка stickman-модели по первому кадру с использованием маски InSPyReNet.

Калибровка головы: строит прямоугольник головы по маске (центр в носу, короткие
стороны равны отрезку ушей 7-8, длинная сторона = 2*len(XN), где X - верхняя
граница маски перпендикулярно линии ушей).

Калибровка торса: строит четырёхугольник плечи-торс по маске, вытягивая стороны
наружу до границы маски, и расширяя вниз до свисающей одежды (через вычитание
модели ног).
"""

import cv2
import numpy as np
import json

from . import config
from .stickman_model import (
    _get_point_px, _rotate90, _build_limb_rect,
    polygon_self_intersects, limb_scale, clamp_head_down_to_shoulders,
    drop_top_edge_below_frame, forearm_degenerate, _build_limb_quad,
    limb_widths_px,
    ARM_PAIRS, SHIN_PAIRS,
    NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
)
# tracking тянет только stickman_model, цикла импорта не возникает
from .tracking import build_neck_quad_from_torso_and_head, neck_sides

# Индексы точек позы для ног
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------
def _find_mask_boundary_x(mask, start_x, y, direction):
    """Идёт от start_x в направлении direction по строке y до границы маски.

    direction: -1 (влево) или +1 (вправо).
    Возвращает x последней точки, принадлежащей маске (границу).
    Если старт вне маски - возвращает start_x.
    """
    h, w = mask.shape
    y = int(round(y))
    if y < 0 or y >= h:
        return start_x
    x = int(round(start_x))
    if x < 0 or x >= w:
        return start_x
    # Если стартовая точка вне маски, возвращаем её
    if mask[y, x] == 0:
        return start_x
    # Идём в направлении direction, пока в маске
    while 0 <= x < w and mask[y, x] > 0:
        x += direction
    # x - первый пиксель вне маски; граница = x - direction
    boundary = x - direction
    boundary = max(0, min(w - 1, boundary))
    return float(boundary)


def _person_component(mask, point):
    """Связная компонента маски под точкой (0/255 -> bool).

    В кадре бывают посторонние силуэты (зрители на заднем плане), и ход вдоль
    строки не должен на них перескакивать. Если точка попала в фон, компонента
    не определяется и возвращается вся маска.
    """
    binary = (mask > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(binary)
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    h, w = binary.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return binary > 0
    lab = int(labels[y, x])
    if lab == 0:
        return binary > 0
    return labels == lab


def _walk_to_mask_edge(inside, start_x, y, direction):
    """Ближайшая граница маски на строке y: наружу, если старт внутри маски,
    иначе внутрь.

    inside -- булева маска. direction: -1 (влево) или +1 (вправо) -- сторона
    "наружу". Возвращает x границы: последний пиксель маски при ходе наружу
    либо первый пиксель маски при ходе внутрь. Если маски на строке нет
    вовсе, возвращает start_x.
    """
    h, w = inside.shape[:2]
    y = int(round(float(y)))
    x = int(round(float(start_x)))
    if not (0 <= y < h):
        return float(start_x)
    x = max(0, min(w - 1, x))
    if inside[y, x]:
        while 0 <= x + direction < w and inside[y, x + direction]:
            x += direction
        return float(x)
    # Старт вне маски -- идём внутрь, до первого пикселя маски.
    step = -direction
    while 0 <= x < w and not inside[y, x]:
        x += step
    if not (0 <= x < w):
        return float(start_x)
    return float(x)


def _find_mask_boundary_along(mask, start_point, direction, max_dist):
    """Идёт от start_point в направлении direction до границы маски.

    direction - единичный вектор. max_dist - максимальное расстояние.
    Возвращает последнюю точку, принадлежащую маске (границу).
    """
    h, w = mask.shape
    step = 1.0
    dist = 0.0
    X = np.array(start_point, dtype=np.float64).copy()
    sx, sy = float(start_point[0]), float(start_point[1])
    # Если старт вне маски, возвращаем старт
    if 0 <= int(round(sx)) < w and 0 <= int(round(sy)) < h:
        if mask[int(round(sy)), int(round(sx))] == 0:
            return X
    else:
        return X
    while dist < max_dist:
        dist += step
        point = np.array([sx, sy], dtype=np.float64) + dist * direction
        px = int(round(point[0]))
        py = int(round(point[1]))
        if px < 0 or px >= w or py < 0 or py >= h:
            break
        if mask[py, px] == 0:
            break
        X = point.copy()
    return X


def _ray_to_mask_edge(mask, start_point, direction, max_dist):
    """Длина луча от start_point до границы маски и чем он остановлен.

    Возвращает (dist, hit_frame). hit_frame=True -- ход прервался выходом за
    кадр: маска в эту сторону не кончилась, её обрезала рамка, и замер ничего
    не говорит о размере тела.

    Отличается от _find_mask_boundary_along только этим признаком: пройденное
    расстояние то же самое.
    """
    h, w = mask.shape[:2]
    sx, sy = float(start_point[0]), float(start_point[1])
    ix, iy = int(round(sx)), int(round(sy))
    if not (0 <= ix < w and 0 <= iy < h) or mask[iy, ix] == 0:
        return 0.0, False          # старт вне маски -- мерить нечего
    dist = 0.0
    reached = 0.0
    while dist < max_dist:
        dist += 1.0
        px = int(round(sx + dist * float(direction[0])))
        py = int(round(sy + dist * float(direction[1])))
        if px < 0 or px >= w or py < 0 or py >= h:
            return reached, True
        if mask[py, px] == 0:
            return reached, False
        reached = dist
    return reached, False


def _segment_hits(barrier, a, b):
    """Задевает ли отрезок a-b залитую область barrier (0/255)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    length = float(np.linalg.norm(b - a))
    h, w = barrier.shape[:2]
    for i in range(int(length) + 1):
        p = a if length < 1e-6 else a + (b - a) * (i / length)
        px, py = int(round(p[0])), int(round(p[1]))
        if 0 <= px < w and 0 <= py < h and barrier[py, px] > 0:
            return True
    return False


def _build_legs_mask(pose_landmarks, region, frame_w, frame_h, S):
    """Строит модель ног (прямоугольники) по точкам позы с дефолтными коэффициентами.

    Возвращает binary маску ног (H, W), uint8.
    """
    legs_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for (a, b), coef in config.CALIBRATION_LEG_COEFS.items():
        A = _get_point_px(pose_landmarks, a, region, frame_w, frame_h)
        B = _get_point_px(pose_landmarks, b, region, frame_w, frame_h)
        if A is None or B is None:
            continue
        rect = _build_limb_rect(A, B, coef * S)
        if rect is not None:
            cv2.fillPoly(legs_mask, [rect], 255)
    return legs_mask


def _legs_visible(pose_landmarks, region, frame_w, frame_h):
    """True, если обе ноги видны: бёдра (23, 24) и колени (25, 26).

    Колени подтверждают, что ноги действительно попали в кадр, а не только
    таз. Лодыжки не требуются -- ноги могут быть обрезаны нижней границей.
    """
    for idx in (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE):
        if _get_point_px(pose_landmarks, idx, region, frame_w, frame_h) is None:
            return False
    return True


def _find_clothing_bottom(mask, pose_landmarks, region, frame_w, frame_h, S, y_hips):
    """Находит нижнюю границу свисающей одежды (y_bottom).

    Строит модель ног, вычитает из маски, идёт вниз от бёдер, пока ширина
    остатка W_extra > порог (CALIBRATION_W_EXTRA_THRESHOLD * S).
    Возвращает y_bottom (float).
    """
    legs_mask = _build_legs_mask(pose_landmarks, region, frame_w, frame_h, S)
    # M_extra = маска БЕЗ ног
    if legs_mask is not None and np.any(legs_mask > 0):
        M_extra = np.logical_and(mask > 0, legs_mask == 0).astype(np.uint8) * 255
    else:
        M_extra = mask.copy()

    threshold = config.CALIBRATION_W_EXTRA_THRESHOLD * S

    y = int(round(y_hips))
    if y < 0:
        y = 0
    if y >= frame_h:
        return y_hips

    y_bottom = float(y_hips)
    max_y = frame_h - 1

    while y <= max_y:
        row = M_extra[y, :]
        nonzero = np.where(row > 0)[0]
        if len(nonzero) == 0:
            W_extra = 0.0
        else:
            W_extra = float(nonzero[-1] - nonzero[0])  # размах по x

        if W_extra > threshold:
            y_bottom = float(y)
            y += 1
        else:
            break

    return y_bottom



def _head_corners(nose, e1, e2, right, left, up, down):
    """Углы прямоугольника головы в том же порядке, что и в calibrate_head.

    [верхний правый, верхний левый, нижний левый, нижний правый].
    Все четыре размера -- расстояния ОТ НОСА: right вдоль +e1, left вдоль -e1,
    up вдоль +e2, down вдоль -e2. Прямоугольник не обязан быть симметричным
    относительно носа: при повороте головы нос смещён с середины линии ушей.
    """
    return np.array([
        nose + right * e1 + up * e2,
        nose - left * e1 + up * e2,
        nose - left * e1 - down * e2,
        nose + right * e1 - down * e2,
    ], dtype=np.float64)


def calibrate_neck(mask, head_corners, torso_quad, levels=None):
    """Профиль шеи: насколько трапецию можно поджать к маске на каждом уровне.

    Трапеция «нижнее ребро головы -- верхнее ребро торса» описывает шею с
    запасом: её боковые стороны прямые, а силуэт в этой полосе вогнутый (плечи
    скошены), поэтому прямая сторона режет фон.

    На levels уровнях от низа головы к линии плеч из точки на осевой линии
    пускается луч к соответствующей стороне трапеции -- но не дальше самой
    стороны. Коэффициент = пройденная доля от полуширины трапеции, то есть
    число от 0 до 1. Единица -- маска доходит до трапеции, ужимать нечего.

    Возвращает {'tl_coefs': [...], 'tr_coefs': [...]} либо None.
    """
    sides = neck_sides(torso_quad, head_corners)
    if sides is None:
        return None
    (top_l, TL), (top_r, TR) = sides
    n = int(levels if levels is not None else config.CALIBRATION_NECK_LEVELS)
    if n < 2:
        return None

    tl_coefs, tr_coefs = [], []
    for i in range(n):
        t = i / (n - 1.0)
        PL = top_l + t * (TL - top_l)
        PR = top_r + t * (TR - top_r)
        M = (PL + PR) / 2.0
        for P, out in ((PL, tl_coefs), (PR, tr_coefs)):
            half = P - M
            limit = float(np.linalg.norm(half))
            if limit < 1e-6:
                out.append(1.0)          # уровень вырожден -- оставляем трапецию
                continue
            hit = _find_mask_boundary_along(mask, M, half / limit, limit)
            reached = float(np.linalg.norm(hit - M))
            # Осевая точка вне маски: замер бессмыслен, оставляем трапецию.
            out.append(1.0 if reached < 1.0 else min(1.0, reached / limit))
    return {'tl_coefs': tl_coefs, 'tr_coefs': tr_coefs}


def _neck_band(nose, e2, torso_quad):
    """Полоса между верхним ребром торса TL-TR и уровнем носа.

    Возвращает (band_poly (4,2), h_torso) либо (None, 0.0), если верх торса
    не ниже носа (полоса вырождена). h_torso -- высота полосы до ближайшей
    из вершин TL/TR.
    """
    TL = np.asarray(torso_quad[0], dtype=np.float64)
    TR = np.asarray(torso_quad[1], dtype=np.float64)
    a_L = -float(np.dot(TL - nose, e2))   # >0, если TL ниже носа
    a_R = -float(np.dot(TR - nose, e2))
    if a_L <= 1.0 or a_R <= 1.0:
        return None, 0.0
    band_poly = np.array([TL, TR, TR + a_R * e2, TL + a_L * e2], dtype=np.float64)
    return band_poly, min(a_L, a_R)


def _fit_down_dist_by_neck_iou(mask, nose, e1, e2, right, left, len_XN, torso_quad):
    """Подбирает down_dist по IoU фигуры (голова + шея) с маской.

    Цель -- маска, ограниченная полосой между верхним ребром торса и уровнем
    носа. Полоса от down_dist НЕ зависит и считается один раз, иначе метрика
    вырождается. Кандидат -- объединение прямоугольника головы и
    четырёхугольника шеи при данном down_dist, обрезанное той же полосой.

    Возвращает (down_dist, iou, curve) или None, если подбор невозможен.
    """
    band_poly, h_torso = _neck_band(nose, e2, torso_quad)
    if band_poly is None:
        return None

    d_lo = config.CALIBRATION_NECK_FIT_MIN_COEF * len_XN
    d_hi = min(config.CALIBRATION_NECK_FIT_MAX_COEF * len_XN,
               h_torso - config.CALIBRATION_NECK_FIT_MARGIN_PX)
    if d_hi <= d_lo:
        return None

    # ROI по bounding box полосы -- растеризовать весь кадр незачем
    h, w = mask.shape[:2]
    x0 = max(0, int(np.floor(band_poly[:, 0].min())) - 2)
    y0 = max(0, int(np.floor(band_poly[:, 1].min())) - 2)
    x1 = min(w, int(np.ceil(band_poly[:, 0].max())) + 3)
    y1 = min(h, int(np.ceil(band_poly[:, 1].max())) + 3)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    org = np.array([x0, y0], dtype=np.float64)

    band_roi = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(band_roi, [(band_poly - org).astype(np.int32)], 255)
    target = cv2.bitwise_and(band_roi, mask[y0:y1, x0:x1])
    area_t = int(cv2.countNonZero(target))
    if area_t == 0:
        return None

    buf = np.empty_like(target)
    step = max(0.25, float(config.CALIBRATION_NECK_FIT_STEP_PX))
    best, curve = None, []
    d = d_lo
    while d <= d_hi + 1e-9:
        corners = _head_corners(nose, e1, e2, right, left, len_XN, d)
        # Профиль шеи меряется заново для каждого кандидата: перебор двигает
        # нижнее ребро головы, а значит и всю полосу. Иначе перебор оценивал бы
        # форму, которой в итоге не будет (см. комментарий у сборки corners).
        neck = build_neck_quad_from_torso_and_head(
            torso_quad, corners, calibrate_neck(mask, corners, torso_quad))
        buf[:] = 0
        cv2.fillPoly(buf, [(corners - org).astype(np.int32)], 255)
        if neck is not None:
            cv2.fillPoly(buf, [(neck - org).astype(np.int32)], 255)
        # фигура обрезается полосой: верх головы выше носа в цель не входит
        cv2.bitwise_and(buf, band_roi, dst=buf)
        area_q = int(cv2.countNonZero(buf))
        inter = int(cv2.countNonZero(cv2.bitwise_and(buf, target)))
        union = area_t + area_q - inter
        iou = inter / union if union > 0 else 0.0
        curve.append((d, iou))
        if best is None or iou > best[1]:
            best = (d, iou)
        d += step

    if best is None:
        return None
    return best[0], best[1], curve


# ------------------------------------------------------------------
# Калибровка головы
# ------------------------------------------------------------------
def calibrate_head(mask, pose_landmarks, region, frame_w, frame_h,
                   chin_point=None, torso_quad=None):
    """Калибрует голову по маске InSPyReNet.

    Прямоугольник головы:
      - центр (опорная точка) в носу (0);
      - короткие стороны параллельны линии ушей (7-8), ширина = отрезок ушей,
        продлённый в обе стороны на СРЕДНЕЕ из двух замеров до границы маски
        (каждый замер -- максимум CALIBRATION_EAR_EXTEND_COEF * |7-8|);
      - длинная сторона: вверх от носа до границы маски (макушка, len_XN);
        вниз от носа до подбородка (chin_point), но не дальше len(XN)  [Вариант B].

    chin_point - координаты подбородка (точка 152 face_landmarker) в тех же
                 координатах, что и mask. Если None - вниз идёт len(XN) (симметрично).

    Возвращает dict с параметрами головы или None.
    """
    nose = _get_point_px(pose_landmarks, NOSE, region, frame_w, frame_h)
    ear_l = _get_point_px(pose_landmarks, LEFT_EAR, region, frame_w, frame_h)
    ear_r = _get_point_px(pose_landmarks, RIGHT_EAR, region, frame_w, frame_h)

    if ear_l is None or ear_r is None:
        return None
    if nose is None:
        return None

    # Ширина по ушам (базовая)
    ear_dist = float(np.linalg.norm(ear_r - ear_l))
    if ear_dist < 1e-6:
        return None

    e1 = (ear_r - ear_l) / ear_dist
    e2 = _rotate90(e1)

    # Направление "вверх" (к макушке) через вектор плечи -> нос
    sh_l = _get_point_px(pose_landmarks, LEFT_SHOULDER, region, frame_w, frame_h)
    sh_r = _get_point_px(pose_landmarks, RIGHT_SHOULDER, region, frame_w, frame_h)
    if sh_l is not None and sh_r is not None:
        shoulder_mid = (sh_l + sh_r) / 2.0
        v = nose - shoulder_mid
        if np.dot(v, e2) < 0:
            e2 = -e2
        S = float(np.linalg.norm(sh_r - sh_l))
    else:
        S = ear_dist

    # Ширина: отрезок ушей растягивается в обе стороны СРАЗУ, на одну и ту же
    # величину, пока оба его конца не упрутся в границу маски. Раз концы
    # уходят синхронно, итоговое удлинение одно на обе стороны -- СРЕДНЕЕ из
    # двух замеров. Максимум каждого замера -- 1.5 * |7-8|.
    max_extend = config.CALIBRATION_EAR_EXTEND_COEF * ear_dist
    ext_l = float(np.linalg.norm(
        _find_mask_boundary_along(mask, ear_l, -e1, max_extend) - ear_l))
    ext_r = float(np.linalg.norm(
        _find_mask_boundary_along(mask, ear_r, e1, max_extend) - ear_r))
    ear_extend = (ext_l + ext_r) / 2.0
    left_boundary = ear_l - ear_extend * e1
    right_boundary = ear_r + ear_extend * e1

    # Запоминаем расстояния ОТ НОСА до каждого конца -- как len_XN у макушки.
    # Нос на линии ушей обычно не посередине (голова повёрнута), поэтому
    # стороны хранятся отдельно, а не одной шириной.
    right_dist = float(np.dot(right_boundary - nose, e1))
    left_dist = float(np.dot(nose - left_boundary, e1))
    head_width = right_dist + left_dist

    # Идём от носа вверх (вдоль e2) до границы маски -> макушка
    step = 1.0
    dist = 0.0
    max_dist = float(max(frame_w, frame_h))
    X = nose.copy()
    while dist < max_dist:
        dist += step
        point = nose + dist * e2
        px = int(round(point[0]))
        py = int(round(point[1]))
        if px < 0 or px >= frame_w or py < 0 or py >= frame_h:
            break
        if mask[py, px] == 0:
            break
        X = point.copy()

    len_XN = float(np.linalg.norm(X - nose))

    # Ограничение len(XN) (защита от слишком большой верхней границы)
    if config.CALIBRATION_EAR_EXTEND_COEF is not None:
        max_len_xn_clamp = config.CALIBRATION_EAR_EXTEND_COEF * S
        if len_XN > max_len_xn_clamp:
            len_XN = max_len_xn_clamp
            X = nose + len_XN * e2

    # Вариант B: вниз от носа ограничено подбородком (но не дальше len_XN)
    if chin_point is not None:
        chin_vec = chin_point - nose
        chin_proj = float(np.dot(chin_vec, e2))  # проекция подбородка на e2
        if chin_proj < 0:
            # подбородок в направлении -e2 (вниз от носа)
            len_chin_along_e2 = abs(chin_proj)
            down_dist = min(len_XN, len_chin_along_e2)
        else:
            # подбородок выше носа (неожиданно) - используем len_XN
            down_dist = len_XN
        down_source = 'chin'
        neck_fit = None
    else:
        # Подбородка нет: опускаем нижнюю границу, пока фигура (голова + шея)
        # не станет точнее всего закрывать маску между торсом и головой.
        down_dist = len_XN            # прежнее поведение как запасное
        down_source = 'symmetric'
        neck_fit = None
        if config.CALIBRATION_NECK_FIT_ENABLED and torso_quad is not None:
            neck_fit = _fit_down_dist_by_neck_iou(
                mask, nose, e1, e2, right_dist, left_dist, len_XN,
                np.asarray(torso_quad, dtype=np.float64))
            if neck_fit is not None:
                down_dist = float(neck_fit[0])
                down_source = 'neck_iou'

    # Фолбэк: низ головы не опускается ниже линии плеч. Подбородок бывает
    # найден на груди (лицо задрано), а перебор по шее на некоторых кадрах
    # уводит нижнюю границу в торс -- голова выходит заметно больше настоящей.
    down_before_clamp = down_dist
    down_dist = clamp_head_down_to_shoulders(nose, e1, e2, right_dist,
                                             left_dist, down_dist, sh_l, sh_r)
    if down_dist < down_before_clamp - 1e-9:
        down_source += '+shoulders'

    # Высота головы: вверх len_XN, вниз down_dist
    head_height = len_XN + down_dist

    # Прямоугольник головы (несимметричный относительно носа).
    # Та же сборка, что и в переборе _fit_down_dist_by_neck_iou -- один источник.
    corners = _head_corners(nose, e1, e2, right_dist, left_dist,
                            len_XN, down_dist)

    return {
        'center': nose,
        'width': head_width,
        'height': head_height,
        'e1': e1,
        'e2': e2,
        'corners': corners,
        'len_XN': len_XN,
        'down_dist': down_dist,
        'down_source': down_source,
        'neck_fit': neck_fit,
        'left_boundary': left_boundary,
        'right_boundary': right_boundary,
        # Ширина хранится двумя расстояниями от носа, как up/down по вертикали
        'right_dist': right_dist,
        'left_dist': left_dist,
        'ear_extend': ear_extend,
        'k_hw': head_width / S if S > 1e-6 else 0.45,
        'k_hh': head_height / S if S > 1e-6 else 0.60,
        'S': S,
    }

# ------------------------------------------------------------------
# Калибровка ширины конечностей
# ------------------------------------------------------------------
# Отрезки, для которых ширина подбирается по маске. Бёдра 23-25 и 24-26 сюда
# не входят: их фигуры строятся от вершин торса (build_thigh_quads).
_WIDTH_PAIRS = ARM_PAIRS + SHIN_PAIRS

# Зеркальные пары точек: (левая, правая). Человек симметричен, а замер по
# маске -- нет: одна рука может быть прижата к телу, вторая на свету.
_MIRROR_POINTS = ((11, 12), (13, 14), (15, 16), (25, 26), (27, 28))


def calibrate_lower_neck(mask, head_corners, frame_w, frame_h):
    """Вся нижняя часть кадра -- шея: плечи (точки 11, 12) не видны.

    Без плеч торса нет, а значит нет и обычной трапеции шеи (она строится от
    верхнего ребра торса). Тогда шеей считается всё, что ниже головы: от
    нижнего ребра прямоугольника головы вниз до последней строки маски, с
    разводом нижних углов до боков силуэта. Развод знаковый (плюс -- наружу)
    и запоминается в долях ширины НИЖНЕГО РЕБРА ГОЛОВЫ: ширины плеч, к которой
    нормируется всё остальное, здесь просто нет.

    Возвращает dict или None (нет головы, вырожденное ребро, маска не ниже
    головы).
    """
    if head_corners is None:
        return None
    head = np.asarray(head_corners, dtype=np.float64)
    if head.shape[0] < 4:
        return None
    # Нижнее ребро прямоугольника головы -- вершины 2 и 3 (см. _head_corners).
    low_l, low_r = (head[2], head[3]) if head[2][0] <= head[3][0] else (head[3], head[2])
    W = float(np.linalg.norm(low_r - low_l))
    if W < 1e-6:
        return None

    person = _person_component(mask, (low_l + low_r) / 2.0)
    rows = np.where(person.any(axis=1))[0]
    if not len(rows):
        return None
    y_bottom = float(min(frame_h - 1, int(rows.max())))
    if y_bottom <= max(low_l[1], low_r[1]) + 1.0:
        return None                        # маска не идёт ниже головы

    x_l = _walk_to_mask_edge(person, low_l[0], y_bottom, -1)
    x_r = _walk_to_mask_edge(person, low_r[0], y_bottom, +1)
    return {
        'quad': np.array([low_l, low_r, [x_r, y_bottom], [x_l, y_bottom]],
                         dtype=np.float64),
        'W': W,
        'y_bottom': y_bottom,
        'out_left_coef': (low_l[0] - x_l) / W,
        'out_right_coef': (x_r - low_r[0]) / W,
    }


def _point_cap(idx):
    """Номинальная ширина точки: минимум по отрезкам, которым она принадлежит.

    Для локтей 13/14 это минимум из плеча и предплечья -- ровно то, чем
    ограничен их K после взятия минимума двух замеров.
    """
    caps = [config.STICKMAN_LIMB_COEFS[p] for p in _WIDTH_PAIRS if idx in p]
    return min(caps) if caps else None


def _limb_width_at(mask, P, u, cap, scale):
    """Коэффициент ширины в точке P для отрезка с направлением u.

    Из P пускаются два луча перпендикулярно отрезку -- в обе стороны, до
    границы маски. A и B -- их длины в долях scale, каждая из них половина
    ширины конечности.

    Луч НЕ ИЗМЕРИЛ конечность в двух случаях:
      * упёрся в край кадра -- маску там обрезала рамка, а не тело;
      * оказался длинным (не меньше половины номинальной ширины cap с запасом
        LIMB_EXTEND_COEF) -- значит ушёл за пределы конечности: в торс или
        вдоль неё. Так всегда происходит в точках плеч 11 и 12, где луч
        внутрь тела идёт через весь торс.

    Такой луч в расчёт не берётся, и ширина меряется по другому -- в удвоенном
    размере, поскольку конечность симметрична относительно своей оси:

        оба измерили        K = A + B
        измерил только B    K = 2 * B
        измерил только A    K = 2 * A
        не измерил ни один  K = cap        -- замер ничего не говорит

    Ветка A + B ограничена сверху не cap, а cap * (1 + LIMB_EXTEND_COEF):
    если конечность на кадре толще номинала, замер это покажет. По той же
    причине столько же ограничивает и удвоенный одиночный луч.

    Если оба луча выродились (точка вне маски -- обычное дело для кистей и
    стоп, которые сегментация теряет), возвращается cap: номинальная
    конечность лучше, чем схлопнутая в линию.
    """
    n = _rotate90(u)
    max_dist = config.CALIBRATION_LIMB_RAY_COEF * scale
    A, a_edge = _ray_to_mask_edge(mask, P, n, max_dist)
    B, b_edge = _ray_to_mask_edge(mask, P, -n, max_dist)
    A /= scale
    B /= scale
    thr = cap / 2.0 * (1.0 + config.LIMB_EXTEND_COEF)

    a_ok = not a_edge and A < thr
    b_ok = not b_edge and B < thr
    if a_ok and b_ok:
        return cap if A + B < config.CALIBRATION_LIMB_WIDTH_MIN_COEF else A + B
    if a_ok or b_ok:
        half = A if a_ok else B
        return cap if 2.0 * half < config.CALIBRATION_LIMB_WIDTH_MIN_COEF else 2.0 * half
    return cap


def calibrate_limb_widths(mask, pose_landmarks, region, frame_w, frame_h):
    """Коэффициенты ширины конечностей по точкам: {индекс точки: K}.

    Замер идёт по каждому отрезку из _WIDTH_PAIRS, для обоих его концов.
    Локти 13/14 входят сразу в два отрезка (плечо и предплечье) -- для них
    берётся МИНИМУМ двух замеров, чтобы локоть не вылез за тело ни в одном из
    двух четырёхугольников. Откат на STICKMAN_LIMB_COEFS при вырожденном
    замере делается внутри _limb_width_at, то есть до взятия минимума.

    Последним шагом левая и правая стороны сводятся к общему значению
    (_mirror_limb_widths).

    Единицы те же, что у STICKMAN_LIMB_COEFS: доли ширины плеч для рук, доли
    ширины таза для ног. Точки, которых не видно, в результат не попадают.
    """
    def point(idx):
        return _get_point_px(pose_landmarks, idx, region, frame_w, frame_h)

    sh_l, sh_r = point(LEFT_SHOULDER), point(RIGHT_SHOULDER)
    if sh_l is None or sh_r is None:
        return {}
    S = float(np.linalg.norm(sh_r - sh_l))
    if S < 1e-6:
        return {}

    hip_l, hip_r = point(LEFT_HIP), point(RIGHT_HIP)
    S_hip = S
    if hip_l is not None and hip_r is not None:
        d_hip = float(np.linalg.norm(hip_r - hip_l))
        if d_hip > 1e-6:
            S_hip = d_hip

    widths = {}
    measured = set()          # точки, где луч действительно что-то намерил
    for pair in _WIDTH_PAIRS:
        cap = config.STICKMAN_LIMB_COEFS.get(pair)
        if cap is None:
            continue
        A, B = point(pair[0]), point(pair[1])
        if A is None or B is None:
            continue
        # Вырожденное предплечье: его собственной длины на кадре нет, лучи
        # поперёк такого отрезка меряют что угодно, только не руку. Замеры
        # не принимаются вовсе -- ни для ширины, ни для зеркала, ни для
        # раздвижения; вместо фигуры там строится квадрат (build_joint_wedges).
        if pair in _FOREARM_PAIRS and forearm_degenerate(
                point(_FOREARM_PAIRS[pair]), A, B):
            continue
        length = float(np.linalg.norm(B - A))
        if length < 1e-6:
            continue
        u = (B - A) / length
        scale = limb_scale(pair, S, S_hip)
        for idx, P in ((pair[0], A), (pair[1], B)):
            k = _limb_width_at(mask, P, u, cap, scale)
            if k != cap:
                measured.add(idx)
            prev = widths.get(idx)
            widths[idx] = k if prev is None else min(prev, k)

    widths = _mirror_limb_widths(widths)
    return widths, _limb_grow_coefs(mask, point, widths, measured, S, S_hip)


# Предплечье -> плечо, от которого оно отходит (для проверки вырожденности).
_FOREARM_PAIRS = {(13, 15): 11, (14, 16): 12}
# Зеркальные точки: замер симметричной стороны считается замером и здесь --
# именно его подставляет _mirror_limb_widths.
_MIRROR_POINT = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15,
                 25: 26, 26: 25, 27: 28, 28: 27}


def _limb_grow_coefs(mask, point, widths, measured, S, S_hip):
    """Раздвижение каждой пары по маске, в долях её масштаба.

    Раздвигается только пара, у которой лучами измерены ОБЕ ширины -- в
    каждом из двух концов, своим замером или замером зеркальной точки (его и
    подставляет _mirror_limb_widths). Если хоть один конец остался
    номинальным, фигура не про эту конечность и раздвигать её не от чего.
    """
    coefs = {}
    if not config.CALIBRATION_LIMB_GROW_ENABLED:
        return coefs

    def ray_measured(idx):
        return idx in measured or _MIRROR_POINT.get(idx, idx) in measured

    for pair in _WIDTH_PAIRS:
        if not (ray_measured(pair[0]) and ray_measured(pair[1])):
            continue
        A, B = point(pair[0]), point(pair[1])
        if A is None or B is None:
            continue
        if pair in _FOREARM_PAIRS and forearm_degenerate(
                point(_FOREARM_PAIRS[pair]), A, B):
            continue
        px = limb_widths_px(pair, S, S_hip, widths)
        if px is None:
            continue
        scale = limb_scale(pair, S, S_hip)
        grow = _grow_limb_width(mask, A, B, px[0], px[1], scale)
        if grow > 0.0 and scale > 1e-6:
            coefs['%d_%d' % pair] = grow / scale
    return coefs


def _grow_limb_width(mask, A, B, width_a, width_b, scale):
    """Насколько фигуру пары можно симметрично раздвинуть по маске.

    Стороны, проходящие через обе точки пары, отодвигаются от оси
    одинаковыми шагами (CALIBRATION_LIMB_GROW_STEP_COEF долей масштаба на
    сторону). Каждый шаг добавляет к фигуре полосу с двух боков; пока в этой
    полосе не меньше CALIBRATION_LIMB_GROW_MIN_FILL маски, шаг принимается.
    Первый же шаг, где полоса пустеет, останавливает рост.

    Возвращает раздвижение НА СТОРОНУ в пикселях.
    """
    step = config.CALIBRATION_LIMB_GROW_STEP_COEF * scale
    limit = config.CALIBRATION_LIMB_GROW_MAX_COEF * scale
    if step <= 0.0 or limit <= 0.0:
        return 0.0
    inside = mask > 0
    grow = 0.0
    prev = _build_limb_quad(A, B, width_a, width_b)
    if prev is None:
        return 0.0
    while grow + step <= limit:
        nxt = _build_limb_quad(A, B, width_a + 2.0 * (grow + step),
                               width_b + 2.0 * (grow + step))
        if nxt is None:
            break
        canvas_prev = np.zeros(mask.shape[:2], dtype=np.uint8)
        canvas_next = np.zeros(mask.shape[:2], dtype=np.uint8)
        cv2.fillPoly(canvas_prev, [np.asarray(prev, dtype=np.int32)], 1)
        cv2.fillPoly(canvas_next, [np.asarray(nxt, dtype=np.int32)], 1)
        strip = np.logical_and(canvas_next > 0, canvas_prev == 0)
        area = int(strip.sum())
        if area == 0:
            break
        if float(np.logical_and(strip, inside).sum()) / area < \
                config.CALIBRATION_LIMB_GROW_MIN_FILL:
            break
        grow += step
        prev = nxt
    return grow


def _mirror_limb_widths(widths):
    """Сводит левую и правую сторону к общему значению.

    K, в точности равный номиналу, означает, что замер ничего не дал: оба
    луча ушли за пределы конечности либо точка вообще вне маски. Тогда
    берётся значение зеркальной точки -- у симметричного человека это
    осмысленнее номинала. Если замерились обе стороны, берётся среднее.
    Если обе оказались номинальными, среднее брать не из чего -- обе
    остаются номинальными.

    Пара, у которой одна из точек не видна, не трогается: подставлять
    ширину для отсутствующей точки нечего, фигуры для неё всё равно нет.
    """
    for left, right in _MIRROR_POINTS:
        kl, kr = widths.get(left), widths.get(right)
        if kl is None or kr is None:
            continue
        if kl == _point_cap(left):
            widths[left] = kr
        elif kr == _point_cap(right):
            widths[right] = kl
        else:
            a = (kl + kr) / 2.0
            widths[left] = a
            widths[right] = a
    return widths


# ------------------------------------------------------------------
# Калибровка торса
# ------------------------------------------------------------------
def calibrate_torso(mask, pose_landmarks, region, frame_w, frame_h,
                    barrier_rects=None, limb_rects=None):
    """Калибрует торс (четырёхугольник плечи-торс) по маске InSPyReNet.

    - Верхние вершины (TL, TR): из каждого плеча пускается луч под
      CALIBRATION_SHOULDER_RAY_DEG к отрезку плеч, наружу и вверх, до границы
      маски. TL строится от точки 12, TR -- от точки 11. Прежнее вытягивание
      вдоль самой линии плеч не используется: оно уходило по поднятой руке.
    - Нижние вершины (BL, BR):
        * ноги видны -- точки бёдер сперва опускаются перпендикулярно линии
          торса 23-24 на STICKMAN_TORSO_EXTEND_COEF * S, а затем от них идём
          наружу ВДОЛЬ отрезка бёдер до границы маски (аналогия с TL/TR на
          отрезке плеч); y_bottom при этом задаётся самими вершинами;
        * ноги не видны -- прежняя логика: y_bottom ищется через вычитание
          модели ног и порог W_extra (свисающая одежда), а BL/BR берутся
          от x-координат бёдер на уровне y_bottom наружу до границы маски.

    barrier_rects -- список прямоугольников (рук и ладоней), сквозь которые
    вытягивание BL/BR не проходит: движение останавливается либо на границе
    маски, либо при входе в такой прямоугольник. Прямоугольники ног в барьер
    НЕ входят. None -- барьера нет.

    - Линия живота (ML, MR): параллельна линии плеч, отстоит от неё на
      CALIBRATION_BELLY_COEF среднего перпендикулярного расстояния до линии
      торса; концы найдены вытягиванием от её середины наружу до границы
      маски. Выключается флагом CALIBRATION_BELLY_ENABLED.

    limb_rects -- четырёхугольники рук и ног. Если луч из середины линии
    живота до границы маски (уже БЕЗ ограничения CALIBRATION_BELLY_EXTEND_COEF)
    задевает такую фигуру, линия живота не строится: сбоку от корпуса на этом
    уровне стоит конечность, и линия ушла бы в неё вместо бока. Тогда торс
    остаётся четырёхугольником плечи-торс. None -- проверка не делается.

    Возвращает dict с параметрами торса или None (если плечи не видны).
    Ключ 'quad' -- ШЕСТИугольник [TL, TR, MR, BR, BL, ML].
    """
    sh_l = _get_point_px(pose_landmarks, LEFT_SHOULDER, region, frame_w, frame_h)
    sh_r = _get_point_px(pose_landmarks, RIGHT_SHOULDER, region, frame_w, frame_h)
    hip_l = _get_point_px(pose_landmarks, LEFT_HIP, region, frame_w, frame_h)
    hip_r = _get_point_px(pose_landmarks, RIGHT_HIP, region, frame_w, frame_h)

    if sh_l is None or sh_r is None:
        return None  # плечи не видны

    S = float(np.linalg.norm(sh_r - sh_l))
    if S < 1e-6:
        return None

    y_shoulders = (sh_l[1] + sh_r[1]) / 2.0

    if hip_l is not None and hip_r is not None:
        y_hips = (hip_l[1] + hip_r[1]) / 2.0
    else:
        y_hips = y_shoulders + S  # фолбэк

    # Направление отрезка плеч (от левого плеча к правому)
    u_shoulder = (sh_r - sh_l) / S

    # Максимальное расстояние вытягивания плеч
    max_extend_shoulder = config.CALIBRATION_SHOULDER_EXTEND_COEF * S

    # Нормаль линии плеч, направленная от торса вверх (к голове). Нужна лучам:
    # поворот на 135 градусов даёт два направления наружу, и берётся верхнее.
    # Нижнее уходит в подмышку и дальше вдоль рукава: на half001 оно дало
    # вершину (380, 412) у нижней кромки рукава вместо (470, 272) на плече.
    n_up = _rotate90(u_shoulder)
    if hip_l is not None and hip_r is not None:
        if float(np.dot((hip_l + hip_r) / 2.0 - (sh_l + sh_r) / 2.0, n_up)) > 0:
            n_up = -n_up
    elif n_up[1] > 0:                   # бёдер нет -- ориентируемся на верх кадра
        n_up = -n_up

    def _arm_spread(shoulder, elbow, u_to_other):
        """Угол при плече между отрезком к другому плечу и плечевой костью."""
        if elbow is None:
            return None
        v = elbow - shoulder
        nv = float(np.linalg.norm(v))
        if nv < 1e-6:
            return None
        return float(np.degrees(np.arccos(
            np.clip(float(np.dot(u_to_other, v / nv)), -1.0, 1.0))))

    def _ray_dir(u_to_other):
        """Луч под CALIBRATION_SHOULDER_RAY_DEG к отрезку плеч, наружу.

        Угол тупой, поэтому оба поворота смотрят наружу (прочь от второго
        плеча) и различаются только знаком по нормали. Берётся верхний: он
        выходит из маски на скате плеча, где и стоит угол торса, а нижний
        уходит в подмышку и дальше по рукаву.
        """
        a = np.radians(config.CALIBRATION_SHOULDER_RAY_DEG)
        c, sn = np.cos(a), np.sin(a)
        d1 = np.array([u_to_other[0] * c - u_to_other[1] * sn,
                       u_to_other[0] * sn + u_to_other[1] * c])
        d2 = np.array([u_to_other[0] * c + u_to_other[1] * sn,
                       -u_to_other[0] * sn + u_to_other[1] * c])
        return d1 if float(np.dot(d1, n_up)) >= float(np.dot(d2, n_up)) else d2

    # Локти: 13 -- у плеча 11, 14 -- у плеча 12.
    elbow_l = _get_point_px(pose_landmarks, 13, region, frame_w, frame_h)
    elbow_r = _get_point_px(pose_landmarks, 14, region, frame_w, frame_h)

    # Верхние вершины строятся ВСЕГДА лучом под CALIBRATION_SHOULDER_RAY_DEG
    # к отрезку плеч, наружу и вверх, от своего плеча: TL -- от точки 12,
    # TR -- от точки 11. Прежнее вытягивание вдоль самой линии плеч убрано:
    # оно уходило по руке, стоило ей подняться к горизонтали, и порог
    # CALIBRATION_ARM_SPREAD_DEG переключал схему рывком, посреди движения.
    # Углы разворота рук считаются по-прежнему, но только для печати.
    spread_11 = _arm_spread(sh_l, elbow_l, u_shoulder)     # угол при 11: 11->12 и 11->13
    spread_12 = _arm_spread(sh_r, elbow_r, -u_shoulder)    # угол при 12: 12->11 и 12->14

    tl_by_ray = tr_by_ray = True
    TL = _find_mask_boundary_along(mask, sh_r, _ray_dir(-u_shoulder),
                                   max_extend_shoulder)
    TR = _find_mask_boundary_along(mask, sh_l, _ray_dir(u_shoulder),
                                   max_extend_shoulder)

    # Нижние вершины (BL, BR).
    #
    # Если ноги видны, свисающей одежды под торсом фактически нет (её
    # закрывают сами ноги), поэтому _find_clothing_bottom не используется:
    # нижнюю сторону строим так же, как верхнюю, только на отрезке бёдер --
    # идём от каждого бедра ВДОЛЬ отрезка 23-24 до границы маски.
    hips_visible = hip_l is not None and hip_r is not None
    legs_visible = (hips_visible
                    and _legs_visible(pose_landmarks, region, frame_w, frame_h))
    # Точек торса нет вовсе -- низ строить не от чего и не по чему: маска ниже
    # плеч принадлежит не торсу, а всему, что попало в кадр. Тогда торс -- это
    # прямоугольник от откалиброванного отрезка плеч, опущенный по нормали до
    # полного выхода нижней стороны за нижнюю кромку кадра.
    torso_rect_below_frame = not hips_visible

    hip_width = (float(np.linalg.norm(hip_r - hip_l))
                 if hip_l is not None and hip_r is not None else 0.0)

    if legs_visible and hip_width > 1e-6:
        u_hip = (hip_r - hip_l) / hip_width
        max_extend_hip = config.CALIBRATION_HIP_EXTEND_COEF * S
        # Барьер: внутрь прямоугольников рук и ладоней не заходим -- вырезаем
        # их из маски, тогда обычный поиск границы сам остановится на них.
        walk_mask = mask
        if barrier_rects:
            barrier = np.zeros(mask.shape[:2], dtype=np.uint8)
            for rect in barrier_rects:
                cv2.fillPoly(barrier, [np.asarray(rect, dtype=np.int32)], 255)
            walk_mask = np.where(barrier > 0, 0, mask).astype(np.uint8)
        # Перед вытягиванием опускаем сами точки бёдер перпендикулярно линии
        # торса 23-24 на STICKMAN_TORSO_EXTEND_COEF * S. Из двух нормалей
        # берём ту, что направлена от плеч (вниз по телу), иначе торс
        # укоротился бы вместо удлинения.
        n_hip = _rotate90(u_hip)
        if np.dot(n_hip, (hip_l + hip_r) / 2.0 - (sh_l + sh_r) / 2.0) < 0:
            n_hip = -n_hip
        drop = config.STICKMAN_TORSO_EXTEND_COEF * S
        hip_l_low = hip_l + n_hip * drop
        hip_r_low = hip_r + n_hip * drop

        # Каждая вершина идёт от бедра СВОЕЙ стороны наружу, прочь от второго
        # бедра. BL лежит со стороны точки 24 (там же, где TL), BR -- со
        # стороны точки 23. Прежняя перекрёстная схема (старт от парной точки
        # насквозь через таз) ломается, как только маска между ногами
        # разорвана: луч встаёт в зазоре, и вершины меняются сторонами.
        BL = _find_mask_boundary_along(walk_mask, hip_r_low, u_hip, max_extend_hip)
        BR = _find_mask_boundary_along(walk_mask, hip_l_low, -u_hip, max_extend_hip)
        # y_bottom больше не ищется по одежде -- он задаётся самими вершинами
        y_bottom = (BL[1] + BR[1]) / 2.0
    elif torso_rect_below_frame:
        legs_visible = False
        BL, BR = drop_top_edge_below_frame(TL, TR, -n_up, frame_h)
        y_bottom = (BL[1] + BR[1]) / 2.0
    else:
        legs_visible = False
        # Ноги не видны: прежняя логика -- нижняя граница свисающей одежды
        y_bottom = _find_clothing_bottom(mask, pose_landmarks, region,
                                         frame_w, frame_h, S, y_hips)

        # Нижние вершины: от x-координаты бедра СВОЕЙ стороны на уровне
        # y_bottom идём наружу, прочь от второго бедра. BL лежит со стороны
        # точки 24 (там же, где TL), BR -- со стороны точки 23. Сторона
        # берётся по самой координате, чтобы не зависеть от разворота
        # человека. Старт от парной точки (насквозь через таз) не годится:
        # на этом уровне силуэт нередко уже разорван на две ноги, ход
        # останавливается в зазоре между ними, и вершины меняются сторонами
        # -- четырёхугольник складывается бантиком.
        dir_bl = 1 if hip_r[0] >= hip_l[0] else -1
        BL = np.array([_find_mask_boundary_x(mask, hip_r[0], y_bottom, dir_bl),
                       y_bottom], dtype=np.float64)
        BR = np.array([_find_mask_boundary_x(mask, hip_l[0], y_bottom, -dir_bl),
                       y_bottom], dtype=np.float64)

    # --- Пятиугольники "плечи-низ" --------------------------------------
    # Половинный кадр: бёдер нет, ниже плеч у модели не остаётся фигур. Для
    # стороны, чья рука не видна, достраивается пятиугольник:
    #
    #   откалиброванная вершина плеча (TL или TR)
    #   -> настоящая точка плеча (12 или 11)
    #   -> низ кадра под ней (по кадру, а не по нормали к линии плеч)
    #   -> низ кадра наружу до границы маски
    #   -> точка A: продление линии 12-11 наружу до границы маски
    #
    # Запоминаются два отношения к ширине плеч: развод по низу (знаковый:
    # плюс -- наружу, минус -- внутрь) и вынос точки A. По ним фигура
    # строится при отслеживании.
    def _arm_seen(ids):
        return any(
            _get_point_px(pose_landmarks, i, region, frame_w, frame_h) is not None
            for i in ids)

    arm_11_seen = _arm_seen((13, 15))       # рука точки 11
    arm_12_seen = _arm_seen((14, 16))       # рука точки 12
    hips_seen = hip_l is not None or hip_r is not None

    side_bottom_quads = {}                  # индекс плеча -> контур (5, 2)
    side_bottom_coefs = {}                  # индекс плеча -> знаковый развод
    side_bottom_a_coefs = {}                # индекс плеча -> вынос точки A

    if not hips_seen and not (arm_11_seen and arm_12_seen):
        # Только силуэт этого человека: в кадре бывают посторонние, и ход вдоль
        # строки не должен на них перескакивать.
        person = _person_component(mask, (TL + TR) / 2.0)
        rows = np.where(person.any(axis=1))[0]
        y_bottom_q = float(min(frame_h - 1,
                               int(rows.max()) if len(rows) else frame_h - 1))
        for idx, shoulder, other, top, seen in (
                (LEFT_SHOULDER, sh_l, sh_r, TR, arm_11_seen),
                (RIGHT_SHOULDER, sh_r, sh_l, TL, arm_12_seen)):
            if seen:
                continue
            # "Наружу" -- прочь от второго плеча. Вдоль линии плеч это точка A,
            # по нижней строке -- сторона кадра, где стоит само плечо.
            u_out = (shoulder - other) / S
            A = _find_mask_boundary_along(mask, shoulder, u_out,
                                          max_extend_shoulder)
            sign = 1 if shoulder[0] >= other[0] else -1
            x_out = _walk_to_mask_edge(person, shoulder[0], y_bottom_q, sign)
            side_bottom_coefs[idx] = (sign * (x_out - shoulder[0]) / S
                                      if S > 1e-6 else 0.0)
            side_bottom_a_coefs[idx] = (float(np.linalg.norm(A - shoulder)) / S
                                        if S > 1e-6 else 0.0)
            side_bottom_quads[idx] = np.array(
                [top, shoulder, [shoulder[0], y_bottom_q],
                 [x_out, y_bottom_q], A], dtype=np.float64)

    # --- Линия живота ---------------------------------------------------
    # Параллельна линии плеч, отстоит от неё на CALIBRATION_BELLY_COEF среднего
    # ПЕРПЕНДИКУЛЯРНОГО расстояния до линии торса. Требования "2/3 расстояния" и
    # "параллельно плечам" совместимы только для трапеции, а торс ей не является
    # (на развёрнутом корпусе линии плеч и торса расходятся на десятки градусов),
    # поэтому смещение задаётся именно по нормали, а не вдоль боковых рёбер.
    # Концы ML/MR ищутся вытягиванием от середины линии наружу до границы
    # маски (почему от середины, а не от плеч -- см. ниже).
    # Выключается флагом CALIBRATION_BELLY_ENABLED: тогда торс остаётся
    # четырёхугольником плечи-торс (в tracking.py -- тот же флаг).
    ML = MR = None
    belly_depth_coef = belly_ext_left_coef = belly_ext_right_coef = 0.0
    belly_ok = False
    belly_reason = 'выключена флагом CALIBRATION_BELLY_ENABLED'

    if torso_rect_below_frame:
        belly_reason = 'торс -- прямоугольник до низа кадра (точек 23-24 нет)'
    elif config.CALIBRATION_BELLY_ENABLED:
        n_sh = _rotate90(u_shoulder)
        if float(np.dot(BL - TL, n_sh)) < 0:
            n_sh = -n_sh                  # нормаль направлена от плеч к торсу
        depth = (float(np.dot(BL - TL, n_sh)) + float(np.dot(BR - TR, n_sh))) / 2.0
        belly_off = config.CALIBRATION_BELLY_COEF * depth
        # Вытягивание идёт ОТ СЕРЕДИНЫ линии живота наружу, а не от точек плеч.
        # Точка, спущенная от плеча, на уровне живота нередко попадает в руку
        # или вовсе за маску (руки на этой высоте прижаты к бокам), и тогда
        # вытягивание останавливается в зазоре между рукой и корпусом либо не
        # начинается вовсе. Середина же лежит на корпусе при любом ракурсе, а
        # остановка в зазоре у руки -- ровно то, что нужно: линия живота
        # заканчивается на боку, а не уходит в руку.
        belly_mid = (sh_l + sh_r) / 2.0 + belly_off * n_sh
        max_extend_belly = config.CALIBRATION_BELLY_EXTEND_COEF * S
        ML = _find_mask_boundary_along(mask, belly_mid, u_shoulder,
                                       max_extend_belly)
        MR = _find_mask_boundary_along(mask, belly_mid, -u_shoulder,
                                       max_extend_belly)

        # Доля именно от ГЛУБИНЫ плечи->торс, а не от S: при отслеживании глубина
        # задаётся восстановленными BL/BR и меняется не пропорционально ширине
        # плеч, поэтому доля от S уводила бы линию (на развёрнутом корпусе -- на
        # десятки пикселей).
        belly_depth_coef = belly_off / depth if abs(depth) > 1e-6 else 0.0
        ext_l = float(np.linalg.norm(ML - belly_mid))
        ext_r = float(np.linalg.norm(MR - belly_mid))
        belly_ext_left_coef = ext_l / S if S > 1e-6 else 0.0
        belly_ext_right_coef = ext_r / S if S > 1e-6 else 0.0

        # Линия живота годится не всегда. Если середина попала ВНЕ маски
        # (сильно развёрнутый или укороченный ракурсом торс), то
        # _find_mask_boundary_along возвращает сам старт, вытягивания не было и
        # линия схлопывается; шестиугольник в таком случае ещё и перекручивается.
        # Тогда честнее отдать прежний четырёхугольник, чем битую фигуру.
        belly_ok = ext_l > 1e-6 and ext_r > 1e-6
        belly_reason = None if belly_ok else 'середина линии вне маски'
        if belly_ok:
            belly_ok = not polygon_self_intersects(
                np.array([TL, TR, MR, BR, BL, ML], dtype=np.float64))
            if not belly_ok:
                belly_reason = 'шестиугольник самопересекается'

        # Луч до границы маски -- уже без ограничения max_extend_belly: сам ML
        # (MR) мог упереться в потолок 1.5*S, не дойдя до руки, а вопрос в том,
        # что стоит сбоку от корпуса на уровне живота. Если свободный луч
        # задевает фигуру руки или ноги, значит конечность прижата к боку без
        # зазора в маске, линия живота уходит в неё, и корректной ширины талии
        # тут не измерить -- отдаём прежний четырёхугольник.
        if belly_ok and limb_rects:
            barrier = np.zeros(mask.shape[:2], dtype=np.uint8)
            for rect in limb_rects:
                cv2.fillPoly(barrier, [np.asarray(rect, dtype=np.int32)], 255)
            reach = float(np.hypot(mask.shape[1], mask.shape[0]))
            for direction in (u_shoulder, -u_shoulder):
                far = _find_mask_boundary_along(mask, belly_mid, direction, reach)
                if _segment_hits(barrier, belly_mid, far):
                    belly_ok = False
                    belly_reason = 'луч упирается в фигуру руки или ноги'
                    break

    # --- Параметры для отслеживания (нормализованные на ширину плеч S) ---
    # Верхние точки: фактическое продление плеч (доли от S). Оставлено для
    # старых потребителей -- луч под углом одним числом не описывается.
    ext_left_actual = float(np.linalg.norm(TL - sh_l))
    ext_right_actual = float(np.linalg.norm(TR - sh_r))
    ext_left_coef = ext_left_actual / S if S > 1e-6 else 0.0
    ext_right_coef = ext_right_actual / S if S > 1e-6 else 0.0

    # Верхние точки хранятся не поодиночке, а целым отрезком TL-TR,
    # привязанным к линии плеч 11-12:
    #   1) отрезок поворачивается вокруг своего центра до параллельности
    #      линии плеч (длина и центр при этом не меняются);
    #   2) из середины линии плеч на него опускается перпендикуляр; его
    #      основание P делит отрезок на две части.
    # Запоминаются три доли ширины плеч: длина перпендикуляра и длины обеих
    # частей. Величины знаковые -- перпендикуляр вдоль нормали к голове
    # (n_up), части вдоль линии плеч (u_shoulder), -- так схема не ломается,
    # когда основание перпендикуляра выходит за пределы самого отрезка.
    C_top = (TL + TR) / 2.0
    half_top = float(np.linalg.norm(TR - TL)) / 2.0
    sh_mid_top = (sh_l + sh_r) / 2.0
    perp_top = float(np.dot(C_top - sh_mid_top, n_up))
    P_top = sh_mid_top + perp_top * n_up          # основание перпендикуляра
    # Поворот сохраняет порядок вершин вдоль отрезка: конец со стороны +u
    # остаётся вершиной TL (она лежит со стороны точки 12).
    dir_tl = 1.0 if float(np.dot(TL - C_top, u_shoulder)) >= 0 else -1.0
    E_TL = C_top + dir_tl * half_top * u_shoulder
    E_TR = C_top - dir_tl * half_top * u_shoulder
    top_perp_coef = perp_top / S if S > 1e-6 else 0.0
    top_left_len_coef = (float(np.dot(E_TL - P_top, u_shoulder)) / S
                         if S > 1e-6 else 0.0)
    top_right_len_coef = (float(np.dot(E_TR - P_top, u_shoulder)) / S
                          if S > 1e-6 else 0.0)

    # Нижние точки: смещение относительно середины ПЛЕЧ (доли от S)
    sh_mid = (sh_l + sh_r) / 2.0
    dx_left_coef = (BL[0] - sh_mid[0]) / S if S > 1e-6 else 0.0
    dy_left_coef = (BL[1] - sh_mid[1]) / S if S > 1e-6 else 0.0
    dx_right_coef = (BR[0] - sh_mid[0]) / S if S > 1e-6 else 0.0
    dy_right_coef = (BR[1] - sh_mid[1]) / S if S > 1e-6 else 0.0

    # Нижние точки: смещение относительно середины БЁДЕР (если бёдра видны)
    has_hip_ref = hip_l is not None and hip_r is not None
    if has_hip_ref:
        hip_mid = (hip_l + hip_r) / 2.0
        dx_left_coef_hip = (BL[0] - hip_mid[0]) / S if S > 1e-6 else 0.0
        dy_left_coef_hip = (BL[1] - hip_mid[1]) / S if S > 1e-6 else 0.0
        dx_right_coef_hip = (BR[0] - hip_mid[0]) / S if S > 1e-6 else 0.0
        dy_right_coef_hip = (BR[1] - hip_mid[1]) / S if S > 1e-6 else 0.0
    else:
        dx_left_coef_hip = 0.0
        dy_left_coef_hip = 0.0
        dx_right_coef_hip = 0.0
        dy_right_coef_hip = 0.0

    # Нижние точки в СОБСТВЕННОМ базисе отрезка бёдер 23-24: вдоль отрезка (u)
    # и по нормали к нему (n). Благодаря этому при отслеживании низ торса
    # поворачивается вместе с отрезком бёдер и остаётся ему параллелен, а не
    # держит фиксированное направление в осях кадра.
    # Базис детерминирован (u из 23->24, n = _rotate90(u)), поэтому калибровка
    # и отслеживание строят его одинаково -- неоднозначности знака нет.
    # Пишется только для ветки, где BL/BR и строились по отрезку бёдер.
    hip_frame_ref = bool(legs_visible and hip_width > 1e-6)
    u_left_coef_hip = n_left_coef_hip = 0.0
    u_right_coef_hip = n_right_coef_hip = 0.0
    if hip_frame_ref and S > 1e-6:
        u_hip_ref = (hip_r - hip_l) / hip_width
        n_hip_ref = _rotate90(u_hip_ref)
        hip_mid_ref = (hip_l + hip_r) / 2.0
        u_left_coef_hip = float(np.dot(BL - hip_mid_ref, u_hip_ref)) / S
        n_left_coef_hip = float(np.dot(BL - hip_mid_ref, n_hip_ref)) / S
        u_right_coef_hip = float(np.dot(BR - hip_mid_ref, u_hip_ref)) / S
        n_right_coef_hip = float(np.dot(BR - hip_mid_ref, n_hip_ref)) / S

    # Шестиугольник плечи-живот-торс, либо прежний четырёхугольник, если
    # линия живота вырождена (см. belly_ok выше).
    if belly_ok:
        quad = np.array([TL, TR, MR, BR, BL, ML], dtype=np.float64)
    else:
        quad = np.array([TL, TR, BR, BL], dtype=np.float64)

    return {
        'quad': quad,
        'TL': TL, 'TR': TR, 'BR': BR, 'BL': BL, 'ML': ML, 'MR': MR,
        # Пятиугольники "плечи-низ" по невидимой руке (половинный кадр
        # без бёдер): индекс плеча -> контур, развод по низу, вынос точки A
        'side_bottom_quads': side_bottom_quads,
        'side_bottom_coefs': side_bottom_coefs,
        'side_bottom_a_coefs': side_bottom_a_coefs,
        'shoulders_bottom_quads': list(side_bottom_quads.values()),
        'has_shoulders_bottom': bool(side_bottom_quads),
        'arm_11_seen': arm_11_seen,
        'arm_12_seen': arm_12_seen,
        # Низ торса -- прямоугольник, опущенный за нижнюю кромку кадра
        'torso_rect_below_frame': torso_rect_below_frame,
        # Верхний отрезок TL-TR через перпендикуляр от середины линии плеч
        'top_frame_ref': True,
        'top_perp_coef': top_perp_coef,
        'top_left_len_coef': top_left_len_coef,
        'top_right_len_coef': top_right_len_coef,
        'tl_by_ray': tl_by_ray,
        'tr_by_ray': tr_by_ray,
        'arm_spread_11': spread_11,
        'arm_spread_12': spread_12,
        'y_shoulders': y_shoulders,
        'y_hips': y_hips,
        'y_bottom': y_bottom,
        'legs_visible': legs_visible,
        'S': S,
        # Нормализованные параметры для отслеживания
        'S_ref': S,
        'ext_left_coef': ext_left_coef,
        'ext_right_coef': ext_right_coef,
        # Привязка к середине плеч (фолбэк)
        'dx_left_coef': dx_left_coef,
        'dy_left_coef': dy_left_coef,
        'dx_right_coef': dx_right_coef,
        'dy_right_coef': dy_right_coef,
        # Привязка к середине бёдер (основная, если бёдра видны)
        'has_hip_ref': has_hip_ref,
        'dx_left_coef_hip': dx_left_coef_hip,
        'dy_left_coef_hip': dy_left_coef_hip,
        'dx_right_coef_hip': dx_right_coef_hip,
        'dy_right_coef_hip': dy_right_coef_hip,
        # Базис отрезка бёдер: низ торса поворачивается вместе с 23-24
        'hip_frame_ref': hip_frame_ref,
        'u_left_coef_hip': u_left_coef_hip,
        'n_left_coef_hip': n_left_coef_hip,
        'u_right_coef_hip': u_right_coef_hip,
        'n_right_coef_hip': n_right_coef_hip,
        # Линия живота (концы -- вершины ML, MR шестиугольника)
        'has_belly': belly_ok,
        'belly_reason': belly_reason,
        'belly_depth_coef': belly_depth_coef,
        'belly_ext_left_coef': belly_ext_left_coef,
        'belly_ext_right_coef': belly_ext_right_coef,
        'sh_mid': sh_mid,
    }


def build_torso_quad_from_params(params, sh_l, sh_r):
    """Строит четырёхугольник торса на текущем кадре по откалиброванным параметрам.

    params - dict с нормализованными параметрами из calibrate_torso:
             ext_left_coef, ext_right_coef, dx_left_coef, dy_left_coef,
             dx_right_coef, dy_right_coef.
    sh_l, sh_r - текущие плечи (точки 11, 12) в координатах текущего кадра.

    Верхние точки (TL, TR): продлеваем текущий отрезок плеч вдоль линии плеч
    на запомненные величины (ext_coef * S_cur).
    Нижние точки (BL, BR): смещаем относительно текущей середины плеч,
    масштабируя запомненные смещения пропорционально текущей ширине плеч.
    Низ четырёхугольника НЕ поворачивается вместе с верхом (используется
    абсолютное смещение в системе координат кадра).

    Возвращает четырёхугольник [TL, TR, BR, BL] как np.array (4, 2) или None.
    """
    sh_l = np.array(sh_l, dtype=np.float64)
    sh_r = np.array(sh_r, dtype=np.float64)
    S_cur = float(np.linalg.norm(sh_r - sh_l))
    if S_cur < 1e-6:
        return None

    u_shoulder = (sh_r - sh_l) / S_cur
    sh_mid = (sh_l + sh_r) / 2.0

    # Верхние точки: продлеваем текущий отрезок плеч
    # TL: от левого плеча в направлении +u_shoulder
    # TR: от правого плеча в направлении -u_shoulder
    ext_left = params['ext_left_coef'] * S_cur
    ext_right = params['ext_right_coef'] * S_cur
    TL = sh_l + ext_left * u_shoulder
    TR = sh_r - ext_right * u_shoulder

    # Нижние точки: смещение относительно середины плеч, масштабированное на S_cur
    BL = sh_mid + np.array([params['dx_left_coef'] * S_cur,
                            params['dy_left_coef'] * S_cur], dtype=np.float64)
    BR = sh_mid + np.array([params['dx_right_coef'] * S_cur,
                            params['dy_right_coef'] * S_cur], dtype=np.float64)

    quad = np.array([TL, TR, BR, BL], dtype=np.float64)
    return quad


def save_calibration_params(filepath, head_result, torso_result,
                            limb_widths=None, neck_result=None,
                            video_path=None, frame_index=None,
                            lower_neck_result=None, limb_grow=None):
    """Сохраняет параметры калибровки в JSON для последующего отслеживания.

    head_result, torso_result - результаты calibrate_head / calibrate_torso
    (могут быть None). limb_widths - результат calibrate_limb_widths,
    neck_result - результат calibrate_neck, lower_neck_result - результат
    calibrate_lower_neck (шея во весь низ кадра, когда плечи не видны).
    Сохраняются нормализованные коэффициенты, пригодные для отслеживания на
    других кадрах.
    """
    data = {
        'metadata': {
            'video_path': video_path,
            'frame_index': frame_index,
        },
        'head': None,
        'torso': None,
        # Ширина конечностей по точкам. Ключи JSON -- строки, при чтении
        # приводятся обратно к int (load_calibration_params).
        'limbs': ({str(k): float(v) for k, v in limb_widths.items()}
                  if limb_widths else None),
        # Профиль шеи: доли полуширины трапеции на каждом уровне.
        'neck': ({'tl_coefs': [float(v) for v in neck_result['tl_coefs']],
                  'tr_coefs': [float(v) for v in neck_result['tr_coefs']]}
                 if neck_result else None),
        # Шея во весь низ кадра: плечи не видны, торса нет. Развод нижних
        # углов -- в долях ширины нижнего ребра головы.
        # Раздвижение фигур конечностей по маске, доли масштаба пары
        'limb_grow': ({k: float(v) for k, v in limb_grow.items()}
                      if limb_grow else None),
        'lower_neck': ({'out_left_coef': float(lower_neck_result['out_left_coef']),
                        'out_right_coef': float(lower_neck_result['out_right_coef'])}
                       if lower_neck_result else None),
    }

    if head_result is not None:
        S = head_result['S']
        data['head'] = {
            'S': float(S),
            'k_hw': float(head_result['k_hw']),
            'k_hh': float(head_result['k_hh']),
            # width_coef оставлен для старых потребителей: полная ширина.
            # Форму задают четыре расстояния от носа -- right/left/up/down.
            'width_coef': float(head_result['width'] / S) if S > 1e-6 else 0.0,
            'right_coef': float(head_result['right_dist'] / S) if S > 1e-6 else 0.0,
            'left_coef': float(head_result['left_dist'] / S) if S > 1e-6 else 0.0,
            'up_coef': float(head_result['len_XN'] / S) if S > 1e-6 else 0.0,
            'down_coef': float(head_result['down_dist'] / S) if S > 1e-6 else 0.0,
        }

    if torso_result is not None:
        data['torso'] = {
            'S_ref': float(torso_result['S_ref']),
            'ext_left_coef': float(torso_result['ext_left_coef']),
            'ext_right_coef': float(torso_result['ext_right_coef']),
            # Привязка к середине плеч (фолбэк)
            'dx_left_coef': float(torso_result['dx_left_coef']),
            'dy_left_coef': float(torso_result['dy_left_coef']),
            'dx_right_coef': float(torso_result['dx_right_coef']),
            'dy_right_coef': float(torso_result['dy_right_coef']),
            # Привязка к середине бёдер (основная, если бёдра видны при калибровке)
            'has_hip_ref': bool(torso_result.get('has_hip_ref', False)),
            'dx_left_coef_hip': float(torso_result.get('dx_left_coef_hip', 0.0)),
            'dy_left_coef_hip': float(torso_result.get('dy_left_coef_hip', 0.0)),
            'dx_right_coef_hip': float(torso_result.get('dx_right_coef_hip', 0.0)),
            'dy_right_coef_hip': float(torso_result.get('dy_right_coef_hip', 0.0)),
            # Базис отрезка бёдер (низ торса поворачивается вместе с 23-24)
            'hip_frame_ref': bool(torso_result.get('hip_frame_ref', False)),
            'u_left_coef_hip': float(torso_result.get('u_left_coef_hip', 0.0)),
            'n_left_coef_hip': float(torso_result.get('n_left_coef_hip', 0.0)),
            'u_right_coef_hip': float(torso_result.get('u_right_coef_hip', 0.0)),
            'n_right_coef_hip': float(torso_result.get('n_right_coef_hip', 0.0)),
            # Пятиугольники "плечи-низ" по невидимой руке. Ключи JSON --
            # строки, поэтому индексы плеч выписаны в именах.
            'side_bottom_11_coef': (
                float(torso_result['side_bottom_coefs'][LEFT_SHOULDER])
                if LEFT_SHOULDER in torso_result.get('side_bottom_coefs', {}) else None),
            'side_bottom_12_coef': (
                float(torso_result['side_bottom_coefs'][RIGHT_SHOULDER])
                if RIGHT_SHOULDER in torso_result.get('side_bottom_coefs', {}) else None),
            'side_bottom_11_a_coef': float(
                torso_result.get('side_bottom_a_coefs', {}).get(LEFT_SHOULDER, 0.0)),
            'side_bottom_12_a_coef': float(
                torso_result.get('side_bottom_a_coefs', {}).get(RIGHT_SHOULDER, 0.0)),
            # Низ торса: прямоугольник от отрезка плеч за нижнюю кромку кадра
            'torso_rect_below_frame': bool(
                torso_result.get('torso_rect_below_frame', False)),
            # Верхний отрезок TL-TR: перпендикуляр от середины линии плеч
            # и две части отрезка, все в долях ширины плеч
            'top_frame_ref': bool(torso_result.get('top_frame_ref', False)),
            'top_perp_coef': float(torso_result.get('top_perp_coef', 0.0)),
            'top_left_len_coef': float(torso_result.get('top_left_len_coef', 0.0)),
            'top_right_len_coef': float(torso_result.get('top_right_len_coef', 0.0)),
            # Линия живота: вершины ML/MR шестиугольника торса
            'has_belly': bool(torso_result.get('has_belly', False)),
            'belly_depth_coef': float(torso_result.get('belly_depth_coef', 0.0)),
            'belly_ext_left_coef': float(torso_result.get('belly_ext_left_coef', 0.0)),
            'belly_ext_right_coef': float(torso_result.get('belly_ext_right_coef', 0.0)),
        }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Параметры калибровки сохранены: {filepath}")


def load_calibration_params(filepath):
    """Загружает параметры калибровки из JSON.

    Возвращает dict с ключами 'metadata', 'head', 'torso', 'limbs'.
    Любая из частей может быть None, если она не была откалибрована; у старых
    файлов секции 'limbs' нет вовсе -- тогда конечности строятся прежними
    прямоугольниками.

    Ключи 'limbs' приводятся к int: в JSON они строки, а искать по ним
    приходится по индексу точки позы.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    limbs = data.get('limbs')
    if limbs:
        data['limbs'] = {int(k): float(v) for k, v in limbs.items()}
    print(f"Параметры калибровки загружены: {filepath}")
    return data
