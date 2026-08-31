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
    polygon_self_intersects, limb_scale,
    ARM_PAIRS, SHIN_PAIRS,
    NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
)
# tracking тянет только stickman_model, цикла импорта не возникает
from .tracking import build_neck_quad_from_torso_and_head

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
        neck = build_neck_quad_from_torso_and_head(torso_quad, corners)
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
      - короткие стороны параллельны линии ушей (7-8), ширина = расстояние между
        границами маски, найденными продлением от ушей вдоль линии ушей
        (максимум CALIBRATION_EAR_EXTEND_COEF * |7-8|);
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
    # уходят синхронно, итоговое удлинение -- большее из двух: тот конец, что
    # вышел из маски раньше, ждёт второй. Максимум -- 1.5 * |7-8|.
    max_extend = config.CALIBRATION_EAR_EXTEND_COEF * ear_dist
    ext_l = float(np.linalg.norm(
        _find_mask_boundary_along(mask, ear_l, -e1, max_extend) - ear_l))
    ext_r = float(np.linalg.norm(
        _find_mask_boundary_along(mask, ear_r, e1, max_extend) - ear_r))
    ear_extend = max(ext_l, ext_r)
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
    границы маски. A и B -- их длины в долях scale. Луч считается коротким,
    пока он меньше порога: половины номинальной ширины cap с запасом
    LIMB_EXTEND_COEF.

        оба коротких        K = A + B          -- меряем конечность как есть
        короткий только B   K = (B + cap) / 2  -- усредняем замер с номиналом
        короткий только A   K = (A + cap) / 2
        оба длинных         K = cap            -- замер ничего не говорит

    Длинный луч означает, что он ушёл за пределы самой конечности (в торс или
    вдоль неё), поэтому в смешанном случае берётся среднее короткого замера и
    номинала. Формула симметрична по A и B, так что какой из лучей считать
    «наружным», роли не играет.

    Ветка A + B ограничена сверху не cap, а cap * (1 + LIMB_EXTEND_COEF):
    если конечность на кадре толще номинала, замер это покажет.

    Если оба луча выродились (точка вне маски -- обычное дело для кистей и
    стоп, которые сегментация теряет), возвращается cap: номинальная
    конечность лучше, чем схлопнутая в линию.
    """
    n = _rotate90(u)
    max_dist = config.CALIBRATION_LIMB_RAY_COEF * scale
    A = float(np.linalg.norm(_find_mask_boundary_along(mask, P, n, max_dist) - P))
    B = float(np.linalg.norm(_find_mask_boundary_along(mask, P, -n, max_dist) - P))
    A /= scale
    B /= scale
    if A + B < config.CALIBRATION_LIMB_WIDTH_MIN_COEF:
        return cap
    thr = cap / 2.0 * (1.0 + config.LIMB_EXTEND_COEF)
    short_a, short_b = A < thr, B < thr
    if short_a and short_b:
        return A + B
    if short_b:
        return (B + cap) / 2.0
    if short_a:
        return (A + cap) / 2.0
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
    for pair in _WIDTH_PAIRS:
        cap = config.STICKMAN_LIMB_COEFS.get(pair)
        if cap is None:
            continue
        A, B = point(pair[0]), point(pair[1])
        if A is None or B is None:
            continue
        length = float(np.linalg.norm(B - A))
        if length < 1e-6:
            continue
        u = (B - A) / length
        scale = limb_scale(pair, S, S_hip)
        for idx, P in ((pair[0], A), (pair[1], B)):
            k = _limb_width_at(mask, P, u, cap, scale)
            prev = widths.get(idx)
            widths[idx] = k if prev is None else min(prev, k)
    return _mirror_limb_widths(widths)


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
                    barrier_rects=None):
    """Калибрует торс (четырёхугольник плечи-торс) по маске InSPyReNet.

    - Верхние вершины (TL, TR): от точек плеч идём наружу (влево/вправо)
      до границы маски на уровне плеч.
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
      торса; концы найдены вытягиванием до границы маски, как у TL/TR.

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

    # Верхние вершины: вытягиваем плечи наружу ВДОЛЬ отрезка плеч
    TL = _find_mask_boundary_along(mask, sh_l, u_shoulder, max_extend_shoulder)
    TR = _find_mask_boundary_along(mask, sh_r, -u_shoulder, max_extend_shoulder)

    # Нижние вершины (BL, BR).
    #
    # Если ноги видны, свисающей одежды под торсом фактически нет (её
    # закрывают сами ноги), поэтому _find_clothing_bottom не используется:
    # нижнюю сторону строим так же, как верхнюю, только на отрезке бёдер --
    # идём от каждого бедра ВДОЛЬ отрезка 23-24 до границы маски.
    legs_visible = (hip_l is not None and hip_r is not None
                    and _legs_visible(pose_landmarks, region, frame_w, frame_h))

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

        BL = _find_mask_boundary_along(walk_mask, hip_l_low, u_hip, max_extend_hip)
        BR = _find_mask_boundary_along(walk_mask, hip_r_low, -u_hip, max_extend_hip)
        # y_bottom больше не ищется по одежде -- он задаётся самими вершинами
        y_bottom = (BL[1] + BR[1]) / 2.0
    else:
        legs_visible = False
        # Ноги не видны: прежняя логика -- нижняя граница свисающей одежды
        y_bottom = _find_clothing_bottom(mask, pose_landmarks, region,
                                         frame_w, frame_h, S, y_hips)

        # Нижние вершины: от x-координат бёдер на уровне y_bottom идём наружу
        x_start_l = hip_l[0] if hip_l is not None else sh_l[0]
        x_start_r = hip_r[0] if hip_r is not None else sh_r[0]
        BL = np.array([_find_mask_boundary_x(mask, x_start_l, y_bottom, direction=-1),
                       y_bottom], dtype=np.float64)
        BR = np.array([_find_mask_boundary_x(mask, x_start_r, y_bottom, direction=+1),
                       y_bottom], dtype=np.float64)

    # --- Линия живота ---------------------------------------------------
    # Параллельна линии плеч, отстоит от неё на CALIBRATION_BELLY_COEF среднего
    # ПЕРПЕНДИКУЛЯРНОГО расстояния до линии торса. Требования "2/3 расстояния" и
    # "параллельно плечам" совместимы только для трапеции, а торс ей не является
    # (на развёрнутом корпусе линии плеч и торса расходятся на десятки градусов),
    # поэтому смещение задаётся именно по нормали, а не вдоль боковых рёбер.
    # Концы ML/MR ищутся вытягиванием до границы маски -- как TL/TR от плеч.
    # ВРЕМЕННО ОТКЛЮЧЕНО: контур торса снова четырёхугольник плечи-торс.
    # Чтобы вернуть шестиугольник -- раскомментировать блок ниже и убрать
    # эти четыре присваивания (в tracking.py -- аналогично).
    ML = MR = None
    belly_depth_coef = belly_ext_left_coef = belly_ext_right_coef = 0.0
    belly_ok = False

    # n_sh = _rotate90(u_shoulder)
    # if float(np.dot(BL - TL, n_sh)) < 0:
    #     n_sh = -n_sh                      # нормаль направлена от плеч к торсу
    # depth = (float(np.dot(BL - TL, n_sh)) + float(np.dot(BR - TR, n_sh))) / 2.0
    # belly_off = config.CALIBRATION_BELLY_COEF * depth
    # belly_start_l = sh_l + belly_off * n_sh
    # belly_start_r = sh_r + belly_off * n_sh
    # max_extend_belly = config.CALIBRATION_BELLY_EXTEND_COEF * S
    # ML = _find_mask_boundary_along(mask, belly_start_l, u_shoulder, max_extend_belly)
    # MR = _find_mask_boundary_along(mask, belly_start_r, -u_shoulder, max_extend_belly)

    # # Доля именно от ГЛУБИНЫ плечи->торс, а не от S: при отслеживании глубина
    # # задаётся восстановленными BL/BR и меняется не пропорционально ширине
    # # плеч, поэтому доля от S уводила бы линию (на развёрнутом корпусе -- на
    # # десятки пикселей).
    # belly_depth_coef = belly_off / depth if abs(depth) > 1e-6 else 0.0
    # ext_l = float(np.linalg.norm(ML - belly_start_l))
    # ext_r = float(np.linalg.norm(MR - belly_start_r))
    # belly_ext_left_coef = ext_l / S if S > 1e-6 else 0.0
    # belly_ext_right_coef = ext_r / S if S > 1e-6 else 0.0

    # # Линия живота годится не всегда. Если стартовая точка вытягивания попала
    # # ВНЕ маски (сильно развёрнутый или укороченный ракурсом торс), то
    # # _find_mask_boundary_along возвращает сам старт, вытягивания не было и
    # # линия схлопывается; шестиугольник в таком случае ещё и перекручивается.
    # # Тогда честнее отдать прежний четырёхугольник, чем битую фигуру.
    # belly_ok = ext_l > 1e-6 and ext_r > 1e-6
    # if belly_ok:
    #     belly_ok = not polygon_self_intersects(
    #         np.array([TL, TR, MR, BR, BL, ML], dtype=np.float64))

    # --- Параметры для отслеживания (нормализованные на ширину плеч S) ---
    # Верхние точки: фактическое продление плеч (доли от S)
    ext_left_actual = float(np.linalg.norm(TL - sh_l))
    ext_right_actual = float(np.linalg.norm(TR - sh_r))
    ext_left_coef = ext_left_actual / S if S > 1e-6 else 0.0
    ext_right_coef = ext_right_actual / S if S > 1e-6 else 0.0

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
                            limb_widths=None,
                            video_path=None, frame_index=None):
    """Сохраняет параметры калибровки в JSON для последующего отслеживания.

    head_result, torso_result - результаты calibrate_head / calibrate_torso
    (могут быть None). limb_widths - результат calibrate_limb_widths.
    Сохраняются нормализованные коэффициенты, пригодные для отслеживания
    на других кадрах.
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
