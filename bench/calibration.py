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
    NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
)

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


# ------------------------------------------------------------------
# Калибровка головы
# ------------------------------------------------------------------
def calibrate_head(mask, pose_landmarks, region, frame_w, frame_h, chin_point=None):
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

    # Продление отрезка ушей до границы маски (максимум 1.5 * |7-8|)
    max_extend = config.CALIBRATION_EAR_EXTEND_COEF * ear_dist
    left_boundary = _find_mask_boundary_along(mask, ear_l, -e1, max_extend)
    right_boundary = _find_mask_boundary_along(mask, ear_r, e1, max_extend)
    head_width = float(np.linalg.norm(right_boundary - left_boundary))

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
    else:
        down_dist = len_XN  # подбородок не доступен - симметрично

    # Высота головы: вверх len_XN, вниз down_dist
    head_height = len_XN + down_dist

    # Прямоугольник головы (несимметричный относительно носа)
    hw = head_width / 2.0
    corners = np.array([
        nose + hw * e1 + len_XN * e2,      # верхний правый
        nose - hw * e1 + len_XN * e2,      # верхний левый
        nose - hw * e1 - down_dist * e2,   # нижний левый
        nose + hw * e1 - down_dist * e2,   # нижний правый
    ], dtype=np.float64)

    return {
        'center': nose,
        'width': head_width,
        'height': head_height,
        'e1': e1,
        'e2': e2,
        'corners': corners,
        'len_XN': len_XN,
        'down_dist': down_dist,
        'left_boundary': left_boundary,
        'right_boundary': right_boundary,
        'k_hw': head_width / S if S > 1e-6 else 0.45,
        'k_hh': head_height / S if S > 1e-6 else 0.60,
        'S': S,
    }

# ------------------------------------------------------------------
# Калибровка торса
# ------------------------------------------------------------------
def calibrate_torso(mask, pose_landmarks, region, frame_w, frame_h):
    """Калибрует торс (четырёхугольник плечи-торс) по маске InSPyReNet.

    - Верхние вершины (TL, TR): от точек плеч идём наружу (влево/вправо)
      до границы маски на уровне плеч.
    - Нижняя граница y_bottom: находится через вычитание модели ног и порог
      W_extra (свисающая одежда).
    - Нижние вершины (BL, BR): от x-координат бёдер на уровне y_bottom идём
      наружу (влево/вправо) до границы маски.

    Возвращает dict с параметрами торса или None (если плечи не видны).
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

    # Нижняя граница свисающей одежды
    y_bottom = _find_clothing_bottom(mask, pose_landmarks, region, frame_w, frame_h, S, y_hips)

    # Нижние вершины: от x-координат бёдер на уровне y_bottom идём наружу
    if hip_l is not None:
        x_BL = _find_mask_boundary_x(mask, hip_l[0], y_bottom, direction=-1)
        BL = np.array([x_BL, y_bottom], dtype=np.float64)
    else:
        x_BL = _find_mask_boundary_x(mask, sh_l[0], y_bottom, direction=-1)
        BL = np.array([x_BL, y_bottom], dtype=np.float64)

    if hip_r is not None:
        x_BR = _find_mask_boundary_x(mask, hip_r[0], y_bottom, direction=+1)
        BR = np.array([x_BR, y_bottom], dtype=np.float64)
    else:
        x_BR = _find_mask_boundary_x(mask, sh_r[0], y_bottom, direction=+1)
        BR = np.array([x_BR, y_bottom], dtype=np.float64)

    # Четырёхугольник: TL, TR, BR, BL (обход по контуру)
    quad = np.array([TL, TR, BR, BL], dtype=np.float64)

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

    # Четырёхугольник: TL, TR, BR, BL (обход по контуру)
    quad = np.array([TL, TR, BR, BL], dtype=np.float64)

    return {
        'quad': quad,
        'TL': TL, 'TR': TR, 'BR': BR, 'BL': BL,
        'y_shoulders': y_shoulders,
        'y_hips': y_hips,
        'y_bottom': y_bottom,
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
                            video_path=None, frame_index=None):
    """Сохраняет параметры калибровки в JSON для последующего отслеживания.

    head_result, torso_result - результаты calibrate_head / calibrate_torso
    (могут быть None). Сохраняются нормализованные коэффициенты, пригодные
    для отслеживания на других кадрах.
    """
    data = {
        'metadata': {
            'video_path': video_path,
            'frame_index': frame_index,
        },
        'head': None,
        'torso': None,
    }

    if head_result is not None:
        S = head_result['S']
        data['head'] = {
            'S': float(S),
            'k_hw': float(head_result['k_hw']),
            'k_hh': float(head_result['k_hh']),
            'width_coef': float(head_result['width'] / S) if S > 1e-6 else 0.0,
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
        }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Параметры калибровки сохранены: {filepath}")


def load_calibration_params(filepath):
    """Загружает параметры калибровки из JSON.

    Возвращает dict с ключами 'metadata', 'head', 'torso'.
    'head' и 'torso' могут быть None, если соответствующая часть
    не была откалибрована.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Параметры калибровки загружены: {filepath}")
    return data
