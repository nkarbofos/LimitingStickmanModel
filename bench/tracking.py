"""Отслеживание stickman-модели по видео с использованием параметров калибровки.

Функции строят модель (голову, торс) на каждом кадре по текущим точкам позы
и откалиброванным параметрам (нормализованным коэффициентам).
"""

import numpy as np

from .stickman_model import (
    _get_point_px, _rotate90,
    NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
)


def build_head_rect_from_params(params, pose_landmarks, region, frame_w, frame_h):
    """Строит прямоугольник головы на текущем кадре по параметрам калибровки.

    params - dict с ключами 'width_coef', 'up_coef', 'down_coef'
             (нормализованные коэффициенты из calibrate_head).
    pose_landmarks - точки позы текущего кадра (список NormalizedLandmark).
    region - (ox, oy, rw, rh) кроп или None (полный кадр).
    frame_w, frame_h - размеры кадра.

    Возвращает corners (4, 2) как np.array или None (если точки не видны).
    """
    nose = _get_point_px(pose_landmarks, NOSE, region, frame_w, frame_h)
    ear_l = _get_point_px(pose_landmarks, LEFT_EAR, region, frame_w, frame_h)
    ear_r = _get_point_px(pose_landmarks, RIGHT_EAR, region, frame_w, frame_h)
    sh_l = _get_point_px(pose_landmarks, LEFT_SHOULDER, region, frame_w, frame_h)
    sh_r = _get_point_px(pose_landmarks, RIGHT_SHOULDER, region, frame_w, frame_h)

    if nose is None or ear_l is None or ear_r is None:
        return None

    ear_dist = float(np.linalg.norm(ear_r - ear_l))
    if ear_dist < 1e-6:
        return None

    # Направление вдоль линии ушей
    e1 = (ear_r - ear_l) / ear_dist
    # Перпендикуляр (для длинной стороны)
    e2 = _rotate90(e1)

    # Направление "вверх" (к макушке) и ширина плеч
    if sh_l is not None and sh_r is not None:
        shoulder_mid = (sh_l + sh_r) / 2.0
        v = nose - shoulder_mid
        if np.dot(v, e2) < 0:
            e2 = -e2  # e2 теперь направлен к макушке
        S_cur = float(np.linalg.norm(sh_r - sh_l))
    else:
        # Плечи не видны - используем расстояние между ушами как масштаб
        S_cur = ear_dist

    # Размеры головы из параметров калибровки (масштабированные на текущую ширину плеч)
    width = params['width_coef'] * S_cur
    up = params['up_coef'] * S_cur
    down = params['down_coef'] * S_cur

    hw = width / 2.0
    corners = np.array([
        nose + hw * e1 + up * e2,      # верхний правый
        nose - hw * e1 + up * e2,      # верхний левый
        nose - hw * e1 - down * e2,    # нижний левый
        nose + hw * e1 - down * e2,    # нижний правый
    ], dtype=np.float64)
    return corners


def build_torso_quad_from_params(params, sh_l, sh_r, hip_l=None, hip_r=None):
    """Строит четырёхугольник торса на текущем кадре по параметрам калибровки.

    params - dict с нормализованными параметрами из calibrate_torso.
    sh_l, sh_r - текущие плечи (точки 11, 12) в координатах кадра.
    hip_l, hip_r - текущие бёдра (точки 23, 24) или None (не видны).

    Верхние точки (TL, TR): продлеваем текущий отрезок плеч вдоль линии плеч
    на запомненные величины (ext_coef * S_cur).

    Нижние точки (BL, BR): если бёдра видны И при калибровке были видны бёдра
    (has_hip_ref=True) — используем смещение относительно середины БЁДЕР.
    Иначе — фолбэк на смещение относительно середины ПЛЕЧ.

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

    # Нижние точки: выбираем привязку (бёдра или плечи)
    use_hip = (hip_l is not None and hip_r is not None
               and params.get('has_hip_ref', False))

    if use_hip:
        # Основная привязка: середина бёдер
        hip_l = np.array(hip_l, dtype=np.float64)
        hip_r = np.array(hip_r, dtype=np.float64)
        hip_mid = (hip_l + hip_r) / 2.0
        BL = hip_mid + np.array([params['dx_left_coef_hip'] * S_cur,
                                 params['dy_left_coef_hip'] * S_cur], dtype=np.float64)
        BR = hip_mid + np.array([params['dx_right_coef_hip'] * S_cur,
                                 params['dy_right_coef_hip'] * S_cur], dtype=np.float64)
    else:
        # Фолбэк: середина плеч
        BL = sh_mid + np.array([params['dx_left_coef'] * S_cur,
                                params['dy_left_coef'] * S_cur], dtype=np.float64)
        BR = sh_mid + np.array([params['dx_right_coef'] * S_cur,
                                params['dy_right_coef'] * S_cur], dtype=np.float64)

    quad = np.array([TL, TR, BR, BL], dtype=np.float64)
    return quad


def build_neck_quad_from_torso_and_head(torso_quad, head_corners):
    """Строит шею как четырёхугольник между верхом торса и низом головы.

    torso_quad   - четырёхугольник торса [TL, TR, BR, BL] (4, 2).
    head_corners - прямоугольник головы [верхний правый, верхний левый,
                   нижний левый, нижний правый] (4, 2).

    Шея: верхние точки = TL, TR торса; нижние точки = нижние углы головы.
    Возвращает четырёхугольник шеи [TL, TR, нижний правый головы, нижний левый головы]
    как np.array (4, 2) или None.
    """
    if torso_quad is None or head_corners is None:
        return None

    TL = torso_quad[0]              # верхняя левая торса
    TR = torso_quad[1]              # верхняя правая торса
    head_bottom_right = head_corners[2]   # нижний левый головы
    head_bottom_left = head_corners[3]  # нижний правый головы

    # Обход: TL -> TR -> нижний правый головы -> нижний левый головы
    neck_quad = np.array([TL, TR, head_bottom_right, head_bottom_left], dtype=np.float64)
    return neck_quad
