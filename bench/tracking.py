"""Отслеживание stickman-модели по видео с использованием параметров калибровки.

Функции строят модель (голову, торс) на каждом кадре по текущим точкам позы
и откалиброванным параметрам (нормализованным коэффициентам).
"""

import numpy as np

from . import config
from .stickman_model import (
    _get_point_px, _rotate90, polygon_self_intersects, torso_vertex,
    clamp_head_down_to_shoulders, drop_top_edge_below_frame,
    NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
)


def build_head_rect_from_params(params, pose_landmarks, region, frame_w, frame_h):
    """Строит прямоугольник головы на текущем кадре по параметрам калибровки.

    params - dict с ключами 'width_coef', 'up_coef', 'down_coef'
             (нормализованные коэффициенты из calibrate_head).
    pose_landmarks - точки позы текущего кадра (список NormalizedLandmark).
    region - (ox, oy, rw, rh) кроп или None (полный кадр).
    frame_w, frame_h - размеры кадра.

    Нижняя граница дополнительно ограничена линией плеч
    (clamp_head_down_to_shoulders) -- как при калибровке.

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

    # Размеры головы из параметров калибровки (масштабированные на текущую
    # ширину плеч). Ширина задана двумя расстояниями от носа: нос на линии
    # ушей обычно не посередине. У старых калибровок этих ключей нет -- там
    # берётся прежняя половина общей ширины на обе стороны.
    up = params['up_coef'] * S_cur
    down = params['down_coef'] * S_cur
    if 'right_coef' in params and 'left_coef' in params:
        right = params['right_coef'] * S_cur
        left = params['left_coef'] * S_cur
    else:
        right = left = params['width_coef'] * S_cur / 2.0

    # Тот же фолбэк, что и при калибровке: низ головы не уходит ниже линии
    # плеч. На кадре поза может измениться так, что запомненный down_coef
    # опустит границу в грудь.
    down = clamp_head_down_to_shoulders(nose, e1, e2, right, left, down,
                                        sh_l, sh_r)

    corners = np.array([
        nose + right * e1 + up * e2,      # верхний правый
        nose - left * e1 + up * e2,       # верхний левый
        nose - left * e1 - down * e2,     # нижний левый
        nose + right * e1 - down * e2,    # нижний правый
    ], dtype=np.float64)
    return corners


def n_up_torso(u_shoulder, sh_mid, hip_l=None, hip_r=None):
    """Нормаль линии плеч, направленная от торса к голове.

    Считается так же, как при калибровке: от бёдер, если они видны, иначе
    просто вверх по кадру.
    """
    n_up = _rotate90(u_shoulder)
    if hip_l is not None and hip_r is not None:
        hip_mid = (np.asarray(hip_l, dtype=np.float64)
                   + np.asarray(hip_r, dtype=np.float64)) / 2.0
        if float(np.dot(hip_mid - sh_mid, n_up)) > 0:
            n_up = -n_up
    elif n_up[1] > 0:
        n_up = -n_up
    return n_up


def build_torso_quad_from_params(params, sh_l, sh_r, hip_l=None, hip_r=None,
                                 frame_h=None):
    """Строит четырёхугольник торса на текущем кадре по параметрам калибровки.

    params - dict с нормализованными параметрами из calibrate_torso.
    sh_l, sh_r - текущие плечи (точки 11, 12) в координатах кадра.
    hip_l, hip_r - текущие бёдра (точки 23, 24) или None (не видны).

    Верхние точки (TL, TR): продлеваем текущий отрезок плеч вдоль линии плеч
    на запомненные величины (ext_coef * S_cur).

    Нижние точки (BL, BR): если бёдра видны И при калибровке были видны бёдра
    (has_hip_ref=True) — используем смещение относительно середины БЁДЕР.
    Иначе — фолбэк на смещение относительно середины ПЛЕЧ.

    Возвращает шестиугольник [TL, TR, MR, BR, BL, ML] как np.array (6, 2);
    для старых калибровок без линии живота -- четырёхугольник (4, 2).
    None, если ширина плеч вырождена.
    """
    sh_l = np.array(sh_l, dtype=np.float64)
    sh_r = np.array(sh_r, dtype=np.float64)
    S_cur = float(np.linalg.norm(sh_r - sh_l))
    if S_cur < 1e-6:
        return None

    u_shoulder = (sh_r - sh_l) / S_cur
    sh_mid = (sh_l + sh_r) / 2.0

    # Верхние точки. Новые калибровки (top_frame_ref) кладут их в базисе
    # линии плеч: TL отложена от СВОЕГО плеча -- точки 12, TR -- от точки 11,
    # вдоль линии (u_shoulder) и по нормали к голове (n_up). Так вершина
    # повторяет луч под углом, которым её строила калибровка. Старые
    # калибровки знают только длину вдоль линии плеч -- для них прежняя схема.
    n_up = n_up_torso(u_shoulder, sh_mid, hip_l, hip_r)
    if params.get('top_frame_ref', False):
        # Верхний отрезок восстанавливается так же, как замерялся: из середины
        # линии плеч 11-12 строится перпендикуляр запомненной длины, а из его
        # конца в обе стороны откладываются две части отрезка параллельно
        # линии плеч. Отрезок поэтому всегда ей параллелен и поворачивается
        # вместе с ней.
        P_top = sh_mid + params['top_perp_coef'] * S_cur * n_up
        TL = P_top + params['top_left_len_coef'] * S_cur * u_shoulder
        TR = P_top + params['top_right_len_coef'] * S_cur * u_shoulder
    else:
        # TL: от левого плеча в направлении +u_shoulder
        # TR: от правого плеча в направлении -u_shoulder
        TL = sh_l + params['ext_left_coef'] * S_cur * u_shoulder
        TR = sh_r - params['ext_right_coef'] * S_cur * u_shoulder

    # Нижние точки: прямоугольник за кадром, привязка к бёдрам или к плечам.
    # Первый вариант -- калибровка без точек 23-24: низ там ничем не измерен,
    # фигура просто уходит вниз за кадр, и на каждом кадре её глубина считается
    # заново от текущего верхнего ребра.
    if params.get('torso_rect_below_frame', False) and frame_h is not None:
        BL, BR = drop_top_edge_below_frame(TL, TR, -n_up_torso(
            u_shoulder, sh_mid, hip_l, hip_r), frame_h)
        return np.array([TL, TR, BR, BL], dtype=np.float64)

    use_hip = (hip_l is not None and hip_r is not None
               and params.get('has_hip_ref', False))

    if use_hip:
        # Основная привязка: середина бёдер
        hip_l = np.array(hip_l, dtype=np.float64)
        hip_r = np.array(hip_r, dtype=np.float64)
        hip_mid = (hip_l + hip_r) / 2.0
        hip_width_cur = float(np.linalg.norm(hip_r - hip_l))
        if params.get('hip_frame_ref', False) and hip_width_cur > 1e-6:
            # Низ торса задан в базисе отрезка бёдер 23-24: поворачивается
            # вместе с ним и остаётся ему параллелен.
            u_hip = (hip_r - hip_l) / hip_width_cur
            n_hip = _rotate90(u_hip)
            BL = hip_mid + S_cur * (params['u_left_coef_hip'] * u_hip
                                    + params['n_left_coef_hip'] * n_hip)
            BR = hip_mid + S_cur * (params['u_right_coef_hip'] * u_hip
                                    + params['n_right_coef_hip'] * n_hip)
        else:
            # Старые калибровки без базиса бёдер -- смещение в осях кадра.
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

    # Линия живота выключена флагом или её не было при калибровке (старые
    # параметры, вырожденная линия) -- прежний четырёхугольник плечи-торс.
    if not (config.CALIBRATION_BELLY_ENABLED
            and params.get('has_belly', False)):
        return np.array([TL, TR, BR, BL], dtype=np.float64)

    # Линия живота: параллельна линии плеч, смещена по нормали к торсу, концы
    # отложены вдоль линии плеч от её середины на запомненные величины.
    n_sh = _rotate90(u_shoulder)
    if float(np.dot(BL - TL, n_sh)) < 0:
        n_sh = -n_sh
    # Глубина берётся с ТЕКУЩЕГО контура, поэтому линия всегда стоит на
    # запомненной доле расстояния плечи->торс, как бы ни изменилась поза.
    depth_cur = (float(np.dot(BL - TL, n_sh)) + float(np.dot(BR - TR, n_sh))) / 2.0
    belly_off = params['belly_depth_coef'] * depth_cur
    # Концы откладываются от СЕРЕДИНЫ линии живота -- так же, как замерялись
    # при калибровке (см. calibrate_torso).
    belly_mid = sh_mid + belly_off * n_sh
    ML = belly_mid + params['belly_ext_left_coef'] * S_cur * u_shoulder
    MR = belly_mid - params['belly_ext_right_coef'] * S_cur * u_shoulder

    # Шестиугольник плечи-живот-торс. На отдельных кадрах поза может увести
    # линию живота так, что контур перекручивается, -- там отдаём прежний
    # четырёхугольник вместо битой фигуры.
    hexagon = np.array([TL, TR, MR, BR, BL, ML], dtype=np.float64)
    if polygon_self_intersects(hexagon):
        return np.array([TL, TR, BR, BL], dtype=np.float64)
    return hexagon


def build_shoulders_bottom_quads(params, torso_quad, sh_l, sh_r,
                                 frame_w, frame_h):
    """Пятиугольники "плечи-низ" на текущем кадре: список (может быть пуст).

    По одному на сторону, чья рука не была видна при калибровке:

        откалиброванная вершина плеча (TL или TR текущего контура торса)
        -> текущая точка плеча (12 или 11)
        -> низ кадра под ней
        -> низ кадра наружу на запомненную долю ширины плеч
        -> точка A: та же доля вдоль линии плеч наружу

    Доля по низу знаковая: плюс разводит наружу, минус сводит внутрь. Обе
    доли считаются от ТЕКУЩЕЙ ширины плеч |11-12| -- в тех же единицах, что
    и при калибровке.
    """
    if torso_quad is None or sh_l is None or sh_r is None:
        return []
    sh_l = np.asarray(sh_l, dtype=np.float64)
    sh_r = np.asarray(sh_r, dtype=np.float64)
    S_cur = float(np.linalg.norm(sh_r - sh_l))
    if S_cur < 1e-6:
        return []
    y_frame = float(frame_h - 1)
    quads = []

    for key, shoulder, other, vertex_name in (
            ('side_bottom_11', sh_l, sh_r, 'TR'),
            ('side_bottom_12', sh_r, sh_l, 'TL')):
        coef = params.get(key + '_coef')
        if coef is None:
            continue
        top = torso_vertex(torso_quad, vertex_name)
        if top is None:
            continue
        u_out = (shoulder - other) / S_cur
        A = shoulder + params.get(key + '_a_coef', 0.0) * S_cur * u_out
        sign = 1 if shoulder[0] >= other[0] else -1
        x_out = min(float(frame_w - 1),
                    max(0.0, shoulder[0] + sign * float(coef) * S_cur))
        quads.append(np.array([top, shoulder, [shoulder[0], y_frame],
                               [x_out, y_frame], A], dtype=np.float64))
    return quads


def build_lower_neck_quad(params, head_corners, frame_w, frame_h):
    """Шея во весь низ кадра: плечи при калибровке не были видны.

    Верхнее ребро -- нижнее ребро текущего прямоугольника головы, нижнее
    лежит на нижней кромке кадра и разведено на запомненную долю ширины этого
    ребра (знак: плюс -- наружу).

    Возвращает контур (4, 2) или None.
    """
    if params is None or head_corners is None:
        return None
    head = np.asarray(head_corners, dtype=np.float64)
    if head.shape[0] < 4:
        return None
    low_l, low_r = ((head[2], head[3]) if head[2][0] <= head[3][0]
                    else (head[3], head[2]))
    W = float(np.linalg.norm(low_r - low_l))
    if W < 1e-6:
        return None
    y_frame = float(frame_h - 1)
    if y_frame <= max(low_l[1], low_r[1]) + 1.0:
        return None
    x_l = max(0.0, low_l[0] - params.get('out_left_coef', 0.0) * W)
    x_r = min(float(frame_w - 1), low_r[0] + params.get('out_right_coef', 0.0) * W)
    return np.array([low_l, low_r, [x_r, y_frame], [x_l, y_frame]],
                    dtype=np.float64)


def neck_sides(torso_quad, head_corners):
    """Две боковые стороны трапеции шеи: (верх, низ) для каждой стороны.

    Трапеция соединяет нижнее ребро головы с верхним ребром торса. Стороны
    перекрёстные: нижний угол головы head[3] (со стороны +e1) соединяется с
    TL, а head[2] -- с TR. Так задан обход в исходной трапеции, и так же
    спарены стороны шеи в build_upper_body_hull.

    Возвращает ((верх_TL, TL), (верх_TR, TR)) либо None.
    """
    if torso_quad is None or head_corners is None:
        return None
    head = np.asarray(head_corners, dtype=np.float64)
    if head.shape[0] < 4:
        return None
    TL = torso_vertex(torso_quad, 'TL')
    TR = torso_vertex(torso_quad, 'TR')
    if TL is None or TR is None:
        return None
    return (head[3], np.asarray(TL, dtype=np.float64)), \
           (head[2], np.asarray(TR, dtype=np.float64))


def build_neck_quad_from_torso_and_head(torso_quad, head_corners,
                                        neck_params=None):
    """Строит шею между верхом торса и низом головы.

    torso_quad   - контур торса; берутся вершины TL и TR.
    head_corners - прямоугольник головы [верхний правый, верхний левый,
                   нижний левый, нижний правый] (4, 2).
    neck_params  - профиль из калибровки {'tl_coefs': [...], 'tr_coefs': [...]}
                   либо None.

    Без профиля -- прежняя трапеция [TL, TR, head[2], head[3]].

    С профилем стороны перестают быть прямыми: на каждом уровне вершина
    сдвигается к осевой линии на свою долю (коэффициент 0..1 от полуширины
    трапеции). Доли меряются по маске при калибровке, поэтому фигура повторяет
    скошенное плечо вместо того, чтобы срезать его по прямой. Единица -- вершина
    осталась на трапеции; коэффициенты не бывают больше единицы, так что фигура
    может только сузиться и новых вылетов за маску не создаёт.

    Масштаб не нужен: доли берутся от самой трапеции, а она уже построена по
    откалиброванным голове и торсу.
    """
    sides = neck_sides(torso_quad, head_corners)
    if sides is None:
        return None
    (top_l, TL), (top_r, TR) = sides

    if not neck_params:
        return np.array([TL, TR, top_r, top_l], dtype=np.float64)

    tl_coefs = neck_params.get('tl_coefs')
    tr_coefs = neck_params.get('tr_coefs')
    if not tl_coefs or not tr_coefs or len(tl_coefs) != len(tr_coefs):
        return np.array([TL, TR, top_r, top_l], dtype=np.float64)

    n = len(tl_coefs)
    if n < 2:
        return np.array([TL, TR, top_r, top_l], dtype=np.float64)

    left_pts, right_pts = [], []
    for i in range(n):
        t = i / (n - 1.0)
        PL = top_l + t * (TL - top_l)      # точка на стороне через TL
        PR = top_r + t * (TR - top_r)      # точка на стороне через TR
        M = (PL + PR) / 2.0                # осевая линия шеи
        left_pts.append(M + float(tl_coefs[i]) * (PL - M))
        right_pts.append(M + float(tr_coefs[i]) * (PR - M))

    poly = np.array(left_pts[::-1] + right_pts, dtype=np.float64)
    if polygon_self_intersects(poly):
        return np.array([TL, TR, top_r, top_l], dtype=np.float64)
    return poly
