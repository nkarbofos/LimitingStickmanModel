"""Калибровка stickman-модели по одному кадру видео.

Пайплайн:
  1. Берём кадр номер CALIB_FRAME_INDEX (по умолчанию 5-й).
  2. YOLO находит box человека, вырезаем кроп с запасом 10%.
  3. На кадре/кропе: InSPyReNet (эталонная маска), pose_landmarker_full
     (скелет), face_landmarker (подбородок, точка 152).
  4. Калибруем голову (с подбородком и продлением ушей) и торс по маске.
  5. Сохраняем нормализованные коэффициенты в JSON и рисуем результат.

Веса моделей докачиваются автоматически в models/ (см. bench/download.py).
Тяжёлые зависимости (ultralytics, transparent-background) ставятся экстрой:

    pip install -e ".[calib]"

Запуск:
    python calibrate_stickman.py
"""

import os

import cv2
import numpy as np
from PIL import Image

# ВАЖНО: transparent_background (torch) обязан импортироваться ДО mediapipe.
# При обратном порядке процесс падает с SIGSEGV на импорте
# transparent_background -- конфликт нативных библиотек torch и mediapipe.
try:
    from transparent_background import Remover as _Remover
except ImportError:
    _Remover = None

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from . import config
from .calibration import (calibrate_head, calibrate_torso,
                          calibrate_limb_widths, calibrate_neck,
                          calibrate_lower_neck, save_calibration_params)
from .download import ensure_models
from .stickman_model import (build_body_rects, build_thigh_quads,
                             build_arm_shoulder_tris)
from .tracking import build_neck_quad_from_torso_and_head
from .visualization import draw_pose_landmarks




# Группы прямоугольников тела: ключ -> (подпись, цвет из config)
_RECT_GROUPS = (
    ('arms',  'руки',    'CALIB_ARM_COLOR'),
    ('legs',  'ноги',    'CALIB_LEG_COLOR'),
    ('palms', 'ладони',  'CALIB_PALM_COLOR'),
    ('feet',  'ступни',  'CALIB_FOOT_COLOR'),
    ('thigh_quads', 'четыр. бёдер', 'CALIB_THIGH_QUAD_COLOR'),
    ('arm_tops_quad', 'четыр. ABCD', 'CALIB_ARM_TOPS_COLOR'),
    ('arm_tris', 'треуг. плеча', 'CALIB_ARM_TRI_COLOR'),
    ('joint_tris', 'треуг. суставов', 'CALIB_JOINT_TRI_COLOR'),
    ('forearm_squares', 'квадр. предплечья', 'CALIB_FOREARM_SQUARE_COLOR'),
)


def detect_face(face_landmarker, image_bgr, origin):
    """Подбородок и овал лица на изображении, в координатах полного кадра.

    origin -- смещение левого верхнего угла image_bgr в полном кадре.
    Возвращает (chin, oval, n_landmarks): точка 152, контур FACE_OVAL (N, 2)
    и число точек разметки (0 -- лицо не найдено). Подбородок и овал -- None,
    если лицо не найдено или разметка короче контура: без овала точки
    подбородка попросту нет.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, None, 0
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = face_landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None, None, 0
    lm = res.face_landmarks[0]
    need = max(max(config.FACE_OVAL), config.CALIB_FACE_CHIN_INDEX)
    if len(lm) <= need:
        return None, None, len(lm)
    h, w = image_bgr.shape[:2]
    oval = np.array([[lm[i].x * w + origin[0], lm[i].y * h + origin[1]]
                     for i in config.FACE_OVAL], dtype=np.float64)
    chin = lm[config.CALIB_FACE_CHIN_INDEX]
    chin_point = np.array([chin.x * w + origin[0], chin.y * h + origin[1]],
                          dtype=np.float64)
    return chin_point, oval, len(lm)


def format_face_oval(oval, n_landmarks):
    """Строка для печати: детектирован ли овал лица FACE_OVAL."""
    if n_landmarks == 0:
        return "НЕ детектирован (лицо не найдено)"
    if oval is None:
        return (f"НЕ детектирован (в разметке {n_landmarks} точек, "
                f"нужно минимум {max(config.FACE_OVAL) + 1})")
    w = oval[:, 0].max() - oval[:, 0].min()
    h = oval[:, 1].max() - oval[:, 1].min()
    return (f"детектирован, {len(oval)} точек, "
            f"габарит {w:.0f}x{h:.0f} px")


def head_crop_box(corners, frame_w, frame_h, expand):
    """Прямоугольник кропа вокруг квадрата головы, увеличенный в expand раз.

    Квадрат головы повёрнут (стороны вдоль линии ушей), а кроп детектору
    нужен по осям кадра, поэтому берётся описанный прямоугольник и
    растягивается от своего центра.

    Возвращает (x0, y0, x1, y1) в координатах полного кадра либо None.
    """
    pts = np.asarray(corners, dtype=np.float64)
    if pts.shape[0] < 3:
        return None
    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) / 2.0 * expand, (y1 - y0) / 2.0 * expand
    bx0 = max(0, int(round(cx - hw)))
    by0 = max(0, int(round(cy - hh)))
    bx1 = min(frame_w, int(round(cx + hw)))
    by1 = min(frame_h, int(round(cy + hh)))
    if bx1 - bx0 < 2 or by1 - by0 < 2:
        return None
    return bx0, by0, bx1, by1


# Подписи точек, для которых калибруется ширина конечности.
_WIDTH_POINT_LABELS = (
    (11, 'плечо L'), (13, 'локоть L'), (15, 'запястье L'),
    (12, 'плечо R'), (14, 'локоть R'), (16, 'запястье R'),
    (25, 'колено L'), (27, 'лодыжка L'),
    (26, 'колено R'), (28, 'лодыжка R'),
)


def print_neck_profile(neck_result):
    """Печатает, насколько трапеция шеи поджалась к маске на каждом уровне."""
    if not neck_result:
        print("\nШея: профиль не построен (нет головы или торса) -- трапеция")
        return
    tl, tr = neck_result['tl_coefs'], neck_result['tr_coefs']
    n = len(tl)
    print("\nШея: доля полуширины трапеции по уровням "
          "(1.00 -- маска доходит до трапеции)")
    print("  уровень:  " + " ".join("%5.2f" % (i / (n - 1.0)) for i in range(n)))
    print("  сторона TL:" + " ".join("%5.2f" % v for v in tl))
    print("  сторона TR:" + " ".join("%5.2f" % v for v in tr))
    trimmed = sum(1 for v in tl + tr if v < 0.99)
    print("  поджато %d из %d вершин, самая узкая %.2f"
          % (trimmed, 2 * n, min(tl + tr)))


def print_limb_widths(limb_widths):
    """Печатает калиброванные коэффициенты ширины по точкам."""
    if not limb_widths:
        print("\nШирина конечностей: не откалибрована "
              "(точки не видны) -- прежние прямоугольники")
        return
    print("\nШирина конечностей (доли плеч для рук, таза для ног):")
    for idx, label in _WIDTH_POINT_LABELS:
        k = limb_widths.get(idx)
        if k is None:
            print(f"  {idx:>2} {label:<10} --")
        else:
            print(f"  {idx:>2} {label:<10} {k:.4f}")


def draw_body_rects(overlay, body_rects):
    """Рисует прямоугольники рук, ног, ладоней и ступней.

    body_rects -- результат stickman_model.build_body_rects (или None).
    Возвращает общее число нарисованных прямоугольников.
    """
    if body_rects is None:
        print("\nПрямоугольники тела: плечи не видны -- ничего не построено")
        return 0

    total = 0
    parts = []
    for key, label, color_name in _RECT_GROUPS:
        rects = body_rects.get(key, [])
        color = getattr(config, color_name)
        for rect in rects:
            cv2.polylines(overlay, [np.asarray(rect, dtype=np.int32)],
                          isClosed=True, color=color,
                          thickness=config.TRACKED_THICKNESS)
        total += len(rects)
        parts.append(f"{label} {len(rects)}")

    print(f"\nПрямоугольники тела: {total} шт ({', '.join(parts)}), "
          f"S={body_rects['S']:.1f} px")
    return total


def load_inspyrenet():
    if _Remover is None:
        print("[!] Не установлен transparent-background (InSPyReNet).")
        print("    Установите: pip install -e \".[calib]\"")
        return None
    print("Загрузка InSPyReNet...")
    return _Remover(mode='base', jit=False)


def get_inspyrenet_mask(remover, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    result_rgba = remover.process(pil_image, type='rgba')
    result_np = np.array(result_rgba)
    alpha = result_np[:, :, 3]
    mask = (alpha > 127).astype(np.uint8) * 255
    return mask


def create_pose_landmarker_full(model_path):
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def create_face_landmarker(model_path):
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def get_person_bbox(yolo_model, frame_bgr):
    """Возвращает [x1, y1, x2, y2] самого большого человека или None."""
    results = yolo_model.predict(frame_bgr, classes=[0], conf=0.3, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    boxes = r.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(areas.argmax())]


def main(video_path=None, frame_index=None, params_path=None, image_path=None,
         show_preview=False, auto_download=True):
    """Калибрует модель по кадру видео и сохраняет параметры в JSON.

    Все аргументы необязательны: None означает "взять значение из bench.config".
    Возвращает 0 при успехе и 1 при ошибке (код возврата процесса).
    """
    video_path = video_path or config.VIDEO_PATH
    frame_index = frame_index if frame_index is not None else config.CALIB_FRAME_INDEX
    params_path = params_path or config.CALIBRATION_PARAMS_OUTPUT_PATH
    image_path = image_path or config.CALIBRATION_RESULT_IMAGE_PATH

    # Веса моделей (докачиваются при отсутствии)
    try:
        yolo_path, pose_path, face_path = ensure_models(
            [config.CALIB_YOLO_MODEL_NAME,
             config.CALIB_POSE_MODEL_NAME,
             config.CALIB_FACE_MODEL_NAME],
            auto_download=auto_download)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[!] {exc}")
        return 1

    # Загрузка нужного кадра
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Не удалось открыть видео: {video_path}")
        return 1
    # frame_index -- 1-based, поэтому пропускаем frame_index - 1 кадров
    for _ in range(frame_index - 1):
        cap.grab()
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        print(f"[!] Не удалось прочитать кадр {frame_index} из {video_path}")
        return 1

    frame_h, frame_w = frame_bgr.shape[:2]
    print(f"Видео: {video_path}")
    print(f"Кадр {frame_index}: {frame_w}x{frame_h}")

    # YOLO -> box человека -> кроп
    print("Загрузка YOLO...")
    from .models import create_yolo
    yolo_model = create_yolo(yolo_path)
    bbox = get_person_bbox(yolo_model, frame_bgr)
    if bbox is None:
        print("[!] YOLO не нашёл человека (фолбэк не делаем)")
        return 1
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x = config.CALIB_YOLO_BBOX_PADDING * bw
    pad_y = config.CALIB_YOLO_BBOX_PADDING * bh
    x1_c = max(0, int(x1 - pad_x))
    y1_c = max(0, int(y1 - pad_y))
    x2_c = min(frame_w, int(x2 + pad_x))
    y2_c = min(frame_h, int(y2 + pad_y))
    crop = frame_bgr[y1_c:y2_c, x1_c:x2_c]
    crop_h, crop_w = crop.shape[:2]
    print(f"YOLO box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], "
          f"кроп: {crop_w}x{crop_h}")

    # InSPyReNet на полном кадре -> эталонная маска
    remover = load_inspyrenet()
    if remover is None:
        return 1
    print("Сегментация полного кадра (InSPyReNet)...")
    mask_full = get_inspyrenet_mask(remover, frame_bgr)
    person_ratio = (mask_full > 0).sum() / mask_full.size * 100.0
    print(f"Маска полного кадра: человек занимает {person_ratio:.1f}%")

    # Pose на кропе
    print("Загрузка pose_landmarker_full...")
    pose_landmarker = create_pose_landmarker_full(pose_path)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
    pose_result = pose_landmarker.detect(mp_image)
    pose_landmarker.close()
    if not pose_result.pose_landmarks:
        print("[!] Поза не обнаружена на кропе")
        return 1
    pose_landmarks = pose_result.pose_landmarks[0]
    print("Поза обнаружена")

    # Face на кропе -> подбородок. Детектор закрываем не здесь: если лицо на
    # кропе человека не нашлось, он ещё понадобится для повторной попытки на
    # тесном кропе головы (после калибровки головы).
    print("Загрузка face_landmarker...")
    face_landmarker = create_face_landmarker(face_path)
    chin_point, face_oval, n_face_lm = detect_face(face_landmarker, crop,
                                                   (x1_c, y1_c))
    print(f"Овал лица (FACE_OVAL): {format_face_oval(face_oval, n_face_lm)}")
    if chin_point is not None:
        print(f"Подбородок найден: ({chin_point[0]:.1f}, {chin_point[1]:.1f})")
    else:
        print("Лицо на кропе человека не обнаружено")

    # Калибровка (точки позы -- в координатах кропа, маска -- полного кадра)
    region = (x1_c, y1_c, crop_w, crop_h)
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ КАЛИБРОВКИ")
    print("=" * 50)

    # Ширина конечностей замеряется по маске ПЕРВОЙ: от неё зависят фигуры
    # рук, а те служат барьером при калибровке торса.
    limb_widths, limb_grow = calibrate_limb_widths(mask_full, pose_landmarks, region,
                                        frame_w, frame_h)
    print_limb_widths(limb_widths)

    # Фигуры частей тела считаем ДО калибровки торса: руки и ладони служат
    # барьером для вытягивания нижних вершин торса (BL, BR).
    body_rects = build_body_rects(pose_landmarks, region, frame_w, frame_h,
                                  limb_widths=limb_widths, limb_grow=limb_grow)
    barrier_rects = None
    limb_rects = None
    if body_rects is not None:
        barrier_rects = body_rects['arms'] + body_rects['palms']
        # Фигуры конечностей: в них не должна упираться линия живота.
        limb_rects = body_rects['arms'] + body_rects['legs']

    torso_result = calibrate_torso(mask_full, pose_landmarks, region,
                                   frame_w, frame_h,
                                   barrier_rects=barrier_rects,
                                   limb_rects=limb_rects)

    # Четырёхугольники верха ног: строятся только после торса -- им нужны
    # его нижние вершины BL и BR.
    if body_rects is not None:
        body_rects['thigh_quads'] = build_thigh_quads(
            pose_landmarks, region, frame_w, frame_h,
            torso_result['quad'] if torso_result is not None else None,
            limb_widths=limb_widths)
        # Треугольники плеча: тоже нужны вершины торса
        body_rects['arm_tris'] = build_arm_shoulder_tris(
            pose_landmarks, region, frame_w, frame_h,
            torso_result['quad'] if torso_result is not None else None,
            limb_widths=limb_widths, limb_grow=limb_grow)

    # Голова считается ПОСЛЕ торса: при отсутствии подбородка нижняя граница
    # головы подбирается по тому, как фигура (голова + шея) ложится на маску,
    # а шея строится из верхнего ребра торса.
    torso_quad = torso_result['quad'] if torso_result is not None else None
    head_result = calibrate_head(mask_full, pose_landmarks, region,
                                 frame_w, frame_h, chin_point=chin_point,
                                 torso_quad=torso_quad)

    # Лицо не нашлось на кропе всего человека -- голова там занимает малую
    # часть картинки. Пробуем ещё раз на тесном кропе вокруг уже построенного
    # квадрата головы, увеличенном на CALIB_HEAD_CROP_EXPAND. Если овал лица
    # там детектится, подбородок известен и голова пересчитывается по нему.
    if chin_point is None and head_result is not None:
        box = head_crop_box(head_result['corners'], frame_w, frame_h,
                            config.CALIB_HEAD_CROP_EXPAND)
        if box is None:
            print("Повторный поиск лица: кроп головы вырожден")
        else:
            bx0, by0, bx1, by1 = box
            print(f"Повторный поиск лица на кропе головы "
                  f"{bx1 - bx0}x{by1 - by0} (+{(config.CALIB_HEAD_CROP_EXPAND - 1) * 100:.0f}%)...")
            chin_point, face_oval, n_face_lm = detect_face(
                face_landmarker, frame_bgr[by0:by1, bx0:bx1], (bx0, by0))
            print(f"  овал лица (FACE_OVAL): "
                  f"{format_face_oval(face_oval, n_face_lm)}")
            if chin_point is None:
                print("  лицо не обнаружено -- нижняя граница остаётся по IoU шеи")
            else:
                print(f"  подбородок найден: ({chin_point[0]:.1f}, "
                      f"{chin_point[1]:.1f}) -- пересчитываем голову")
                head_result = calibrate_head(mask_full, pose_landmarks, region,
                                             frame_w, frame_h,
                                             chin_point=chin_point,
                                             torso_quad=torso_quad)

    face_landmarker.close()

    # Плечи не видны -- торса нет, и всё, что ниже головы, считается шеей.
    lower_neck_result = None
    if torso_result is None and head_result is not None:
        lower_neck_result = calibrate_lower_neck(
            mask_full, head_result['corners'], frame_w, frame_h)

    # Профиль шеи считается последним: ему нужны и готовая голова, и торс.
    neck_result = calibrate_neck(
        mask_full,
        head_result['corners'] if head_result is not None else None,
        torso_result['quad'] if torso_result is not None else None)
    print_neck_profile(neck_result)

    if limb_grow:
        print("\nРаздвижение фигур по маске (доли масштаба пары, на сторону):")
        for key in sorted(limb_grow):
            print("  %-7s %.4f" % (key, limb_grow[key]))
    elif config.CALIBRATION_LIMB_GROW_ENABLED:
        print("\nРаздвижение фигур по маске: ни одна пара не раздвинулась")
    else:
        print("\nРаздвижение фигур по маске: выключено флагом "
              "CALIBRATION_LIMB_GROW_ENABLED")

    extended = [k for k, v in (body_rects or {}).get('arm_extend', {}).items() if v]
    if extended:
        print("\nПродление плечевых костей за кадр (предплечья не видно, "
              "считается на каждом кадре): "
              + ", ".join(sorted(extended)))

    if head_result is not None:
        print("\nГолова:")
        print(f"  Ширина (продление ушей): {head_result['width']:.1f} px "
              f"(удлинение отрезка ушей {head_result['ear_extend']:.1f} px "
              f"в каждую сторону)")
        print(f"  От носа вправо/влево:    {head_result['right_dist']:.1f} / "
              f"{head_result['left_dist']:.1f} px")
        print(f"  Высота:                  {head_result['height']:.1f} px")
        print(f"  len(XN) вверх:           {head_result['len_XN']:.1f} px")
        src = head_result['down_source']
        src_ru = {'chin': 'по подбородку',
                  'neck_iou': 'подобран по шее',
                  'symmetric': 'заглушка len_XN'}.get(src.split('+')[0], src)
        if src.endswith('+shoulders'):
            src_ru += ', урезан линией плеч'
        print(f"  down_dist:               {head_result['down_dist']:.1f} px ({src_ru})")
        fit = head_result.get('neck_fit')
        if fit is not None:
            d_best, iou_best, curve = fit
            at_bound = (d_best <= curve[0][0] + 1e-6
                        or d_best >= curve[-1][0] - 1e-6)
            print(f"    IoU(голова+шея) = {iou_best:.3f}, кандидатов {len(curve)} "
                  f"в диапазоне {curve[0][0]:.1f}..{curve[-1][0]:.1f} px; "
                  f"заглушка дала бы {head_result['len_XN']:.1f} px")
            if at_bound:
                print("    [!] оптимум на границе диапазона -- подбор вырожден")
            if config.CALIBRATION_NECK_FIT_DEBUG:
                print("    кривая IoU(d): "
                      + " ".join(f"{d:.0f}:{v:.3f}" for d, v in curve))
        print(f"  k_hw: {head_result['k_hw']:.3f}, k_hh: {head_result['k_hh']:.3f}")
        print(f"  S (ширина плеч): {head_result['S']:.1f} px")
    else:
        print("\nГолова: пропущена (уши или нос не видны)")

    if torso_result is not None:
        print("\nТорс:")
        print(f"  y_shoulders: {torso_result['y_shoulders']:.1f} px")
        print(f"  y_hips:      {torso_result['y_hips']:.1f} px")
        print(f"  y_bottom:    {torso_result['y_bottom']:.1f} px")
        if torso_result['legs_visible']:
            _bottom_how = 'BL/BR по отрезку бёдер'
        elif torso_result.get('torso_rect_below_frame'):
            _bottom_how = 'прямоугольник от отрезка плеч за низ кадра'
        else:
            _bottom_how = 'BL/BR по свисающей одежде'
        print(f"  ноги видны:  {torso_result['legs_visible']} ({_bottom_how})")
        print(f"  S (ширина плеч): {torso_result['S']:.1f} px")
        print(f"  Контур торса ({len(torso_result['quad'])} вершин):")
        print(f"    TL = {torso_result['TL'].astype(int).tolist()}")
        print(f"    TR = {torso_result['TR'].astype(int).tolist()}")
        print(f"    BR = {torso_result['BR'].astype(int).tolist()}")
        print(f"    BL = {torso_result['BL'].astype(int).tolist()}")
        if torso_result.get('has_belly'):
            print(f"  Линия живота (вершины шестиугольника):")
            print(f"    ML = {torso_result['ML'].astype(int).tolist()}")
            print(f"    MR = {torso_result['MR'].astype(int).tolist()}")
            print(f"    belly_depth_coef:     {torso_result['belly_depth_coef']:.4f}")
            print(f"    belly_ext_left_coef:  {torso_result['belly_ext_left_coef']:.4f}")
            print(f"    belly_ext_right_coef: {torso_result['belly_ext_right_coef']:.4f}")
        else:
            print(f"  Линия живота: не построена "
                  f"({torso_result.get('belly_reason', 'причина не указана')})")
        for idx in sorted(torso_result.get('side_bottom_quads', {})):
            q = torso_result['side_bottom_quads'][idx]
            print(f"  Пятиугольник плечи-низ (рука точки {idx} не видна):")
            print("    " + " -> ".join(v.astype(int).tolist().__repr__()
                                       for v in q))
            print(f"    развод по низу / вынос точки A (доли ширины плеч): "
                  f"{torso_result['side_bottom_coefs'][idx]:.4f} / "
                  f"{torso_result['side_bottom_a_coefs'][idx]:.4f}")
        def _spread(v):
            return '--' if v is None else f'{v:.1f}'
        print(f"  Верхние вершины: луч "
              f"{config.CALIBRATION_SHOULDER_RAY_DEG:.0f} град. к отрезку плеч "
              f"(TR от точки 11, TL от точки 12)")
        print(f"    разворот рук (угол при плече, справочно): "
              f"11-13 {_spread(torso_result.get('arm_spread_11'))}, "
              f"12-14 {_spread(torso_result.get('arm_spread_12'))}")
        print(f"  Параметры для отслеживания (нормализованные):")
        print(f"    Верхний отрезок TL-TR (доли ширины плеч):")
        print(f"      перпендикуляр от середины 11-12: "
              f"{torso_result['top_perp_coef']:.4f}")
        print(f"      части отрезка (к TL / к TR):     "
              f"{torso_result['top_left_len_coef']:.4f} / "
              f"{torso_result['top_right_len_coef']:.4f}")
        print(f"    ext_left_coef:  {torso_result['ext_left_coef']:.4f}")
        print(f"    ext_right_coef: {torso_result['ext_right_coef']:.4f}")
        print(f"    Привязка к плечам (фолбэк):")
        print(f"      dx_left_coef:   {torso_result['dx_left_coef']:.4f}, "
              f"dy_left_coef:  {torso_result['dy_left_coef']:.4f}")
        print(f"      dx_right_coef:  {torso_result['dx_right_coef']:.4f}, "
              f"dy_right_coef: {torso_result['dy_right_coef']:.4f}")
        print(f"    Привязка к бёдрам (has_hip_ref={torso_result['has_hip_ref']}):")
        if torso_result['has_hip_ref']:
            print(f"      dx_left_coef_hip:   {torso_result['dx_left_coef_hip']:.4f}, "
                  f"dy_left_coef_hip:  {torso_result['dy_left_coef_hip']:.4f}")
            print(f"      dx_right_coef_hip:  {torso_result['dx_right_coef_hip']:.4f}, "
                  f"dy_right_coef_hip: {torso_result['dy_right_coef_hip']:.4f}")
        else:
            print(f"      (бёдра не видны при калибровке - используется фолбэк на плечи)")
    else:
        print("\nТорс: пропущен (плечи не видны)")
        if lower_neck_result is not None:
            q = lower_neck_result['quad']
            print("\nШея во весь низ кадра (плечи не видны, торса нет):")
            print(f"  верх = {q[0].astype(int).tolist()} .. {q[1].astype(int).tolist()}")
            print(f"  низ  = {q[3].astype(int).tolist()} .. {q[2].astype(int).tolist()}")
            print(f"  развод наружу (доли ширины низа головы): "
                  f"{lower_neck_result['out_left_coef']:.4f} / "
                  f"{lower_neck_result['out_right_coef']:.4f}")
        elif head_result is not None:
            print("Шея во весь низ кадра: не построена "
                  "(маска не идёт ниже головы)")

    if head_result is None and torso_result is None:
        print("\n[!] Ни голова, ни торс не откалиброваны -- нечего сохранять.")
        return 1

    # Сохранение параметров калибровки в JSON
    os.makedirs(os.path.dirname(os.path.abspath(params_path)) or ".", exist_ok=True)
    save_calibration_params(params_path, head_result, torso_result,
                            limb_widths=limb_widths, neck_result=neck_result,
                            video_path=video_path, frame_index=frame_index,
                            lower_neck_result=lower_neck_result,
                            limb_grow=limb_grow)

    # Визуализация на полном кадре
    overlay = frame_bgr.copy()

    # Маска (полупрозрачная зелёная)
    mask_bool = mask_full > 0
    colored_mask = np.zeros_like(frame_bgr)
    colored_mask[mask_bool] = (0, 255, 0)
    blended = cv2.addWeighted(overlay, 1.0 - config.CALIB_MASK_OVERLAY_ALPHA,
                              colored_mask, config.CALIB_MASK_OVERLAY_ALPHA, 0)
    overlay[mask_bool] = blended[mask_bool]

    # YOLO box
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Голова
    if head_result is not None:
        corners_full = head_result['corners'].astype(np.int32)
        cv2.polylines(overlay, [corners_full], isClosed=True,
                      color=(0, 200, 255), thickness=2)

    # Торс
    if torso_result is not None:
        quad_full = torso_result['quad'].astype(np.int32)
        cv2.polylines(overlay, [quad_full], isClosed=True,
                      color=(255, 100, 0), thickness=2)

    # Фигуры "плечи-низ" (половинный кадр без бёдер): полная и половинные
    if torso_result is not None:
        for quad in torso_result.get('shoulders_bottom_quads', []):
            cv2.polylines(overlay, [np.asarray(quad).astype(np.int32)],
                          isClosed=True,
                          color=config.CALIB_SHOULDERS_BOTTOM_COLOR,
                          thickness=2)

    # Шея во весь низ кадра (плечи не видны)
    if lower_neck_result is not None:
        cv2.polylines(overlay, [lower_neck_result['quad'].astype(np.int32)],
                      isClosed=True, color=config.CALIB_SHOULDERS_BOTTOM_COLOR,
                      thickness=2)

    # Линия живота (хорда ML-MR внутри шестиугольника: сами ML и MR уже
    # обведены контуром торса, а сама линия -- нет).
    if torso_result is not None and torso_result.get('has_belly'):
        cv2.line(overlay,
                 tuple(torso_result['ML'].astype(int).tolist()),
                 tuple(torso_result['MR'].astype(int).tolist()),
                 color=config.CALIB_BELLY_COLOR, thickness=2)

    # Шея (из контура торса и прямоугольника головы, поджатая по профилю).
    # Рисуется всегда, как голова и торс выше: DRAW_TRACKED_NECK управляет
    # отрисовкой поверх маски при отслеживании, к картинке калибровки
    # отношения не имеет.
    if head_result is not None and torso_result is not None:
        neck_quad = build_neck_quad_from_torso_and_head(
            torso_result['quad'], head_result['corners'], neck_result)
        if neck_quad is not None:
            cv2.polylines(overlay, [neck_quad.astype(np.int32)],
                          isClosed=True,
                          color=config.TRACKED_NECK_COLOR,
                          thickness=config.TRACKED_THICKNESS)

    # Руки, ноги, ладони, ступни (прямоугольники частей тела)
    if config.DRAW_CALIB_LIMBS:
        draw_body_rects(overlay, body_rects)

    # Скелет (поверх всего)
    overlay = draw_pose_landmarks(overlay, pose_result.pose_landmarks, region)

    os.makedirs(os.path.dirname(os.path.abspath(image_path)) or ".", exist_ok=True)
    cv2.imwrite(image_path, overlay)
    print(f"Результат сохранён: {image_path}")

    if show_preview:
        try:
            cv2.imshow("Calibration result", overlay)
            print("Нажмите любую клавишу для выхода...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("Предпросмотр недоступен (OpenCV без GUI). Результат сохранён в файл.")

    return 0
