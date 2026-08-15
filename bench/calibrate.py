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

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from . import config
from .calibration import calibrate_head, calibrate_torso, save_calibration_params
from .download import ensure_models
from .tracking import build_neck_quad_from_torso_and_head
from .visualization import draw_pose_landmarks


def load_inspyrenet():
    try:
        from transparent_background import Remover
    except ImportError:
        print("[!] Не установлен transparent-background (InSPyReNet).")
        print("    Установите: pip install -e \".[calib]\"")
        return None
    print("Загрузка InSPyReNet...")
    return Remover(mode='base', jit=False)


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

    # Face на кропе -> подбородок
    print("Загрузка face_landmarker...")
    face_landmarker = create_face_landmarker(face_path)
    face_result = face_landmarker.detect(mp_image)
    face_landmarker.close()
    chin_point = None
    if face_result.face_landmarks:
        face_lm = face_result.face_landmarks[0]
        if len(face_lm) > config.CALIB_FACE_CHIN_INDEX:
            chin_lm = face_lm[config.CALIB_FACE_CHIN_INDEX]
            # Координаты кропа -> координаты полного кадра
            chin_point = np.array([chin_lm.x * crop_w + x1_c,
                                   chin_lm.y * crop_h + y1_c], dtype=np.float64)
            print(f"Подбородок найден: ({chin_point[0]:.1f}, {chin_point[1]:.1f})")
        else:
            print("Подбородок: точка недоступна")
    else:
        print("Лицо не обнаружено (подбородок не используется)")

    # Калибровка (точки позы -- в координатах кропа, маска -- полного кадра)
    region = (x1_c, y1_c, crop_w, crop_h)
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ КАЛИБРОВКИ")
    print("=" * 50)

    head_result = calibrate_head(mask_full, pose_landmarks, region,
                                 frame_w, frame_h, chin_point=chin_point)
    if head_result is not None:
        print("\nГолова:")
        print(f"  Ширина (продление ушей): {head_result['width']:.1f} px")
        print(f"  Высота:                  {head_result['height']:.1f} px")
        print(f"  len(XN) вверх:           {head_result['len_XN']:.1f} px")
        print(f"  down_dist (до подбородка): {head_result['down_dist']:.1f} px")
        print(f"  k_hw: {head_result['k_hw']:.3f}, k_hh: {head_result['k_hh']:.3f}")
        print(f"  S (ширина плеч): {head_result['S']:.1f} px")
    else:
        print("\nГолова: пропущена (уши или нос не видны)")

    torso_result = calibrate_torso(mask_full, pose_landmarks, region, frame_w, frame_h)
    if torso_result is not None:
        print("\nТорс:")
        print(f"  y_shoulders: {torso_result['y_shoulders']:.1f} px")
        print(f"  y_hips:      {torso_result['y_hips']:.1f} px")
        print(f"  y_bottom:    {torso_result['y_bottom']:.1f} px")
        print(f"  S (ширина плеч): {torso_result['S']:.1f} px")
        print(f"  Четырёхугольник:")
        print(f"    TL = {torso_result['TL'].astype(int).tolist()}")
        print(f"    TR = {torso_result['TR'].astype(int).tolist()}")
        print(f"    BR = {torso_result['BR'].astype(int).tolist()}")
        print(f"    BL = {torso_result['BL'].astype(int).tolist()}")
        print(f"  Параметры для отслеживания (нормализованные):")
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

    if head_result is None and torso_result is None:
        print("\n[!] Ни голова, ни торс не откалиброваны -- нечего сохранять.")
        return 1

    # Сохранение параметров калибровки в JSON
    os.makedirs(os.path.dirname(os.path.abspath(params_path)) or ".", exist_ok=True)
    save_calibration_params(params_path, head_result, torso_result,
                            video_path=video_path, frame_index=frame_index)

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

    # Шея (из четырёхугольника торса и прямоугольника головы)
    if config.DRAW_TRACKED_NECK and head_result is not None and torso_result is not None:
        neck_quad = build_neck_quad_from_torso_and_head(
            torso_result['quad'], head_result['corners'])
        if neck_quad is not None:
            cv2.polylines(overlay, [neck_quad.astype(np.int32)],
                          isClosed=True,
                          color=config.TRACKED_NECK_COLOR,
                          thickness=config.TRACKED_THICKNESS)

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
