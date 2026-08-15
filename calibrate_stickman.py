"""Калибровка stickman-модели по 5-му кадру видео.

Пайплайн:
  1. Берём 5-й кадр видео.
  2. YOLO находит box человека, вырезаем кроп с запасом 10%.
  3. На кропе: InSPyReNet (маска), pose_landmarker_full (скелет),
     face_landmarker (подбородок, точка 152).
  4. Калибруем голову (с подбородком и продлением ушей) и торс по кропу.
  5. Визуализируем на полном кадре.

Установка:
    pip install transparent-background

Скачать модели (если нет):
    wget -O pose_landmarker_full.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
    wget -O face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

Запуск:
    pytКак скачать yolo26s.pthon calibrate_stickman.py
"""

import os
import cv2
import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from bench.calibration import calibrate_head, calibrate_torso, save_calibration_params
from bench.visualization import draw_pose_landmarks
from bench.tracking import build_neck_quad_from_torso_and_head
from bench import config


# --- Настройки ---
VIDEO_PATH = "example.mp4"
YOLO_MODEL_PATH = "yolo26s.pt"
POSE_MODEL_PATH = "pose_landmarker_full.task"
FACE_MODEL_PATH = "face_landmarker.task"
OUTPUT_IMAGE_PATH = "calibration_result.png"
CALIBRATION_PARAMS_PATH = "calibration_params.json"
SHOW_PREVIEW = True
MASK_OVERLAY_ALPHA = 0.5
FRAME_INDEX = 5          # номер кадра для калибровки (1-based)
YOLO_BBOX_PADDING = 0.10  # запас вокруг box-а: 10% от размера
FACE_CHIN_INDEX = 152     # индекс подбородка (chin tip) в face_landmarker


def load_inspyrenet():
    try:
        from transparent_background import Remover
    except ImportError:
        print("Установите transparent_background: pip install transparent-background")
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
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def create_face_landmarker(model_path):
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
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


def main():
    # Проверка моделей
    for path, hint in [
        (YOLO_MODEL_PATH, "YOLO (yolo26s.pt)"),
        (POSE_MODEL_PATH, "pose_landmarker_full.task"),
        (FACE_MODEL_PATH, "face_landmarker.task"),
    ]:
        if not os.path.exists(path):
            print(f"[!] Модель не найдена: {path} ({hint})")
            return

    # Загрузка 5-го кадра
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {VIDEO_PATH}")
        return
    # Пропускаем кадры до нужного (FRAME_INDEX=5 -> пропускаем 4)
    for _ in range(FRAME_INDEX - 1):
        cap.grab()
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        print(f"Не удалось прочитать кадр {FRAME_INDEX}")
        return

    frame_h, frame_w = frame_bgr.shape[:2]
    print(f"Кадр {FRAME_INDEX}: {frame_w}x{frame_h}")

    # YOLO -> box человека -> кроп
    print("Загрузка YOLO...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    bbox = get_person_bbox(yolo_model, frame_bgr)
    if bbox is None:
        print("YOLO не нашёл человека (фолбэк не делаем)")
        return
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x = YOLO_BBOX_PADDING * bw
    pad_y = YOLO_BBOX_PADDING * bh
    x1_c = max(0, int(x1 - pad_x))
    y1_c = max(0, int(y1 - pad_y))
    x2_c = min(frame_w, int(x2 + pad_x))
    y2_c = min(frame_h, int(y2 + pad_y))
    crop = frame_bgr[y1_c:y2_c, x1_c:x2_c]
    crop_h, crop_w = crop.shape[:2]
    print(f"YOLO box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], "
          f"кроп: {crop_w}x{crop_h}")

    # InSPyReNet на кадре
    remover = load_inspyrenet()
    if remover is None:
        return
    print("Сегментация полного кадра (InSPyReNet)...")
    mask_full = get_inspyrenet_mask(remover, frame_bgr)
    person_ratio = (mask_full > 0).sum() / mask_full.size * 100.0
    print(f"Маска полного кадра: человек занимает {person_ratio:.1f}%")

    # Pose на кропе
    print("Загрузка pose_landmarker_full...")
    pose_landmarker = create_pose_landmarker_full(POSE_MODEL_PATH)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
    pose_result = pose_landmarker.detect(mp_image)
    pose_landmarker.close()
    if not pose_result.pose_landmarks:
        print("Поза не обнаружена на кропе")
        return
    pose_landmarks = pose_result.pose_landmarks[0]
    print("Поза обнаружена")

    # Face на кропе -> подбородок
    print("Загрузка face_landmarker...")
    face_landmarker = create_face_landmarker(FACE_MODEL_PATH)
    face_result = face_landmarker.detect(mp_image)
    face_landmarker.close()
    chin_point = None
    if face_result.face_landmarks:
        face_lm = face_result.face_landmarks[0]
        if len(face_lm) > FACE_CHIN_INDEX:
            chin_lm = face_lm[FACE_CHIN_INDEX]
            # Было: chin_point = np.array([chin_lm.x * crop_w, chin_lm.y * crop_h], ...)
            # Стало: преобразуем из кропа в полный кадр
            chin_point = np.array([chin_lm.x * crop_w + x1_c,
                                   chin_lm.y * crop_h + y1_c], dtype=np.float64)
            print(f"Подбородок найден: ({chin_point[0]:.1f}, {chin_point[1]:.1f})")
        else:
            print("Подбородок: точка недоступна")
    else:
        print("Лицо не обнаружено (подбородок не используется)")

    # Калибровка (в координатах кропа, region=None)
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
        # НОВОЕ: привязка к бёдрам
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

    # Сохранение параметров калибровки в JSON
    save_calibration_params(CALIBRATION_PARAMS_PATH, head_result, torso_result,
                            video_path=VIDEO_PATH, frame_index=FRAME_INDEX)

    # Визуализация на полном кадре
    overlay = frame_bgr.copy()

    # Маска (полупрозрачная зелёная)
    mask_bool = mask_full > 0
    colored_mask = np.zeros_like(frame_bgr)
    colored_mask[mask_bool] = (0, 255, 0)
    blended = cv2.addWeighted(overlay, 1.0 - MASK_OVERLAY_ALPHA,
                              colored_mask, MASK_OVERLAY_ALPHA, 0)
    overlay[mask_bool] = blended[mask_bool]

    # YOLO box
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Голова
    if head_result is not None:
        corners_full = head_result['corners'].astype(np.int32)
        cv2.polylines(overlay, [corners_full], isClosed=True, color=(0, 200, 255), thickness=2)

    # Торс
    if torso_result is not None:
        quad_full = torso_result['quad'].astype(np.int32)
        cv2.polylines(overlay, [quad_full], isClosed=True, color=(255, 100, 0), thickness=2)

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

    cv2.imwrite(OUTPUT_IMAGE_PATH, overlay)
    print(f"\nРезультат сохранён: {OUTPUT_IMAGE_PATH}")

    if SHOW_PREVIEW:
        try:
            cv2.imshow("Calibration result", overlay)
            print("Нажмите любую клавишу для выхода...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("Предпросмотр недоступен (OpenCV без GUI). Результат сохранён в файл.")


if __name__ == "__main__":
    main()