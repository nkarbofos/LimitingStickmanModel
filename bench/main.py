"""Основной цикл бенчмарка."""

import os
import time
import statistics
import cv2
import numpy as np

import mediapipe as mp

from . import config
from .models import (create_yolo, create_segmenter, create_pose_landmarker,
                     create_face_landmarker)
from .yolo_crop import get_person_bbox, get_region
from .head_constraint import (compute_head_ellipse_from_pose,
                              apply_head_ellipse_constraint,
                              draw_head_ellipse_arc)
from .visualization import (category_mask_to_colored, confidence_masks_to_heatmap,
                            draw_pose_landmarks, draw_face_oval, compute_face_oval_size, 
                            fill_poly_with_alpha)
from .stats import print_latency_stats
from .download import ensure_models
from .stickman_model import (build_stickman_mask, overlay_stickman,
                             build_body_rects)
from .calibration import load_calibration_params
from .tracking import (build_head_rect_from_params, build_torso_quad_from_params,
                       build_neck_quad_from_torso_and_head,
                       build_shoulders_bottom_quads, build_lower_neck_quad)
from .stickman_model import LEFT_SHOULDER, RIGHT_SHOULDER, _get_point_px
from .calibration import LEFT_HIP, RIGHT_HIP


def enhance_contrast_clahe(frame_bgr, clip_limit=2.0, grid_size=8):
    """Повышает контрастность через CLAHE в канале L (LAB)."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(grid_size, grid_size))
    l_eq = clahe.apply(l)
    
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def main(video_path=None, output_path=None, calibration_params_path=None,
         max_frames=None, show_preview=None, use_yolo_crop=None,
         save_output_video=None, enable_segmentation=None, auto_download=True):
    """Прогоняет пайплайн по видео и печатает статистику скорости.

    Все аргументы необязательны: None означает "взять значение из bench.config".
    """
    video_path = video_path or config.VIDEO_PATH
    output_path = output_path or config.OUTPUT_VIDEO_PATH
    if calibration_params_path is None:
        calibration_params_path = config.default_calibration_params_path()
    if max_frames is None:
        max_frames = config.MAX_FRAMES
    if show_preview is None:
        show_preview = config.SHOW_PREVIEW
    if use_yolo_crop is None:
        use_yolo_crop = config.USE_YOLO_CROP
    if save_output_video is None:
        save_output_video = config.SAVE_OUTPUT_VIDEO
    if enable_segmentation is None:
        enable_segmentation = config.ENABLE_SEGMENTATION

    if not (enable_segmentation or config.ENABLE_POSE or config.ENABLE_FACE):
        print("Все MediaPipe-бенчмарки выключены.")
        return

    # Недостающие веса докачиваются автоматически в models/
    required = []
    if enable_segmentation:
        required.append(config.SEG_MODEL_NAME)
    if config.ENABLE_POSE:
        required.append(config.POSE_MODEL_NAME)
    if config.ENABLE_FACE:
        required.append(config.FACE_MODEL_NAME)
    if use_yolo_crop:
        required.append(config.YOLO_MODEL_NAME)
    try:
        ensure_models(required, auto_download=auto_download)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[!] {exc}")
        return

    yolo_model = create_yolo() if use_yolo_crop else None
    segmenter = create_segmenter() if enable_segmentation else None
    pose_landmarker = create_pose_landmarker() if config.ENABLE_POSE else None
    face_landmarker = create_face_landmarker() if config.ENABLE_FACE else None

    loaded = []
    if yolo_model: loaded.append("YOLO (crop)")
    if segmenter: loaded.append("SelfieMulticlass")
    if pose_landmarker: loaded.append("PoseLandmarker-Lite")
    if face_landmarker: loaded.append("FaceLandmarker")
    print(f"Загружены модели: {', '.join(loaded)}")
    print(f"USE_YOLO_CROP={use_yolo_crop}, "
        f"HEAD_CONSTRAINT={config.ENABLE_HEAD_CONSTRAINT} "
        f"(u_coef={config.HEAD_ELLIPSE_RADIUS_U_COEF}, "
        f"n_coef={config.HEAD_ELLIPSE_RADIUS_N_COEF})\n")

    # Загрузка параметров калибровки для отслеживания модели
    calib_params = None
    if config.ENABLE_STICKMAN_TRACKING:
        try:
            calib_params = load_calibration_params(calibration_params_path)
        except FileNotFoundError:
            print(f"[!] Файл калибровки не найден: {calibration_params_path}")
            print("    Запустите calibrate_stickman.py для создания параметров.")
            calib_params = None

    # Ширина конечностей по точкам. У старых калибровок секции нет -- тогда
    # конечности строятся прежними прямоугольниками постоянной ширины.
    calib_limb_widths = calib_params.get('limbs') if calib_params else None
    # Профиль шеи. У старых калибровок секции нет -- шея строится трапецией.
    calib_neck = calib_params.get('neck') if calib_params else None

    n_mediapipe = sum(x is not None for x in (segmenter, pose_landmarker, face_landmarker))
    n_active = n_mediapipe + (1 if use_yolo_crop else 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return
    print(f"Видео-вход: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Видео: {width}x{height}, {fps:.1f} FPS, {total_frames} кадров\n")

    out = None
    if save_output_video:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"[!] Не удалось создать видео-файл: {output_path}")
            out = None

    yolo_lat, seg_lat, pose_lat, face_lat, total_lat = [], [], [], [], []
    face_sizes = []
    frame_idx = processed = skipped = 0
    no_person_count = 0

    frame_processing_times = []

    print(f"{'Frame':>6} | {'YOLO':>6} | {'Seg':>6} | {'Pose':>6} | {'Face':>6} | "
          f"{'Models':>7} | {'Frame':>7} | crop")
    print("-" * 72)

    while True:
        frame_start_time = time.perf_counter()

        ret, frame_bgr = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        # new
        # frame_bgr = enhance_contrast_clahe(frame_bgr, clip_limit=2.0, grid_size=8)

        timestamp_ms = int(frame_idx * 1000 / fps)

        # --- Шаг 0: YOLO -> bbox -> кроп ---
        region = None
        yolo_ms = 0.0
        if yolo_model is not None:
            t0 = time.perf_counter()
            bbox = get_person_bbox(frame_bgr, yolo_model)
            yolo_ms = (time.perf_counter() - t0) * 1000.0
            region = get_region(bbox, width, height, config.CROP_PADDING)
            if region is None:
                no_person_count += 1

        if region is not None:
            ox, oy, rw, rh = region
            mp_input_bgr = frame_bgr[oy:oy + rh, ox:ox + rw]
        else:
            mp_input_bgr = frame_bgr

        frame_rgb = cv2.cvtColor(mp_input_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        seg_ms = pose_ms = face_ms = 0.0
        seg_result = pose_result = face_result = None

        # --- Шаг 1: сегментация ---
        if segmenter is not None:
            t0 = time.perf_counter()
            seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
            seg_ms = (time.perf_counter() - t0) * 1000.0

        # --- Шаг 2: поза ---
        if pose_landmarker is not None:
            t0 = time.perf_counter()
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_ms = (time.perf_counter() - t0) * 1000.0

        # --- Шаг 3: face mesh ---
        if face_landmarker is not None:
            t0 = time.perf_counter()
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = yolo_ms + seg_ms + pose_ms + face_ms

        # --- Эллипс головы (один раз на кадр) ---
        head_ellipse = None
        if ((config.ENABLE_HEAD_CONSTRAINT or config.DRAW_HEAD_ELLIPSE)
                and pose_result is not None and pose_result.pose_landmarks):
            head_ellipse = compute_head_ellipse_from_pose(
                pose_result.pose_landmarks, region, width, height,
                config.HEAD_ELLIPSE_RADIUS_U_COEF,
                config.HEAD_ELLIPSE_RADIUS_N_COEF)

        # --- Размер FACE_OVAL ---
        face_size_str = "-"
        if face_result is not None and face_result.face_landmarks:
            rw_in = region[2] if region else width
            rh_in = region[3] if region else height
            size = compute_face_oval_size(face_result.face_landmarks, rw_in, rh_in)
            if size is not None:
                face_sizes.append(size)
                face_size_str = f"{size[0]:.0f}x{size[1]:.0f}"

        crop_str = f"{region[2]}x{region[3]}" if region else "full"

        # --- Статистика ---
        if frame_idx < config.WARMUP_FRAMES:
            skipped += 1
        else:
            processed += 1
            if yolo_model is not None:
                yolo_lat.append(yolo_ms)
            if segmenter is not None:
                seg_lat.append(seg_ms)
            if pose_landmarker is not None:
                pose_lat.append(pose_ms)
            if face_landmarker is not None:
                face_lat.append(face_ms)
            if n_active >= 2:
                total_lat.append(total_ms)

            frame_end_time = time.perf_counter()
            frame_processing_time = (frame_end_time - frame_start_time) * 1000.0  # в миллисекундах
            frame_processing_times.append(frame_processing_time)

            if processed % 30 == 0 or frame_idx == config.WARMUP_FRAMES:
                print(f"{frame_idx:>6} | {yolo_ms:>6.1f} | {seg_ms:>6.1f} | "
                      f"{pose_ms:>6.1f} | {face_ms:>6.1f} | {total_ms:>7.1f} | "
                      f"{frame_processing_time:>7.1f} | {crop_str}")

        # --- Визуализация: отрисовка + запись в видео (всегда) ---
        if frame_idx >= config.WARMUP_FRAMES:
            overlay = frame_bgr.copy()

            # Маска сегментации (полный размер)
            if config.SHOW_CURR_MASK and seg_result is not None:
                if config.SHOW_CONFIDENCE_HEATMAP:
                    heatmap_small, conf_small = confidence_masks_to_heatmap(
                        seg_result.confidence_masks,
                        colormap=config.HEATMAP_COLORMAP)

                    mask_full = np.zeros((height, width, 3), dtype=np.uint8)
                    conf_full = np.zeros((height, width), dtype=np.uint8)
                    if region is not None:
                        ox, oy, rw, rh = region
                        heatmap = cv2.resize(heatmap_small, (rw, rh), interpolation=cv2.INTER_LINEAR)
                        conf_resized = cv2.resize(conf_small, (rw, rh), interpolation=cv2.INTER_LINEAR)
                        mask_full[oy:oy + rh, ox:ox + rw] = heatmap
                        conf_full[oy:oy + rh, ox:ox + rw] = conf_resized
                    else:
                        mask_full = cv2.resize(heatmap_small, (width, height), interpolation=cv2.INTER_LINEAR)
                        conf_full = cv2.resize(conf_small, (width, height), interpolation=cv2.INTER_LINEAR)

                    alpha = (conf_full.astype(np.float32) / 255.0)[:, :, np.newaxis]
                    strength = config.HEATMAP_STRENGTH
                    overlay_f = overlay.astype(np.float32)
                    mask_f = mask_full.astype(np.float32)
                    blended_f = overlay_f * (1.0 - alpha * strength) + mask_f * (alpha * strength)
                    overlay = blended_f.astype(np.uint8)
                else:
                    try:
                        cat_mask = seg_result.category_mask.numpy_view()
                    except AttributeError:
                        cat_mask = seg_result.category_mask.numpy()
                    cat_mask = np.squeeze(cat_mask).astype(np.uint8)
                    colored_small = category_mask_to_colored(cat_mask)

                    mask_full = np.zeros((height, width, 3), dtype=np.uint8)
                    if region is not None:
                        ox, oy, rw, rh = region
                        colored = cv2.resize(colored_small, (rw, rh), interpolation=cv2.INTER_NEAREST)
                        mask_full[oy:oy + rh, ox:ox + rw] = colored
                    else:
                        mask_full = cv2.resize(colored_small, (width, height), interpolation=cv2.INTER_NEAREST)

                    if config.ENABLE_HEAD_CONSTRAINT and head_ellipse is not None:
                        mask_full = apply_head_ellipse_constraint(mask_full, head_ellipse)

                    mask_bool = np.any(mask_full > 0, axis=2)
                    blended = cv2.addWeighted(overlay, 0.5, mask_full, 0.5, 0)
                    overlay[mask_bool] = blended[mask_bool]

            # Отслеживаемый торс считаем заранее: он нужен и маске stickman
            # (треугольники торс-нога), и блоку отслеживания ниже.
            tracked_head_corners = None
            if calib_params is not None and calib_params.get('head') is not None \
                    and pose_result is not None and pose_result.pose_landmarks:
                tracked_head_corners = build_head_rect_from_params(
                    calib_params['head'], pose_result.pose_landmarks[0],
                    region, width, height)

            tracked_torso_quad = None
            if calib_params is not None and calib_params.get('torso') is not None \
                    and pose_result is not None and pose_result.pose_landmarks:
                _lm = pose_result.pose_landmarks[0]
                _sh_l = _get_point_px(_lm, LEFT_SHOULDER, region, width, height)
                _sh_r = _get_point_px(_lm, RIGHT_SHOULDER, region, width, height)
                if _sh_l is not None and _sh_r is not None:
                    tracked_torso_quad = build_torso_quad_from_params(
                        calib_params['torso'], _sh_l, _sh_r,
                        hip_l=_get_point_px(_lm, LEFT_HIP, region, width, height),
                        hip_r=_get_point_px(_lm, RIGHT_HIP, region, width, height),
                        frame_h=height)

            # Шея: нужна и маске stickman, и блоку отслеживания ниже.
            tracked_neck_quad = None
            if tracked_head_corners is not None and tracked_torso_quad is not None:
                tracked_neck_quad = build_neck_quad_from_torso_and_head(
                    tracked_torso_quad, tracked_head_corners, calib_neck)

            # Фигуры "плечи-низ": есть только у калибровок с половинного
            # кадра без бёдер (целиком либо половинками по невидимой руке).
            tracked_shoulders_bottom = []
            if calib_params is not None and calib_params.get('torso') is not None \
                    and pose_result is not None and pose_result.pose_landmarks:
                _lm = pose_result.pose_landmarks[0]
                tracked_shoulders_bottom = build_shoulders_bottom_quads(
                    calib_params['torso'], tracked_torso_quad,
                    _get_point_px(_lm, LEFT_SHOULDER, region, width, height),
                    _get_point_px(_lm, RIGHT_SHOULDER, region, width, height),
                    width, height)

            # Шея во весь низ кадра: плечи не видны, торса нет.
            tracked_lower_neck = None
            if calib_params is not None and calib_params.get('lower_neck') is not None \
                    and tracked_torso_quad is None:
                tracked_lower_neck = build_lower_neck_quad(
                    calib_params['lower_neck'], tracked_head_corners,
                    width, height)

            # --- Модель stickman (поверх тепловой карты, под скелетом) ---
            if config.DRAW_STICKMAN and pose_result is not None and pose_result.pose_landmarks:
                stickman_mask = build_stickman_mask(
                    pose_result.pose_landmarks, region, width, height,
                    torso_quad=tracked_torso_quad,
                    head_corners=tracked_head_corners,
                    limb_widths=calib_limb_widths,
                    neck_quad=tracked_neck_quad,
                    shoulders_bottom_quads=tracked_shoulders_bottom,
                    lower_neck_quad=tracked_lower_neck,
                    limb_grow=(calib_params or {}).get('limb_grow'))
                if stickman_mask is not None:
                    overlay = overlay_stickman(
                        overlay, stickman_mask,
                        config.STICKMAN_COLOR, config.STICKMAN_ALPHA)

            # Поза и овал лица
            if config.DRAW_POSE and pose_result is not None and pose_result.pose_landmarks:
                overlay = draw_pose_landmarks(overlay, pose_result.pose_landmarks, region)
            if config.DRAW_FACE_OVAL and face_result is not None and face_result.face_landmarks:
                overlay = draw_face_oval(overlay, face_result.face_landmarks, region)

            # --- Отслеживание stickman-модели ---
            if config.ENABLE_STICKMAN_TRACKING and calib_params is not None \
                    and pose_result is not None and pose_result.pose_landmarks:
                pose_lm = pose_result.pose_landmarks[0]

                head_corners = None
                torso_quad = None

                # Голова -- посчитана выше (нужна была маске stickman)
                head_corners = tracked_head_corners

                # Торс -- посчитан выше (нужен был маске stickman)
                torso_quad = tracked_torso_quad

                # Шея -- посчитана выше (нужна была маске stickman)
                neck_quad = tracked_neck_quad

                # Все фигуры ниже уже залиты в маску stickman, поэтому по
                # умолчанию не рисуются: заливка легла бы вторым слоем поверх
                # маски, а контуры дублировали бы её границу. Каждый флаг
                # DRAW_TRACKED_* включает свою фигуру обратно -- заливку и
                # контур сразу.
                for enabled, poly, color in (
                        (config.DRAW_TRACKED_TORSO, torso_quad, config.TRACKED_TORSO_COLOR),
                        (config.DRAW_TRACKED_NECK, neck_quad, config.TRACKED_NECK_COLOR),
                        (config.DRAW_TRACKED_HEAD, head_corners, config.TRACKED_HEAD_COLOR)):
                    if not enabled or poly is None:
                        continue
                    overlay = fill_poly_with_alpha(overlay, poly,
                                                   config.STICKMAN_COLOR,
                                                   config.STICKMAN_ALPHA)
                    cv2.polylines(overlay, [np.asarray(poly, dtype=np.int32)],
                                  isClosed=True, color=color,
                                  thickness=config.TRACKED_THICKNESS)

                # Ладони и ступни: строятся прямо из точек позы, калибровка
                # для них не нужна (ширина -- доля от плеч / таза).
                if config.DRAW_TRACKED_PALMS or config.DRAW_TRACKED_FEET:
                    body_rects = build_body_rects(pose_lm, region, width, height,
                                                  limb_widths=calib_limb_widths)
                    tracked_extra = []
                    if body_rects is not None:
                        if config.DRAW_TRACKED_PALMS:
                            tracked_extra.append((body_rects['palms'], config.TRACKED_PALM_COLOR))
                        if config.DRAW_TRACKED_FEET:
                            tracked_extra.append((body_rects['feet'], config.TRACKED_FOOT_COLOR))
                    for rects_group, color in tracked_extra:
                        for rect in rects_group:
                            overlay = fill_poly_with_alpha(
                                overlay, np.asarray(rect, dtype=np.float64),
                                config.STICKMAN_COLOR, config.STICKMAN_ALPHA)
                            cv2.polylines(overlay, [np.asarray(rect, dtype=np.int32)],
                                          isClosed=True, color=color,
                                          thickness=config.TRACKED_THICKNESS)

            # Прямоугольник YOLO
            if config.DRAW_YOLO_BBOX and region is not None:
                ox, oy, rw, rh = region
                cv2.rectangle(overlay, (ox, oy), (ox + rw, oy + rh), (255, 0, 0), 2)

            # Верхний полуэллипс головы
            if config.DRAW_HEAD_ELLIPSE and head_ellipse is not None:
                overlay = draw_head_ellipse_arc(overlay, head_ellipse)


            # info = (f"f{frame_idx} yolo={yolo_ms:.0f} seg={seg_ms:.0f} "
            #         f"pose={pose_ms:.0f} face={face_ms:.0f} tot={total_ms:.0f}ms")
            # cv2.putText(overlay, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            # if face_size_str != "-":
            #     cv2.putText(overlay, f"FACE_OVAL {face_size_str}px", (10, 55),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            # cv2.imshow("MediaPipe benchmark (head constraint)", overlay)

            # --- ЗАПИСЬ В ВИДЕО (всегда, если writer создан) ---
            if out is not None:
                out.write(overlay)

            # --- ПРЕДПРОСМОТР (только если включён) ---
            if show_preview:
                cv2.imshow("MediaPipe benchmark (head constraint)", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nПрервано пользователем (q).")
                    break

        frame_idx += 1

    cap.release()
    if segmenter: segmenter.close()
    if pose_landmarker: pose_landmarker.close()
    if face_landmarker: face_landmarker.close()
    if show_preview: cv2.destroyAllWindows()

    if out is not None:
        out.release()
        print(f"\nВыходное видео сохранено: {output_path}")

    print("\n" + "=" * 58)
    print("РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    print("=" * 58)
    print(f"Всего кадров: {total_frames} | warm-up: {skipped} | обработано: {processed}")
    print(f"Кадры, где человек не найден: {no_person_count}")

    if yolo_model is not None:
        print_latency_stats("YOLO (детекция bbox)", yolo_lat)
    if segmenter is not None:
        print_latency_stats("SelfieMulticlass (segmentation)", seg_lat)
    if pose_landmarker is not None:
        print_latency_stats("PoseLandmarker-Lite (pose)", pose_lat)
    if face_landmarker is not None:
        print_latency_stats("FaceLandmarker (face mesh)", face_lat)
    if n_active >= 2:
        print_latency_stats("СУММАРНО (весь пайплайн на кадр)", total_lat)

    if frame_processing_times:
        mean_frame_time = statistics.mean(frame_processing_times)
        median_frame_time = statistics.median(frame_processing_times)
        min_frame_time = min(frame_processing_times)
        max_frame_time = max(frame_processing_times)
        
        print("\n" + "=" * 58)
        print("ВРЕМЯ ОБРАБОТКИ КАДРА (полный цикл)")
        print("=" * 58)
        print(f"Среднее время на кадр:  {mean_frame_time:.2f} ms")
        print(f"Медианное время:        {median_frame_time:.2f} ms")
        print(f"Минимальное время:      {min_frame_time:.2f} ms")
        print(f"Максимальное время:     {max_frame_time:.2f} ms")
        print(f"Эквивалент FPS:         {1000/mean_frame_time:.1f} FPS")
        print("=" * 58)

    if face_sizes:
        ws = [s[0] for s in face_sizes]
        hs = [s[1] for s in face_sizes]
        print("\nFACE_OVAL (размер в пикселях):")
        print(f"  ширина: mean={statistics.mean(ws):.1f}  min={min(ws):.1f}  max={max(ws):.1f}")
        print(f"  высота: mean={statistics.mean(hs):.1f}  min={min(hs):.1f}  max={max(hs):.1f}")

    print("\n" + "=" * 58)
    print("ОЦЕНКА")
    print("=" * 58)
    if total_lat:
        mean_total = statistics.mean(total_lat)
        print(f"Среднее время всего пайплайна на кадр: {mean_total:.2f} ms "
              f"(~{1000/mean_total:.1f} FPS)")
        if mean_total < 50:
            print("Пайплайн реально тянуть per-frame на CPU.")
        elif mean_total < 100:
            print("Per-frame на CPU возможно, ~10 FPS.")
        else:
            print("Тяжело для per-frame; лучше шаблон/ключевые кадры.")


if __name__ == "__main__":
    main()
    