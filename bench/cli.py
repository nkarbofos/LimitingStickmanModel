"""Разбор аргументов командной строки для обоих скриптов.

Единственное место, где описаны опции: и файлы calibrate_stickman.py /
track_stickman.py в корне репозитория, и консольные команды
stickman-calibrate / stickman-track вызывают функции отсюда.
"""

import argparse

from . import config


# ------------------------------------------------------------------
# track
# ------------------------------------------------------------------
def build_track_parser(prog="track_stickman.py"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Накладывает откалиброванную stickman-модель на видео.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--video", default=config.VIDEO_PATH,
        help="входное видео")
    parser.add_argument(
        "--output", default=config.OUTPUT_VIDEO_PATH,
        help="куда сохранить видео с наложенной моделью")
    parser.add_argument(
        "--params", default=None,
        help="коэффициенты калибровки (JSON); по умолчанию "
             "output/calibration_params.json, иначе демо-файл "
             "example/calibration_params.json")
    parser.add_argument(
        "--max-frames", type=int, default=config.MAX_FRAMES,
        help="ограничить число кадров (по умолчанию -- всё видео)")
    parser.add_argument(
        "--yolo-crop", action="store_true", default=config.USE_YOLO_CROP,
        help="предварительно вырезать человека детектором YOLO")
    parser.add_argument(
        "--no-segmentation", action="store_true",
        help="не запускать SelfieMulticlass: для наложения модели он не нужен "
             "(его результат виден только при config.SHOW_CURR_MASK=True), "
             "но занимает большую часть времени кадра")
    parser.add_argument(
        "--no-save", action="store_true",
        help="не писать выходное видео, только замерить скорость")
    parser.add_argument(
        "--preview", action="store_true",
        help="показывать кадры в окне OpenCV (нужен GUI)")
    parser.add_argument(
        "--no-download", action="store_true",
        help="не скачивать недостающие модели автоматически")
    return parser


def track(argv=None, prog="stickman-track"):
    """Точка входа отслеживания. Возвращает код возврата процесса."""
    from .main import main

    args = build_track_parser(prog).parse_args(argv)
    main(
        video_path=args.video,
        output_path=args.output,
        calibration_params_path=args.params,
        max_frames=args.max_frames,
        show_preview=args.preview,
        use_yolo_crop=args.yolo_crop,
        save_output_video=not args.no_save,
        enable_segmentation=not args.no_segmentation,
        auto_download=not args.no_download,
    )
    return 0


# ------------------------------------------------------------------
# calibrate
# ------------------------------------------------------------------
def build_calibrate_parser(prog="calibrate_stickman.py"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Калибрует stickman-модель по одному кадру видео.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--video", default=config.VIDEO_PATH,
        help="видео, по кадру которого калибруемся")
    parser.add_argument(
        "--frame", type=int, default=config.CALIB_FRAME_INDEX,
        help="номер кадра для калибровки (1-based)")
    parser.add_argument(
        "--params", default=config.CALIBRATION_PARAMS_OUTPUT_PATH,
        help="куда сохранить коэффициенты калибровки (JSON)")
    parser.add_argument(
        "--image", default=config.CALIBRATION_RESULT_IMAGE_PATH,
        help="куда сохранить картинку с результатом")
    parser.add_argument(
        "--preview", action="store_true",
        help="показать результат в окне OpenCV (нужен GUI)")
    parser.add_argument(
        "--no-download", action="store_true",
        help="не скачивать недостающие модели автоматически")
    return parser


def calibrate(argv=None, prog="stickman-calibrate"):
    """Точка входа калибровки. Возвращает код возврата процесса."""
    from .calibrate import main

    args = build_calibrate_parser(prog).parse_args(argv)
    return main(
        video_path=args.video,
        frame_index=args.frame,
        params_path=args.params,
        image_path=args.image,
        show_preview=args.preview,
        auto_download=not args.no_download,
    )
