"""Наложение откалиброванной stickman-модели на видео + замер скорости.

Пайплайн на кадр (CPU):
  0) YOLO (опционально, --yolo-crop) -- находим bbox человека и вырезаем кроп
  1) SelfieMulticlass (image segmenter)  -- на кропе
  2) PoseLandmarker Lite (pose)          -- на кропе
  3) Голова / шея / торс строятся по коэффициентам калибровки

Если своей калибровки ещё нет, берётся демо-калибровка
example/calibration_params.json, поэтому скрипт работает сразу после
клонирования репозитория.

Логика лежит в bench/main.py, разбор аргументов -- в bench/cli.py.

Запуск:
    python track_stickman.py
    python track_stickman.py --video my.mp4 --output out.mp4 --max-frames 100
    python track_stickman.py --help
"""

from bench.cli import track

if __name__ == "__main__":
    raise SystemExit(track(prog="track_stickman.py"))
