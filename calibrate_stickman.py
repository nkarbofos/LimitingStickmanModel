"""Калибровка stickman-модели по одному кадру видео.

Один раз подгоняет размеры головы и торса модели под конкретного человека
(эталон -- маска InSPyReNet) и сохраняет нормализованные коэффициенты в JSON.
Дальше track_stickman.py накладывает модель на всё видео по этим коэффициентам.

Требует дополнительных зависимостей:
    pip install -e ".[calib]"

Логика лежит в bench/calibrate.py, разбор аргументов -- в bench/cli.py.

Запуск:
    python calibrate_stickman.py
    python calibrate_stickman.py --video my.mp4 --frame 10 --preview
    python calibrate_stickman.py --help
"""

from bench.cli import calibrate

if __name__ == "__main__":
    raise SystemExit(calibrate(prog="calibrate_stickman.py"))
