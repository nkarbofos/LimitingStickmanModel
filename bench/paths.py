"""Пути проекта.

Все пути считаются от корня репозитория, а не от текущей рабочей директории,
поэтому скрипты можно запускать откуда угодно:

    python calibrate_stickman.py
    python /path/to/LimitingStickmanModel/track_stickman.py

Раскладка:
    <корень>/models/   -- веса моделей (скачиваются автоматически, в git не хранятся)
    <корень>/example/  -- демо-данные, идущие вместе с репозиторием
    <корень>/output/   -- результаты работы скриптов (в git не хранятся)

Каталоги models/ и output/ можно переопределить переменными окружения
STICKMAN_MODELS_DIR и STICKMAN_OUTPUT_DIR.
"""

import os
from pathlib import Path

# bench/paths.py -> bench/ -> корень репозитория
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = Path(os.environ.get("STICKMAN_MODELS_DIR",
                                 PROJECT_ROOT / "models")).expanduser()
OUTPUT_DIR = Path(os.environ.get("STICKMAN_OUTPUT_DIR",
                                 PROJECT_ROOT / "output")).expanduser()
EXAMPLE_DIR = PROJECT_ROOT / "example"


def model_path(name):
    """Полный путь к весам модели в models/ (файла может ещё не быть)."""
    return MODELS_DIR / name


def output_path(name):
    """Полный путь к файлу в output/, создавая каталог при необходимости."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def example_path(name):
    """Полный путь к файлу демо-данных в example/."""
    return EXAMPLE_DIR / name
