"""Автоматическая загрузка весов моделей в models/.

Веса в git не хранятся: MediaPipe-модели лежат на storage.googleapis.com,
YOLO-модели -- в релизах ultralytics/assets. Скрипты вызывают ensure_model()
и докачивают недостающее при первом запуске.

Всё сразу можно скачать заранее:

    python -m bench.download          # все модели
    python -m bench.download --group track
    stickman-download-models          # после pip install
"""

import argparse
import shutil
import sys
import urllib.error
import urllib.request

from .paths import MODELS_DIR, model_path

_MP_BASE = "https://storage.googleapis.com/mediapipe-models"
_YOLO_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

#: имя файла -> (URL, короткое описание)
MODEL_REGISTRY = {
    "selfie_multiclass_256x256.tflite": (
        f"{_MP_BASE}/image_segmenter/selfie_multiclass_256x256/float32/latest"
        f"/selfie_multiclass_256x256.tflite",
        "MediaPipe Image Segmenter (маска человека по 6 классам)",
    ),
    "pose_landmarker_lite.task": (
        f"{_MP_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest"
        f"/pose_landmarker_lite.task",
        "MediaPipe Pose Landmarker Lite (скелет, быстрый -- для отслеживания)",
    ),
    "pose_landmarker_full.task": (
        f"{_MP_BASE}/pose_landmarker/pose_landmarker_full/float16/latest"
        f"/pose_landmarker_full.task",
        "MediaPipe Pose Landmarker Full (скелет, точный -- для калибровки)",
    ),
    "face_landmarker.task": (
        f"{_MP_BASE}/face_landmarker/face_landmarker/float16/latest"
        f"/face_landmarker.task",
        "MediaPipe Face Landmarker (подбородок, точка 152)",
    ),
    "yolo26s.pt": (
        f"{_YOLO_BASE}/yolo26s.pt",
        "YOLO26s -- детекция человека при калибровке",
    ),
    "yolo26n.pt": (
        f"{_YOLO_BASE}/yolo26n.pt",
        "YOLO26n -- детекция человека для кропа при отслеживании (опционально)",
    ),
    "yolo26n-seg.pt": (
        f"{_YOLO_BASE}/yolo26n-seg.pt",
        "YOLO26n-seg -- сегментация, для сравнения качества",
    ),
}

#: наборы моделей под конкретные сценарии
MODEL_GROUPS = {
    "track": [
        "selfie_multiclass_256x256.tflite",
        "pose_landmarker_lite.task",
    ],
    "calibrate": [
        "pose_landmarker_full.task",
        "face_landmarker.task",
        "yolo26s.pt",
    ],
    "optional": [
        "yolo26n.pt",
        "yolo26n-seg.pt",
    ],
}


def _report(done, total):
    if total <= 0:
        sys.stdout.write(f"\r    {done / 1e6:.1f} MB")
    else:
        pct = 100.0 * done / total
        sys.stdout.write(f"\r    {pct:5.1f}%  ({done / 1e6:.1f} / {total / 1e6:.1f} MB)")
    sys.stdout.flush()


def download_model(name, force=False):
    """Скачивает модель `name` в models/ и возвращает путь к ней.

    Скачивание идёт во временный файл `.part` и переименовывается только
    после успешного завершения, поэтому оборванная загрузка не оставляет
    битых весов.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Неизвестная модель: {name}. "
                       f"Известные: {', '.join(sorted(MODEL_REGISTRY))}")

    dest = model_path(name)
    if dest.exists() and not force:
        return dest

    url, description = MODEL_REGISTRY[name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    print(f"[>] Скачиваю {name} -- {description}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as fh:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                _report(done, total)
        print()
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Не удалось скачать {name} с {url}: {exc}\n"
            f"    Скачайте файл вручную и положите в {MODELS_DIR}"
        ) from exc

    shutil.move(str(tmp), str(dest))
    print(f"    сохранено: {dest}")
    return dest


def ensure_model(name, auto_download=True):
    """Возвращает путь к модели, при необходимости скачав её.

    Если auto_download=False, а файла нет -- поднимает FileNotFoundError
    с подсказкой, чем его добрать.
    """
    dest = model_path(name)
    if dest.exists():
        return dest
    if auto_download:
        return download_model(name)
    raise FileNotFoundError(
        f"Модель не найдена: {dest}\n"
        f"    Скачайте её командой: python -m bench.download {name}"
    )


def ensure_models(names, auto_download=True):
    """ensure_model() для списка моделей. Возвращает список путей."""
    return [ensure_model(name, auto_download=auto_download) for name in names]


def cli(argv=None):
    parser = argparse.ArgumentParser(
        prog="stickman-download-models",
        description="Скачивает веса моделей в models/.")
    parser.add_argument(
        "models", nargs="*", metavar="MODEL",
        help="конкретные модели (по умолчанию -- все)")
    parser.add_argument(
        "--group", choices=sorted(MODEL_GROUPS),
        help="скачать только набор под сценарий: "
             "track (отслеживание), calibrate (калибровка), optional (прочее)")
    parser.add_argument(
        "--force", action="store_true",
        help="перекачать, даже если файл уже есть")
    parser.add_argument(
        "--list", action="store_true",
        help="показать список моделей и выйти")
    args = parser.parse_args(argv)

    if args.list:
        print(f"Каталог моделей: {MODELS_DIR}\n")
        for name, (_, description) in MODEL_REGISTRY.items():
            mark = "есть" if model_path(name).exists() else " -- "
            print(f"  [{mark}] {name:<38} {description}")
        return 0

    if args.models:
        names = args.models
    elif args.group:
        names = MODEL_GROUPS[args.group]
    else:
        names = list(MODEL_REGISTRY)

    try:
        for name in names:
            path = download_model(name, force=args.force)
            if not args.force:
                print(f"[ok] {name}: {path}")
    except (KeyError, RuntimeError) as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 1

    print(f"\nГотово. Модели лежат в {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
