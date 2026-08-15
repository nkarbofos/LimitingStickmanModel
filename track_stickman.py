"""
Бенчмарк пайплайна на CPU:
  0) YOLO            -> находим bbox человека и вырезаем кроп
  1) SelfieMulticlass (image segmenter)      -- на кропе
  2) PoseLandmarker Lite (pose)              -- на кропе
  3) FaceLandmarker (face mesh + FACE_OVAL)  -- на кропе
  4) Ограничение головы кругом (верхний полукруг)

Вся логика разнесена по модулям пакета bench/.

Запуск:
    python benchmark_selfie_multiclass.py
"""

from bench.main import main

if __name__ == "__main__":
    main()
