"""Все настройки и константы бенчмарка."""

import cv2

# --- Пути к моделям и видео ---
SEG_MODEL_PATH = "selfie_multiclass_256x256.tflite"
POSE_MODEL_PATH = "pose_landmarker_lite.task"
FACE_MODEL_PATH = "face_landmarker.task"
YOLO_MODEL_PATH = "yolo26n.pt"
YOLO_SEG_MODEL_PATH = "yolo26n-seg.pt"
VIDEO_PATH = "example.mp4"

SHOW_CURR_MASK = False

# --- YOLO-кроп ---
USE_YOLO_CROP = False
CROP_PADDING = 0.1
YOLO_CONF = 0.3

# --- Включение моделей ---
ENABLE_SEGMENTATION = True
ENABLE_POSE = True
ENABLE_FACE = False

# --- Ограничение головы эллипсом ---
HEAD_ELLIPSE_RADIUS_U_COEF = 0.5  # полуось ВДОЛЬ плеч (полуширина головы), a = coef * S
HEAD_ELLIPSE_RADIUS_N_COEF = 0.8   # полуось К ГОЛОВЕ (полувысота, нос->макушка), b = coef * S
ENABLE_HEAD_CONSTRAINT = False      # применять ограничение к маске
DRAW_HEAD_ELLIPSE = False           # рисовать верхний полуэллипс на кадре

# --- Общие ---
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = "output_selfie_multiclass_skeleton.mp4"
SHOW_PREVIEW = False   # окно можно держать выключенным
WARMUP_FRAMES = 3
DRAW_POSE = True
DRAW_FACE_OVAL = False
DRAW_YOLO_BBOX = True
MAX_FRAMES = None               # None = всё видео; число = ограничить


# --- Категории сегментации ---
CATEGORIES = {0: "background", 1: "hair", 2: "body-skin",
              3: "face-skin", 4: "clothes", 5: "others"}
CATEGORY_COLORS = {
    0: (0, 0, 0), 1: (0, 165, 255), 2: (0, 255, 255),
    3: (255, 255, 0), 4: (128, 0, 128), 5: (0, 255, 0),
}

# --- Соединения скелета (упрощённо, без лица) ---
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

# --- Овал лица (FACE_OVAL), по контуру ---
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]

# --- Индексы точек позы для круга головы ---
POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

# --- Тепловая карта уверенности (временно вместо цветовой классификации) ---
SHOW_CONFIDENCE_HEATMAP = True        # True = heatmap уверенности, False = цвета категорий
HEATMAP_COLORMAP = cv2.COLORMAP_JET   # JET, INFERNO, MAGMA, TURBO и др.
HEATMAP_STRENGTH = 0.8                # сила наложения heatmap (0..1)

# --- Модель stickman (stickman + rectangles) ---
DRAW_STICKMAN = True                    # рисовать модель тела
STICKMAN_HEAD_W_COEF = 0 #0.45             # k_hw: ширина головы = k_hw * S
STICKMAN_HEAD_H_COEF = 0 #0.60             # k_hh: высота головы = k_hh * S
STICKMAN_TORSO_SCALE = 0 #1.2              # масштаб торса относительно его центра (1.0 = без изменений)
STICKMAN_PALM_COEF = 0.40               # k_palm: сторона квадрата ладони = k_palm * S
STICKMAN_LIMB_COEFS = {                 # ширина каждой конечности = coef * S
    (11, 13): 0.36,   # левое плечо (верх руки)
    (13, 15): 0.33,   # левое предплечье
    (12, 14): 0.36,   # правое плечо (верх руки)
    (14, 16): 0.33,   # правое предплечье
    (23, 25): 0.40,   # левое бедро
    (25, 27): 0.36,   # левая голень
    (24, 26): 0.40,   # правое бедро
    (26, 28): 0.36,   # правая голень
}
STICKMAN_FALLBACK_TORSO_ANGLE = 75.0    # угол фолбэк-трапеции торса
STICKMAN_FALLBACK_LENGTH_FACTOR = 3.0   # глубина фолбэк-трапеции
STICKMAN_COLOR = (255, 100, 0)          # цвет модели (BGR)
STICKMAN_ALPHA = 0.5                    # прозрачность заполнения

# --- Калибровка stickman-модели по первому кадру ---
CALIBRATION_W_EXTRA_THRESHOLD = 0.05   # порог W_extra: 5% от ширины плеч S
CALIBRATION_EAR_EXTEND_COEF = 1.5      # макс. продление от ушей: 1.5 × |7-8|
CALIBRATION_SHOULDER_EXTEND_COEF = 1.5
CALIBRATION_LEG_COEFS = {              # дефолтные коэффициенты ног для вычитания
    (23, 25): 0.20,   # левое бедро
    (25, 27): 0.16,   # левая голень
    (24, 26): 0.20,   # правое бедро
    (26, 28): 0.16,   # правая голень
}

# --- Отслеживание stickman-модели ---
ENABLE_STICKMAN_TRACKING = True                      # применять модель при отслеживании
CALIBRATION_PARAMS_PATH = "calibration_params.json"  # путь к параметрам калибровки
DRAW_TRACKED_HEAD = True                             # рисовать отслеживаемую голову
DRAW_TRACKED_TORSO = True                            # рисовать отслеживаемый торс
DRAW_TRACKED_NECK = True                       # рисовать отслеживаемую шею
TRACKED_NECK_COLOR = (255, 100, 0)             # цвет шеи (BGR)
TRACKED_HEAD_COLOR = (255, 100, 0)                   # цвет головы (BGR)
TRACKED_TORSO_COLOR = (255, 100, 0)                  # цвет торса (BGR)
TRACKED_THICKNESS = 2                                # толщина контуров
