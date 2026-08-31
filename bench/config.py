"""Все настройки и константы бенчмарка.

Пути собраны в bench/paths.py и считаются от корня репозитория, поэтому
скрипты работают из любой рабочей директории. Значения ниже -- умолчания;
почти все они переопределяются аргументами командной строки.
"""

import os

import cv2

from .paths import OUTPUT_DIR, example_path, model_path

# --- Имена моделей (файлы лежат в models/, скачиваются bench/download.py) ---
SEG_MODEL_NAME = "selfie_multiclass_256x256.tflite"
POSE_MODEL_NAME = "pose_landmarker_lite.task"
FACE_MODEL_NAME = "face_landmarker.task"
YOLO_MODEL_NAME = "yolo26n.pt"
YOLO_SEG_MODEL_NAME = "yolo26n-seg.pt"

# --- Пути к моделям и видео ---
SEG_MODEL_PATH = str(model_path(SEG_MODEL_NAME))
POSE_MODEL_PATH = str(model_path(POSE_MODEL_NAME))
FACE_MODEL_PATH = str(model_path(FACE_MODEL_NAME))
YOLO_MODEL_PATH = str(model_path(YOLO_MODEL_NAME))
YOLO_SEG_MODEL_PATH = str(model_path(YOLO_SEG_MODEL_NAME))
VIDEO_PATH = str(example_path("example.mp4"))

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
OUTPUT_VIDEO_PATH = str(OUTPUT_DIR / "example_tracked.mp4")
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
# Опускание точек бёдер перпендикулярно линии торса 23-24 (в долях S)
STICKMAN_PALM_COEF = 0.2               # k_palm: сторона квадрата ладони = k_palm * S
STICKMAN_FOOT_COEF = 0.5915               # k_foot: толщина ступни = k_foot * S_hip(таз)
# Удлинение отрезка наружу от сустава: 1.0 = без удлинения, 1.3 = +30%
STICKMAN_TORSO_EXTEND_COEF = 0 #0.4
STICKMAN_PALM_EXTEND_COEF = 1.4         # отрезок ладони: 15 -> середина 19 и 17 (16 -> 20 и 18)
STICKMAN_FOOT_EXTEND_COEF = 1.3         # отрезок ступни 27-31 / 28-32
STICKMAN_LIMB_COEFS = {                 # руки: coef * S(плечи), ноги: coef * S_hip(таз)
    (11, 13): 0.36,   # левое плечо (верх руки)
    (13, 15): 0.33,   # левое предплечье
    (12, 14): 0.36,   # правое плечо (верх руки)
    (14, 16): 0.33,   # правое предплечье
    (23, 25): 0.676,   # левое бедро
    (25, 27): 0.6084,   # левая голень
    (24, 26): 0.676,   # правое бедро
    (26, 28): 0.6084,   # правая голень
}
STICKMAN_FALLBACK_TORSO_ANGLE = 75.0    # угол фолбэк-трапеции торса
STICKMAN_FALLBACK_LENGTH_FACTOR = 3.0   # глубина фолбэк-трапеции
STICKMAN_COLOR = (255, 100, 0)          # цвет модели (BGR)
STICKMAN_ALPHA = 0.5                    # прозрачность заполнения

# --- Калибровка stickman-модели по первому кадру ---
CALIBRATION_W_EXTRA_THRESHOLD = 0.05   # порог W_extra: 5% от ширины плеч S
CALIBRATION_EAR_EXTEND_COEF = 1.5      # макс. продление от ушей: 1.5 × |7-8|
CALIBRATION_SHOULDER_EXTEND_COEF = 1.5
CALIBRATION_HIP_EXTEND_COEF = 1.5      # макс. вытягивание отрезка бёдер (в долях S)
# Линия живота: параллельна линии плеч, на доле расстояния плечи -> торс
CALIBRATION_BELLY_COEF = 2.0 / 3.0     # доля перпендикулярного расстояния
CALIBRATION_BELLY_EXTEND_COEF = 1.5    # макс. вытягивание до границы маски (в долях S)
# Ширина конечности в каждой её точке: луч из точки перпендикулярно отрезку
# до границы маски. Единицы -- те же, что у STICKMAN_LIMB_COEFS: доли ширины
# плеч для рук, доли ширины таза для ног.
CALIBRATION_LIMB_RAY_COEF = 1.5            # макс. длина самого луча
CALIBRATION_LIMB_WIDTH_MIN_COEF = 0.02     # ниже -- замер вырожден, берём STICKMAN_LIMB_COEFS
# Запас к порогу «луч короткий»: луч считается коротким, пока он меньше
# половины номинальной ширины, увеличенной на LIMB_EXTEND_COEF.
LIMB_EXTEND_COEF = 0.4
# --- Подбор нижней границы головы, когда подбородок не найден ---
# Опускаем down_dist, максимизируя IoU фигуры (голова + шея) с маской
# в фиксированной полосе между верхним ребром торса и уровнем носа.
CALIBRATION_NECK_FIT_ENABLED = True     # включить подбор (иначе -- симметрично len_XN)
CALIBRATION_NECK_FIT_MIN_COEF = 0.15    # нижняя граница перебора (в долях len_XN)
CALIBRATION_NECK_FIT_MAX_COEF = 1.20    # верхняя граница перебора (в долях len_XN)
CALIBRATION_NECK_FIT_STEP_PX = 1.0      # шаг перебора, px
CALIBRATION_NECK_FIT_MARGIN_PX = 1.0    # запас над верхним ребром торса, px
CALIBRATION_NECK_FIT_DEBUG = False      # печатать всю кривую IoU(d)

CALIBRATION_LEG_COEFS = {              # дефолтные коэффициенты ног для вычитания
    (23, 25): 0.20,   # левое бедро
    (25, 27): 0.16,   # левая голень
    (24, 26): 0.20,   # правое бедро
    (26, 28): 0.16,   # правая голень
}

# --- Отслеживание stickman-модели ---
ENABLE_STICKMAN_TRACKING = True    # применять модель при отслеживании

# Куда calibrate_stickman.py кладёт результат калибровки
CALIBRATION_PARAMS_OUTPUT_PATH = str(OUTPUT_DIR / "calibration_params.json")
CALIBRATION_RESULT_IMAGE_PATH = str(OUTPUT_DIR / "calibration_result.png")
# Готовая калибровка для example.mp4, идущая вместе с репозиторием
CALIBRATION_PARAMS_EXAMPLE_PATH = str(example_path("calibration_params.json"))


def default_calibration_params_path():
    """Путь к параметрам калибровки, используемый при отслеживании.

    Своя калибровка из output/ имеет приоритет; если её ещё нет, берётся
    демо-калибровка из example/, чтобы track_stickman.py работал сразу
    после клонирования.
    """
    if os.path.exists(CALIBRATION_PARAMS_OUTPUT_PATH):
        return CALIBRATION_PARAMS_OUTPUT_PATH
    return CALIBRATION_PARAMS_EXAMPLE_PATH


CALIBRATION_PARAMS_PATH = CALIBRATION_PARAMS_OUTPUT_PATH  # обратная совместимость

# --- Скрипт калибровки (calibrate_stickman.py) ---
CALIB_YOLO_MODEL_NAME = "yolo26s.pt"          # детектор человека (точнее, чем n)
CALIB_POSE_MODEL_NAME = "pose_landmarker_full.task"  # точный скелет
CALIB_FACE_MODEL_NAME = "face_landmarker.task"       # подбородок
CALIB_YOLO_MODEL_PATH = str(model_path(CALIB_YOLO_MODEL_NAME))
CALIB_POSE_MODEL_PATH = str(model_path(CALIB_POSE_MODEL_NAME))
CALIB_FACE_MODEL_PATH = str(model_path(CALIB_FACE_MODEL_NAME))

CALIB_FRAME_INDEX = 5           # номер кадра для калибровки (1-based)
CALIB_YOLO_BBOX_PADDING = 0.10  # запас вокруг box-а: 10% от размера
CALIB_FACE_CHIN_INDEX = 152     # индекс подбородка (chin tip) в face_landmarker
# Повторный поиск лица, когда на кропе всего человека оно не нашлось: кроп
# берётся вокруг построенного квадрата головы, увеличенного в столько раз.
CALIB_HEAD_CROP_EXPAND = 1.15
CALIB_MASK_OVERLAY_ALPHA = 0.5  # прозрачность маски InSPyReNet на визуализации
# --- Прямоугольники конечностей на визуализации калибровки ---
DRAW_CALIB_LIMBS = True                # рисовать прямоугольники рук и ног
CALIB_ARM_COLOR = (255, 0, 255)        # цвет рук (BGR, пурпурный)
CALIB_LEG_COLOR = (255, 255, 0)        # цвет ног (BGR, голубой)
CALIB_PALM_COLOR = (255, 255, 255)     # цвет ладоней (BGR, белый)
CALIB_FOOT_COLOR = (128, 128, 255)     # цвет ступней (BGR, светло-красный)
CALIB_HIP_TRI_COLOR = (0, 0, 0)        # цвет треугольников таза (BGR, чёрный)
CALIB_TORSO_TRI_COLOR = (0, 0, 128)    # цвет треугольников торс-нога (BGR, тёмно-красный)
CALIB_THIGH_QUAD_COLOR = (0, 0, 128)   # цвет четырёхугольников бёдер (BGR, тёмно-красный)
CALIB_UPPER_HULL_COLOR = (0, 128, 0)   # цвет многоугольника XABCDY (BGR, тёмно-зелёный)
CALIB_ARM_TOPS_COLOR = (0, 128, 0)     # цвет четырёхугольника ABCD (BGR, тёмно-зелёный)
DRAW_TRACKED_HEAD = True                             # рисовать отслеживаемую голову
DRAW_TRACKED_TORSO = True                            # рисовать отслеживаемый торс
DRAW_TRACKED_NECK = True                       # рисовать отслеживаемую шею
TRACKED_NECK_COLOR = (255, 100, 0)             # цвет шеи (BGR)
TRACKED_HEAD_COLOR = (255, 100, 0)                   # цвет головы (BGR)
TRACKED_TORSO_COLOR = (255, 100, 0)                  # цвет торса (BGR)
DRAW_TRACKED_PALMS = True                      # рисовать ладони при отслеживании
DRAW_TRACKED_FEET = True                       # рисовать ступни при отслеживании
TRACKED_PALM_COLOR = (255, 255, 255)           # цвет ладоней (BGR, белый)
TRACKED_FOOT_COLOR = (128, 128, 255)           # цвет ступней (BGR, светло-красный)
TRACKED_THICKNESS = 2                                # толщина контуров
