"""Создание моделей MediaPipe и YOLO."""

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from . import config


def create_segmenter():
    opts = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=config.SEG_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        output_category_mask=True,
        output_confidence_masks=True,
    )
    return vision.ImageSegmenter.create_from_options(opts)


def create_pose_landmarker():
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=config.POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def create_face_landmarker():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def create_yolo():
    return YOLO(config.YOLO_MODEL_PATH)

