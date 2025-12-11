"""
Configuration settings for optical flow project.
Centralized configuration for paths, model parameters, and runtime settings.
"""
import os
from pathlib import Path


class Settings:
    """Configuration settings for the optical flow application."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    
    # Video processing settings
    VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
    FRAME_PREFIX = "frame_"
    FRAME_FORMAT = "png"
    FRAME_PADDING = 4  # Number of digits in frame numbering (e.g., 0001, 0002)
    
    # YOLO model settings
    YOLO_MODEL = "yolov8n.pt"  # Default YOLOv8 nano model
    YOLO_CONFIDENCE_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    
    # Optical flow settings
    OPTICAL_FLOW_METHOD = "farneback"  # Options: farneback, lucaskanade
    OPTICAL_FLOW_PYR_SCALE = 0.5
    OPTICAL_FLOW_LEVELS = 3
    OPTICAL_FLOW_WINSIZE = 15
    OPTICAL_FLOW_ITERATIONS = 3
    OPTICAL_FLOW_POLY_N = 5
    OPTICAL_FLOW_POLY_SIGMA = 1.2
    OPTICAL_FLOW_FLAGS = 0
    
    # Motion detection settings
    MOTION_THRESHOLD = 1.0  # Magnitude threshold for detecting motion
    MIN_CONTOUR_AREA = 500  # Minimum area for motion contours
    
    # Kalman filter settings
    KALMAN_PROCESS_NOISE = 0.03
    KALMAN_MEASUREMENT_NOISE = 1.0
    
    # Trajectory prediction settings
    TRAJECTORY_MAX_LENGTH = 30  # Maximum trajectory history
    PREDICTION_FRAMES = 30  # Number of frames to predict ahead
    
    # Output settings
    OUTPUT_VIDEO_CODEC = 'mp4v'
    OUTPUT_VIDEO_FPS = 20
    OUTPUT_JSON_INDENT = 2
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_video_path(cls, path: str) -> bool:
        """Validate if the path points to a supported video file."""
        path_obj = Path(path)
        return path_obj.exists() and path_obj.suffix.lower() in cls.VIDEO_FORMATS


# Create a singleton instance
settings = Settings()
