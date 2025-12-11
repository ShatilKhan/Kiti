"""
Kiti Autonomous Vehicle - Complete computer vision pipeline.

This package provides modules for:
- Video frame extraction
- Area marking and ROI-based obstacle detection
- YOLO-based object detection
- Optical flow analysis and trajectory prediction
"""

__version__ = '1.0.0'
__author__ = 'Kiti Team'

from .video_processor import VideoProcessor
from .area_marking import AreaMarker
from .object_detection import ObjectDetector
from .optical_flow import OpticalFlowAnalyzer
from .main import AutonomousVehiclePipeline

__all__ = [
    'VideoProcessor',
    'AreaMarker',
    'ObjectDetector',
    'OpticalFlowAnalyzer',
    'AutonomousVehiclePipeline'
]
