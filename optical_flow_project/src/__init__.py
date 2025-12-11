"""
Optical Flow Project - Video processing for autonomous vehicles.

This package provides modules for:
- Video frame extraction
- YOLO-based object detection
- Optical flow analysis and trajectory prediction
"""

__version__ = '1.0.0'
__author__ = 'Kiti Team'

from .video_processor import VideoProcessor
from .object_detection import ObjectDetector
from .optical_flow import OpticalFlowAnalyzer
from .main import OpticalFlowPipeline

__all__ = [
    'VideoProcessor',
    'ObjectDetector',
    'OpticalFlowAnalyzer',
    'OpticalFlowPipeline'
]
