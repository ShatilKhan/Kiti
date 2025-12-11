"""
Area marking module for defining regions of interest (ROI) in video frames.
Handles ROI definition, obstacle detection within marked areas, and visualization.
"""
import logging
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

from config.settings import settings


logger = logging.getLogger(__name__)


class AreaMarker:
    """Define and manage regions of interest for obstacle detection."""
    
    def __init__(self, roi_start_ratio: float = 0.3, roi_end_ratio: float = 0.7):
        """
        Initialize area marker.
        
        Args:
            roi_start_ratio: Start of ROI as ratio of frame width (default 0.3 = 30%)
            roi_end_ratio: End of ROI as ratio of frame width (default 0.7 = 70%)
        """
        self.roi_start_ratio = roi_start_ratio
        self.roi_end_ratio = roi_end_ratio
        self.background_subtractor = None
        
        logger.info(f"Initialized AreaMarker with ROI: {roi_start_ratio*100:.0f}% - {roi_end_ratio*100:.0f}%")
    
    def get_roi_bounds(self, frame_width: int) -> Tuple[int, int]:
        """
        Calculate ROI boundaries for a given frame width.
        
        Args:
            frame_width: Width of the video frame
            
        Returns:
            Tuple[int, int]: (start_x, end_x) pixel coordinates
        """
        start_x = int(self.roi_start_ratio * frame_width)
        end_x = int(self.roi_end_ratio * frame_width)
        return start_x, end_x
    
    def draw_roi(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 3) -> np.ndarray:
        """
        Draw the ROI rectangle on a frame.
        
        Args:
            frame: Input frame (BGR format)
            color: Rectangle color in BGR (default green)
            thickness: Line thickness
            
        Returns:
            np.ndarray: Frame with ROI drawn
        """
        height, width = frame.shape[:2]
        start_x, end_x = self.get_roi_bounds(width)
        
        result = frame.copy()
        cv2.rectangle(result, (start_x, 0), (end_x, height), color, thickness)
        
        return result
    
    def is_in_roi(self, x: int, y: int, frame_width: int) -> bool:
        """
        Check if a point is within the ROI.
        
        Args:
            x: X coordinate
            y: Y coordinate (unused for horizontal ROI)
            frame_width: Width of the frame
            
        Returns:
            bool: True if point is within ROI
        """
        start_x, end_x = self.get_roi_bounds(frame_width)
        return start_x <= x <= end_x
    
    def is_bbox_in_roi(self, bbox: Tuple[int, int, int, int], frame_width: int) -> bool:
        """
        Check if a bounding box center is within the ROI.
        
        Args:
            bbox: Bounding box (x, y, w, h)
            frame_width: Width of the frame
            
        Returns:
            bool: True if bbox center is within ROI
        """
        x, y, w, h = bbox
        center_x = x + w // 2
        return self.is_in_roi(center_x, y, frame_width)
    
    def initialize_background_subtractor(self, history: int = 100, var_threshold: int = 50):
        """
        Initialize background subtractor for motion detection.
        
        Args:
            history: Number of frames for background model
            var_threshold: Variance threshold for background subtraction
        """
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold
        )
        logger.info("Background subtractor initialized")
    
    def detect_motion(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect motion in frame using background subtraction.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes (x, y, w, h)
        """
        if self.background_subtractor is None:
            self.initialize_background_subtractor()
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Threshold and clean up
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes for significant contours
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= settings.MIN_CONTOUR_AREA:
                bboxes.append((x, y, w, h))
        
        return bboxes
    
    def detect_obstacles_in_roi(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect obstacles within the ROI using motion detection.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List[Dict]: List of detected obstacles with bbox and position info
        """
        height, width = frame.shape[:2]
        bboxes = self.detect_motion(frame)
        
        obstacles = []
        for bbox in bboxes:
            x, y, w, h = bbox
            center_x = x + w // 2
            center_y = y + h // 2
            
            if self.is_bbox_in_roi(bbox, width):
                obstacles.append({
                    'bbox': bbox,
                    'center': (center_x, center_y),
                    'in_roi': True,
                    'area': w * h
                })
        
        return obstacles
    
    def annotate_frame(self, frame: np.ndarray, 
                       draw_roi: bool = True,
                       detect_obstacles: bool = True,
                       roi_color: Tuple[int, int, int] = (0, 255, 0),
                       obstacle_color: Tuple[int, int, int] = (0, 0, 255)) -> Tuple[np.ndarray, List[Dict]]:
        """
        Annotate frame with ROI and detected obstacles.
        
        Args:
            frame: Input frame (BGR format)
            draw_roi: Whether to draw ROI rectangle
            detect_obstacles: Whether to detect and annotate obstacles
            roi_color: Color for ROI rectangle
            obstacle_color: Color for obstacle annotations
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: Annotated frame and list of obstacles
        """
        result = frame.copy()
        obstacles = []
        
        # Draw ROI
        if draw_roi:
            result = self.draw_roi(result, color=roi_color)
        
        # Detect and annotate obstacles
        if detect_obstacles:
            obstacles = self.detect_obstacles_in_roi(frame)
            
            for obs in obstacles:
                x, y, w, h = obs['bbox']
                
                # Draw obstacle bounding box
                cv2.rectangle(result, (x, y), (x + w, y + h), obstacle_color, 2)
                
                # Add label
                cv2.putText(result, 'Obstacle Detected', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, obstacle_color, 2)
        
        return result, obstacles
    
    def process_video(self, video_processor, output_path: str,
                      draw_roi: bool = True,
                      detect_obstacles: bool = True) -> Tuple[int, int]:
        """
        Process entire video with area marking and obstacle detection.
        
        Args:
            video_processor: VideoProcessor instance with opened video
            output_path: Path to save annotated video
            draw_roi: Whether to draw ROI
            detect_obstacles: Whether to detect obstacles
            
        Returns:
            Tuple[int, int]: (frames_processed, total_obstacles_detected)
        """
        # Get video properties
        width = video_processor.width
        height = video_processor.height
        fps = video_processor.fps or settings.OUTPUT_VIDEO_FPS
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*settings.OUTPUT_VIDEO_CODEC)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Failed to create output video: {output_path}")
            return 0, 0
        
        frames_processed = 0
        total_obstacles = 0
        video_processor.reset()
        
        logger.info("Processing video with area marking...")
        
        while True:
            ret, frame = video_processor.read_frame()
            
            if not ret:
                break
            
            annotated_frame, obstacles = self.annotate_frame(
                frame, draw_roi=draw_roi, detect_obstacles=detect_obstacles
            )
            
            out.write(annotated_frame)
            frames_processed += 1
            total_obstacles += len(obstacles)
            
            if frames_processed % settings.PROGRESS_REPORT_INTERVAL == 0:
                logger.info(f"Processed {frames_processed} frames, {total_obstacles} obstacles detected")
        
        out.release()
        logger.info(f"Video processing complete. Saved to: {output_path}")
        logger.info(f"Total: {frames_processed} frames, {total_obstacles} obstacles detected")
        
        return frames_processed, total_obstacles
    
    def reset(self):
        """Reset area marker state."""
        self.background_subtractor = None
        logger.info("Area marker reset")
