"""
Video processing module for extracting frames from video files.
Handles different video formats and manages frame numbering and storage.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from config.settings import settings


logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handle video frame extraction and processing."""
    
    def __init__(self, video_path: str):
        """
        Initialize video processor.
        
        Args:
            video_path: Path to the video file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video format is not supported
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not settings.validate_video_path(str(self.video_path)):
            raise ValueError(f"Unsupported video format: {self.video_path.suffix}")
        
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        
        logger.info(f"Initialized VideoProcessor for: {video_path}")
    
    def open(self) -> bool:
        """
        Open the video file and read metadata.
        
        Returns:
            bool: True if video opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(str(self.video_path))
            
            if not self.cap.isOpened():
                logger.error("Failed to open video file")
                return False
            
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video opened - Frames: {self.frame_count}, "
                       f"FPS: {self.fps}, Resolution: {self.width}x{self.height}")
            
            return True
        except Exception as e:
            logger.error(f"Error opening video: {e}")
            return False
    
    def extract_frames(self, output_dir: str, 
                      frame_skip: int = 1,
                      start_frame: int = 0,
                      end_frame: Optional[int] = None) -> int:
        """
        Extract frames from video and save as images.
        
        Args:
            output_dir: Directory to save extracted frames
            frame_skip: Extract every nth frame (1 = all frames)
            start_frame: Frame number to start extraction
            end_frame: Frame number to end extraction (None = end of video)
            
        Returns:
            int: Number of frames extracted
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Video not opened. Call open() first.")
            return 0
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extracted_count = 0
        current_frame = 0
        
        if end_frame is None:
            end_frame = self.frame_count
        
        # Set starting position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        logger.info(f"Extracting frames {start_frame} to {end_frame} "
                   f"(skip={frame_skip}) to {output_dir}")
        
        while current_frame < end_frame:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            frames_since_start = current_frame - start_frame
            if frames_since_start % frame_skip == 0:
                frame_filename = (f"{settings.FRAME_PREFIX}"
                                f"{current_frame:0{settings.FRAME_PADDING}d}."
                                f"{settings.FRAME_FORMAT}")
                frame_path = output_path / frame_filename
                
                cv2.imwrite(str(frame_path), frame)
                extracted_count += 1
            
            current_frame += 1
        
        logger.info(f"Extracted {extracted_count} frames")
        return extracted_count
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and frame data
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Video not opened. Call open() first.")
            return False, None
        
        return self.cap.read()
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number.
        
        Args:
            frame_number: Frame number to retrieve
            
        Returns:
            Optional[np.ndarray]: Frame data or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Video not opened. Call open() first.")
            return None
        
        if frame_number < 0 or frame_number >= self.frame_count:
            logger.error(f"Frame number {frame_number} out of range")
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def reset(self):
        """Reset video to the beginning."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logger.info("Video reset to beginning")
    
    def close(self):
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Video resources released")
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def get_video_info(self) -> dict:
        """
        Get video metadata.
        
        Returns:
            dict: Video information
        """
        return {
            'path': str(self.video_path),
            'frame_count': self.frame_count,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'duration_seconds': self.frame_count / self.fps if self.fps > 0 else 0
        }
