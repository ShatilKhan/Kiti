"""
Object detection module using YOLO models.
Handles YOLO model integration, object detection on video frames,
and annotation generation.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

from config.settings import settings


logger = logging.getLogger(__name__)


class ObjectDetector:
    """Handle YOLO-based object detection on video frames."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to YOLO model file (uses default if None)
        """
        self.model_path = model_path or settings.YOLO_MODEL
        self.model = None
        self.confidence_threshold = settings.YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.YOLO_IOU_THRESHOLD
        
        logger.info(f"Initialized ObjectDetector with model: {self.model_path}")
    
    def load_model(self) -> bool:
        """
        Load the YOLO model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            from ultralytics import YOLO
            
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded successfully: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def detect(self, frame: np.ndarray, 
               conf_threshold: Optional[float] = None) -> List[Dict]:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: Input frame (BGR format)
            conf_threshold: Confidence threshold (uses default if None)
            
        Returns:
            List[Dict]: List of detections with bbox, class, and confidence
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return []
        
        conf = conf_threshold if conf_threshold is not None else self.confidence_threshold
        
        try:
            results = self.model(frame, conf=conf, iou=self.iou_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'class_id': int(box.cls[0].cpu().numpy()),
                        'class_name': result.names[int(box.cls[0])]
                    }
                    detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def detect_and_annotate(self, frame: np.ndarray,
                           conf_threshold: Optional[float] = None,
                           draw_labels: bool = True,
                           draw_conf: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects and return annotated frame.
        
        Args:
            frame: Input frame (BGR format)
            conf_threshold: Confidence threshold (uses default if None)
            draw_labels: Whether to draw class labels
            draw_conf: Whether to draw confidence scores
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: Annotated frame and detections
        """
        detections = self.detect(frame, conf_threshold)
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            label_parts = []
            if draw_labels:
                label_parts.append(class_name)
            if draw_conf:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw background rectangle for text
                cv2.rectangle(annotated_frame,
                            (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1),
                            (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(annotated_frame, label,
                          (x1, y1 - baseline - 2),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (0, 0, 0), 1)
        
        return annotated_frame, detections
    
    def detect_video(self, video_processor, output_path: str,
                    show_preview: bool = False) -> int:
        """
        Process entire video with object detection.
        
        Args:
            video_processor: VideoProcessor instance with opened video
            output_path: Path to save annotated video
            show_preview: Whether to show preview window (not supported in headless)
            
        Returns:
            int: Number of frames processed
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return 0
        
        # Get video properties
        width = video_processor.width
        height = video_processor.height
        fps = video_processor.fps or settings.OUTPUT_VIDEO_FPS
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*settings.OUTPUT_VIDEO_CODEC)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Failed to create output video: {output_path}")
            return 0
        
        frames_processed = 0
        video_processor.reset()
        
        logger.info(f"Processing video with object detection...")
        
        while True:
            ret, frame = video_processor.read_frame()
            
            if not ret:
                break
            
            annotated_frame, _ = self.detect_and_annotate(frame)
            out.write(annotated_frame)
            
            frames_processed += 1
            
            if frames_processed % settings.PROGRESS_REPORT_INTERVAL == 0:
                logger.info(f"Processed {frames_processed} frames")
        
        out.release()
        logger.info(f"Video processing complete. Saved to: {output_path}")
        
        return frames_processed
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold
        }
