"""
Main application module for optical flow video processing pipeline.
Orchestrates video processing, object detection, and optical flow analysis.
"""
import os
import logging
import argparse
import json
from pathlib import Path
from typing import Optional

import cv2

from config.settings import settings
from video_processor import VideoProcessor
from object_detection import ObjectDetector
from optical_flow import OpticalFlowAnalyzer


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class OpticalFlowPipeline:
    """Main pipeline for optical flow video processing."""
    
    def __init__(self, video_path: str, output_dir: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            video_path: Path to input video file
            output_dir: Output directory (uses default if None)
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir) if output_dir else settings.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.video_processor = VideoProcessor(video_path)
        self.object_detector = ObjectDetector()
        self.optical_flow = OpticalFlowAnalyzer()
        
        logger.info(f"Pipeline initialized for video: {video_path}")
    
    def extract_frames(self, frame_skip: int = 1) -> int:
        """
        Extract frames from video.
        
        Args:
            frame_skip: Extract every nth frame
            
        Returns:
            int: Number of frames extracted
        """
        logger.info("Starting frame extraction...")
        
        frames_dir = self.output_dir / "frames"
        
        with self.video_processor as vp:
            count = vp.extract_frames(str(frames_dir), frame_skip=frame_skip)
        
        logger.info(f"Frame extraction complete: {count} frames")
        return count
    
    def run_object_detection(self, output_filename: str = "annotated_video.mp4") -> bool:
        """
        Run object detection on video.
        
        Args:
            output_filename: Name of output video file
            
        Returns:
            bool: True if successful
        """
        logger.info("Starting object detection...")
        
        # Load model
        if not self.object_detector.load_model():
            logger.error("Failed to load YOLO model")
            return False
        
        # Process video
        output_path = self.output_dir / output_filename
        
        with self.video_processor as vp:
            frames_processed = self.object_detector.detect_video(
                vp, str(output_path)
            )
        
        if frames_processed > 0:
            logger.info(f"Object detection complete: {frames_processed} frames")
            return True
        else:
            logger.error("Object detection failed")
            return False
    
    def run_optical_flow(self, input_video: Optional[str] = None,
                        output_filename: str = "optical_flow_video.mp4",
                        use_kalman: bool = False) -> bool:
        """
        Run optical flow analysis on video.
        
        Args:
            input_video: Path to input video (uses original if None)
            output_filename: Name of output video file
            use_kalman: Whether to use Kalman filter for prediction
            
        Returns:
            bool: True if successful
        """
        logger.info(f"Starting optical flow analysis (Kalman={use_kalman})...")
        
        # Use specified video or original
        video_path = input_video or self.video_path
        video_proc = VideoProcessor(video_path)
        
        output_path = self.output_dir / output_filename
        
        try:
            with video_proc as vp:
                # Setup video writer
                fourcc = cv2.VideoWriter_fourcc(*settings.OUTPUT_VIDEO_CODEC)
                fps = vp.fps or settings.OUTPUT_VIDEO_FPS
                out = cv2.VideoWriter(str(output_path), fourcc, fps, 
                                     (vp.width, vp.height))
                
                if not out.isOpened():
                    logger.error(f"Failed to create output video: {output_path}")
                    return False
                
                frames_processed = 0
                
                while True:
                    ret, frame = vp.read_frame()
                    
                    if not ret:
                        break
                    
                    # Process frame with optical flow
                    processed_frame, results = self.optical_flow.process_frame_with_prediction(
                        frame, draw_trajectory=True, draw_prediction=True, use_kalman=use_kalman
                    )
                    
                    out.write(processed_frame)
                    frames_processed += 1
                    
                    if frames_processed % settings.PROGRESS_REPORT_INTERVAL == 0:
                        logger.info(f"Processed {frames_processed} frames")
                
                out.release()
                logger.info(f"Optical flow analysis complete: {frames_processed} frames")
                logger.info(f"Output saved to: {output_path}")
                
                return True
        except Exception as e:
            logger.error(f"Optical flow analysis failed: {e}")
            return False
    
    def run_full_pipeline(self, extract_frames_flag: bool = False,
                         run_detection: bool = True,
                         run_flow: bool = True,
                         use_kalman: bool = False) -> dict:
        """
        Run the complete processing pipeline.
        
        Args:
            extract_frames_flag: Whether to extract frames
            run_detection: Whether to run object detection
            run_flow: Whether to run optical flow
            use_kalman: Whether to use Kalman filter
            
        Returns:
            dict: Pipeline execution results
        """
        results = {
            'video_path': self.video_path,
            'output_dir': str(self.output_dir),
            'frames_extracted': 0,
            'detection_success': False,
            'optical_flow_success': False
        }
        
        logger.info("=" * 60)
        logger.info("Starting full pipeline execution")
        logger.info("=" * 60)
        
        # Extract frames if requested
        if extract_frames_flag:
            results['frames_extracted'] = self.extract_frames()
        
        # Run object detection
        if run_detection:
            results['detection_success'] = self.run_object_detection()
        
        # Run optical flow (on detected video if available)
        if run_flow:
            input_video = None
            if run_detection and results['detection_success']:
                input_video = str(self.output_dir / "annotated_video.mp4")
            
            results['optical_flow_success'] = self.run_optical_flow(
                input_video=input_video,
                output_filename="optical_flow_prediction.mp4",
                use_kalman=use_kalman
            )
        
        # Save results
        results_path = self.output_dir / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=settings.OUTPUT_JSON_INDENT)
        
        logger.info("=" * 60)
        logger.info("Pipeline execution complete")
        logger.info(f"Results saved to: {results_path}")
        logger.info("=" * 60)
        
        return results


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Optical Flow Video Processing Pipeline for Autonomous Vehicles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on a video
  python main.py --video path/to/video.mp4 --full
  
  # Extract frames only
  python main.py --video path/to/video.mp4 --extract-frames
  
  # Run object detection only
  python main.py --video path/to/video.mp4 --detect
  
  # Run optical flow with Kalman filter
  python main.py --video path/to/video.mp4 --flow --kalman
  
  # Custom output directory
  python main.py --video path/to/video.mp4 --full --output /path/to/output
        """
    )
    
    parser.add_argument('--video', '-v', required=True,
                       help='Path to input video file')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory (default: output/)')
    
    # Processing modes
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline (detection + optical flow)')
    parser.add_argument('--extract-frames', action='store_true',
                       help='Extract frames from video')
    parser.add_argument('--detect', action='store_true',
                       help='Run object detection only')
    parser.add_argument('--flow', action='store_true',
                       help='Run optical flow analysis only')
    
    # Options
    parser.add_argument('--kalman', action='store_true',
                       help='Use Kalman filter for trajectory prediction')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Extract every nth frame (default: 1)')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Create pipeline
    pipeline = OpticalFlowPipeline(args.video, args.output)
    
    # Determine what to run
    if args.full:
        # Run everything
        pipeline.run_full_pipeline(
            extract_frames_flag=args.extract_frames,
            run_detection=True,
            run_flow=True,
            use_kalman=args.kalman
        )
    else:
        # Run specific components
        if args.extract_frames:
            pipeline.extract_frames(frame_skip=args.frame_skip)
        
        if args.detect:
            pipeline.run_object_detection()
        
        if args.flow:
            pipeline.run_optical_flow(use_kalman=args.kalman)
        
        # If nothing specified, show help
        if not any([args.extract_frames, args.detect, args.flow]):
            parser.print_help()


if __name__ == '__main__':
    main()
