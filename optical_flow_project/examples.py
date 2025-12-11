"""
Example usage of the Optical Flow Project modules.

This script demonstrates how to use the different modules in the project.
Note: This requires a video file to be present in the data/ directory.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config.settings import settings


def example_1_video_info():
    """Example 1: Get video information"""
    print("=" * 60)
    print("Example 1: Video Information")
    print("=" * 60)
    
    from video_processor import VideoProcessor
    
    # This is just an example - replace with actual video path
    video_path = "data/sample_video.mp4"
    
    print(f"Video path: {video_path}")
    print(f"Note: This example requires a video file at {video_path}")
    print()
    
    # Example code (commented to avoid errors when no video present):
    """
    with VideoProcessor(video_path) as vp:
        info = vp.get_video_info()
        print("Video Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    """


def example_2_object_detection():
    """Example 2: Object detection setup"""
    print("=" * 60)
    print("Example 2: Object Detection")
    print("=" * 60)
    
    from object_detection import ObjectDetector
    
    detector = ObjectDetector()
    print(f"Model path: {detector.model_path}")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    print(f"IOU threshold: {detector.iou_threshold}")
    print()
    
    # Load model (commented to avoid downloading during example)
    """
    detector.load_model()
    print("Model loaded successfully!")
    """


def example_3_optical_flow():
    """Example 3: Optical flow analyzer"""
    print("=" * 60)
    print("Example 3: Optical Flow Analyzer")
    print("=" * 60)
    
    from optical_flow import OpticalFlowAnalyzer
    
    analyzer = OpticalFlowAnalyzer()
    print(f"Trajectory max length: {settings.TRAJECTORY_MAX_LENGTH}")
    print(f"Motion threshold: {settings.MOTION_THRESHOLD}")
    print(f"Prediction frames: {settings.PREDICTION_FRAMES}")
    print()


def example_4_full_pipeline():
    """Example 4: Full pipeline setup"""
    print("=" * 60)
    print("Example 4: Full Pipeline")
    print("=" * 60)
    
    from main import OpticalFlowPipeline
    
    video_path = "data/sample_video.mp4"
    output_dir = "output/example"
    
    print(f"Video path: {video_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Example pipeline setup (commented to avoid errors)
    """
    pipeline = OpticalFlowPipeline(video_path, output_dir)
    
    # Run specific components
    pipeline.extract_frames(frame_skip=5)
    pipeline.run_object_detection()
    pipeline.run_optical_flow(use_kalman=True)
    
    # Or run everything at once
    results = pipeline.run_full_pipeline(
        extract_frames_flag=False,
        run_detection=True,
        run_flow=True,
        use_kalman=True
    )
    print(results)
    """


def show_configuration():
    """Show current configuration"""
    print("=" * 60)
    print("Current Configuration")
    print("=" * 60)
    
    print(f"Project root: {settings.PROJECT_ROOT}")
    print(f"Data directory: {settings.DATA_DIR}")
    print(f"Output directory: {settings.OUTPUT_DIR}")
    print()
    print(f"YOLO model: {settings.YOLO_MODEL}")
    print(f"Confidence threshold: {settings.YOLO_CONFIDENCE_THRESHOLD}")
    print()
    print(f"Optical flow method: {settings.OPTICAL_FLOW_METHOD}")
    print(f"Motion threshold: {settings.MOTION_THRESHOLD}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Optical Flow Project - Usage Examples")
    print("=" * 60 + "\n")
    
    show_configuration()
    print()
    
    example_1_video_info()
    print()
    
    example_2_object_detection()
    print()
    
    example_3_optical_flow()
    print()
    
    example_4_full_pipeline()
    print()
    
    print("=" * 60)
    print("For actual usage, run: python run.py --help")
    print("=" * 60)


if __name__ == '__main__':
    main()
