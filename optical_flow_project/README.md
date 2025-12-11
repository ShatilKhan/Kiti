# Optical Flow Video Processing Project

A production-ready Python project for autonomous vehicle computer vision tasks, featuring video processing, YOLO object detection, and optical flow analysis with trajectory prediction.

## Overview

This project converts video streams into actionable insights for autonomous vehicle applications by:
- Extracting and processing video frames
- Detecting objects using YOLOv8
- Analyzing motion through optical flow
- Predicting future trajectories using Linear Regression or Kalman Filtering

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **YOLO Integration**: State-of-the-art object detection using Ultralytics YOLOv8
- **Optical Flow Analysis**: Motion detection and tracking using Farneback algorithm
- **Trajectory Prediction**: 
  - Linear Regression for basic path prediction
  - Kalman Filter for advanced smoothed predictions
- **Flexible CLI**: Easy-to-use command-line interface
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Configuration Management**: Centralized settings for easy customization
- **Batch Processing**: Support for processing multiple videos

## Project Structure

```
optical_flow_project/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Main pipeline orchestration
│   ├── video_processor.py       # Video frame extraction
│   ├── object_detection.py      # YOLO object detection
│   └── optical_flow.py          # Optical flow analysis
├── config/
│   ├── __init__.py              # Config package initialization
│   └── settings.py              # Centralized configuration
├── data/
│   └── README.md                # Data directory information
├── output/
│   └── README.md                # Output directory information
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup script
├── README.md                    # This file
└── run.py                       # CLI entry point
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ShatilKhan/Kiti.git
cd Kiti/optical_flow_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install as a package:
```bash
pip install -e .
```

## Usage

### Quick Start

Run the full pipeline on a video:
```bash
python run.py --video data/your_video.mp4 --full
```

### CLI Options

```bash
python run.py --help
```

#### Basic Commands

**Extract frames only:**
```bash
python run.py --video path/to/video.mp4 --extract-frames
```

**Run object detection:**
```bash
python run.py --video path/to/video.mp4 --detect
```

**Run optical flow analysis:**
```bash
python run.py --video path/to/video.mp4 --flow
```

**Full pipeline (detection + optical flow):**
```bash
python run.py --video path/to/video.mp4 --full
```

#### Advanced Options

**Use Kalman filter for trajectory prediction:**
```bash
python run.py --video path/to/video.mp4 --flow --kalman
```

**Custom output directory:**
```bash
python run.py --video path/to/video.mp4 --full --output /custom/path
```

**Extract every 5th frame:**
```bash
python run.py --video path/to/video.mp4 --extract-frames --frame-skip 5
```

### Python API

You can also use the modules programmatically:

```python
from src.main import OpticalFlowPipeline

# Create pipeline
pipeline = OpticalFlowPipeline('path/to/video.mp4', output_dir='output/')

# Run full pipeline
results = pipeline.run_full_pipeline(
    extract_frames_flag=False,
    run_detection=True,
    run_flow=True,
    use_kalman=True
)

print(results)
```

### Individual Modules

```python
from src.video_processor import VideoProcessor
from src.object_detection import ObjectDetector
from src.optical_flow import OpticalFlowAnalyzer

# Video processing
with VideoProcessor('video.mp4') as vp:
    vp.extract_frames('frames/', frame_skip=1)

# Object detection
detector = ObjectDetector()
detector.load_model()
detections = detector.detect(frame)

# Optical flow
analyzer = OpticalFlowAnalyzer()
processed_frame, results = analyzer.process_frame_with_prediction(frame)
```

## Configuration

Edit `config/settings.py` to customize:

- **Paths**: Default data and output directories
- **YOLO Parameters**: Model path, confidence thresholds
- **Optical Flow**: Algorithm parameters
- **Output Settings**: Video codec, FPS, format options

Example customization:
```python
# In config/settings.py
class Settings:
    YOLO_MODEL = "yolov8m.pt"  # Use medium model instead of nano
    YOLO_CONFIDENCE_THRESHOLD = 0.5  # Increase confidence threshold
    MOTION_THRESHOLD = 2.0  # Adjust motion sensitivity
```

## Output

The pipeline generates:

1. **Extracted Frames** (if requested): Individual PNG images in `output/frames/`
2. **Annotated Video**: `annotated_video.mp4` with object detection bounding boxes
3. **Optical Flow Video**: `optical_flow_prediction.mp4` with motion analysis and trajectory prediction
4. **Results JSON**: `pipeline_results.json` with processing statistics

Example `pipeline_results.json`:
```json
{
  "video_path": "data/sample.mp4",
  "output_dir": "output",
  "frames_extracted": 0,
  "detection_success": true,
  "optical_flow_success": true
}
```

## Technical Details

### Video Processing
- Supports multiple video formats (MP4, AVI, MOV, MKV)
- Frame extraction with configurable skip rate
- Proper resource management with context managers

### Object Detection
- YOLOv8 integration via Ultralytics
- Configurable confidence and IOU thresholds
- Automatic model downloading
- Bounding box visualization

### Optical Flow
- Farneback dense optical flow algorithm
- Motion magnitude thresholding
- Contour-based object tracking
- Dual prediction methods:
  - **Linear Regression**: Fast, simple trajectory extrapolation
  - **Kalman Filter**: Smoothed predictions with noise filtering

### Trajectory Prediction
- **Linear Regression**: Projects future positions based on historical trajectory
- **Kalman Filter**: 4-state filter (position + velocity) for robust tracking

## Dependencies

Core libraries:
- `opencv-python`: Video processing and computer vision
- `ultralytics`: YOLOv8 object detection
- `numpy`: Numerical computations
- `scikit-learn`: Linear regression for trajectory prediction
- `torch` & `torchvision`: Deep learning backend
- `matplotlib`: Visualization support

See `requirements.txt` for complete list.

## Future Enhancements

Potential improvements:
- [ ] Multi-object tracking
- [ ] Real-time processing support
- [ ] GPU acceleration options
- [ ] Support for additional optical flow algorithms
- [ ] Export trajectory data to CSV/JSON
- [ ] Web-based visualization dashboard
- [ ] Docker containerization
- [ ] Unit and integration tests

## Troubleshooting

**Video won't open:**
- Verify the video file exists and is not corrupted
- Check that the format is supported (MP4, AVI, MOV, MKV)

**YOLO model download fails:**
- Ensure internet connectivity
- Model will auto-download on first use

**Out of memory:**
- Process videos at lower resolution
- Increase frame skip rate
- Process shorter video segments

**No motion detected:**
- Adjust `MOTION_THRESHOLD` in settings.py
- Check video has actual motion
- Verify camera is not completely static

## Contributing

Contributions are welcome! Areas for contribution:
- Additional optical flow algorithms
- Performance optimizations
- Additional trajectory prediction methods
- Documentation improvements
- Bug fixes

## License

This project is part of the Kiti autonomous vehicle prototype pipeline.

## Authors

Kiti Team - Autonomous Vehicle Computer Vision

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- Scikit-learn developers
