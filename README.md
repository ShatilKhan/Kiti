# Kiti - Autonomous Vehicle Computer Vision Pipeline

Kiti is a **production-ready Python project** for autonomous vehicle computer vision tasks, combining **Area Marking**, **Object Detection**, and **Optical Flow** analysis with trajectory prediction.

## Overview

This project provides a complete pipeline for autonomous vehicle perception by:
- **Area Marking & ROI Detection** - Define central 40% region of interest with motion-based obstacle detection
- **Object Detection** - YOLO-based real-time object detection and classification
- **Motion Analysis** - Optical flow for tracking moving objects
- **Trajectory Prediction** - Linear Regression and Kalman Filter methods for path forecasting

## Features

- **Complete Vision Pipeline**: From video input to trajectory prediction
- **Modular Python Architecture**: Clean separation of concerns with dedicated modules
- **Area Marking Module**: ROI definition with background subtraction for obstacle detection
- **YOLO Integration**: State-of-the-art object detection using Ultralytics YOLOv8
- **Optical Flow Analysis**: Motion detection using Farneback algorithm
- **Dual Prediction Methods**: Linear Regression and Kalman Filter
- **CLI Support**: Full command-line interface for all operations

## Project Structure

```
Kiti/
├── src/                          # Python source modules
│   ├── __init__.py              
│   ├── main.py                  # Main pipeline orchestration (AutonomousVehiclePipeline)
│   ├── video_processor.py       # Video frame extraction (VideoProcessor)
│   ├── area_marking.py          # ROI & obstacle detection (AreaMarker)
│   ├── object_detection.py      # YOLO object detection (ObjectDetector)
│   └── optical_flow.py          # Optical flow analysis (OpticalFlowAnalyzer)
├── config/
│   ├── __init__.py              
│   └── settings.py              # Centralized configuration
├── data/                        # Input data directory
├── output/                      # Output directory
├── requirements.txt             # Python dependencies
├── run.py                       # CLI entry point
├── setup.py                     # Package setup
├── examples.py                  # Usage examples
├── validate.py                  # Project validation
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ShatilKhan/Kiti.git
cd Kiti
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

### Command Line Interface

```bash
# Show help
python run.py --help

# Run full pipeline (area marking + YOLO detection + optical flow)
python run.py --video data/your_video.mp4 --full

# Run area marking with obstacle detection only
python run.py --video path/to/video.mp4 --area-marking

# Run YOLO object detection only
python run.py --video path/to/video.mp4 --detect

# Run optical flow analysis only
python run.py --video path/to/video.mp4 --flow

# Run optical flow with Kalman filter
python run.py --video path/to/video.mp4 --flow --kalman

# Extract frames only
python run.py --video path/to/video.mp4 --extract-frames
```

### Python API

```python
from src.main import AutonomousVehiclePipeline
from src.area_marking import AreaMarker
from src.object_detection import ObjectDetector
from src.optical_flow import OpticalFlowAnalyzer

# Create full pipeline
pipeline = AutonomousVehiclePipeline('path/to/video.mp4', output_dir='output/')

# Run full pipeline
results = pipeline.run_full_pipeline(
    run_area_marking=True,
    run_detection=True,
    run_flow=True,
    use_kalman=True
)

print(results)

# Or use individual modules
from src.video_processor import VideoProcessor

# Area marking example
area_marker = AreaMarker(roi_start_ratio=0.3, roi_end_ratio=0.7)
with VideoProcessor('video.mp4') as vp:
    frames, obstacles = area_marker.process_video(vp, 'output/marked.mp4')

# Object detection example
detector = ObjectDetector()
detector.load_model()
detections = detector.detect(frame)

# Optical flow example
analyzer = OpticalFlowAnalyzer()
processed_frame, results = analyzer.process_frame_with_prediction(frame, use_kalman=True)
```

## Configuration

Edit `config/settings.py` to customize:

- **Paths**: Default data and output directories
- **YOLO Parameters**: Model path, confidence thresholds
- **Optical Flow**: Algorithm parameters
- **Output Settings**: Video codec, FPS, format options

## Dependencies

Core libraries:
- `opencv-python`: Video processing and computer vision
- `ultralytics`: YOLOv8 object detection
- `numpy`: Numerical computations
- `scikit-learn`: Linear regression for trajectory prediction
- `torch` & `torchvision`: Deep learning backend
- `matplotlib`: Visualization support

See `requirements.txt` for the complete list.

## Pipeline Components

### 1. Area Marking
Defines a central 40% region of interest (ROI) for prioritized obstacle detection. Objects within this zone trigger alerts.

### 2. Motion Detection
Uses background subtraction (MOG2) for detecting moving objects without deep learning.

### 3. YOLO Object Detection
Accurate object detection with class labels using YOLOv5/YOLOv8.

### 4. Optical Flow Analysis
Farneback dense optical flow for motion tracking between frames.

### 5. Trajectory Prediction
- **Linear Regression**: Fast, simple trajectory extrapolation
- **Kalman Filter**: Smoothed predictions with noise filtering

## Output

The pipeline generates:
- **Annotated Video**: Video with detection bounding boxes
- **Optical Flow Video**: Motion analysis visualization
- **Trajectory Prediction**: Path forecasting visualization
- **Results JSON**: Processing statistics

## Contributing

Contributions are welcome! Areas for contribution:
- Additional optical flow algorithms
- Performance optimizations
- Real-time processing support
- Documentation improvements

## License

MIT License

## Authors

Kiti Team - Autonomous Vehicle Computer Vision
