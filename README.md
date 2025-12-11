# Kiti - Autonomous Vehicle Computer Vision Pipeline

Kiti is a production-ready Python project for autonomous vehicle computer vision tasks, combining **Area Marking**, **Object Detection**, and **Optical Flow** analysis with trajectory prediction.

## Overview

This project provides a complete pipeline for autonomous vehicle perception by:
- **Defining a Region of Interest (ROI)** - Central 40% area marking for prioritized obstacle detection
- **Object Detection** - YOLO-based real-time object detection and classification
- **Motion Analysis** - Optical flow for tracking moving objects
- **Trajectory Prediction** - Linear Regression and Kalman Filter methods for path forecasting

## Features

- **Complete Vision Pipeline**: From video input to trajectory prediction
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **YOLO Integration**: State-of-the-art object detection using Ultralytics YOLOv8
- **Optical Flow Analysis**: Motion detection using Farneback algorithm
- **Dual Prediction Methods**: Linear Regression and Kalman Filter
- **Jupyter Notebook**: Interactive notebook for experimentation (`Kiti_Autonomous_Vehicle.ipynb`)
- **CLI Support**: Easy-to-use command-line interface

## Project Structure

```
Kiti/
├── src/                          # Python source modules
│   ├── __init__.py              
│   ├── main.py                  # Main pipeline orchestration
│   ├── video_processor.py       # Video frame extraction
│   ├── object_detection.py      # YOLO object detection
│   └── optical_flow.py          # Optical flow analysis
├── config/
│   ├── __init__.py              
│   └── settings.py              # Centralized configuration
├── data/                        # Input data directory
├── output/                      # Output directory
├── Kiti_Autonomous_Vehicle.ipynb # Combined notebook
├── Area_Marking.ipynb           # Original area marking notebook
├── Kiti_Optical_flow.ipynb      # Original optical flow notebook
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

### Jupyter Notebook (Recommended for experimentation)

Open `Kiti_Autonomous_Vehicle.ipynb` in Jupyter or Google Colab for an interactive experience.

### Command Line Interface

```bash
# Show help
python run.py --help

# Run full pipeline
python run.py --video data/your_video.mp4 --full

# Extract frames only
python run.py --video path/to/video.mp4 --extract-frames

# Run object detection
python run.py --video path/to/video.mp4 --detect

# Run optical flow with Kalman filter
python run.py --video path/to/video.mp4 --flow --kalman
```

### Python API

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
