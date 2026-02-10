# Navirox Radar Tester

**NaviUI - Autonomous Navigation System Dashboard**

A sophisticated PyQt6-based maritime radar visualization and object detection testing platform with a dark-themed command interface.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Components](#components)
- [Model Pipeline](#model-pipeline)
- [Configuration](#configuration)
- [Development](#development)
- [Author](#author)

---

## 🌊 Overview

Navirox Radar Tester is a comprehensive maritime autonomous navigation system designed for radar visualization, object detection, tracking, and threat analysis. The system provides a real-time tactical map interface with GIS-based polar coordinate support, dynamic obstacle generation, and multi-stage AI model pipeline for detecting and tracking maritime objects such as vessels, persons, debris, and buoys.

### Key Capabilities

- **Real-time Radar Visualization**: Interactive tactical map with polar coordinates and configurable range rings
- **Object Detection & Tracking**: Multi-stage YOLO-based detection pipeline with ByteTrack and BoT-SORT tracking
- **Multi-Camera Support**: 4-camera grid (Forward, Aft, Port, Starboard) with zoom controls
- **Dynamic Obstacle Management**: Real-time obstacle generation, classification, and visualization
- **GIS Integration**: Topographical and satellite map overlays with coordinate conversion
- **Picture-in-Picture (PIP)**: Detailed obstacle view with camera imagery and telemetry
- **System Monitoring**: Live console logging and status displays

---

## ✨ Features

### 🆕 Sensor Data Fusion (NEW!)
- **Camera + Lidar Fusion**: Combines visual detection with precise ranging
- **Two Data Modes**: 
  - **Dummy Mode**: Simulated obstacles for testing
  - **Real Mode**: Live camera and lidar data integration
- **Intelligent Matching**: Automatic temporal and spatial data alignment
- **Three Fusion Strategies**: Matched, camera-only, and lidar-only detections
- **Persistent Tracking**: Maintains object IDs across frames
- **See**: [`docs/DATA_FUSION_GUIDE.md`](docs/DATA_FUSION_GUIDE.md) for integration details

### Radar & Visualization
- **Polar Coordinate System**: Converts between polar (angle, distance) and screen coordinates
- **Configurable Radar Parameters**:
  - Radar height (0.5-100m)
  - Start/End range (0-5000m)
  - Beam angle (±0-90°)
  - Azimuth (10-360°)
  - Transparency controls
- **Multiple Overlay Modes**:
  - Topographical view with depth/relief
  - Satellite imagery
  - Heatmap overlays (thermal, sonar, depth)
- **Dynamic Range Rings**: Auto-adjusting concentric circles with distance labels
- **Radar Sweep Visualization**: Animated sweep cone with configurable beam width

### Object Detection & Tracking
- **6 Obstacle Types Supported**:
  - BOAT (Blue, #29B6F6)
  - PERSON (Red, #FF1744)
  - DEBRIS (Orange, #FF9100)
  - VESSEL (Green, #00E676)
  - BUOY (Yellow, #FFEB3B)
  - UNKNOWN (Gray, #9E9E9E)
- **Multi-Stage Detection Pipeline**:
  - Stage 1: General object detection (YOLO)
  - Stage 2: Depth estimation
  - Stage 3: Direction estimation (RAFT)
- **Advanced Tracking**:
  - ByteTrack (fast, real-time)
  - BoT-SORT (robust, accurate)
  - Persistent tracking across frames
  - Global ID assignment

### User Interface
- **3-Panel Layout**:
  - **Left Panel**: Camera grid, radar controls, sensor settings
  - **Center Panel**: Tactical map with coordinate display
  - **Right Panel**: System status and console logs
- **Interactive Elements**:
  - Click-to-inspect obstacles
  - Real-time camera feed integration
  - Draggable PIP window
  - Custom toggle switches and sliders
- **Dark Theme**: Professional maritime command center aesthetic

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Navirox Radar Tester                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Left Panel  │  │ Center Panel │  │ Right Panel  │        │
│  │             │  │              │  │              │        │
│  │ • Cameras   │  │ • Radar Map  │  │ • Status     │        │
│  │ • Controls  │  │ • Obstacles  │  │ • Console    │        │
│  │ • Settings  │  │ • PIP View   │  │ • Logs       │        │
│  └─────────────┘  └──────────────┘  └──────────────┘        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    Services Layer                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Model Pipeline (Cascade Processing)          │   │
│  │                                                      │   │
│  │  Stage 1 → Stage 2 → Stage 3                         │   │
│  │  Detection  Depth     Direction                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐    │
│  │   Trackers   │  │  Annotation   │  │  Data Loaders  │    │
│  │              │  │   Renderers   │  │                │    │
│  │ • ByteTrack  │  │ • Detection   │  │ • Detection    │    │
│  │ • BoT-SORT   │  │ • Direction   │  │ • Loader       │    │
│  └──────────────┘  └───────────────┘  └────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- PyQt6
- PyTorch (for YOLO models)
- CUDA (optional, for GPU acceleration)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd navirox-radar-tester
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- PyQt6
- ultralytics (YOLO)
- torch
- numpy
- opencv-python (cv2)

### Step 3: Download Model Weights

Place YOLO model weights in the appropriate directories:
- `services/model/cfgs/stage1/yolo11n.pt` (or your custom model)

---

## 🚀 Usage

### Running the Application

```bash
python main.py
```

### Basic Operations

#### 1. Radar Configuration
- **Adjust Radar Height**: Left Panel → Radar Controls → Radar Height (affects zoom)
- **Set Range**: Configure Start Range and End Range to define detection boundaries
- **Beam Angle**: Adjust the radar sweep cone width (±beam angle)
- **Azimuth**: Set the radar sweep center direction

#### 2. Camera Controls
- **Select Camera**: Click any of the 4 camera cells (FWD, AFT, PORT, STBD)
- **Zoom**: Use the zoom slider to adjust camera magnification

#### 3. Viewing Obstacles
- **Auto-Detection**: Obstacles are generated automatically every 3 seconds
- **Inspect**: Click on any obstacle marker to open the PIP window
- **PIP Window**: Shows obstacle details, camera view, and telemetry data

#### 4. Map Overlays
- **Heatmap**: Select from Thermal, Sonar, Depth, or Weather overlays
- **Topographical View**: Toggle "Curvature of Earth" for 3D terrain visualization
- **Transparency**: Adjust overlay opacity with the transparency slider

---

## 📁 Project Structure

```
navirox-radar-tester/
│
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
│
├── constants/                       # Application constants
│   ├── color.py                     # Color definitions
│   └── detections_constant.py       # Detection keys and constants
│
├── naviui/                          # UI components (PyQt6)
│   ├── __init__.py                  # Package initialization
│   ├── app.py                       # Main window and layout
│   ├── styles.py                    # Dark theme stylesheet
│   │
│   ├── panels/                      # UI panels
│   │   ├── header.py                # Top header bar
│   │   ├── left_panel.py            # Camera grid & radar controls
│   │   ├── center_panel.py          # Tactical map view
│   │   └── right_panel.py           # Status & console
│   │
│   ├── scenes/                      # Graphics scenes
│   │   └── tactical_map.py          # Radar visualization scene
│   │
│   ├── widgets/                     # Custom widgets
│   │   ├── camera_cell.py           # Camera selector widget
│   │   ├── heatmap_row.py           # Heatmap overlay selector
│   │   ├── pip_window.py            # Picture-in-Picture overlay
│   │   └── toggle_switch.py         # Custom toggle switch
│   │
│   └── utils/                       # UI utilities
│       └── pixmap_helpers.py        # Map generation functions
│
└── services/                        # Backend services
    ├── common/
    │   └── models/
    │       └── pipe_structure.py    # Pipeline structure models
    │
    ├── loaders/                     # Data loaders
    │   ├── idata_loader.py          # Data loader interface
    │   └── detection_data_loader.py # Detection data loader
    │
    ├── managers/                    # Management services
    │   ├── color_manager.py         # Color assignment manager
    │   ├── model_strategy_manager.py # Model selection strategy
    │   ├── tracker_factory.py       # Tracker instantiation
    │   └── tracker_manager.py       # Tracker lifecycle management
    │
    ├── model/                       # AI model pipeline
    │   ├── cfgs/
    │   │   ├── ibase_stage.py       # Base stage interface
    │   │   ├── model_pipeline.py    # Cascade pipeline
    │   │   │
    │   │   ├── stage1/              # Object detection
    │   │   │   ├── general_object_detection_detector.py
    │   │   │   ├── general_object_detection_tracker.py
    │   │   │   └── test/
    │   │   │
    │   │   ├── stage2/              # Depth estimation
    │   │   │   ├── depth_estimation_stage2.py
    │   │   │   └── test/
    │   │   │
    │   │   └── stage3/              # Direction estimation
    │   │       └── raft_direction_estimation_stage3.py
    │   │
    │   └── test/
    │       └── run_calculate_distance.py
    │
    ├── trackers/                    # Object tracking
    │   ├── itracker.py              # Tracker interface
    │   ├── base_tracker.py          # Base tracker implementation
    │   └── yolo_trackers.py         # ByteTrack & BoT-SORT
    │
    └── visualization/               # Annotation renderers
        ├── iannotation_renderer.py  # Renderer interface
        ├── master_annotation_renderer.py # Base renderer
        ├── detection_annotation_renderer.py
        └── direction_annotation_renderer.py
```

---

## 🔧 Components

### UI Components

#### TacticalMapScene (`naviui/scenes/tactical_map.py`)
The core radar visualization component with:
- **Polar coordinate system** for angle/distance calculations
- **Dynamic obstacle management** with auto-generation and cleanup
- **Interactive obstacle markers** with click detection
- **Configurable radar rings** and sweep visualization
- **Background map rendering** (topographical or satellite)
- **Coordinate overlay grid** with latitude/longitude markers

**Key Methods:**
- `polar_to_screen(angle, distance)`: Convert polar to screen coordinates
- `update_radar_height(height)`: Adjust radar zoom level
- `update_range(start, end)`: Modify detection range
- `update_angles(beam, azimuth)`: Configure sweep cone
- `update_heatmap(color)`: Change overlay color
- `update_topographical_view(enabled)`: Toggle 3D terrain view

#### PIP Window (`naviui/widgets/pip_window.py`)
Floating overlay displaying obstacle details:
- **Camera imagery** fetched from remote URLs
- **Obstacle classification** with color-coded labels
- **Telemetry data**: Distance, bearing, speed
- **Threat level indicators** (Safe/Caution/Danger)
- **Auto-hide timer** (dismisses after 15 seconds)

### Service Components

#### Model Pipeline (`services/model/cfgs/model_pipeline.py`)
Cascade processing pipeline supporting:
- **Sequential stage execution** with dependency management
- **Result aggregation** across multiple models
- **Dynamic model loading** via importlib
- **Configurable model order** using PipeStructure

**Pipeline Flow:**
```python
Stage 1 (Detection) → Stage 2 (Depth) → Stage 3 (Direction)
```

#### Trackers (`services/trackers/`)

**ByteTrack:**
- Fast, real-time object tracking
- Low computational overhead
- Suitable for embedded systems

**BoT-SORT:**
- More robust tracking with re-identification
- Better handling of occlusions
- Higher accuracy, slightly slower

#### Annotation Renderers (`services/visualization/`)
Render bounding boxes, labels, and tracking IDs on frames:
- **Color-coded bounding boxes** per object type
- **Tracking ID display** (T_xxx, G_xxx)
- **Confidence scores** and class labels
- **Region annotations** for zone-based detection
- **Timestamp overlays**

---

## 🤖 Model Pipeline

### Stage 1: General Object Detection
**File:** `services/model/cfgs/stage1/general_object_detection_detector.py`

- **Model**: YOLO11n (or custom YOLO variant)
- **Purpose**: Detect maritime objects (boats, persons, debris, vessels)
- **Output**: Bounding boxes with class labels and confidence scores

### Stage 2: Depth Estimation
**File:** `services/model/cfgs/stage2/depth_estimation_stage2.py`

- **Purpose**: Estimate distance/depth of detected objects
- **Dependency**: Requires Stage 1 detection results
- **Output**: Depth maps or distance values

### Stage 3: Direction Estimation (RAFT)
**File:** `services/model/cfgs/stage3/raft_direction_estimation_stage3.py`

- **Purpose**: Estimate movement direction and velocity
- **Method**: RAFT (Recurrent All-Pairs Field Transforms) optical flow
- **Output**: Direction vectors and speed estimates

### Pipeline Configuration

```python
from services.model.cfgs.model_pipeline import ModelPipeline
from services.common.models.pipe_structure import PipeStructure

# Define stages
stage1 = PipeStructure(
    order=1,
    model_id="detector",
    model=GeneralObjectDetectorStage1(...),
    lead_by=None
)

stage2 = PipeStructure(
    order=2,
    model_id="depth",
    model=DepthEstimationStage2(...),
    lead_by="detector"
)

stage3 = PipeStructure(
    order=3,
    model_id="direction",
    model=RAFTDirectionEstimationStage3(...),
    lead_by="depth"
)

# Create pipeline
pipeline = ModelPipeline([stage1, stage2, stage3])

# Run inference
results = pipeline(image)
```

---

## ⚙️ Configuration

### Radar Settings

Edit default values in `naviui/scenes/tactical_map.py`:

```python
self.radar_height = 4.5      # meters
self.start_range = 50         # meters
self.end_range = 500          # meters
self.beam_angle = 25          # degrees (±)
self.azimuth = 120            # degrees
self.transparency = 70        # percentage
self.meters_per_pixel = 2.0   # scale factor
```

### Obstacle Generation

Modify obstacle spawn rate in `naviui/scenes/tactical_map.py`:

```python
self.obstacle_timer.start(3000)   # milliseconds (3 seconds)
self.cleanup_timer.start(8000)    # milliseconds (8 seconds)
```

### Color Scheme

Update obstacle colors in `naviui/scenes/tactical_map.py`:

```python
OBSTACLE_TYPES = [
    {"type": "BOAT", "color": "#29B6F6", "size": 16},
    {"type": "PERSON", "color": "#FF1744", "size": 14},
    # Add or modify types...
]
```

---

## 🛠 Development

### Adding a New Detection Stage

1. **Create Stage Class** in `services/model/cfgs/stageX/`:

```python
from services.model.cfgs.ibase_stage import BaseStage

class CustomStage(BaseStage):
    def __init__(self, model_path, model_id, device=None):
        super().__init__(model_id)
        # Initialize model
    
    def forward(self, image, prev_results=None):
        # Process image and prev_results
        return detections
```

2. **Register in Pipeline**:

```python
custom_stage = PipeStructure(
    order=4,
    model_id="custom",
    model=CustomStage(...),
    lead_by="direction"
)
```

### Extending UI Components

Add custom widgets to `naviui/widgets/`:

```python
from PyQt6.QtWidgets import QWidget

class CustomWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Widget implementation
```

### Custom Annotation Renderers

Implement `IAnnotationRenderer` in `services/visualization/`:

```python
from services.visualization.master_annotation_renderer import MasterAnnotationRenderer

class CustomRenderer(MasterAnnotationRenderer):
    def render(self, frame, detection, regions, color_manager, **kwargs):
        # Custom rendering logic
        return frame
```

---

## 👨‍💻 Author

**Shaikh Azan Asim**

Version: 1.0.0

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **PyQt6**: Cross-platform GUI framework
- **Ultralytics YOLO**: State-of-the-art object detection
- **ByteTrack/BoT-SORT**: Advanced object tracking algorithms
- **RAFT**: Optical flow for motion estimation

---

## 📞 Support

For issues, questions, or contributions, please contact the development team or open an issue in the repository.

---

## 🔮 Future Enhancements

- [ ] Real camera feed integration
- [ ] AIS (Automatic Identification System) integration
- [ ] Historical track playback
- [ ] Multi-vessel collision prediction
- [ ] Weather overlay with real-time data
- [ ] Export detection logs to CSV/JSON
- [ ] Remote monitoring dashboard
- [ ] WebSocket API for external integrations

---

**Happy Navigation! ⚓🌊**
