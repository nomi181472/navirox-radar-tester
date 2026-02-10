# Code Architecture Overview

## Project Structure
This is a **NaviUI Autonomous Navigation System** - a PyQt6-based maritime radar and object detection dashboard that fuses radar data with computer vision (YOLO) detections.

---

## 🔗 File Connections & Data Flow

### 1. **Application Entry Point**
```
main.py
  └─> Initializes QApplication
  └─> Creates MainWindow from naviui/app.py
  └─> Applies dark theme stylesheet
```

### 2. **Main UI Architecture** (`naviui/`)

**naviui/app.py (MainWindow)**
- Creates 3-panel layout: `LeftPanel | CenterPanel | RightPanel`
- Connects radar control signals between left panel and center panel
- Central orchestrator of the UI

```
MainWindow
  ├─> LeftPanel (naviui/panels/left_panel.py)
  │    ├─> CameraCell widgets (4 cameras: FWD, AFT, PORT, STBD)
  │    ├─> Radar control sliders (zoom, height, range, beam angle, etc.)
  │    └─> HeatmapRow widgets for sensor visualization
  │
  ├─> CenterPanel (naviui/panels/center_panel.py) ⭐ CORE FUSION LOGIC
  │    ├─> TacticalMapScene (tactical_map.py) - Radar visualization
  │    ├─> FusionManager (services/managers/fusion_manager.py) - Data fusion
  │    ├─> InferenceService (services/inferenced_services/) - YOLO detections
  │    └─> PIPWindow (widgets/pip_window.py) - Detail overlay on click
  │
  └─> RightPanel (naviui/panels/right_panel.py)
       ├─> System status display
       └─> Live console logs with timestamps
```

---

## 3. **Data Fusion Pipeline** (The Heart of the System)

### **CenterPanel** orchestrates the fusion:

```
┌─────────────────────────────────────────────────┐
│          CenterPanel (center_panel.py)          │
│                                                 │
│  ┌──────────────────┐    ┌──────────────────┐  │
│  │ TacticalMapScene │───>│  Radar Detections│  │
│  │ (tactical_map.py)│    │  {rtrack_id,     │  │
│  │                  │    │   camera_id,     │  │
│  │ - Generates radar│    │   angle, dist}   │  │
│  │   detections     │    └─────────┬────────┘  │
│  │ - Displays on    │              │           │
│  │   polar grid     │              │           │
│  └──────────────────┘              │           │
│                                    │           │
│  ┌──────────────────┐              │           │
│  │ InferenceService │              │           │
│  │ (inference_      │    ┌─────────▼────────┐  │
│  │  service.py)     │───>│ FusionManager    │  │
│  │                  │    │ (managers/       │  │
│  │ - Dummy YOLO     │    │  fusion_manager  │  │
│  │   detections     │    │  .py)            │  │
│  │ - Runs every 3s  │    │                  │  │
│  │ - Provides       │    │ MATCHES by:      │  │
│  │   {track_id,     │    │ 1. camera_id     │  │
│  │    camera_id,    │    │ 2. timestamp     │  │
│  │    bbox, class}  │    │ 3. angle         │  │
│  └──────────────────┘    └─────────┬────────┘  │
│                                    │           │
│                          ┌─────────▼────────┐  │
│                          │ Fused Detection  │  │
│                          │ {rtrack_id,      │  │
│                          │  track_id,       │  │
│                          │  camera_id,      │  │
│                          │  angle, distance,│  │
│                          │  bbox, class,    │  │
│                          │  confidence}     │  │
│                          └─────────┬────────┘  │
│                                    │           │
│                          ┌─────────▼────────┐  │
│                          │ Update Tactical  │  │
│                          │ Map with Labels  │  │
│                          └──────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Key Connection Points:**
1. `TacticalMapScene.radar_detections_updated` signal → `FusionManager.update_radar_detections()`
2. `InferenceService.run_inference()` called by timer → `FusionManager.update_yolo_detections()`
3. `FusionManager.get_fused_detections()` → Updates `TacticalMapScene` markers with class labels

---

## 4. **Model Pipeline** (`services/model/`)

```
ModelPipeline (model_pipeline.py)
  └─> Orchestrates cascade of ML models
  └─> Uses PipeStructure (pipe_structure.py) to define:
      - Model execution order
      - Dependencies between stages
      - Region definitions (polygon/bbox/line)

Example pipeline stages:
  Stage 1: general_object_detection_detector.py → Object Detection (YOLO)
  Stage 1: general_object_detection_tracker.py → Object Tracking
  Stage 2: depth_estimation_stage2.py → Depth Estimation
  Stage 3: raft_direction_estimation_stage3.py → Direction Estimation
```

**Key Files:**
- `services/model/cfgs/ibase_stage.py` - Base interface for pipeline stages
- `services/common/models/pipe_structure.py` - Data models for pipeline configuration

---

## 5. **Visualization Services** (`services/visualization/`)

```
MasterAnnotationRenderer (master_annotation_renderer.py)
  ├─> Base class for all renderers
  ├─> Adds timestamp overlay
  └─> Uses ColorManager for consistent color schemes

├─> DetectionAnnotationRenderer - Draws bounding boxes
├─> DirectionAnnotationRenderer - Draws direction arrows
└─> IAnnotationRenderer - Interface contract
```

**Connection:** Used by inference services to annotate video frames with detections

---

## 6. **Manager Components** (`services/managers/`)

1. **FusionManager** (`fusion_manager.py`) ⭐
   - Matches radar detections with YOLO detections
   - Criteria: camera_id, timestamp (±6s), angle (±45°)
   - Outputs fused detection objects

2. **ColorManager** (`color_manager.py`)
   - Manages consistent color schemes for object classes

3. **TrackerManager** / **TrackerFactory** (`tracker_manager.py`, `tracker_factory.py`)
   - Creates and manages object trackers (YOLO-based)
   - Supports different tracker types: BoT-SORT, ByteTrack, etc.

4. **ModelStrategyManager** (`model_strategy_manager.py`)
   - Selects appropriate model strategy based on config

---

## 7. **Data Loaders** (`services/loaders/`)

```
IDataLoader (idata_loader.py) - Interface
  └─> DetectionDataLoader (detection_data_loader.py)
      - Loads and parses YOLO detection results
      - Converts to internal detection format
```

---

## 8. **Widget Components** (`naviui/widgets/`)

- **CameraCell** - Individual camera preview with status indicators
- **HeatmapRow** - Sensor heatmap visualization row
- **PIPWindow** - Picture-in-picture detail overlay (shows on radar marker click)
- **ToggleSwitch** - Custom toggle switch UI component

---

## 🔄 Complete Data Flow Example

**Scenario:** Object detected in camera view

```
1. TacticalMapScene generates radar detection
   └─> {rtrack_id: 1, camera_id: 2, angle: 45°, distance: 200m}
   └─> Emits radar_detections_updated signal

2. InferenceService.run_inference() runs (timer-based, every 3s)
   └─> Generates YOLO detection
   └─> {track_id: 5, camera_id: 2, bbox: [100,200,300,400], 
        class: "vessel-ship", confidence: 0.89}

3. FusionManager receives both streams
   └─> Matches by camera_id=2, similar angle (bbox center → angle)
   └─> Creates fused detection:
       {rtrack_id: 1, track_id: 5, camera_id: 2, 
        angle: 45°, distance: 200m, bbox: [...],
        class: "vessel-ship", confidence: 0.89}

4. CenterPanel receives fused detection
   └─> Updates TacticalMapScene marker with class label
   └─> Marker now shows "vessel-ship" instead of generic radar blip

5. User clicks marker on TacticalMapScene
   └─> Emits obstacle_clicked signal
   └─> CenterPanel shows PIPWindow with detailed info
```

---

## 🎨 UI Theme & Styling

- **styles.py** - Contains DARK_STYLESHEET with maritime-themed colors
- **Constants:**
  - `constants/color.py` - Color definitions
  - `constants/detections_constant.py` - Detection-related constants

---

## 🧪 Testing Structure

```
tests/ - Unit and integration tests
services/model/cfgs/stage1/test/ - Stage 1 model tests
services/model/cfgs/stage2/test/ - Stage 2 model tests
```

---

## 📦 Key Dependencies (from requirements.txt)

- **PyQt6** - UI framework
- **torch** - Deep learning models
- **ultralytics** - YOLO models
- **opencv-python** - Image processing
- **numpy** - Numerical computations

---

## 🎯 Critical Integration Points

1. **Radar → Fusion:** `TacticalMapScene.radar_detections_updated` signal
2. **YOLO → Fusion:** `InferenceService.run_inference()` → `FusionManager.update_yolo_detections()`
3. **Fusion → UI:** `FusionManager.get_fused_detections()` → `TacticalMapScene` marker updates
4. **UI Controls → Radar:** `LeftPanel` slider signals → `TacticalMapScene.update_radar_parameters()`
5. **User Interaction:** `TacticalMapScene.obstacle_clicked` → `PIPWindow.show()`

---

## 📝 Notes for Development

- Currently using **dummy data** in `InferenceService` and `FusionManager`
- Real YOLO model integration: Replace `InferenceService.run_inference()` body
- Real radar feed: Replace `TacticalMapScene._generate_random_obstacle()` with actual radar data stream
- Fusion thresholds tunable in `fusion_manager.py`: `MAX_TIME_DELTA_S`, `MAX_ANGLE_DELTA_DEG`
