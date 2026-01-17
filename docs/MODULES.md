# Module Reference

Developer guide to the codebase modules and their responsibilities.

---

## `main.py` - Orchestration Loop

**Purpose:** Main entry point and pipeline orchestration.

**Key Responsibilities:**
- Discovers videos from directory structure
- Manages the processing loop (frame-by-frame)
- Coordinates between Vision Agent, Tracker Factory, and Renderer
- Handles video I/O (reading input, writing output)
- Manages Kalman filtering for smooth positioning
- Implements re-query logic on tracking failure
- Provides CLI interface with argparse

**Key Classes:**
- `MedicalAnnotationPipeline`: Main pipeline orchestrator
- `VideoTask`: Dataclass representing a video processing task

**Key Methods:**
- `discover_videos()`: Scans `data_dir` for video files
- `analyze_first_frame()`: Calls Vision Agent on first frame
- `process_video()`: Core processing loop for one video
- `process_all()`: Batch processing for all discovered videos

**Dependencies:**
- `vision_agent.py` - Modality detection and ROI identification
- `tracker_factory.py` - Tracker selection and initialization
- `utils.py` - Kalman filter, rendering, video utilities

**Example Usage:**
```python
from main import MedicalAnnotationPipeline

pipeline = MedicalAnnotationPipeline(
    data_dir="./data",
    output_dir="./output",
    use_mock_api=False,
    device="cuda"
)
results = pipeline.process_all()
```

---

## `src/vision_agent.py` - Gemini API Interface

**Purpose:** Interface to Google Gemini 2.0 Flash for medical image analysis.

**Key Responsibilities:**
- Converts OpenCV frames to PIL images for Gemini API
- Sends analysis prompts with folder hints
- Parses JSON responses with multiple extraction strategies
- Validates ROI coordinates (clamps to image bounds)
- Implements automatic model fallback (2.0-flash-exp → 1.5-flash → ...)
- Provides mock agent for testing without API key

**Key Classes:**
- `VisionAgent`: Real Gemini API interface
- `MockVisionAgent`: Fallback for testing (uses folder name heuristics)
- `VisionAnalysisResult`: Dataclass for analysis results
- `Modality`: Enum for imaging modalities (ULTRASOUND, LAPAROSCOPY, OTHER)

**Key Methods:**
- `analyze_frame(frame, folder_hint)`: Single frame analysis
- `analyze_frame_with_retry()`: Retry logic with exponential backoff
- `_parse_response()`: Multi-strategy JSON parsing (handles code blocks, braces, etc.)
- `_validate_response()`: Clamps coordinates and validates fields

**API Details:**
- **Model:** `gemini-2.0-flash-exp` (primary), with fallback chain
- **Prompt:** Structured JSON request for modality, ROI, label, confidence
- **Response Format:** JSON with `{modality, roi: {x, y}, label, confidence}`

**Example Usage:**
```python
from vision_agent import get_vision_agent

agent = get_vision_agent(use_mock=False)
result = agent.analyze_frame(frame, folder_hint="Echo")
print(f"Modality: {result.modality.value}, ROI: {result.roi}")
```

---

## `src/tracker_factory.py` - Model Routing Logic

**Purpose:** Factory pattern for creating and switching between tracking models based on modality.

**Key Responsibilities:**
- Checks availability of CoTracker3 and SAM 2
- Routes to appropriate tracker based on `Modality` enum
- Provides fallback to OpenCV trackers (CSRT, KCF, MIL)
- Encapsulates tracker initialization and lifecycle
- Documents reasoning for tracker selection

**Key Classes:**
- `TrackerFactory`: Factory for creating trackers
- `BaseTracker`: Abstract base class (interface)
- `CoTrackerWrapper`: Wrapper for CoTracker3 (point tracking)
- `SAM2Wrapper`: Wrapper for SAM 2 (mask tracking)
- `FallbackTracker`: OpenCV-based tracker (CSRT, KCF, MIL)
- `TrackingResult`: Dataclass for tracking output

**Key Methods:**
- `create_tracker(modality)`: Factory method returning tracker instance
- `get_tracker_info(modality)`: Returns metadata about selected tracker
- `_get_reasoning(modality)`: Human-readable explanation for routing

**Tracker Selection Logic:**
```
Ultrasound → CoTracker3 (if available) → FallbackTracker (else)
Laparoscopy → SAM 2 (if available) → FallbackTracker (else)
Other → FallbackTracker
```

**Example Usage:**
```python
from tracker_factory import TrackerFactory
from vision_agent import Modality

factory = TrackerFactory(device="cuda")
tracker = factory.create_tracker(Modality.ULTRASOUND)
tracker.initialize(frame, roi_x=320, roi_y=240)
result = tracker.update(next_frame)
```

---

## `src/utils.py` - Utilities & Rendering

**Purpose:** Helper modules for filtering, rendering, and video utilities.

**Key Components:**

### `KalmanFilter2D`
**Purpose:** 2D Kalman filter for smoothing tracking positions.

**Features:**
- Constant velocity model (state: `[x, y, vx, vy]`)
- Configurable process/measurement noise
- Predicts position during tracking gaps

**Methods:**
- `update(measurement)`: Update with new measurement
- `predict()`: Predict next position without measurement
- `reset(position)`: Reset filter state

---

### `AnnotationRenderer`
**Purpose:** Renders medical annotations on video frames.

**Features:**
- Color-coded labels by confidence (green/yellow/red)
- Draws masks (SAM 2) or point grids (CoTracker)
- Displays performance metrics overlay
- Handles label positioning with frame-bound clamping

**Methods:**
- `draw_label()`: Main label rendering
- `draw_mask_overlay()`: SAM 2 mask visualization
- `draw_point_grid()`: CoTracker point visualization
- `draw_metrics()`: FPS, latency, tracker info overlay

---

### `FrameRateTracker`
**Purpose:** Tracks FPS and latency metrics.

**Methods:**
- `tick(latency_ms)`: Record frame timing
- `get_metrics()`: Returns `PerformanceMetrics` dataclass

---

### Video Utilities

**Functions:**
- `extract_first_frame(video_path)`: Extracts first frame as numpy array
- `get_video_info(video_path)`: Returns metadata (width, height, FPS, duration)

---

## `server.py` - WebRTC Streaming Server (Bonus)

**Purpose:** Optional WebRTC server for collaborative real-time viewing.

**Features:**
- Streams annotated video frames via WebRTC
- Web interface for viewing stream
- Real-time statistics (FPS, latency, frame count)
- Test pattern mode when no video specified

**Key Classes:**
- `WebRTCServer`: Main server class
- `AnnotatedVideoTrack`: WebRTC video track that processes frames

**Usage:**
```bash
python server.py --video ./data/Echo/echo1.mp4 --port 8080
# Open http://localhost:8080 in browser
```

**Dependencies:**
- `aiohttp`: Async web framework
- `aiortc`: WebRTC implementation
- `av`: Video frame handling

---

## Module Interactions

```
main.py
  ├── vision_agent.py (analyze first frame)
  ├── tracker_factory.py (get tracker)
  └── utils.py (Kalman filter, renderer, video I/O)

tracker_factory.py
  ├── vision_agent.py (Modality enum)
  └── CoTracker3/SAM2 (external libraries)

utils.py
  └── OpenCV, NumPy (standard libraries)
```

---

## Testing Individual Modules

### Test Vision Agent
```bash
cd src
python vision_agent.py
```

### Test Tracker Factory
```bash
cd src
python tracker_factory.py
```

### Test Utils
```bash
cd src
python utils.py
```

---

## Extension Guide

### Adding a New Tracker

1. **Implement `BaseTracker` interface:**
   ```python
   class MyTracker(BaseTracker):
       def initialize(self, frame, roi_x, roi_y): ...
       def update(self, frame) -> TrackingResult: ...
       def reset(self): ...
       @property
       def tracker_type(self) -> str: ...
   ```

2. **Register in `TrackerFactory.create_tracker()`:**
   ```python
   elif modality == Modality.MY_MODALITY:
       return MyTracker(self.device)
   ```

### Adding a New Modality

1. **Add to `Modality` enum in `vision_agent.py`**
2. **Update `Modality.from_string()` method**
3. **Add routing logic in `TrackerFactory`**
4. **Update Gemini prompt if needed**

---

**For more details, see inline code comments and docstrings.**
