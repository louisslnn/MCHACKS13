# HoloRay Engine

**Real-Time Object Tracking & Annotation SDK for AR/VR Applications**

McHacks 13 HoloXR Challenge - Proof of Concept Demo

---

## Quick Start

### Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the interactive demo
python main_demo.py
```

### Basic Usage

```bash
# Use webcam (default)
python main_demo.py

# Use specific webcam index
python main_demo.py --source 1

# Use video file
python main_demo.py --source video.mp4

# Disable GPU (CPU fallback)
python main_demo.py --no-gpu

# Change annotation style
python main_demo.py --style gaming
```

---

## Overview

HoloRay Engine is a computer vision SDK that tracks objects in real-time video and renders sticky annotations that persist through:

- **Camera movement** - Annotations follow objects as camera moves
- **Object occlusion** - Labels fade when objects are blocked (e.g., hand in front)
- **Frame re-entry** - Objects are re-identified when they return to frame

### Key Features

✅ **Zero-latency video capture** - Threaded pipeline for real-time performance  
✅ **GPU-accelerated tracking** - CoTracker3 for high-speed point tracking  
✅ **Smart occlusion detection** - Automatically detects when objects are blocked  
✅ **Re-identification** - Finds objects when they re-enter the frame  
✅ **Multiple annotation styles** - Minimal, Standard, Detailed, Gaming  
✅ **CPU fallback** - Optical flow tracking when GPU unavailable  

---

## Interactive Demo Controls

- **LEFT CLICK** - Add tracker at cursor position
- **RIGHT CLICK** - Remove nearest tracker (within 100px)
- **R** - Reset all trackers
- **S** - Cycle annotation styles (minimal → standard → detailed → gaming)
- **Q / ESC** - Quit

---

## Project Structure

```
MCHACKS13/
├── main_demo.py              # Interactive demo application
├── requirements.txt          # Python dependencies
├── src/
│   └── holoray/
│       ├── holoray_core.py       # Object tracking engine
│       ├── video_pipeline.py     # Threaded video capture
│       └── annotation_layer.py   # Annotation rendering
└── docs/                     # Documentation
```

---

## Technology Stack

- **Computer Vision:** OpenCV, PyTorch
- **Tracking:** CoTracker3 (GPU) / Lucas-Kanade Optical Flow (CPU)
- **Feature Matching:** ORB descriptors for re-identification
- **Python:** 3.8+

---

## Documentation

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Technical design and pipeline flow
- **[MODULES.md](./MODULES.md)** - API reference and module documentation

---

## Use Case

Originally designed for **Checkmate AR** (VR Chess application), the engine can track chess pieces and maintain annotations as players move pieces, move the camera, or temporarily block pieces with their hands.

**Status:** Proof of Concept - Demonstrates core tracking capabilities with interactive demo.

---

## License

McHacks 13 Submission - HoloXR Challenge
