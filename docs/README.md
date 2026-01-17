# HoloXR Motion Tracker (Hybrid Engine)

**McHacks 13 HoloXR Challenge Submission**

A real-time medical video annotation system that dynamically tracks anatomical structures across video frames, solving the critical problem of static annotations on moving medical imagery.

---

## Quick Start

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google API Key (for Gemini Vision):**
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

3. **Run the pipeline:**
   ```bash
   python main.py --data-dir ./data
   ```

### Basic Usage

Process all videos in the data directory:
```bash
python main.py --data-dir ./data
```

Process a single video:
```bash
python main.py --video ./data/Echo/echo1.mp4 --folder-hint Echo
```

Disable live preview (headless mode):
```bash
python main.py --data-dir ./data --no-display
```

---

## Project Goal

### The Problem

Traditional medical image annotation systems place static labels that don't follow anatomical structures as they move during:
- **Ultrasound examinations** (probe movement, breathing motion, cardiac cycles)
- **Laparoscopic surgeries** (organ manipulation, camera movement, tool interaction)
- **Real-time imaging** (where frame-by-frame re-annotation is impractical)

This creates confusing and potentially misleading overlays that can compromise clinical interpretation.

### Our Solution

The **HoloXR Motion Tracker** uses a hybrid AI approach to:

1. **Detect** the imaging modality and identify clinically significant regions of interest (ROI) using Google Gemini 2.0 Flash Vision API
2. **Route** to the optimal tracking algorithm based on modality:
   - **CoTracker3** for Ultrasound (handles speckle noise and texture motion)
   - **SAM 2** for Laparoscopy (handles occlusion and deformation)
3. **Track** the ROI across video frames with sub-pixel accuracy
4. **Annotate** with smooth, persistent labels that follow anatomical structures in real-time

---

## Current Status

### ✅ Supported Modalities

- **Ultrasound** (via CoTracker3)
  - Echocardiography
  - POCUS (Point-of-Care Ultrasound)
  - Obstetric ultrasound
  - Abdominal ultrasound

- **Laparoscopy** (via SAM 2)
  - Cholecystectomy
  - General surgical procedures

### 🎯 Features

- **Real-time processing** with performance metrics (FPS, latency)
- **Automatic modality detection** using vision AI
- **Hybrid tracking engine** that switches models based on image characteristics
- **Kalman filtering** for smooth label positioning
- **Confidence-aware visualization** (color-coded by tracking quality)
- **Robust error handling** with automatic fallbacks

### 📊 Output

Processed videos are saved to the `./output/` directory with format:
```
{Modality}_{VideoName}_annotated.mp4
```

Example: `Echo_echo3_annotated.mp4`

---

## Project Structure

```
MCHACKS13/
├── main.py              # Main orchestration pipeline
├── server.py            # WebRTC streaming server (bonus feature)
├── requirements.txt     # Python dependencies
├── data/                # Input video directory
│   ├── Echo/
│   ├── Lapchole/
│   └── POCUS/
├── output/              # Annotated video output
├── models/              # Model checkpoints (SAM 2, CoTracker3)
└── src/
    ├── vision_agent.py      # Gemini Vision API interface
    ├── tracker_factory.py   # Model routing logic
    └── utils.py             # Kalman filter, rendering, utilities
```

---

## Documentation

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Technical deep-dive into the pipeline design
- **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** - Known issues and solutions
- **[MODULES.md](./MODULES.md)** - Developer reference for codebase modules

---

## Technology Stack

- **Vision AI:** Google Gemini 2.0 Flash Experimental
- **Tracking Models:**
  - CoTracker3 (Meta Research) - Point-based tracking
  - SAM 2 (Meta Research) - Mask-based segmentation
- **Framework:** PyTorch, OpenCV
- **Streaming:** WebRTC (optional bonus feature)

---

## License

McHacks 13 Submission - HoloXR Challenge

---

**Built with ❤️ for improving medical imaging workflows**
