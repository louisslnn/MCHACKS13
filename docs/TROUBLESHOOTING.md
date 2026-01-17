# Troubleshooting Guide

This document covers known issues, fixes, and solutions for the HoloXR Motion Tracker pipeline.

---

## Gemini API Issues

### Issue: 404 Not Found Errors

**Symptom:**
```
google.api_core.exceptions.NotFound: 404 models/gemini-1.5-flash not found
```

**Root Cause:**
The `gemini-1.5-flash` endpoint was deprecated or unavailable during development, returning 404 errors on API calls.

**Solution:**
We migrated to **`gemini-2.0-flash-exp`** (Gemini 2.0 Flash Experimental), which is:
- Faster than 1.5 Flash
- More reliable endpoint availability
- Better vision-language understanding

**Implementation:**
The `VisionAgent` class in `src/vision_agent.py` includes automatic model fallback:
```python
GEMINI_MODEL_PRIORITY = [
    "gemini-2.0-flash-exp",          # Primary
    "models/gemini-2.0-flash-exp",   # With prefix variant
    "gemini-1.5-flash-latest",       # Fallback
    ...
]
```

**How to Verify:**
Check logs for:
```
Successfully initialized model: gemini-2.0-flash-exp
```

---

### Issue: Missing API Key

**Symptom:**
```
ValueError: API key required. Set GOOGLE_API_KEY env var or pass api_key parameter.
```

**Solution:**
```bash
export GOOGLE_API_KEY="your-key-here"
```

**Alternative:**
Use mock mode for testing (no API calls):
```bash
python main.py --mock-api --data-dir ./data
```

---

## CoTracker Tensor Shape Mismatch

### Issue: Tensor Shape Mismatch Error

**Symptom:**
```
RuntimeError: Expected tensor of size [B, T, C, H, W] where H and W are divisible by 16
```

**Root Cause:**
CoTracker3 uses a convolutional backbone with stride=16. Input frames must have dimensions divisible by 16, otherwise tensor operations fail.

**Common Failure Cases:**
- Video resolution: 480x640 → ✅ (both divisible by 16)
- Video resolution: 480x637 → ❌ (637 % 16 ≠ 0)
- Cropped frames: 360x480 → ✅
- Cropped frames: 361x479 → ❌

### Solution: Explicit Padding (Not Resizing)

**Why Padding, Not Resizing?**
- **Resizing distorts medical images** → alters pixel ratios → clinical inaccuracy
- **Padding preserves original resolution** → no distortion → maintains medical validity

**Implementation:**
The `CoTrackerWrapper` in `src/tracker_factory.py` should pad frames before tensor conversion:

```python
def _pad_to_divisible(frame: np.ndarray, divisor: int = 16) -> Tuple[np.ndarray, int, int]:
    """Pad frame to be divisible by divisor, return padded frame and crop offsets."""
    h, w = frame.shape[:2]
    
    # Calculate padding needed
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    
    # Pad symmetrically (or use cv2.BORDER_REPLICATE for better edges)
    padded = cv2.copyMakeBorder(
        frame,
        pad_h // 2, pad_h - pad_h // 2,
        pad_w // 2, pad_w - pad_w // 2,
        cv2.BORDER_REPLICATE
    )
    
    return padded, pad_h // 2, pad_w // 2
```

**During Output:**
Crop the padded region from tracking results before rendering:
```python
# After tracking
track_result = tracker.update(padded_frame)

# Crop back to original dimensions
track_result.x -= pad_w_offset
track_result.y -= pad_h_offset
```

**Status:** This fix was implemented to ensure compatibility with all video resolutions.

---

## Model Loading Issues

### Issue: CoTracker3 Not Found

**Symptom:**
```
ImportError: cannot import name 'CoTrackerOnlinePredictor' from 'cotracker'
```

**Solution:**
Install CoTracker3 from source:
```bash
git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker
pip install -e .
```

Or via direct install:
```bash
pip install git+https://github.com/facebookresearch/co-tracker.git
```

---

### Issue: SAM 2 Checkpoint Missing

**Symptom:**
```
FileNotFoundError: models/sam2_hiera_small.pt not found
```

**Solution:**
1. Download SAM 2 checkpoint from [official repository](https://github.com/facebookresearch/segment-anything-2)
2. Place in `models/` directory:
   ```bash
   mkdir -p models
   # Download checkpoint to models/sam2_hiera_small.pt
   ```

Or specify custom checkpoint path:
```python
tracker = SAM2Wrapper(
    device="cuda",
    checkpoint="/path/to/sam2_hiera_small.pt"
)
```

---

## Performance Issues

### Issue: Slow Processing (CPU Mode)

**Symptom:**
- Processing speed: < 5 FPS
- High CPU usage
- GPU not detected

**Diagnosis:**
Check device:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

**Solution:**
1. Ensure CUDA is installed and PyTorch has CUDA support
2. Use fallback tracker for CPU-only systems:
   ```bash
   python main.py --fallback-tracker --data-dir ./data
   ```

---

### Issue: Out of Memory (OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Use smaller models:**
   - SAM 2 Small instead of Large
   - Reduce CoTracker grid size (default: 7x7 → try 5x5)

2. **Process shorter video clips**

3. **Use fallback tracker** (lowest memory):
   ```bash
   python main.py --fallback-tracker --data-dir ./data
   ```

---

## Video I/O Issues

### Issue: Cannot Open Video File

**Symptom:**
```
Failed to open video: data/Echo/echo1.mp4
```

**Solutions:**
1. **Check file path:**
   ```bash
   ls -la data/Echo/echo1.mp4
   ```

2. **Verify codec support:**
   - OpenCV requires codecs installed on system
   - Try converting video to H.264/MP4:
     ```bash
     ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
     ```

3. **Check file permissions:**
   ```bash
   chmod 644 data/Echo/echo1.mp4
   ```

---

## Tracking Quality Issues

### Issue: Labels Jump Around (Jittery)

**Symptom:**
- Labels rapidly change position between frames
- Visual jitter even when object is stationary

**Solution:**
Adjust Kalman filter parameters in `main.py`:
```python
kalman = KalmanFilter2D(
    process_noise=0.005,      # Lower = smoother (try 0.001)
    measurement_noise=0.1,    # Lower = more responsive (try 0.05)
    initial_position=(x, y)
)
```

---

### Issue: Tracking Lost Frequently

**Symptom:**
- Low confidence values (< 0.3)
- Labels defaulting to center position
- Frequent re-queries to Gemini API

**Solutions:**
1. **Lower confidence threshold:**
   ```bash
   python main.py --confidence-threshold 0.2 --data-dir ./data
   ```

2. **Enable re-query on failure** (default: enabled)
   - System automatically re-queries Gemini after 30 frames of low confidence

3. **Check video quality:**
   - Low resolution or blurry videos are harder to track
   - Try preprocessing with deblur or upscaling

---

## Logging & Debugging

### Enable Debug Logging

```bash
python main.py --log-level DEBUG --data-dir ./data
```

### View Model Initialization

Check startup logs for:
```
INFO - CoTracker3 is available
INFO - SAM 2 is available
INFO - Successfully initialized model: gemini-2.0-flash-exp
```

### Check Tracker Selection

Logs show routing decisions:
```
INFO - Creating CoTracker3 for Ultrasound modality
INFO - Reasoning: CoTracker3 selected: Ultrasound has speckle noise...
```

---

## Common Workarounds

### Testing Without Models

Use fallback tracker and mock API:
```bash
python main.py --mock-api --fallback-tracker --data-dir ./data
```

### Quick Test on Single Video

```bash
python main.py --video ./data/Echo/echo1.mp4 --folder-hint Echo --no-display
```

### Headless Server Mode

```bash
python main.py --data-dir ./data --no-display --log-level INFO
```

---

## Getting Help

1. **Check logs:** Enable `DEBUG` logging to see detailed error messages
2. **Verify dependencies:** Run `pip install -r requirements.txt --upgrade`
3. **Test components individually:** See `MODULES.md` for module-level testing
4. **Check GitHub Issues:** (if repository is public)

---

**Next:** See [MODULES.md](./MODULES.md) for module-level documentation.
