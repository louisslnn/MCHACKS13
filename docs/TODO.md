# HoloRay Development Roadmap

## 🚀 Core Tracking Engine
- [x] **Optimize Latency:** Implement downscaled Optical Flow for >30 FPS performance.
- [ ] **Robust Re-Identification:** Implement Hybrid Architecture (Optical Flow + SIFT/ORB) to handle:
    - [ ] Object disappearance (clean vanish when out of frame).
    - [ ] Object reappearance (snap-back using feature matching).
- [ ] **Model Upgrade:** Integrate improved feature matching logic (Feature "Fingerprinting") to fix drift accuracy.

## 🧪 Testing & Validation
- [ ] **Build Test Suite:** Create automated test cases for:
    - [ ] **Exit/Entry:** Verify tracker recovers object after 3 seconds off-screen.
    - [ ] **Occlusion:** Verify tracker handles partial blockage without drift.
    - [ ] **Shake:** Verify stability during rapid camera motion.

## 🎨 UI/UX Features
- [ ] **Dynamic Labeling:** Add an input placeholder/dialog upon clicking an object to specify its custom name (e.g., "White King", "Tumor").
- [ ] **Visual Feedback:** Implement distinct visual states for "Tracking" (Green), "Occluded" (Yellow/Ghost), and "Lost" (Hidden).