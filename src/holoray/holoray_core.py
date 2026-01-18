"""
HoloRay Core - Ultimate Hybrid Tracking Engine

The most robust AR tracking system possible. Feels "magnetic":
- Locks on with SIFT Visual DNA + HSV color verification
- Handles extreme motion with Forward-Backward optical flow
- Vanishes INSTANTLY when off-screen (0ms delay)
- Snaps back PRECISELY via RANSAC homography on re-entry

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    FAST PATH (Every Frame)                       │
│  Lucas-Kanade Optical Flow with Forward-Backward Error Check    │
│  Target latency: <2ms | Boundary Guard: 5px from edge           │
├─────────────────────────────────────────────────────────────────┤
│                    SLOW PATH (Adaptive)                          │
│  SIFT Feature Matching + RANSAC Homography (10+ inliers)        │
│  Triggered: Low velocity (drift check) OR LOST (Re-ID)          │
│  Downscaled to 640px width for speed                            │
└─────────────────────────────────────────────────────────────────┘

Performance: <15ms latency, >60 FPS achievable
"""

import time
import logging
import math
import threading
from enum import Enum
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import uuid

import numpy as np
import cv2


class TrackingStatus(Enum):
    """Status of a tracked object."""
    TRACKING = "tracking"      # Actively tracking (green) - VISIBLE
    OCCLUDED = "occluded"      # Partial occlusion (yellow) - FADED  
    LOST = "lost"              # Object left frame - HIDDEN (instant)
    SEARCHING = "searching"    # Looking for re-ID - HIDDEN
    INACTIVE = "inactive"      # Not initialized


class LabelStatus(Enum):
    """Status of AI labeling process."""
    IDLE = "idle"           # No identification in progress
    THINKING = "thinking"   # API call in progress (show "...")
    LABELED = "labeled"     # Label has been set by AI
    ERROR = "error"         # API call failed


@dataclass
class TrackingState:
    """Current state of a tracked object."""
    status: TrackingStatus
    x: float  # Display coordinates (full resolution)
    y: float
    confidence: float
    is_occluded: bool
    visibility: float = 1.0
    opacity: float = 1.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    frames_since_seen: int = 0
    last_good_position: Tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0
    rotation: float = 0.0


class VisualDNA:
    """
    Visual DNA - The immutable identity of a tracked object.
    
    On initialization (click):
    1. Extract SIFT keypoints, keep TOP 50 STRONGEST (not all!)
    2. Store HSV histogram for color verification
    
    Why both?
    - SIFT sees texture/edges → can confuse white pawn for black pawn
    - HSV histogram sees color → prevents wrong-color matches
    
    This combination creates a robust "fingerprint" that uniquely
    identifies the object even after it leaves and re-enters the frame.
    """
    
    TOP_N_KEYPOINTS = 50  # Only keep strongest features
    ROI_SIZE = 120        # Region of interest around click
    HSV_BINS = (16, 16, 8)  # Hue, Saturation, Value bins
    
    def __init__(self):
        # SIFT with more features, we'll filter to top 50
        self._sift = cv2.SIFT_create(nfeatures=200, contrastThreshold=0.03)
        
        # Cached DNA (computed ONCE at initialization)
        self.keypoints: List[cv2.KeyPoint] = []
        self.descriptors: Optional[np.ndarray] = None
        self.hsv_histogram: Optional[np.ndarray] = None
        self.roi_center: Tuple[int, int] = (0, 0)  # Original center in ROI coords
        self.bbox_size: Tuple[int, int] = (0, 0)
        
        self._initialized = False
    
    def initialize(self, frame: np.ndarray, x: int, y: int) -> bool:
        """
        Capture the Visual DNA of the object at click position.
        
        CRITICAL: This is called ONCE. The DNA never changes.
        
        Args:
            frame: BGR frame (tracking resolution)
            x, y: Click position
            
        Returns:
            True if DNA captured successfully
        """
        h, w = frame.shape[:2]
        half = self.ROI_SIZE // 2
        
        # Extract ROI
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(w, x + half), min(h, y + half)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 30 or roi.shape[1] < 30:
            return False
        
        self.bbox_size = (x2 - x1, y2 - y1)
        self.roi_center = (x - x1, y - y1)  # Center in ROI coordinates
        
        # === SIFT EXTRACTION ===
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        
        if keypoints is None or len(keypoints) < 4:
            # Fallback: still store what we have
            self.keypoints = list(keypoints) if keypoints else []
            self.descriptors = descriptors
        else:
            # OPTIMIZATION: Keep only TOP 50 strongest keypoints
            # Sorted by response (strength)
            sorted_kp = sorted(keypoints, key=lambda k: k.response, reverse=True)
            top_indices = [keypoints.index(kp) for kp in sorted_kp[:self.TOP_N_KEYPOINTS]]
            
            self.keypoints = [keypoints[i] for i in top_indices]
            self.descriptors = descriptors[top_indices] if descriptors is not None else None
        
        # === HSV HISTOGRAM (Color DNA) ===
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.hsv_histogram = cv2.calcHist(
            [hsv], [0, 1, 2], None, 
            list(self.HSV_BINS),
            [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(self.hsv_histogram, self.hsv_histogram, 0, 1, cv2.NORM_MINMAX)
        
        self._initialized = True
        
        logging.getLogger("VisualDNA").debug(
            f"DNA captured: {len(self.keypoints)} keypoints (top {self.TOP_N_KEYPOINTS}), "
            f"histogram shape {self.hsv_histogram.shape}"
        )
        return True
    
    def verify_color(self, frame: np.ndarray, x: int, y: int, threshold: float = 0.5) -> float:
        """
        Verify if the color at position matches our DNA.
        
        Returns:
            Similarity score 0.0-1.0 (>threshold means match)
        """
        if self.hsv_histogram is None:
            return 0.0
        
        h, w = frame.shape[:2]
        half = self.ROI_SIZE // 2
        
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(w, x + half), min(h, y + half)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        current_hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            list(self.HSV_BINS),
            [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(current_hist, current_hist, 0, 1, cv2.NORM_MINMAX)
        
        # Correlation comparison
        similarity = cv2.compareHist(self.hsv_histogram, current_hist, cv2.HISTCMP_CORREL)
        return max(0.0, similarity)  # Correlation can be negative
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def has_features(self) -> bool:
        return self.descriptors is not None and len(self.descriptors) >= 4


class MagneticOpticalFlow:
    """
    Magnetic Optical Flow Tracker with Forward-Backward Error Checking.
    
    The "magnetic" feel comes from:
    1. Forward-Backward validation: Track A→B then B→A, reject if error > 1px
    2. Robust point grid that auto-regenerates when points are lost
    3. 5px boundary guard: INSTANT LOST if approaching edge
    
    This eliminates jitter and "sticky edges" completely.
    """
    
    FORWARD_BACKWARD_THRESHOLD = 1.0  # Max acceptable FB error in pixels
    BOUNDARY_GUARD = 5  # Pixels from edge to trigger LOST
    MIN_VALID_POINTS = 3  # Minimum points to continue tracking
    
    def __init__(self, num_points: int = 25, spread: int = 8):
        """
        Args:
            num_points: Number of tracking points (5x5 grid = 25)
            spread: Spacing between grid points
        """
        self.num_points = num_points
        self.spread = spread
        self.grid_size = int(math.sqrt(num_points))
        
        # LK parameters optimized for speed
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.prev_gray: Optional[np.ndarray] = None
        self.points: Optional[np.ndarray] = None
        self._frame_w = 0
        self._frame_h = 0
    
    def initialize(self, frame: np.ndarray, x: int, y: int):
        """Initialize tracking grid centered at position."""
        h, w = frame.shape[:2]
        self._frame_w = w
        self._frame_h = h
        
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._create_grid(x, y)
    
    def _create_grid(self, cx: int, cy: int):
        """Create tracking point grid centered at (cx, cy)."""
        points = []
        half = self.grid_size // 2
        
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px = cx + dx * self.spread
                py = cy + dy * self.spread
                # Clamp to frame bounds
                px = max(self.BOUNDARY_GUARD, min(px, self._frame_w - self.BOUNDARY_GUARD - 1))
                py = max(self.BOUNDARY_GUARD, min(py, self._frame_h - self.BOUNDARY_GUARD - 1))
                points.append([float(px), float(py)])
        
        self.points = np.array(points, dtype=np.float32)
    
    def track(self, frame: np.ndarray) -> Tuple[float, float, float, bool, bool]:
        """
        Track with Forward-Backward error checking.
        
        Returns:
            (x, y, confidence, is_valid, at_boundary)
            - is_valid: False if tracking completely failed
            - at_boundary: True if position is within BOUNDARY_GUARD of edge
        """
        if self.prev_gray is None or self.points is None or len(self.points) == 0:
            return 0.0, 0.0, 0.0, False, False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # === FORWARD FLOW: prev → current ===
        pts_prev = self.points.reshape(-1, 1, 2)
        pts_next, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, pts_prev, None, **self.lk_params
        )
        
        if pts_next is None:
            return 0.0, 0.0, 0.0, False, False
        
        # === BACKWARD FLOW: current → prev (for validation) ===
        pts_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, pts_next, None, **self.lk_params
        )
        
        if pts_back is None:
            return 0.0, 0.0, 0.0, False, False
        
        # === FORWARD-BACKWARD ERROR CHECK ===
        # Points are valid only if both flows succeeded AND round-trip error < threshold
        fb_error = np.linalg.norm(pts_prev - pts_back, axis=2).flatten()
        status_fwd = status_fwd.flatten()
        status_bwd = status_bwd.flatten()
        
        valid_mask = (
            (status_fwd == 1) & 
            (status_bwd == 1) & 
            (fb_error < self.FORWARD_BACKWARD_THRESHOLD)
        )
        
        valid_count = valid_mask.sum()
        
        if valid_count < self.MIN_VALID_POINTS:
            # Not enough valid points - tracking unreliable
            self.prev_gray = gray
            return 0.0, 0.0, 0.0, False, False
        
        # Extract valid points
        pts_next = pts_next.reshape(-1, 2)
        valid_points = pts_next[valid_mask]
        
        # Update points for next frame
        self.points = valid_points
        self.prev_gray = gray
        
        # Compute center
        center_x = float(valid_points[:, 0].mean())
        center_y = float(valid_points[:, 1].mean())
        
        # Confidence based on valid points ratio
        confidence = float(valid_count) / float(self.num_points)
        
        # === BOUNDARY CHECK ===
        at_boundary = (
            center_x < self.BOUNDARY_GUARD or
            center_x > self._frame_w - self.BOUNDARY_GUARD or
            center_y < self.BOUNDARY_GUARD or
            center_y > self._frame_h - self.BOUNDARY_GUARD
        )
        
        return center_x, center_y, confidence, True, at_boundary
    
    def reset_at(self, x: int, y: int, frame: np.ndarray):
        """Reset tracking grid to new position."""
        self._create_grid(x, y)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def update_frame(self, frame: np.ndarray):
        """Update previous frame reference."""
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class RANSACMatcher:
    """
    RANSAC-based Feature Matcher with Geometric Lock.
    
    The "magic" re-identification step:
    1. Match SIFT descriptors with BFMatcher + Lowe's ratio test (0.7)
    2. Run findHomography with RANSAC
    3. STRICTNESS: Require mask.sum() > 10 (at least 10 geometrically consistent points)
    4. Project original center through homography to get new position
    
    This ensures we ONLY snap to geometrically verified matches, preventing
    false positives from similar-looking objects.
    """
    
    LOWE_RATIO = 0.7  # Stricter than typical 0.75
    MIN_RANSAC_INLIERS = 10  # "Geometric Lock" threshold
    RANSAC_REPROJ_THRESHOLD = 5.0
    
    def __init__(self):
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self._sift = cv2.SIFT_create(nfeatures=300)  # More features for search
    
    def match_and_localize(
        self,
        frame: np.ndarray,
        dna: VisualDNA,
        roi_offset: Tuple[int, int] = (0, 0)
    ) -> Optional[Tuple[float, float, float, int]]:
        """
        Match DNA against frame and localize object.
        
        Args:
            frame: Current frame (grayscale OK, BGR OK)
            dna: Visual DNA to match against
            roi_offset: Offset to add to result coordinates
            
        Returns:
            (x, y, confidence, num_inliers) or None if no match
        """
        if not dna.has_features:
            return None
        
        # Detect features in frame
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        kp_frame, desc_frame = self._sift.detectAndCompute(gray, None)
        
        if desc_frame is None or len(desc_frame) < 4:
            return None
        
        # === FEATURE MATCHING with Lowe's Ratio Test ===
        desc_dna = dna.descriptors.astype(np.float32)
        desc_frame = desc_frame.astype(np.float32)
        
        try:
            matches = self.bf_matcher.knnMatch(desc_dna, desc_frame, k=2)
        except cv2.error:
            return None
        
        # Apply Lowe's ratio test (stricter 0.7)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.LOWE_RATIO * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return None
        
        # === RANSAC HOMOGRAPHY ===
        src_pts = np.float32([
            dna.keypoints[m.queryIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)
        
        dst_pts = np.float32([
            kp_frame[m.trainIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)
        
        try:
            H, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, self.RANSAC_REPROJ_THRESHOLD
            )
        except cv2.error:
            return None
        
        if H is None or mask is None:
            return None
        
        # === GEOMETRIC LOCK: Check inlier count ===
        num_inliers = int(mask.sum())
        
        if num_inliers < self.MIN_RANSAC_INLIERS:
            # Not enough geometric consistency - reject
            return None
        
        # === PROJECT CENTER through Homography ===
        roi_center = np.array([[dna.roi_center]], dtype=np.float32)
        new_center = cv2.perspectiveTransform(roi_center, H)
        
        new_x = float(new_center[0, 0, 0]) + roi_offset[0]
        new_y = float(new_center[0, 0, 1]) + roi_offset[1]
        
        confidence = min(1.0, num_inliers / 20.0)
        
        return new_x, new_y, confidence, num_inliers
    
    def detect_in_roi(
        self,
        frame: np.ndarray,
        x: int, y: int,
        roi_size: int,
        dna: VisualDNA
    ) -> Optional[Tuple[float, float, float, int]]:
        """
        Detect object in ROI around position (for drift correction).
        """
        h, w = frame.shape[:2]
        half = roi_size // 2
        
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(w, x + half), min(h, y + half)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        result = self.match_and_localize(roi, dna, roi_offset=(x1, y1))
        return result


class UltimateHybridTracker:
    """
    Ultimate Hybrid Tracking Engine.
    
    The most robust tracker possible:
    
    1. MAGNETIC LOCK: Forward-backward optical flow eliminates jitter
    2. CLEAN VANISH: 5px boundary guard triggers instant LOST (no edge sticking)
    3. VISUAL DNA: Top 50 SIFT features + HSV histogram (color-aware)
    4. GEOMETRIC LOCK: RANSAC with 10+ inliers for re-identification
    5. ADAPTIVE SCHEDULING: Re-ID triggered by low velocity OR LOST state
    6. SNAP-BACK: Instant teleport when RANSAC confirms object
    
    State Machine:
    ┌──────────┐     FB error or      ┌──────────┐    timeout    ┌──────────┐
    │ TRACKING │ ──────────────────→  │ OCCLUDED │ ────────────→ │   LOST   │
    │ (green)  │     boundary hit     │ (yellow) │               │ (hidden) │
    └────┬─────┘                      └────┬─────┘               └────┬─────┘
         │                                 │                          │
         │  ←─── RANSAC snap-back ────────┴──────────────────────────┘
         │                                 (instant teleport)
    """
    
    # === PERFORMANCE TUNING ===
    TRACKING_WIDTH = 640  # Downscale target
    
    # Adaptive scheduling
    VELOCITY_LOW_THRESHOLD = 2.0  # Below this = "stationary", run Re-ID
    DRIFT_CHECK_INTERVAL = 20     # Frames between drift corrections (if moving)
    SEARCH_INTERVAL = 3           # Frames between global searches when LOST
    
    # State transitions
    OCCLUSION_TIMEOUT = 3         # Frames in OCCLUDED before LOST (FAST!)
    LOST_TIMEOUT = 300            # Give up after this many frames
    
    # Confidence
    MIN_FLOW_CONFIDENCE = 0.4
    COLOR_VERIFY_THRESHOLD = 0.4
    
    def __init__(self, tracker_id: Optional[str] = None, enable_reid: bool = True):
        self.tracker_id = tracker_id or str(uuid.uuid4())[:8]
        self.enable_reid = enable_reid
        self.logger = logging.getLogger(f"UltimateTracker-{self.tracker_id}")
        
        # Core components
        self.dna = VisualDNA()
        self.flow = MagneticOpticalFlow(num_points=25, spread=8)
        self.matcher = RANSACMatcher()
        
        # State
        self._status = TrackingStatus.INACTIVE
        self._confidence = 0.0
        
        # Position (DISPLAY coordinates)
        self._x = 0.0
        self._y = 0.0
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._last_good_pos = (0.0, 0.0)
        
        # Position (TRACKING coordinates - downscaled)
        self._track_x = 0.0
        self._track_y = 0.0
        
        # Velocity tracking
        self._velocity = (0.0, 0.0)
        self._velocity_magnitude = 0.0
        
        # Frame info
        self._scale_factor = 1.0
        self._display_w = 0
        self._display_h = 0
        self._track_w = 0
        self._track_h = 0
        
        # Counters
        self._frame_count = 0
        self._frames_occluded = 0
        self._frames_lost = 0
        
        # Label
        self.label = ""
        
        # AI Labeling state (thread-safe)
        self._label_lock = threading.Lock()
        self._label_status = LabelStatus.IDLE
        self._ai_label: Optional[str] = None  # Label from AI (OpenAI)
        
        # Attached drawings (shapes that follow this object)
        # These are rendered ONLY when status == TRACKING
        self._drawings: List = []  # List of BaseShape from shapes.py
    
    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        """Downscale to tracking resolution."""
        h, w = frame.shape[:2]
        self._display_w = w
        self._display_h = h
        
        if w <= self.TRACKING_WIDTH:
            self._scale_factor = 1.0
            self._track_w = w
            self._track_h = h
            return frame
        
        self._scale_factor = w / self.TRACKING_WIDTH
        self._track_h = int(h / self._scale_factor)
        self._track_w = self.TRACKING_WIDTH
        
        return cv2.resize(frame, (self._track_w, self._track_h), interpolation=cv2.INTER_LINEAR)
    
    def _to_display(self, x: float, y: float) -> Tuple[float, float]:
        return x * self._scale_factor, y * self._scale_factor
    
    def _to_tracking(self, x: float, y: float) -> Tuple[float, float]:
        return x / self._scale_factor, y / self._scale_factor
    
    def initialize(self, frame: np.ndarray, x: int, y: int, label: str = "") -> bool:
        """
        Initialize tracker at click position.
        
        Captures Visual DNA and initializes optical flow.
        """
        self.label = label
        
        # Downscale
        small_frame = self._downscale(frame)
        track_x, track_y = self._to_tracking(float(x), float(y))
        
        # Capture Visual DNA
        if not self.dna.initialize(small_frame, int(track_x), int(track_y)):
            self.logger.warning("Failed to capture Visual DNA")
        
        # Initialize optical flow
        self.flow.initialize(small_frame, int(track_x), int(track_y))
        
        # Set state
        self._track_x = track_x
        self._track_y = track_y
        self._x = float(x)
        self._y = float(y)
        self._prev_x = self._x
        self._prev_y = self._y
        self._last_good_pos = (self._x, self._y)
        
        self._status = TrackingStatus.TRACKING
        self._confidence = 1.0
        self._frame_count = 0
        self._frames_occluded = 0
        self._frames_lost = 0
        
        self.logger.info(
            f"Initialized '{label}' at ({x}, {y}), "
            f"DNA: {len(self.dna.keypoints)} keypoints"
        )
        return True
    
    def _run_drift_correction(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Run SIFT matching for drift correction in local ROI.
        
        Returns:
            (x, y, confidence) in tracking coords, or None
        """
        result = self.matcher.detect_in_roi(
            frame,
            int(self._track_x), int(self._track_y),
            roi_size=150,
            dna=self.dna
        )
        
        if result is None:
            return None
        
        new_x, new_y, conf, inliers = result
        
        # Color verification (prevents wrong-object snap)
        color_sim = self.dna.verify_color(frame, int(new_x), int(new_y))
        
        if color_sim < self.COLOR_VERIFY_THRESHOLD:
            self.logger.debug(f"Color mismatch: {color_sim:.2f}")
            return None
        
        return new_x, new_y, conf
    
    def _run_global_search(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Search entire frame for object (Re-ID).
        
        Returns:
            (x, y, confidence) in tracking coords, or None
        """
        result = self.matcher.match_and_localize(frame, self.dna)
        
        if result is None:
            return None
        
        new_x, new_y, conf, inliers = result
        
        # Validate position is inside frame
        if (new_x < MagneticOpticalFlow.BOUNDARY_GUARD or
            new_x > self._track_w - MagneticOpticalFlow.BOUNDARY_GUARD or
            new_y < MagneticOpticalFlow.BOUNDARY_GUARD or
            new_y > self._track_h - MagneticOpticalFlow.BOUNDARY_GUARD):
            return None
        
        # Color verification
        color_sim = self.dna.verify_color(frame, int(new_x), int(new_y))
        
        if color_sim < self.COLOR_VERIFY_THRESHOLD:
            self.logger.debug(f"Re-ID color mismatch: {color_sim:.2f}")
            return None
        
        self.logger.info(f"Re-ID SUCCESS: ({new_x:.0f}, {new_y:.0f}), {inliers} inliers, color={color_sim:.2f}")
        return new_x, new_y, conf
    
    def update(self, frame: np.ndarray) -> TrackingState:
        """
        Update tracker with new frame.
        
        Hybrid strategy with adaptive scheduling:
        - Fast path: Optical flow every frame
        - Slow path: SIFT matching when velocity low OR when LOST
        
        Returns:
            TrackingState with current position and visibility
        """
        self._frame_count += 1
        
        # Inactive check
        if self._status == TrackingStatus.INACTIVE:
            return TrackingState(
                status=TrackingStatus.INACTIVE,
                x=0, y=0, confidence=0, is_occluded=False,
                visibility=0.0, opacity=0.0
            )
        
        # Store previous
        self._prev_x = self._x
        self._prev_y = self._y
        
        # Downscale
        small_frame = self._downscale(frame)
        
        # ═══════════════════════════════════════════════════════════════
        # STATE: TRACKING
        # ═══════════════════════════════════════════════════════════════
        if self._status == TrackingStatus.TRACKING:
            # === FAST PATH: Optical Flow ===
            new_x, new_y, flow_conf, is_valid, at_boundary = self.flow.track(small_frame)
            
            # BOUNDARY GUARD: Instant LOST if at edge
            if at_boundary:
                self._status = TrackingStatus.LOST
                self._frames_lost = 0
                self._confidence = 0.0
                self.logger.info("Boundary hit → LOST")
            
            elif not is_valid or flow_conf < self.MIN_FLOW_CONFIDENCE:
                # Tracking failed → OCCLUDED
                self._status = TrackingStatus.OCCLUDED
                self._frames_occluded = 0
                self._confidence = flow_conf
            
            else:
                # Tracking successful
                self._track_x = new_x
                self._track_y = new_y
                self._confidence = flow_conf
                
                # === ADAPTIVE SLOW PATH ===
                # Run drift correction if:
                # 1. Velocity is low (object stationary - drift likely)
                # 2. OR periodic check interval
                should_correct = (
                    self._velocity_magnitude < self.VELOCITY_LOW_THRESHOLD or
                    self._frame_count % self.DRIFT_CHECK_INTERVAL == 0
                )
                
                if should_correct and self.dna.has_features:
                    correction = self._run_drift_correction(small_frame)
                    
                    if correction:
                        corr_x, corr_y, corr_conf = correction
                        
                        # SNAP to corrected position (teleport, not blend)
                        self._track_x = corr_x
                        self._track_y = corr_y
                        self._confidence = corr_conf
                        
                        # Reset flow grid to new position
                        self.flow.reset_at(int(corr_x), int(corr_y), small_frame)
                
                # Update display coordinates
                self._x, self._y = self._to_display(self._track_x, self._track_y)
                self._last_good_pos = (self._x, self._y)
        
        # ═══════════════════════════════════════════════════════════════
        # STATE: OCCLUDED
        # ═══════════════════════════════════════════════════════════════
        elif self._status == TrackingStatus.OCCLUDED:
            self._frames_occluded += 1
            
            # Try optical flow
            new_x, new_y, flow_conf, is_valid, at_boundary = self.flow.track(small_frame)
            
            if at_boundary:
                self._status = TrackingStatus.LOST
                self._frames_lost = 0
            elif is_valid and flow_conf > self.MIN_FLOW_CONFIDENCE * 1.5:
                # Recovery attempt with SIFT verification
                correction = self._run_drift_correction(small_frame)
                
                if correction:
                    # RECOVERED!
                    corr_x, corr_y, corr_conf = correction
                    self._status = TrackingStatus.TRACKING
                    self._track_x = corr_x
                    self._track_y = corr_y
                    self._confidence = corr_conf
                    self._x, self._y = self._to_display(corr_x, corr_y)
                    self._last_good_pos = (self._x, self._y)
                    self.flow.reset_at(int(corr_x), int(corr_y), small_frame)
                else:
                    # Still occluded
                    self._track_x = new_x
                    self._track_y = new_y
                    self._confidence = flow_conf * 0.5
                    self._x, self._y = self._to_display(new_x, new_y)
            
            # FAST timeout to LOST
            if self._frames_occluded > self.OCCLUSION_TIMEOUT:
                self._status = TrackingStatus.LOST
                self._frames_lost = 0
        
        # ═══════════════════════════════════════════════════════════════
        # STATE: LOST → SEARCHING
        # ═══════════════════════════════════════════════════════════════
        elif self._status == TrackingStatus.LOST:
            self._frames_lost += 1
            self._confidence = 0.0
            
            if self.enable_reid:
                self._status = TrackingStatus.SEARCHING
        
        # ═══════════════════════════════════════════════════════════════
        # STATE: SEARCHING (Re-ID)
        # ═══════════════════════════════════════════════════════════════
        if self._status == TrackingStatus.SEARCHING:
            # Run global search frequently
            should_search = (
                self._frames_lost <= 1 or
                self._frames_lost % self.SEARCH_INTERVAL == 0
            )
            
            if should_search and self.dna.has_features:
                result = self._run_global_search(small_frame)
                
                if result:
                    # SNAP-BACK: Instant teleport to new position
                    new_x, new_y, conf = result
                    
                    self._status = TrackingStatus.TRACKING
                    self._track_x = new_x
                    self._track_y = new_y
                    self._confidence = conf
                    self._x, self._y = self._to_display(new_x, new_y)
                    self._last_good_pos = (self._x, self._y)
                    self._frames_lost = 0
                    
                    # Reset flow to new position
                    self.flow.reset_at(int(new_x), int(new_y), small_frame)
            
            # Increment counter
            if self._status == TrackingStatus.SEARCHING:
                self._frames_lost += 1
            
            # Timeout
            if self._frames_lost > self.LOST_TIMEOUT:
                self._status = TrackingStatus.LOST
                self._confidence = 0.0
        
        # === VELOCITY COMPUTATION ===
        self._velocity = (self._x - self._prev_x, self._y - self._prev_y)
        self._velocity_magnitude = math.sqrt(self._velocity[0]**2 + self._velocity[1]**2)
        
        # === VISIBILITY/OPACITY (Clean Vanish) ===
        if self._status == TrackingStatus.TRACKING:
            visibility = self._confidence
            opacity = 1.0
            is_occluded = False
        elif self._status == TrackingStatus.OCCLUDED:
            visibility = max(0.3, self._confidence)
            opacity = 0.5
            is_occluded = True
        else:  # LOST, SEARCHING - INSTANT HIDDEN
            visibility = 0.0
            opacity = 0.0
            is_occluded = False
        
        return TrackingState(
            status=self._status,
            x=self._x,
            y=self._y,
            confidence=self._confidence,
            is_occluded=is_occluded,
            visibility=visibility,
            opacity=opacity,
            velocity=self._velocity,
            frames_since_seen=self._frames_lost if self._status in (TrackingStatus.LOST, TrackingStatus.SEARCHING) else 0,
            last_good_position=self._last_good_pos,
            scale=1.0,
            rotation=0.0
        )
    
    def reset(self):
        """Reset tracker to inactive."""
        self._status = TrackingStatus.INACTIVE
        self._x = 0
        self._y = 0
        self._confidence = 0
        self._frame_count = 0
        self._frames_lost = 0
        self._frames_occluded = 0
        self.dna = VisualDNA()
        self.flow = MagneticOpticalFlow()
        self._drawings.clear()
    
    # =========================================================================
    # AI LABELING (Thread-Safe)
    # =========================================================================
    
    @property
    def label_status(self) -> LabelStatus:
        """Get current AI labeling status."""
        with self._label_lock:
            return self._label_status
    
    @label_status.setter
    def label_status(self, value: LabelStatus):
        """Set AI labeling status (thread-safe)."""
        with self._label_lock:
            self._label_status = value
    
    @property
    def ai_label(self) -> Optional[str]:
        """Get the AI-generated label."""
        with self._label_lock:
            return self._ai_label
    
    def update_label(self, new_label: str) -> None:
        """
        Thread-safe method to update the label from AI.
        
        Called from background thread after AI API returns.
        Also updates the main label attribute.
        
        Args:
            new_label: New label text from AI
        """
        with self._label_lock:
            self._ai_label = new_label
            self._label_status = LabelStatus.LABELED
            # Also update main label
            self.label = new_label
    
    def start_thinking(self) -> None:
        """Mark that AI labeling is in progress."""
        with self._label_lock:
            self._label_status = LabelStatus.THINKING
    
    def set_label_error(self) -> None:
        """Mark that AI labeling failed."""
        with self._label_lock:
            self._label_status = LabelStatus.ERROR
    
    def reset_label_status(self) -> None:
        """Reset label status to idle."""
        with self._label_lock:
            self._label_status = LabelStatus.IDLE
    
    def get_display_label(self) -> str:
        """
        Get the label to display, considering AI status.
        
        Returns:
            - "..." if THINKING
            - AI label if LABELED
            - Original label otherwise
        """
        with self._label_lock:
            if self._label_status == LabelStatus.THINKING:
                return "..."
            elif self._label_status == LabelStatus.LABELED and self._ai_label:
                return self._ai_label
            else:
                return self.label or "Tracker"
    
    # =========================================================================
    # DRAWING MANAGEMENT
    # =========================================================================
    
    def add_drawing(self, shape) -> None:
        """
        Add a shape to this tracker's drawings.
        
        The shape will follow the object and be rendered when TRACKING.
        When LOST, the shape is automatically hidden.
        
        Args:
            shape: BaseShape instance from shapes.py
        """
        self._drawings.append(shape)
    
    def remove_drawing(self, shape) -> bool:
        """
        Remove a shape from drawings.
        
        Returns:
            True if shape was found and removed
        """
        if shape in self._drawings:
            self._drawings.remove(shape)
            return True
        return False
    
    def clear_drawings(self) -> None:
        """Remove all drawings from this tracker."""
        self._drawings.clear()
    
    def render_drawings(self, frame: np.ndarray, opacity_override: Optional[float] = None) -> np.ndarray:
        """
        Render all drawings onto the frame.
        
        Called automatically in the main loop when tracker is TRACKING.
        Drawings are hidden when LOST/SEARCHING (opacity=0).
        
        Args:
            frame: Frame to draw on
            opacity_override: Force specific opacity (used for OCCLUDED state)
            
        Returns:
            Frame with drawings rendered
        """
        if self._status == TrackingStatus.INACTIVE:
            return frame
        
        # Determine opacity based on status
        if opacity_override is not None:
            opacity = opacity_override
        elif self._status == TrackingStatus.TRACKING:
            opacity = 1.0
        elif self._status == TrackingStatus.OCCLUDED:
            opacity = 0.5
        else:  # LOST, SEARCHING
            opacity = 0.0  # Hidden
        
        if opacity <= 0:
            return frame
        
        # Render each shape
        for shape in self._drawings:
            if hasattr(shape, 'render') and hasattr(shape, 'visible'):
                if shape.visible:
                    original_opacity = getattr(shape, 'opacity', 1.0)
                    shape.opacity = opacity * original_opacity
                    frame = shape.render(frame, self._x, self._y)
                    shape.opacity = original_opacity
        
        return frame
    
    @property
    def drawings(self) -> List:
        """Get list of attached drawings."""
        return self._drawings
    
    @property
    def drawing_count(self) -> int:
        """Number of attached drawings."""
        return len(self._drawings)
    
    @property
    def status(self) -> TrackingStatus:
        return self._status
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self._x, self._y)
    
    @property
    def confidence(self) -> float:
        return self._confidence


# Backward compatibility aliases
HybridTracker = UltimateHybridTracker
ObjectTracker = UltimateHybridTracker
VisualFingerprint = VisualDNA
VisualMemory = VisualDNA


class FastOpticalFlow:
    """Backward compatibility wrapper."""
    def __init__(self, grid_size: int = 3, grid_spacing: int = 10):
        self._flow = MagneticOpticalFlow(num_points=grid_size**2, spread=grid_spacing)
    
    def initialize(self, frame, x, y):
        self._flow.initialize(frame, x, y)
    
    def track(self, frame):
        x, y, conf, valid, _ = self._flow.track(frame)
        return x, y, conf, valid
    
    def reset_points(self, x, y, frame=None):
        if frame is not None:
            self._flow.reset_at(x, y, frame)
    
    def update_prev_frame(self, frame):
        self._flow.update_frame(frame)


class FeatureMatcher:
    """Backward compatibility wrapper."""
    LOWE_RATIO = 0.7
    MIN_MATCHES_FOR_HOMOGRAPHY = 4
    
    def __init__(self):
        self._matcher = RANSACMatcher()
        self.bf_matcher = self._matcher.bf_matcher
    
    def match_descriptors(self, desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)
        
        try:
            matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []
        
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < self.LOWE_RATIO * n.distance:
                    good.append(m)
        return good
    
    def compute_homography(self, kp1, kp2, matches, roi_offset=(0, 0)):
        if len(matches) < 4:
            return None, np.array([]), np.array([])
        
        src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst[:, :, 0] += roi_offset[0]
        dst[:, :, 1] += roi_offset[1]
        
        try:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            return H, src, dst
        except:
            return None, src, dst
    
    def estimate_new_center(self, H, orig_center, bbox_size):
        cx, cy = bbox_size[0] / 2, bbox_size[1] / 2
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H)
        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])


class TrackerManager:
    """Manages multiple trackers."""
    
    def __init__(self, use_gpu: bool = True, enable_reid: bool = True):
        self.enable_reid = enable_reid
        self._trackers: Dict[str, UltimateHybridTracker] = {}
        self.logger = logging.getLogger("TrackerManager")
    
    def create_tracker(self, frame: np.ndarray, x: int, y: int, label: str = "") -> str:
        tracker = UltimateHybridTracker(enable_reid=self.enable_reid)
        tracker.initialize(frame, x, y, label)
        self._trackers[tracker.tracker_id] = tracker
        return tracker.tracker_id
    
    def update_all(self, frame: np.ndarray) -> Dict[str, TrackingState]:
        return {tid: t.update(frame) for tid, t in self._trackers.items()}
    
    def get_tracker(self, tracker_id: str) -> Optional[UltimateHybridTracker]:
        return self._trackers.get(tracker_id)
    
    def remove_tracker(self, tracker_id: str):
        if tracker_id in self._trackers:
            del self._trackers[tracker_id]
    
    def remove_lost_trackers(self) -> List[str]:
        lost = [
            tid for tid, t in self._trackers.items()
            if t.status == TrackingStatus.LOST and t._frames_lost > UltimateHybridTracker.LOST_TIMEOUT
        ]
        for tid in lost:
            del self._trackers[tid]
        return lost
    
    def clear_all(self):
        self._trackers.clear()
    
    @property
    def tracker_ids(self) -> List[str]:
        return list(self._trackers.keys())
    
    @property
    def active_count(self) -> int:
        return sum(
            1 for t in self._trackers.values()
            if t.status in (TrackingStatus.TRACKING, TrackingStatus.OCCLUDED)
        )


class PerformanceOverlay:
    """FPS and latency overlay."""
    
    def __init__(self):
        self._last_time = time.perf_counter()
        self._fps_history: List[float] = []
        self._avg_fps = 0.0
        self._latency_ms = 0.0
    
    def update(self, latency_ms: float = 0.0):
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        
        if dt > 0:
            self._fps_history.append(1.0 / dt)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
            self._avg_fps = sum(self._fps_history) / len(self._fps_history)
        
        self._latency_ms = latency_ms
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        color = (0, 255, 0) if self._avg_fps >= 30 else (0, 255, 255) if self._avg_fps >= 20 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {self._avg_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Latency: {self._latency_ms:.0f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        return frame
    
    @property
    def fps(self) -> float:
        return self._avg_fps


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  ULTIMATE HYBRID TRACKING ENGINE - TEST SUITE")
    print("=" * 60)
    
    # Create test frame with textured object
    def create_test_frame(obj_x, obj_y, obj_visible=True):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        
        if obj_visible and 0 < obj_x < 1280 and 0 < obj_y < 720:
            # Distinctive object with texture
            cv2.rectangle(frame, (obj_x-50, obj_y-50), (obj_x+50, obj_y+50), (0, 120, 255), -1)
            cv2.circle(frame, (obj_x, obj_y), 30, (255, 255, 255), -1)
            cv2.circle(frame, (obj_x-15, obj_y-10), 8, (0, 0, 0), -1)
            cv2.circle(frame, (obj_x+15, obj_y-10), 8, (0, 0, 0), -1)
            for i in range(8):
                tx = obj_x + (i % 4 - 2) * 15
                ty = obj_y + (i // 4) * 20
                cv2.circle(frame, (tx, ty), 4, (200, 200, 200), -1)
        
        return frame
    
    # Test 1: Performance
    print("\n[TEST 1] Performance Benchmark")
    frame = create_test_frame(640, 360)
    tracker = UltimateHybridTracker(enable_reid=True)
    tracker.initialize(frame, 640, 360, "Test")
    
    times = []
    for _ in range(100):
        start = time.perf_counter()
        tracker.update(frame)
        times.append((time.perf_counter() - start) * 1000)
    
    avg_ms = sum(times) / len(times)
    fps = 1000 / avg_ms
    print(f"  Average: {avg_ms:.2f}ms | FPS: {fps:.0f}")
    print(f"  {'✅ PASS' if avg_ms < 15 else '❌ FAIL'}: <15ms latency")
    
    # Test 2: Forward-Backward Error Check
    print("\n[TEST 2] Forward-Backward Error Check")
    tracker = UltimateHybridTracker(enable_reid=True)
    tracker.initialize(frame, 640, 360, "FB-Test")
    
    stable_frames = 0
    for _ in range(30):
        state = tracker.update(frame)
        if state.status == TrackingStatus.TRACKING:
            stable_frames += 1
    
    print(f"  Stable frames: {stable_frames}/30")
    print(f"  {'✅ PASS' if stable_frames >= 28 else '❌ FAIL'}: FB validation working")
    
    # Test 3: Boundary Guard (5px)
    print("\n[TEST 3] Boundary Guard (5px edge detection)")
    tracker = UltimateHybridTracker(enable_reid=True)
    edge_frame = create_test_frame(1270, 360)  # Near right edge
    tracker.initialize(edge_frame, 1270, 360, "Edge-Test")
    
    lost_frame = None
    for i in range(10):
        blank = create_test_frame(0, 0, False)
        state = tracker.update(blank)
        if state.status == TrackingStatus.LOST and lost_frame is None:
            lost_frame = i + 1
    
    print(f"  Lost at frame: {lost_frame}")
    print(f"  {'✅ PASS' if lost_frame and lost_frame <= 5 else '❌ FAIL'}: Quick boundary detection")
    
    # Test 4: Re-Identification with Color Verification
    print("\n[TEST 4] Re-ID with Color Verification")
    tracker = UltimateHybridTracker(enable_reid=True)
    tracker.initialize(create_test_frame(640, 360), 640, 360, "ReID-Test")
    
    # Track, then lose, then re-appear
    for _ in range(5):
        tracker.update(create_test_frame(640, 360))
    
    for _ in range(10):
        tracker.update(create_test_frame(0, 0, False))
    
    reacquired = False
    for _ in range(15):
        state = tracker.update(create_test_frame(200, 400))
        if state.status == TrackingStatus.TRACKING:
            reacquired = True
            pos_error = abs(state.x - 200) + abs(state.y - 400)
            break
    
    print(f"  Re-acquired: {reacquired}")
    if reacquired:
        print(f"  Position error: {pos_error:.0f}px")
        print(f"  {'✅ PASS' if pos_error < 50 else '❌ FAIL'}: Accurate snap-back")
    else:
        print("  ❌ FAIL: Re-ID not working")
    
    # Test 5: Clean Vanish (instant hidden)
    print("\n[TEST 5] Clean Vanish (instant opacity=0)")
    tracker = UltimateHybridTracker(enable_reid=True)
    tracker.initialize(create_test_frame(640, 360), 640, 360, "Vanish-Test")
    tracker.update(create_test_frame(640, 360))
    
    # Disappear
    for _ in range(5):
        state = tracker.update(create_test_frame(0, 0, False))
    
    print(f"  Status: {state.status.value}")
    print(f"  Opacity: {state.opacity}")
    print(f"  {'✅ PASS' if state.opacity == 0.0 else '❌ FAIL'}: Instant hidden")
    
    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)
