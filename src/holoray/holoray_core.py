"""
HoloRay Core - Hybrid Fast-Slow Object Tracking System

Architecture:
- Fast Path (Every Frame): Lucas-Kanade Optical Flow for smooth, high-FPS updates
- Slow Path (Every 15 frames OR on Loss): SIFT Feature Matching for re-anchoring and Re-ID

Features:
- Clean disappearance: Object exits frame → immediate LOST, no edge drawing
- Robust re-identification: Object re-enters → snap back to correct position
- Drift correction: Periodic SIFT matching corrects optical flow drift
- Performance: >30 FPS with hybrid strategy
"""

import time
import logging
import math
from enum import Enum
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import uuid

import numpy as np
import cv2


class TrackingStatus(Enum):
    """Status of a tracked object."""
    TRACKING = "tracking"      # Actively tracking (green)
    OCCLUDED = "occluded"      # Partially visible (yellow)
    LOST = "lost"              # Out of frame or lost (hidden)
    SEARCHING = "searching"    # Searching for re-ID (hidden)
    INACTIVE = "inactive"      # Not initialized


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


class VisualFingerprint:
    """
    Visual identity of the tracked object.
    
    Stores the "DNA" of the object at initialization:
    - SIFT keypoints and descriptors (feature fingerprint)
    - Bounding box dimensions (size reference)
    - Reference patch for template matching fallback
    
    Used for:
    1. Drift correction: Re-anchor optical flow every 15 frames
    2. Re-identification: Find object after it leaves and re-enters frame
    """
    
    def __init__(self, roi_size: int = 100):
        self.roi_size = roi_size
        
        # SIFT detector - moderate features for balance of speed/accuracy
        self.sift = cv2.SIFT_create(nfeatures=150, contrastThreshold=0.04)
        
        # Stored fingerprint
        self.keypoints: List[cv2.KeyPoint] = []
        self.descriptors: Optional[np.ndarray] = None
        self.bbox_w_h: Tuple[int, int] = (0, 0)
        self.reference_patch: Optional[np.ndarray] = None
        
        self._initialized = False
    
    def initialize(self, frame: np.ndarray, x: int, y: int) -> bool:
        """
        Capture the visual fingerprint of the object.
        
        Called ONCE on first user click. Extracts SIFT features
        from a region around the click point.
        
        Args:
            frame: Full frame (BGR)
            x, y: Click position (tracking resolution coordinates)
            
        Returns:
            True if fingerprint was successfully captured
        """
        h, w = frame.shape[:2]
        half = self.roi_size // 2
        
        # Calculate ROI bounds (clamped to frame)
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w, x + half)
        y2 = min(h, y + half)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return False
        
        # Store bbox dimensions
        self.bbox_w_h = (x2 - x1, y2 - y1)
        
        # Convert to grayscale for SIFT
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Store reference patch for template matching fallback
        self.reference_patch = gray_roi.copy()
        
        # Extract SIFT keypoints and descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray_roi, None)
        
        if keypoints is None or len(keypoints) < 4:
            # Not enough features - object might be too plain
            # Store what we have anyway
            self.keypoints = list(keypoints) if keypoints else []
            self.descriptors = descriptors
            self._initialized = True
            return True
        
        self.keypoints = list(keypoints)
        self.descriptors = descriptors
        self._initialized = True
        
        logging.getLogger("VisualFingerprint").debug(
            f"Captured fingerprint: {len(self.keypoints)} keypoints, "
            f"bbox {self.bbox_w_h}"
        )
        return True
    
    def detect_features(self, frame: np.ndarray, x: int, y: int) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Detect SIFT features at current position.
        
        Args:
            frame: Current frame (BGR)
            x, y: Current estimated position
            
        Returns:
            (keypoints, descriptors, roi_bounds) where roi_bounds = (x1, y1, x2, y2)
        """
        h, w = frame.shape[:2]
        half = self.roi_size // 2
        
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w, x + half)
        y2 = min(h, y + half)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return [], None, (x1, y1, x2, y2)
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray_roi, None)
        
        return list(keypoints) if keypoints else [], descriptors, (x1, y1, x2, y2)
    
    def detect_features_fullframe(self, frame: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Detect SIFT features on the entire frame (for global search).
        
        WARNING: This is expensive! Only use when object is LOST.
        
        Returns:
            (keypoints, descriptors) for entire frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return list(keypoints) if keypoints else [], descriptors
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def has_features(self) -> bool:
        return self.descriptors is not None and len(self.descriptors) >= 4


class FeatureMatcher:
    """
    Feature matching using Brute Force matcher with Lowe's ratio test.
    
    Uses BFMatcher for more accurate matching (vs FLANN which is faster but less accurate).
    Implements Lowe's ratio test to filter out ambiguous matches.
    Computes homography for geometric validation.
    """
    
    LOWE_RATIO = 0.75  # Lowe's ratio threshold (lower = stricter)
    MIN_MATCHES_FOR_HOMOGRAPHY = 4  # Minimum matches to compute homography
    
    def __init__(self):
        # BFMatcher with L2 norm for SIFT
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def match_descriptors(
        self,
        desc_reference: np.ndarray,
        desc_current: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Match descriptors using KNN + Lowe's ratio test.
        
        Args:
            desc_reference: Reference (stored) descriptors
            desc_current: Current frame descriptors
            
        Returns:
            List of good matches passing ratio test
        """
        if desc_reference is None or desc_current is None:
            return []
        if len(desc_reference) < 2 or len(desc_current) < 2:
            return []
        
        # Ensure float32
        desc_reference = desc_reference.astype(np.float32)
        desc_current = desc_current.astype(np.float32)
        
        try:
            # KNN match with k=2 for ratio test
            matches = self.bf_matcher.knnMatch(desc_reference, desc_current, k=2)
        except cv2.error:
            return []
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.LOWE_RATIO * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def compute_homography(
        self,
        kp_reference: List[cv2.KeyPoint],
        kp_current: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        roi_offset: Tuple[int, int] = (0, 0)
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Compute homography from matched keypoints.
        
        The homography maps points from reference to current frame,
        accounting for ROI offset.
        
        Args:
            kp_reference: Reference keypoints (from fingerprint ROI)
            kp_current: Current keypoints (from current ROI)
            matches: Good matches from ratio test
            roi_offset: (x, y) offset of current ROI in frame
            
        Returns:
            (homography_matrix, src_points, dst_points)
            homography_matrix is None if not enough matches
        """
        if len(matches) < self.MIN_MATCHES_FOR_HOMOGRAPHY:
            return None, np.array([]), np.array([])
        
        # Extract matched keypoint positions
        src_pts = np.float32([
            kp_reference[m.queryIdx].pt for m in matches
        ]).reshape(-1, 1, 2)
        
        dst_pts = np.float32([
            kp_current[m.trainIdx].pt for m in matches
        ]).reshape(-1, 1, 2)
        
        # Add ROI offset to destination points
        dst_pts[:, :, 0] += roi_offset[0]
        dst_pts[:, :, 1] += roi_offset[1]
        
        if len(src_pts) < 4:
            return None, src_pts, dst_pts
        
        try:
            # Compute homography with RANSAC
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H, src_pts, dst_pts
        except cv2.error:
            return None, src_pts, dst_pts
    
    def estimate_new_center(
        self,
        H: np.ndarray,
        original_center: Tuple[int, int],
        bbox_size: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Use homography to estimate new object center.
        
        Args:
            H: 3x3 homography matrix
            original_center: Original center in reference ROI
            bbox_size: Original bbox (width, height)
            
        Returns:
            (new_x, new_y) in frame coordinates
        """
        # Original center in ROI coordinates
        cx, cy = bbox_size[0] / 2, bbox_size[1] / 2
        
        # Transform through homography
        pt = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pt, H)
        
        new_x = float(transformed[0, 0, 0])
        new_y = float(transformed[0, 0, 1])
        
        return new_x, new_y


class FastOpticalFlow:
    """
    Optimized Lucas-Kanade optical flow tracker.
    
    Uses a small grid of tracking points for minimal computation.
    Typical execution time: <1ms per frame on 640x360.
    """
    
    def __init__(self, grid_size: int = 3, grid_spacing: int = 10):
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        
        # Optimized LK parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.prev_gray: Optional[np.ndarray] = None
        self.track_points: Optional[np.ndarray] = None
        self._frame_size: Tuple[int, int] = (0, 0)
    
    def initialize(self, frame: np.ndarray, x: int, y: int):
        """
        Initialize tracking points around position.
        
        Args:
            frame: Frame (BGR)
            x, y: Initial position
        """
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        
        # Create grid of tracking points
        points = []
        half = self.grid_size // 2
        
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px = max(0, min(x + dx * self.grid_spacing, w - 1))
                py = max(0, min(y + dy * self.grid_spacing, h - 1))
                points.append([float(px), float(py)])
        
        self.track_points = np.array(points, dtype=np.float32)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def track(self, frame: np.ndarray) -> Tuple[float, float, float, bool]:
        """
        Track points to new frame using optical flow.
        
        Returns:
            (x, y, confidence, is_valid)
            is_valid is False if tracking failed completely
        """
        if self.prev_gray is None or self.track_points is None:
            return 0.0, 0.0, 0.0, False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optical flow
        points = self.track_points.reshape(-1, 1, 2)
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, points, None, **self.lk_params
        )
        
        self.prev_gray = gray
        
        if next_points is None:
            return 0.0, 0.0, 0.0, False
        
        status = status.flatten()
        next_points = next_points.reshape(-1, 2)
        
        # Filter valid points
        valid_mask = status == 1
        valid_count = valid_mask.sum()
        
        if valid_count == 0:
            return 0.0, 0.0, 0.0, False
        
        valid_points = next_points[valid_mask]
        
        # Update tracking points for next frame
        self.track_points = valid_points
        
        # Compute center from valid points
        center_x = float(valid_points[:, 0].mean())
        center_y = float(valid_points[:, 1].mean())
        
        # Confidence based on how many points survived
        confidence = float(valid_count) / float(self.grid_size ** 2)
        
        return center_x, center_y, confidence, True
    
    def reset_points(self, x: int, y: int, frame: Optional[np.ndarray] = None):
        """
        Reset tracking points to a new position.
        
        Called after re-identification to re-anchor the tracker.
        """
        w, h = self._frame_size
        if w == 0 or h == 0:
            return
        
        points = []
        half = self.grid_size // 2
        
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px = max(0, min(x + dx * self.grid_spacing, w - 1))
                py = max(0, min(y + dy * self.grid_spacing, h - 1))
                points.append([float(px), float(py)])
        
        self.track_points = np.array(points, dtype=np.float32)
        
        if frame is not None:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def update_prev_frame(self, frame: np.ndarray):
        """Update previous frame reference."""
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class HybridTracker:
    """
    Hybrid Fast-Slow Object Tracker.
    
    Architecture:
    - Fast Path (Every Frame): Lucas-Kanade Optical Flow (~1ms)
    - Slow Path (Every 15 frames OR on Loss): SIFT Matching (~20-50ms)
    
    State Machine:
    - TRACKING: Active tracking, annotation visible (green)
    - OCCLUDED: Low confidence, annotation faded (yellow)
    - LOST: Object left frame, annotation hidden
    - SEARCHING: Looking for object to re-identify
    
    Key Behaviors:
    1. Object exits frame → immediate LOST (no edge drawing)
    2. Object re-enters → snap to correct position via SIFT
    3. Periodic drift correction every 15 frames
    """
    
    # === CONFIGURATION ===
    TRACKING_WIDTH = 640           # Downscale to this width
    CORRECTION_INTERVAL = 15       # Run SIFT every N frames
    SEARCH_INTERVAL = 5            # How often to search when SEARCHING (faster)
    
    # Thresholds
    FLOW_MIN_CONFIDENCE = 0.3      # Below this → OCCLUDED
    FEATURE_MIN_MATCHES = 6        # Need this many good matches for Re-ID (lowered for robustness)
    OCCLUSION_TIMEOUT = 5          # Frames before OCCLUDED → LOST (quick transition!)
    LOST_TIMEOUT = 180             # Frames before giving up search
    
    # Frame bounds margin (to detect exit)
    EDGE_MARGIN = 15
    
    def __init__(self, tracker_id: Optional[str] = None, enable_reid: bool = True):
        self.tracker_id = tracker_id or str(uuid.uuid4())[:8]
        self.enable_reid = enable_reid
        self.logger = logging.getLogger(f"HybridTracker-{self.tracker_id}")
        
        # Core components
        self.fingerprint = VisualFingerprint(roi_size=100)
        self.optical_flow = FastOpticalFlow(grid_size=3, grid_spacing=10)
        self.matcher = FeatureMatcher()
        
        # State
        self._status = TrackingStatus.INACTIVE
        self._confidence = 0.0
        
        # Position in DISPLAY coordinates
        self._x = 0.0
        self._y = 0.0
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._velocity = (0.0, 0.0)
        self._last_good_position = (0.0, 0.0)
        
        # Position in TRACKING coordinates (downscaled)
        self._track_x = 0.0
        self._track_y = 0.0
        
        # Scale factor: display = tracking * scale_factor
        self._scale_factor = 1.0
        
        # Frame dimensions
        self._display_w = 0
        self._display_h = 0
        self._track_w = 0
        self._track_h = 0
        
        # Counters
        self._frame_count = 0
        self._frames_lost = 0
        self._frames_occluded = 0
        
        # Label
        self.label = ""
    
    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        """Downscale frame to tracking resolution."""
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
        """Convert tracking coords to display coords."""
        return x * self._scale_factor, y * self._scale_factor
    
    def _to_tracking(self, x: float, y: float) -> Tuple[float, float]:
        """Convert display coords to tracking coords."""
        return x / self._scale_factor, y / self._scale_factor
    
    def _is_outside_frame(self, x: float, y: float) -> bool:
        """Check if position is outside frame bounds."""
        return (x < self.EDGE_MARGIN or 
                x > self._track_w - self.EDGE_MARGIN or
                y < self.EDGE_MARGIN or 
                y > self._track_h - self.EDGE_MARGIN)
    
    def initialize(self, frame: np.ndarray, x: int, y: int, label: str = "") -> bool:
        """
        Initialize tracker with a click position.
        
        Captures the visual fingerprint and starts tracking.
        
        Args:
            frame: Full resolution frame
            x, y: Click position in display coordinates
            label: Optional label for annotation
            
        Returns:
            True if initialization successful
        """
        self.label = label
        
        # Downscale frame
        small_frame = self._downscale(frame)
        
        # Convert click to tracking coordinates
        track_x, track_y = self._to_tracking(float(x), float(y))
        
        # Capture visual fingerprint
        if not self.fingerprint.initialize(small_frame, int(track_x), int(track_y)):
            self.logger.warning("Failed to capture visual fingerprint")
            # Continue anyway - we can still use optical flow
        
        # Initialize optical flow
        self.optical_flow.initialize(small_frame, int(track_x), int(track_y))
        
        # Set initial state
        self._track_x = track_x
        self._track_y = track_y
        self._x = float(x)
        self._y = float(y)
        self._prev_x = self._x
        self._prev_y = self._y
        self._last_good_position = (self._x, self._y)
        
        self._status = TrackingStatus.TRACKING
        self._confidence = 1.0
        self._frame_count = 0
        self._frames_lost = 0
        self._frames_occluded = 0
        
        self.logger.info(
            f"Initialized '{label}' at display({x}, {y}) "
            f"track({track_x:.0f}, {track_y:.0f}) "
            f"fingerprint={self.fingerprint.has_features}"
        )
        return True
    
    def _run_correction(self, frame: np.ndarray) -> Tuple[bool, float, float, float]:
        """
        Run SIFT feature matching for drift correction.
        
        Returns:
            (success, new_x, new_y, confidence) in tracking coords
        """
        if not self.fingerprint.has_features:
            return False, self._track_x, self._track_y, self._confidence
        
        # Detect features at current position
        kp_current, desc_current, roi_bounds = self.fingerprint.detect_features(
            frame, int(self._track_x), int(self._track_y)
        )
        
        if desc_current is None or len(desc_current) < 4:
            return False, self._track_x, self._track_y, 0.3
        
        # Match against fingerprint
        matches = self.matcher.match_descriptors(
            self.fingerprint.descriptors, desc_current
        )
        
        if len(matches) < self.FEATURE_MIN_MATCHES:
            return False, self._track_x, self._track_y, len(matches) / 20.0
        
        # Compute homography
        H, src_pts, dst_pts = self.matcher.compute_homography(
            self.fingerprint.keypoints,
            kp_current,
            matches,
            roi_offset=(roi_bounds[0], roi_bounds[1])
        )
        
        if H is None:
            # No valid homography, but we have some matches
            # Use average of matched points as estimate
            if len(dst_pts) > 0:
                avg_x = float(dst_pts[:, 0, 0].mean())
                avg_y = float(dst_pts[:, 0, 1].mean())
                return True, avg_x, avg_y, len(matches) / 20.0
            return False, self._track_x, self._track_y, 0.3
        
        # Estimate new center using homography
        new_x, new_y = self.matcher.estimate_new_center(
            H, 
            (self.fingerprint.roi_size // 2, self.fingerprint.roi_size // 2),
            self.fingerprint.bbox_w_h
        )
        
        confidence = min(1.0, len(matches) / 15.0)
        
        return True, new_x, new_y, confidence
    
    def _run_global_search(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Search entire frame for the object.
        
        EXPENSIVE! Only run when LOST.
        
        Returns:
            (x, y, confidence) in tracking coords, or None if not found
        """
        if not self.fingerprint.has_features:
            return None
        
        # Detect features on entire frame
        kp_frame, desc_frame = self.fingerprint.detect_features_fullframe(frame)
        
        if desc_frame is None or len(desc_frame) < 4:
            return None
        
        # Match against fingerprint
        matches = self.matcher.match_descriptors(
            self.fingerprint.descriptors, desc_frame
        )
        
        if len(matches) < self.FEATURE_MIN_MATCHES:
            return None
        
        # Compute homography (ROI offset is 0,0 since we searched full frame)
        H, src_pts, dst_pts = self.matcher.compute_homography(
            self.fingerprint.keypoints,
            kp_frame,
            matches,
            roi_offset=(0, 0)
        )
        
        if H is not None:
            # Use homography to find center
            new_x, new_y = self.matcher.estimate_new_center(
                H,
                (self.fingerprint.roi_size // 2, self.fingerprint.roi_size // 2),
                self.fingerprint.bbox_w_h
            )
            confidence = min(1.0, len(matches) / 15.0)
            return new_x, new_y, confidence
        
        # Fallback: average position of matched keypoints
        if len(dst_pts) >= self.FEATURE_MIN_MATCHES:
            avg_x = float(dst_pts[:, 0, 0].mean())
            avg_y = float(dst_pts[:, 0, 1].mean())
            return avg_x, avg_y, len(matches) / 20.0
        
        return None
    
    def update(self, frame: np.ndarray) -> TrackingState:
        """
        Update tracker with new frame.
        
        Hybrid Strategy:
        1. Every frame: Run fast optical flow
        2. Every 15 frames: Run SIFT correction
        3. When LOST: Run global SIFT search
        
        Returns:
            Current tracking state
        """
        self._frame_count += 1
        
        # Handle inactive state
        if self._status == TrackingStatus.INACTIVE:
            return TrackingState(
                status=TrackingStatus.INACTIVE,
                x=0, y=0, confidence=0, is_occluded=False,
                visibility=0.0, opacity=0.0
            )
        
        # Store previous position
        self._prev_x = self._x
        self._prev_y = self._y
        
        # Downscale frame
        small_frame = self._downscale(frame)
        
        # =========================================
        # STATE: TRACKING
        # =========================================
        if self._status == TrackingStatus.TRACKING:
            # === FAST PATH: Optical Flow ===
            new_x, new_y, flow_conf, valid = self.optical_flow.track(small_frame)
            
            if not valid or flow_conf < self.FLOW_MIN_CONFIDENCE:
                # Optical flow failed → OCCLUDED
                self._status = TrackingStatus.OCCLUDED
                self._frames_occluded = 0
                self._confidence = flow_conf
            else:
                # Check if object exited frame
                if self._is_outside_frame(new_x, new_y):
                    # Object LEFT the frame → immediately LOST
                    self._status = TrackingStatus.LOST
                    self._frames_lost = 0
                    self._confidence = 0.0
                    self.logger.info(f"Object exited frame at ({new_x:.0f}, {new_y:.0f})")
                else:
                    # Still tracking
                    self._track_x = new_x
                    self._track_y = new_y
                    self._confidence = flow_conf
                    
                    # === SLOW PATH: Periodic Correction ===
                    if self._frame_count % self.CORRECTION_INTERVAL == 0:
                        success, corr_x, corr_y, corr_conf = self._run_correction(small_frame)
                        
                        if success and corr_conf > 0.5:
                            # Apply correction (blend with optical flow result)
                            blend = 0.7  # Weight towards SIFT correction
                            self._track_x = self._track_x * (1 - blend) + corr_x * blend
                            self._track_y = self._track_y * (1 - blend) + corr_y * blend
                            self._confidence = (flow_conf + corr_conf) / 2
                            
                            # Reset optical flow points to corrected position
                            self.optical_flow.reset_points(
                                int(self._track_x), int(self._track_y), small_frame
                            )
                        elif not success:
                            # SIFT failed but optical flow is still OK
                            # Might be occluded
                            self._confidence = flow_conf * 0.8
                    
                    # Update display coordinates
                    self._x, self._y = self._to_display(self._track_x, self._track_y)
                    self._last_good_position = (self._x, self._y)
        
        # =========================================
        # STATE: OCCLUDED
        # =========================================
        elif self._status == TrackingStatus.OCCLUDED:
            self._frames_occluded += 1
            
            # Try optical flow
            new_x, new_y, flow_conf, valid = self.optical_flow.track(small_frame)
            
            if valid and flow_conf > self.FLOW_MIN_CONFIDENCE * 1.5:
                # Check if outside frame
                if self._is_outside_frame(new_x, new_y):
                    self._status = TrackingStatus.LOST
                    self._frames_lost = 0
                else:
                    # Try SIFT verification
                    success, corr_x, corr_y, corr_conf = self._run_correction(small_frame)
                    
                    if success and corr_conf > 0.6:
                        # Recovered!
                        self._status = TrackingStatus.TRACKING
                        self._track_x = corr_x
                        self._track_y = corr_y
                        self._confidence = corr_conf
                        self._x, self._y = self._to_display(corr_x, corr_y)
                        self._last_good_position = (self._x, self._y)
                        
                        self.optical_flow.reset_points(int(corr_x), int(corr_y), small_frame)
                    else:
                        # Still occluded but tracking position
                        self._track_x = new_x
                        self._track_y = new_y
                        self._confidence = flow_conf * 0.5
                        self._x, self._y = self._to_display(new_x, new_y)
            
            # Timeout → LOST
            if self._frames_occluded > self.OCCLUSION_TIMEOUT:
                self._status = TrackingStatus.LOST
                self._frames_lost = 0
        
        # =========================================
        # STATE: LOST
        # =========================================
        elif self._status == TrackingStatus.LOST:
            self._frames_lost += 1
            self._confidence = 0.0
            
            if self.enable_reid:
                # Transition to SEARCHING and run search immediately
                self._status = TrackingStatus.SEARCHING
                # Fall through to SEARCHING logic below
        
        # =========================================
        # STATE: SEARCHING
        # =========================================
        if self._status == TrackingStatus.SEARCHING:
            # Run global search on first frame of SEARCHING, or periodically
            should_search = (
                self._frames_lost <= 1 or  # First frame in SEARCHING
                self._frames_lost % self.SEARCH_INTERVAL == 0
            )
            
            if should_search:
                result = self._run_global_search(small_frame)
                
                if result:
                    new_x, new_y, conf = result
                    
                    # Validate position is inside frame
                    if not self._is_outside_frame(new_x, new_y):
                        # FOUND! Re-acquired object
                        self._status = TrackingStatus.TRACKING
                        self._track_x = new_x
                        self._track_y = new_y
                        self._confidence = conf
                        self._x, self._y = self._to_display(new_x, new_y)
                        self._last_good_position = (self._x, self._y)
                        self._frames_lost = 0
                        
                        # Reset optical flow
                        self.optical_flow.reset_points(int(new_x), int(new_y), small_frame)
                        
                        self.logger.info(f"Re-acquired object at ({new_x:.0f}, {new_y:.0f})")
            
            # Increment lost counter for SEARCHING state (not just LOST)
            if self._status == TrackingStatus.SEARCHING:
                self._frames_lost += 1
            
            # Timeout - give up searching
            if self._frames_lost > self.LOST_TIMEOUT:
                self._status = TrackingStatus.LOST
                self._confidence = 0.0
        
        # Compute velocity
        self._velocity = (self._x - self._prev_x, self._y - self._prev_y)
        
        # Compute visibility/opacity for annotation layer
        if self._status == TrackingStatus.TRACKING:
            visibility = self._confidence
            opacity = 1.0
            is_occluded = False
        elif self._status == TrackingStatus.OCCLUDED:
            visibility = max(0.3, self._confidence)
            opacity = 0.5  # Yellow, 50% opacity
            is_occluded = True
        else:  # LOST, SEARCHING
            visibility = 0.0
            opacity = 0.0  # Hidden
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
            last_good_position=self._last_good_position,
            scale=1.0,
            rotation=0.0
        )
    
    def reset(self):
        """Reset tracker to inactive state."""
        self._status = TrackingStatus.INACTIVE
        self._x = 0
        self._y = 0
        self._confidence = 0
        self._frame_count = 0
        self._frames_lost = 0
        self._frames_occluded = 0
        self.fingerprint = VisualFingerprint()
        self.optical_flow = FastOpticalFlow()
    
    @property
    def status(self) -> TrackingStatus:
        return self._status
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self._x, self._y)
    
    @property
    def confidence(self) -> float:
        return self._confidence


# Alias for backward compatibility
ObjectTracker = HybridTracker


class TrackerManager:
    """
    Manages multiple HybridTrackers.
    
    Provides a simple interface to:
    - Create trackers with a click
    - Update all trackers each frame
    - Remove lost trackers
    """
    
    def __init__(self, use_gpu: bool = True, enable_reid: bool = True):
        """
        Args:
            use_gpu: Legacy parameter (not used, kept for compatibility)
            enable_reid: Enable re-identification when objects are lost
        """
        self.enable_reid = enable_reid
        self._trackers: Dict[str, HybridTracker] = {}
        self.logger = logging.getLogger("TrackerManager")
    
    def create_tracker(self, frame: np.ndarray, x: int, y: int, label: str = "") -> str:
        """
        Create and initialize a new tracker.
        
        Args:
            frame: Current frame
            x, y: Click position (display coordinates)
            label: Optional label
            
        Returns:
            Tracker ID
        """
        tracker = HybridTracker(enable_reid=self.enable_reid)
        tracker.initialize(frame, x, y, label)
        self._trackers[tracker.tracker_id] = tracker
        self.logger.info(f"Created tracker {tracker.tracker_id} for '{label}'")
        return tracker.tracker_id
    
    def update_all(self, frame: np.ndarray) -> Dict[str, TrackingState]:
        """
        Update all trackers with new frame.
        
        Returns:
            Dict of tracker_id -> TrackingState
        """
        results = {}
        for tracker_id, tracker in self._trackers.items():
            results[tracker_id] = tracker.update(frame)
        return results
    
    def get_tracker(self, tracker_id: str) -> Optional[HybridTracker]:
        """Get tracker by ID."""
        return self._trackers.get(tracker_id)
    
    def remove_tracker(self, tracker_id: str):
        """Remove a tracker."""
        if tracker_id in self._trackers:
            del self._trackers[tracker_id]
    
    def remove_lost_trackers(self) -> List[str]:
        """Remove trackers that have been lost too long."""
        lost = [
            tid for tid, t in self._trackers.items()
            if t.status == TrackingStatus.LOST and t._frames_lost > HybridTracker.LOST_TIMEOUT
        ]
        for tid in lost:
            del self._trackers[tid]
        return lost
    
    def clear_all(self):
        """Remove all trackers."""
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


# Performance overlay (moved here for convenience)
class PerformanceOverlay:
    """Renders FPS and latency overlay on frame."""
    
    def __init__(self):
        self._last_time = time.perf_counter()
        self._fps_values: List[float] = []
        self._avg_fps = 0.0
        self._latency_ms = 0.0
    
    def update(self, latency_ms: float = 0.0):
        """Update metrics."""
        current_time = time.perf_counter()
        dt = current_time - self._last_time
        self._last_time = current_time
        
        if dt > 0:
            fps = 1.0 / dt
            self._fps_values.append(fps)
            if len(self._fps_values) > 30:
                self._fps_values.pop(0)
            self._avg_fps = sum(self._fps_values) / len(self._fps_values)
        
        self._latency_ms = latency_ms
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS and latency on frame."""
        # Color based on FPS
        if self._avg_fps >= 30:
            color = (0, 255, 0)  # Green
        elif self._avg_fps >= 20:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # FPS
        cv2.putText(
            frame, f"FPS: {self._avg_fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
        )
        
        # Latency
        cv2.putText(
            frame, f"Latency: {self._latency_ms:.0f}ms",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA
        )
        
        return frame
    
    @property
    def fps(self) -> float:
        return self._avg_fps


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Hybrid Fast-Slow Tracker...")
    
    # Create test frame with distinctive object
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    
    # Draw object with texture (for SIFT features)
    cv2.rectangle(frame, (900, 500), (1000, 600), (0, 120, 255), -1)
    cv2.circle(frame, (950, 550), 25, (255, 255, 255), -1)
    cv2.circle(frame, (935, 540), 8, (0, 0, 0), -1)
    cv2.circle(frame, (965, 540), 8, (0, 0, 0), -1)
    
    # Create tracker
    tracker = HybridTracker(enable_reid=True)
    tracker.initialize(frame, 950, 550, label="Test Object")
    
    print(f"Display size: {tracker._display_w}x{tracker._display_h}")
    print(f"Tracking size: {tracker._track_w}x{tracker._track_h}")
    print(f"Scale factor: {tracker._scale_factor:.2f}")
    print(f"Fingerprint has features: {tracker.fingerprint.has_features}")
    if tracker.fingerprint.has_features:
        print(f"  Keypoints: {len(tracker.fingerprint.keypoints)}")
    
    # Benchmark
    import time
    times = []
    correction_times = []
    
    for i in range(100):
        start = time.perf_counter()
        state = tracker.update(frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        
        if (i + 1) % 15 == 0:
            correction_times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    avg_correction = sum(correction_times) / len(correction_times) if correction_times else 0
    
    print(f"\nPerformance (100 frames):")
    print(f"  Average frame: {avg_time:.2f}ms")
    print(f"  Min: {min(times):.2f}ms")
    print(f"  Max: {max(times):.2f}ms")
    print(f"  Correction frames (every 15th): {avg_correction:.2f}ms")
    print(f"  Estimated FPS: {1000 / avg_time:.1f}")
    
    if 1000 / avg_time >= 30:
        print("\n✅ PASS: Real-time performance (>30 FPS)")
    else:
        print("\n⚠️  Below 30 FPS target")
    
    print(f"\nFinal state: {state.status.value}, confidence: {state.confidence:.2f}")
    print("Test complete.")
