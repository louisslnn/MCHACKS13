"""
HoloRay Core - Object Tracking Engine with Occlusion Detection and Re-Identification

The brain of the HoloRay AR annotation system. Tracks objects clicked by users
and maintains sticky annotations even through occlusion and leaving/re-entering frame.
"""

import time
import logging
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import uuid

import numpy as np
import cv2
import torch
import torch.nn.functional as F


class TrackingStatus(Enum):
    """Status of a tracked object."""
    TRACKING = "tracking"      # Actively tracking, high confidence
    OCCLUDED = "occluded"      # Object temporarily blocked (e.g., by hand)
    LOST = "lost"              # Object left frame or tracking failed
    SEARCHING = "searching"    # Looking for re-identification
    INACTIVE = "inactive"      # Tracker not initialized


@dataclass
class TrackingState:
    """Current state of a tracked object."""
    status: TrackingStatus
    x: float
    y: float
    confidence: float
    is_occluded: bool
    velocity: Tuple[float, float] = (0.0, 0.0)
    frames_since_seen: int = 0
    last_good_position: Tuple[float, float] = (0.0, 0.0)


@dataclass
class FeaturePatch:
    """Feature descriptor for Re-Identification."""
    patch: np.ndarray  # 64x64 grayscale patch
    keypoints: List[cv2.KeyPoint] = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    histogram: Optional[np.ndarray] = None


class SmartPadding:
    """
    Padding utility for making frame dimensions divisible by stride.
    Used for CoTracker compatibility.
    """

    def __init__(self, stride: int = 16):
        self.stride = stride
        self.pad_h = 0
        self.pad_w = 0
        self.original_h = 0
        self.original_w = 0

    def compute_padding(self, height: int, width: int) -> Tuple[int, int, int, int]:
        """Compute padding needed."""
        self.original_h = height
        self.original_w = width
        self.pad_h = (self.stride - (height % self.stride)) % self.stride
        self.pad_w = (self.stride - (width % self.stride)) % self.stride

        pad_top = self.pad_h // 2
        pad_bottom = self.pad_h - pad_top
        pad_left = self.pad_w // 2
        pad_right = self.pad_w - pad_left

        return (pad_left, pad_right, pad_top, pad_bottom)

    def pad_frame(self, frame: np.ndarray) -> np.ndarray:
        """Pad frame to be divisible by stride."""
        h, w = frame.shape[:2]
        pad_left, pad_right, pad_top, pad_bottom = self.compute_padding(h, w)

        if self.pad_h == 0 and self.pad_w == 0:
            return frame

        if len(frame.shape) == 3:
            return np.pad(frame, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
        return np.pad(frame, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')

    def unpad_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """Convert padded coordinates to original space."""
        pad_left = self.pad_w // 2
        pad_top = self.pad_h // 2
        return (
            max(0, min(x - pad_left, self.original_w - 1)),
            max(0, min(y - pad_top, self.original_h - 1))
        )

    def pad_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """Convert original coordinates to padded space."""
        pad_left = self.pad_w // 2
        pad_top = self.pad_h // 2
        return (x + pad_left, y + pad_top)


class FeatureExtractor:
    """
    Extracts features for Re-Identification using ORB descriptors.
    """

    def __init__(self, patch_size: int = 64):
        self.patch_size = patch_size
        self.orb = cv2.ORB_create(nfeatures=50)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract_patch(self, frame: np.ndarray, x: int, y: int) -> np.ndarray:
        """Extract a patch centered at (x, y)."""
        h, w = frame.shape[:2]
        half = self.patch_size // 2

        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w, x + half)
        y2 = min(h, y + half)

        patch = frame[y1:y2, x1:x2]

        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))

        return patch

    def extract_features(self, frame: np.ndarray, x: int, y: int) -> FeaturePatch:
        """Extract feature patch with ORB descriptors."""
        patch = self.extract_patch(frame, x, y)

        # Convert to grayscale if needed
        if len(patch.shape) == 3:
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray_patch = patch

        # Extract ORB features
        keypoints, descriptors = self.orb.detectAndCompute(gray_patch, None)

        # Compute color histogram for additional matching
        if len(patch.shape) == 3:
            hist = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
        else:
            hist = cv2.calcHist([gray_patch], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

        return FeaturePatch(
            patch=gray_patch,
            keypoints=list(keypoints) if keypoints else [],
            descriptors=descriptors,
            histogram=hist
        )

    def match_features(self, feat1: FeaturePatch, feat2: FeaturePatch) -> float:
        """
        Compute similarity score between two feature patches.

        Returns:
            Similarity score (0.0 to 1.0), higher = more similar
        """
        score = 0.0
        weights_sum = 0.0

        # ORB descriptor matching
        if feat1.descriptors is not None and feat2.descriptors is not None:
            try:
                matches = self.bf_matcher.match(feat1.descriptors, feat2.descriptors)
                if matches:
                    avg_distance = sum(m.distance for m in matches) / len(matches)
                    # Normalize: lower distance = higher score
                    orb_score = max(0, 1.0 - (avg_distance / 100.0))
                    score += orb_score * 0.5
                    weights_sum += 0.5
            except cv2.error:
                pass

        # Histogram comparison
        if feat1.histogram is not None and feat2.histogram is not None:
            hist_score = cv2.compareHist(feat1.histogram, feat2.histogram, cv2.HISTCMP_CORREL)
            hist_score = (hist_score + 1.0) / 2.0  # Normalize to 0-1
            score += hist_score * 0.3
            weights_sum += 0.3

        # Template matching on patches
        if feat1.patch is not None and feat2.patch is not None:
            result = cv2.matchTemplate(feat2.patch, feat1.patch, cv2.TM_CCOEFF_NORMED)
            template_score = (result.max() + 1.0) / 2.0
            score += template_score * 0.2
            weights_sum += 0.2

        return score / weights_sum if weights_sum > 0 else 0.0


class ObjectTracker:
    """
    HoloRay Object Tracker - Tracks a single object through video frames.

    Features:
    - Point tracking using CoTracker3 (GPU) or Lucas-Kanade optical flow (CPU fallback)
    - Smart Occlusion Detection: Detects when object is temporarily blocked
    - Re-Identification: Recovers tracking when object re-enters frame

    Usage:
        tracker = ObjectTracker()
        tracker.initialize(frame, x, y, label="Pawn")

        while running:
            state = tracker.update(frame)
            if state.status == TrackingStatus.TRACKING:
                draw_annotation(state.x, state.y)
    """

    # Thresholds
    CONFIDENCE_HIGH = 0.7
    CONFIDENCE_LOW = 0.3
    OCCLUSION_CONFIDENCE_DROP = 0.4  # If confidence drops by this much suddenly
    LOST_FRAMES_THRESHOLD = 30  # Frames before marking as LOST
    REID_MATCH_THRESHOLD = 0.6  # Minimum similarity for re-identification
    EDGE_MARGIN = 50  # Pixels from edge to search for re-entry

    def __init__(
        self,
        tracker_id: Optional[str] = None,
        use_gpu: bool = True,
        enable_reid: bool = True
    ):
        """
        Initialize ObjectTracker.

        Args:
            tracker_id: Unique identifier (auto-generated if None)
            use_gpu: Use GPU-accelerated tracking if available
            enable_reid: Enable re-identification feature
        """
        self.tracker_id = tracker_id or str(uuid.uuid4())[:8]
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_reid = enable_reid

        self.logger = logging.getLogger(f"ObjectTracker-{self.tracker_id}")

        # State
        self._status = TrackingStatus.INACTIVE
        self._x = 0.0
        self._y = 0.0
        self._confidence = 0.0
        self._velocity = (0.0, 0.0)
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._frames_since_seen = 0
        self._last_good_position = (0.0, 0.0)
        self._prev_confidence = 1.0

        # Frame info
        self._frame_width = 0
        self._frame_height = 0
        self._prev_frame: Optional[np.ndarray] = None

        # Tracking points (for optical flow)
        self._track_points: Optional[np.ndarray] = None
        self._initial_points: Optional[np.ndarray] = None

        # CoTracker (GPU)
        self._cotracker = None
        self._cotracker_initialized = False
        self._padding = SmartPadding(stride=16)

        # Re-ID
        self._feature_extractor = FeatureExtractor() if enable_reid else None
        self._reference_features: Optional[FeaturePatch] = None

        # Label
        self.label: str = ""

    def _init_cotracker(self):
        """Initialize CoTracker3 if available."""
        if not self.use_gpu or self._cotracker is not None:
            return

        try:
            from cotracker.predictor import CoTrackerOnlinePredictor
            self._cotracker = CoTrackerOnlinePredictor(checkpoint=None)
            self._cotracker = self._cotracker.cuda()
            self.logger.info("CoTracker3 initialized on GPU")
        except ImportError:
            self.logger.info("CoTracker3 not available, using optical flow fallback")
            self._cotracker = None
        except Exception as e:
            self.logger.warning(f"CoTracker3 init failed: {e}")
            self._cotracker = None

    def _create_point_grid(self, x: int, y: int, grid_size: int = 5, spacing: int = 10) -> np.ndarray:
        """Create a grid of tracking points around the center."""
        points = []
        half = grid_size // 2

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px = x + dx * spacing
                py = y + dy * spacing
                # Clamp to frame bounds
                px = max(0, min(px, self._frame_width - 1))
                py = max(0, min(py, self._frame_height - 1))
                points.append([px, py])

        return np.array(points, dtype=np.float32)

    def initialize(self, frame: np.ndarray, x: int, y: int, label: str = "") -> bool:
        """
        Initialize tracking on an object.

        Args:
            frame: Current video frame (BGR)
            x, y: Initial object position (from user click)
            label: Label text for annotation

        Returns:
            True if initialization successful
        """
        self._frame_height, self._frame_width = frame.shape[:2]
        self._x = float(x)
        self._y = float(y)
        self._prev_x = self._x
        self._prev_y = self._y
        self._last_good_position = (self._x, self._y)
        self.label = label

        # Create tracking points grid
        self._track_points = self._create_point_grid(x, y)
        self._initial_points = self._track_points.copy()

        # Store grayscale for optical flow
        self._prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract reference features for Re-ID
        if self._feature_extractor:
            self._reference_features = self._feature_extractor.extract_features(frame, x, y)

        # Try to init CoTracker
        self._init_cotracker()

        if self._cotracker:
            try:
                # Prepare frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_padded = self._padding.pad_frame(frame_rgb)
                frame_tensor = torch.from_numpy(frame_padded).permute(2, 0, 1).float()
                frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0).cuda()

                # Prepare queries (with padding offset)
                padded_coords = self._padding.pad_coordinates(x, y)
                queries = torch.tensor([[[0, padded_coords[0], padded_coords[1]]]], dtype=torch.float32).cuda()

                self._cotracker(frame_tensor, is_first_step=True, queries=queries)
                self._cotracker_initialized = True
            except Exception as e:
                self.logger.warning(f"CoTracker init error: {e}")
                self._cotracker = None

        self._status = TrackingStatus.TRACKING
        self._confidence = 1.0
        self._prev_confidence = 1.0
        self._frames_since_seen = 0

        self.logger.info(f"Initialized at ({x}, {y}) with label '{label}'")
        return True

    def _track_optical_flow(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Track points using Lucas-Kanade optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._track_points is None or len(self._track_points) == 0:
            return np.array([]), np.array([])

        # Lucas-Kanade parameters
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Track points
        points = self._track_points.reshape(-1, 1, 2)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_frame, gray, points, None, **lk_params
        )

        self._prev_frame = gray

        if next_points is None:
            return np.array([]), np.array([])

        status = status.flatten()
        next_points = next_points.reshape(-1, 2)

        return next_points, status

    def _track_cotracker(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """Track using CoTracker3."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_padded = self._padding.pad_frame(frame_rgb)
            frame_tensor = torch.from_numpy(frame_padded).permute(2, 0, 1).float()
            frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0).cuda()

            pred_tracks, pred_vis = self._cotracker(frame_tensor, is_first_step=False)

            if pred_tracks is not None and pred_tracks.numel() > 0:
                # Get latest position
                pos = pred_tracks[0, -1, 0].cpu().numpy()
                vis = pred_vis[0, -1, 0].cpu().item()

                # Unpad coordinates
                x, y = self._padding.unpad_coordinates(pos[0], pos[1])
                return x, y, vis

        except Exception as e:
            self.logger.warning(f"CoTracker error: {e}")

        return self._x, self._y, 0.0

    def _detect_occlusion(self, new_confidence: float) -> bool:
        """
        Detect if object is occluded based on confidence drop.

        Logic: If confidence drops suddenly but object hasn't left frame,
        it's likely occluded (e.g., hand passing in front).
        """
        confidence_drop = self._prev_confidence - new_confidence

        # Check for sudden confidence drop
        if confidence_drop > self.OCCLUSION_CONFIDENCE_DROP:
            # Object still in frame bounds?
            if (self.EDGE_MARGIN < self._x < self._frame_width - self.EDGE_MARGIN and
                self.EDGE_MARGIN < self._y < self._frame_height - self.EDGE_MARGIN):
                return True

        return False

    def _check_frame_exit(self) -> bool:
        """Check if object has exited the frame."""
        margin = self.EDGE_MARGIN // 2
        return (self._x < margin or self._x > self._frame_width - margin or
                self._y < margin or self._y > self._frame_height - margin)

    def _search_for_reentry(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Search frame edges for re-entering object using feature matching.

        Returns:
            (x, y) if object found, None otherwise
        """
        if not self._feature_extractor or self._reference_features is None:
            return None

        h, w = frame.shape[:2]
        margin = self.EDGE_MARGIN
        best_match = None
        best_score = self.REID_MATCH_THRESHOLD

        # Search regions: top, bottom, left, right edges
        search_regions = [
            (0, 0, w, margin),           # Top
            (0, h - margin, w, h),       # Bottom
            (0, margin, margin, h - margin),      # Left
            (w - margin, margin, w, h - margin),  # Right
        ]

        for x1, y1, x2, y2 in search_regions:
            # Sample points in this region
            for test_x in range(x1 + 32, x2 - 32, 32):
                for test_y in range(y1 + 32, y2 - 32, 32):
                    # Extract features at this point
                    test_features = self._feature_extractor.extract_features(frame, test_x, test_y)

                    # Match against reference
                    score = self._feature_extractor.match_features(
                        self._reference_features, test_features
                    )

                    if score > best_score:
                        best_score = score
                        best_match = (float(test_x), float(test_y))

        if best_match:
            self.logger.info(f"Re-ID match found at {best_match} with score {best_score:.2f}")

        return best_match

    def update(self, frame: np.ndarray) -> TrackingState:
        """
        Update tracker with new frame.

        Args:
            frame: Current video frame (BGR)

        Returns:
            TrackingState with current position and status
        """
        if self._status == TrackingStatus.INACTIVE:
            return TrackingState(
                status=TrackingStatus.INACTIVE,
                x=0, y=0, confidence=0, is_occluded=False
            )

        # Track position
        if self._cotracker and self._cotracker_initialized:
            new_x, new_y, confidence = self._track_cotracker(frame)
        else:
            # Optical flow fallback
            next_points, status = self._track_optical_flow(frame)

            if len(next_points) > 0 and status.sum() > 0:
                # Filter valid points
                valid_mask = status == 1
                valid_points = next_points[valid_mask]

                if len(valid_points) > 0:
                    new_x = valid_points[:, 0].mean()
                    new_y = valid_points[:, 1].mean()
                    confidence = valid_mask.mean()

                    # Update tracking points
                    self._track_points = valid_points
                else:
                    new_x, new_y = self._x, self._y
                    confidence = 0.1
            else:
                new_x, new_y = self._x, self._y
                confidence = 0.0

        # Calculate velocity
        self._velocity = (new_x - self._x, new_y - self._y)

        # Detect occlusion
        is_occluded = self._detect_occlusion(confidence)

        # Update state based on confidence and position
        if confidence > self.CONFIDENCE_HIGH:
            self._status = TrackingStatus.TRACKING
            self._x = new_x
            self._y = new_y
            self._last_good_position = (new_x, new_y)
            self._frames_since_seen = 0

        elif confidence > self.CONFIDENCE_LOW:
            if is_occluded:
                self._status = TrackingStatus.OCCLUDED
                # Keep using predicted position based on velocity
                self._x = self._last_good_position[0] + self._velocity[0] * 0.5
                self._y = self._last_good_position[1] + self._velocity[1] * 0.5
            else:
                self._status = TrackingStatus.TRACKING
                self._x = new_x
                self._y = new_y
            self._frames_since_seen += 1

        else:
            # Low confidence - check if lost or just occluded
            self._frames_since_seen += 1

            if self._check_frame_exit():
                self._status = TrackingStatus.LOST
            elif self._frames_since_seen > self.LOST_FRAMES_THRESHOLD:
                self._status = TrackingStatus.LOST
            else:
                self._status = TrackingStatus.OCCLUDED
                is_occluded = True

        # Try re-identification if lost
        if self._status == TrackingStatus.LOST and self.enable_reid:
            self._status = TrackingStatus.SEARCHING
            reid_pos = self._search_for_reentry(frame)

            if reid_pos:
                # Re-initialize at new position
                self._x, self._y = reid_pos
                self._status = TrackingStatus.TRACKING
                self._frames_since_seen = 0
                self._confidence = 0.8

                # Re-create tracking points
                self._track_points = self._create_point_grid(int(self._x), int(self._y))
                self._prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                self.logger.info(f"Re-acquired object at ({self._x:.0f}, {self._y:.0f})")

        # Store for next iteration
        self._prev_confidence = confidence
        self._confidence = confidence
        self._prev_x = self._x
        self._prev_y = self._y

        return TrackingState(
            status=self._status,
            x=self._x,
            y=self._y,
            confidence=self._confidence,
            is_occluded=is_occluded,
            velocity=self._velocity,
            frames_since_seen=self._frames_since_seen,
            last_good_position=self._last_good_position
        )

    def reset(self):
        """Reset tracker to inactive state."""
        self._status = TrackingStatus.INACTIVE
        self._x = 0
        self._y = 0
        self._confidence = 0
        self._track_points = None
        self._prev_frame = None
        self._frames_since_seen = 0
        self._cotracker_initialized = False

    @property
    def status(self) -> TrackingStatus:
        """Get current tracking status."""
        return self._status

    @property
    def position(self) -> Tuple[float, float]:
        """Get current position."""
        return (self._x, self._y)

    @property
    def confidence(self) -> float:
        """Get current confidence."""
        return self._confidence


class TrackerManager:
    """
    Manages multiple ObjectTrackers for tracking many objects simultaneously.
    """

    def __init__(self, use_gpu: bool = True, enable_reid: bool = True):
        self.use_gpu = use_gpu
        self.enable_reid = enable_reid
        self._trackers: Dict[str, ObjectTracker] = {}
        self.logger = logging.getLogger("TrackerManager")

    def create_tracker(self, frame: np.ndarray, x: int, y: int, label: str = "") -> str:
        """
        Create and initialize a new tracker.

        Returns:
            Tracker ID
        """
        tracker = ObjectTracker(
            use_gpu=self.use_gpu,
            enable_reid=self.enable_reid
        )
        tracker.initialize(frame, x, y, label)
        self._trackers[tracker.tracker_id] = tracker
        self.logger.info(f"Created tracker {tracker.tracker_id} for '{label}'")
        return tracker.tracker_id

    def update_all(self, frame: np.ndarray) -> Dict[str, TrackingState]:
        """Update all trackers with new frame."""
        results = {}
        for tracker_id, tracker in self._trackers.items():
            results[tracker_id] = tracker.update(frame)
        return results

    def get_tracker(self, tracker_id: str) -> Optional[ObjectTracker]:
        """Get tracker by ID."""
        return self._trackers.get(tracker_id)

    def remove_tracker(self, tracker_id: str):
        """Remove a tracker."""
        if tracker_id in self._trackers:
            del self._trackers[tracker_id]

    def remove_lost_trackers(self):
        """Remove all trackers with LOST status."""
        lost = [tid for tid, t in self._trackers.items() if t.status == TrackingStatus.LOST]
        for tid in lost:
            del self._trackers[tid]
        return lost

    @property
    def tracker_ids(self) -> List[str]:
        """Get all tracker IDs."""
        return list(self._trackers.keys())

    @property
    def active_count(self) -> int:
        """Get count of active (non-lost) trackers."""
        return sum(1 for t in self._trackers.values() if t.status != TrackingStatus.LOST)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Testing ObjectTracker...")

    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Draw a white square as "object"
    cv2.rectangle(test_frame, (300, 220), (340, 260), (255, 255, 255), -1)

    # Initialize tracker
    tracker = ObjectTracker(use_gpu=False, enable_reid=True)
    tracker.initialize(test_frame, 320, 240, label="TestObject")

    print(f"Status: {tracker.status}")
    print(f"Position: {tracker.position}")
    print(f"Confidence: {tracker.confidence}")

    # Simulate update
    state = tracker.update(test_frame)
    print(f"\nAfter update:")
    print(f"Status: {state.status}")
    print(f"Position: ({state.x:.1f}, {state.y:.1f})")
    print(f"Confidence: {state.confidence:.2f}")
    print(f"Occluded: {state.is_occluded}")

    print("\nTest complete.")
