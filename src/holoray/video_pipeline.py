"""
HoloRay Video Pipeline - High-Performance Threaded Video Reader

Provides zero-latency frame access by continuously capturing frames
in a background thread and always returning the freshest frame.
"""

import threading
import time
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging

import cv2
import numpy as np


class VideoSource(Enum):
    """Video source types."""
    WEBCAM = "webcam"
    FILE = "file"
    RTSP = "rtsp"


@dataclass
class FrameMetadata:
    """Metadata for a captured frame."""
    timestamp: float
    frame_number: int
    width: int
    height: int
    fps: float
    dropped_frames: int


class ThreadedVideoCapture:
    """
    High-performance threaded video capture with zero-latency frame access.

    Key Features:
    - Background thread continuously captures frames
    - `.latest_frame` always returns the freshest frame (no buffering lag)
    - Automatically drops old frames to maintain real-time performance
    - Thread-safe access with minimal locking overhead

    Usage:
        cap = ThreadedVideoCapture(source=0)  # Webcam
        cap.start()

        while True:
            frame = cap.latest_frame
            if frame is not None:
                cv2.imshow("Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.stop()
    """

    def __init__(
        self,
        source: int | str = 0,
        target_fps: Optional[float] = None,
        resolution: Optional[Tuple[int, int]] = None,
        buffer_size: int = 1
    ):
        """
        Initialize threaded video capture.

        Args:
            source: Camera index (int) or video file path / RTSP URL (str)
            target_fps: Target frame rate (None = use source native FPS)
            resolution: Target resolution as (width, height), None = native
            buffer_size: Internal buffer size (1 = always fresh, no delay)
        """
        self.source = source
        self.target_fps = target_fps
        self.resolution = resolution
        self.buffer_size = buffer_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Metrics
        self._frame_count = 0
        self._dropped_frames = 0
        self._start_time = 0.0
        self._last_frame_time = 0.0
        self._actual_fps = 0.0

        # Source info
        self._width = 0
        self._height = 0
        self._native_fps = 0.0

        self.logger = logging.getLogger(__name__)

    def _init_capture(self) -> bool:
        """Initialize the video capture device."""
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            self.logger.error(f"Failed to open video source: {self.source}")
            return False

        # Set resolution if specified
        if self.resolution:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Set buffer size to minimum for lowest latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        # Get actual properties
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.logger.info(
            f"Video source initialized: {self._width}x{self._height} @ {self._native_fps:.1f}fps"
        )
        return True

    def _capture_loop(self):
        """Background thread capture loop."""
        frame_interval = 1.0 / (self.target_fps or self._native_fps)
        last_capture_time = 0.0

        while self._running:
            current_time = time.perf_counter()

            # Rate limiting if target_fps is set
            if self.target_fps and (current_time - last_capture_time) < frame_interval:
                time.sleep(0.001)  # Brief sleep to prevent busy-waiting
                continue

            ret, frame = self._cap.read()

            if ret:
                with self._frame_lock:
                    # Drop old frame (if any) and store new one
                    if self._frame is not None:
                        self._dropped_frames += 1
                    self._frame = frame
                    self._frame_count += 1
                    self._last_frame_time = current_time

                last_capture_time = current_time

                # Update FPS calculation
                elapsed = current_time - self._start_time
                if elapsed > 0:
                    self._actual_fps = self._frame_count / elapsed
            else:
                # End of video file or capture error
                if isinstance(self.source, str):
                    self.logger.info("End of video file reached")
                    self._running = False
                else:
                    self.logger.warning("Frame capture failed, retrying...")
                    time.sleep(0.01)

    def start(self) -> bool:
        """
        Start the capture thread.

        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            return True

        if not self._init_capture():
            return False

        self._running = True
        self._start_time = time.perf_counter()
        self._frame_count = 0
        self._dropped_frames = 0

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        self.logger.info("Video capture thread started")
        return True

    def stop(self):
        """Stop the capture thread and release resources."""
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._cap:
            self._cap.release()
            self._cap = None

        self.logger.info("Video capture stopped")

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame.

        This is the KEY FEATURE - always returns the freshest frame
        with zero buffering delay. Old frames are dropped.

        Returns:
            Latest frame as numpy array (BGR), or None if no frame available
        """
        with self._frame_lock:
            if self._frame is None:
                return None
            # Return copy to prevent race conditions
            return self._frame.copy()

    @property
    def latest_frame_no_copy(self) -> Optional[np.ndarray]:
        """
        Get latest frame without copying (faster but not thread-safe for modification).

        Returns:
            Reference to latest frame, or None
        """
        with self._frame_lock:
            return self._frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        OpenCV-compatible read method.

        Returns:
            Tuple of (success, frame)
        """
        frame = self.latest_frame
        return (frame is not None, frame)

    @property
    def metadata(self) -> FrameMetadata:
        """Get current frame metadata."""
        return FrameMetadata(
            timestamp=self._last_frame_time,
            frame_number=self._frame_count,
            width=self._width,
            height=self._height,
            fps=self._actual_fps,
            dropped_frames=self._dropped_frames
        )

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get frame size as (width, height)."""
        return (self._width, self._height)

    @property
    def fps(self) -> float:
        """Get actual frames per second."""
        return self._actual_fps

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class VideoFileReader(ThreadedVideoCapture):
    """
    Specialized reader for video files with seeking support.
    """

    def __init__(self, filepath: str, loop: bool = False, **kwargs):
        """
        Initialize video file reader.

        Args:
            filepath: Path to video file
            loop: If True, loop back to start when reaching end
            **kwargs: Additional arguments for ThreadedVideoCapture
        """
        super().__init__(source=filepath, **kwargs)
        self.filepath = filepath
        self.loop = loop
        self._total_frames = 0

    def _init_capture(self) -> bool:
        if not super()._init_capture():
            return False
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True

    def _capture_loop(self):
        """Override to add looping support."""
        while self._running:
            ret, frame = self._cap.read()

            if ret:
                with self._frame_lock:
                    self._frame = frame
                    self._frame_count += 1
                    self._last_frame_time = time.perf_counter()
            else:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.logger.info("Video looped")
                else:
                    self._running = False
                    break

            # Control playback speed
            if self._native_fps > 0:
                time.sleep(1.0 / self._native_fps)

    def seek(self, frame_number: int):
        """Seek to specific frame."""
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def seek_time(self, seconds: float):
        """Seek to specific time in seconds."""
        if self._cap and self._native_fps > 0:
            frame_num = int(seconds * self._native_fps)
            self.seek(frame_num)

    @property
    def total_frames(self) -> int:
        """Get total frame count."""
        return self._total_frames

    @property
    def duration(self) -> float:
        """Get video duration in seconds."""
        if self._native_fps > 0:
            return self._total_frames / self._native_fps
        return 0.0


class FrameProcessor:
    """
    Utility class for common frame processing operations.
    """

    @staticmethod
    def resize(frame: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
        """Resize frame maintaining aspect ratio."""
        h, w = frame.shape[:2]

        if width and not height:
            ratio = width / w
            new_size = (width, int(h * ratio))
        elif height and not width:
            ratio = height / h
            new_size = (int(w * ratio), height)
        elif width and height:
            new_size = (width, height)
        else:
            return frame

        return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def extract_patch(frame: np.ndarray, x: int, y: int, size: int = 64) -> np.ndarray:
        """
        Extract a square patch centered at (x, y).

        Args:
            frame: Source frame
            x, y: Center coordinates
            size: Patch size (will be size x size)

        Returns:
            Extracted patch, resized to size x size
        """
        h, w = frame.shape[:2]
        half = size // 2

        # Calculate bounds with clamping
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w, x + half)
        y2 = min(h, y + half)

        patch = frame[y1:y2, x1:x2]

        # Resize to consistent size
        if patch.shape[0] != size or patch.shape[1] != size:
            patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)

        return patch


if __name__ == "__main__":
    # Demo: Test threaded video capture
    logging.basicConfig(level=logging.INFO)

    print("Testing ThreadedVideoCapture with webcam...")
    print("Press 'q' to quit")

    with ThreadedVideoCapture(source=0) as cap:
        # Wait for first frame
        while cap.latest_frame is None:
            time.sleep(0.01)

        while True:
            frame = cap.latest_frame
            if frame is None:
                continue

            # Display metadata
            meta = cap.metadata
            info_text = f"FPS: {meta.fps:.1f} | Frame: {meta.frame_number} | Dropped: {meta.dropped_frames}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("HoloRay Video Pipeline Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("Test complete.")
