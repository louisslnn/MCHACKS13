"""
HoloRay - Tracking & Annotation Engine for AR Applications

A modular SDK for real-time object tracking with sticky annotations.
Designed for integration with AR/VR applications like the VR Chess App.

Features:
- Zero-latency threaded video capture
- Point tracking with CoTracker3 (GPU) or optical flow (CPU)
- Smart occlusion detection
- Object re-identification on frame re-entry
- Smooth, animated annotation rendering

Quick Start:
    from holoray import TrackerManager, AnnotationRenderer, ThreadedVideoCapture

    # Setup video
    video = ThreadedVideoCapture(source=0)
    video.start()

    # Setup tracking
    tracker_manager = TrackerManager()
    renderer = AnnotationRenderer()

    # Track an object
    frame = video.latest_frame
    tracker_id = tracker_manager.create_tracker(frame, x=320, y=240, label="Object")
    renderer.create_annotation(tracker_id, label="Object", x=320, y=240)

    # Update loop
    while True:
        frame = video.latest_frame
        states = tracker_manager.update_all(frame)
        frame = renderer.render_all(frame, states)
        display(frame)
"""

__version__ = "1.0.0"
__author__ = "HoloRay Team - McHacks 13"

# Core tracking
from .holoray_core import (
    ObjectTracker,
    HybridTracker,
    TrackerManager,
    TrackingStatus,
    TrackingState,
    VisualFingerprint,
    FeatureMatcher,
    FastOpticalFlow,
)

# Backward compatibility alias
VisualMemory = VisualFingerprint

# Video pipeline
from .video_pipeline import (
    ThreadedVideoCapture,
    VideoFileReader,
    FrameProcessor,
    FrameMetadata,
    VideoSource,
    PerformanceOverlay,
)

# Annotation rendering
from .annotation_layer import (
    Annotation,
    AnnotationRenderer,
    AnnotationStyle,
    ColorScheme,
    draw_tracking_annotation,
    # Shape Engine
    ShapeType,
    RelativeShape,
    ShapeEngine,
    create_pointer_arrow,
    create_highlight_circle,
    create_bounding_box,
)

__all__ = [
    # Version
    "__version__",

    # Core Tracking
    "ObjectTracker",
    "HybridTracker",
    "TrackerManager",
    "TrackingStatus",
    "TrackingState",
    "VisualFingerprint",
    "VisualMemory",  # Backward compat alias
    "FeatureMatcher",
    "FastOpticalFlow",

    # Video
    "ThreadedVideoCapture",
    "VideoFileReader",
    "FrameProcessor",
    "FrameMetadata",
    "VideoSource",
    "PerformanceOverlay",

    # Annotation
    "Annotation",
    "AnnotationRenderer",
    "AnnotationStyle",
    "ColorScheme",
    "draw_tracking_annotation",
    
    # Shape Engine
    "ShapeType",
    "RelativeShape",
    "ShapeEngine",
    "create_pointer_arrow",
    "create_highlight_circle",
    "create_bounding_box",
]
