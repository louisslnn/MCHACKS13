#!/usr/bin/env python3
"""
HoloRay Engine - Proof of Concept Demo

Demonstrates the tracking and annotation capabilities:
1. Opens webcam feed
2. User clicks on objects to track them
3. Labels stick to objects through movement
4. Labels fade when objects are occluded (hand in front)
5. Labels reappear when objects re-enter frame

Usage:
    python main_demo.py

Controls:
    - LEFT CLICK: Add tracker at cursor position
    - RIGHT CLICK: Remove nearest tracker
    - R: Reset all trackers
    - S: Cycle annotation styles
    - Q/ESC: Quit

This is NOT the chess game - it's a generic tracking demo
that proves the HoloRay engine works for the VR Chess team.
"""

import sys
import time
import logging
import argparse
from typing import List

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]

# Add src to path
sys.path.insert(0, 'src')

from holoray.video_pipeline import ThreadedVideoCapture
from holoray.holoray_core import TrackerManager, TrackingStatus
from holoray.annotation_layer import AnnotationRenderer, AnnotationStyle


# Default labels for demo (simulating chess pieces)
DEMO_LABELS = [
    "Pawn", "Rook", "Knight", "Bishop", "Queen", "King",
    "Object A", "Object B", "Object C", "Target 1", "Target 2"
]


class HoloRayDemo:
    """
    Interactive demo of the HoloRay Tracking & Annotation Engine.
    """

    WINDOW_NAME = "HoloRay Engine Demo"

    def __init__(
        self,
        source: int | str = 0,
        use_gpu: bool = True,
        enable_reid: bool = True,
        resolution: tuple = None,
        style: AnnotationStyle = AnnotationStyle.STANDARD
    ):
        """
        Initialize demo.

        Args:
            source: Camera index or video file path
            use_gpu: Use GPU acceleration if available
            enable_reid: Enable re-identification
            resolution: Target resolution (width, height)
            style: Default annotation style
        """
        self.source = source
        self.use_gpu = use_gpu
        self.enable_reid = enable_reid
        self.resolution = resolution
        self.current_style = style

        # Components
        self.video = None
        self.tracker_manager = None
        self.renderer = None

        # State
        self._running = False
        self._label_index = 0
        self._click_position = None
        self._right_click_position = None

        self.logger = logging.getLogger("HoloRayDemo")

    def _get_next_label(self) -> str:
        """Get next label from rotation."""
        label = DEMO_LABELS[self._label_index % len(DEMO_LABELS)]
        self._label_index += 1
        return label

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_position = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._right_click_position = (x, y)

    def _handle_left_click(self, frame: np.ndarray):
        """Handle left click - add new tracker."""
        if self._click_position is None:
            return

        x, y = self._click_position
        self._click_position = None

        label = self._get_next_label()

        # Create tracker
        tracker_id = self.tracker_manager.create_tracker(
            frame, x, y, label=label
        )

        # Create annotation
        self.renderer.create_annotation(
            tracker_id=tracker_id,
            label=label,
            x=x, y=y,
            style=self.current_style
        )

        self.logger.info(f"Added tracker '{label}' at ({x}, {y})")

    def _handle_right_click(self):
        """Handle right click - remove nearest tracker."""
        if self._right_click_position is None:
            return

        x, y = self._right_click_position
        self._right_click_position = None

        # Find nearest tracker
        min_dist = float('inf')
        nearest_id = None

        for tracker_id in self.tracker_manager.tracker_ids:
            tracker = self.tracker_manager.get_tracker(tracker_id)
            if tracker:
                tx, ty = tracker.position
                dist = ((tx - x) ** 2 + (ty - y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_id = tracker_id

        if nearest_id and min_dist < 100:  # Within 100 pixels
            self.tracker_manager.remove_tracker(nearest_id)
            self.renderer.remove_annotation(nearest_id)
            self.logger.info(f"Removed tracker {nearest_id}")

    def _cycle_style(self):
        """Cycle through annotation styles."""
        styles = list(AnnotationStyle)
        current_idx = styles.index(self.current_style)
        self.current_style = styles[(current_idx + 1) % len(styles)]
        self.logger.info(f"Switched to style: {self.current_style.value}")

        # Update all annotations
        for tracker_id in self.tracker_manager.tracker_ids:
            if tracker_id in self.renderer._annotations:
                self.renderer._annotations[tracker_id].style = self.current_style

    def _reset_all(self):
        """Reset all trackers."""
        for tracker_id in list(self.tracker_manager.tracker_ids):
            self.tracker_manager.remove_tracker(tracker_id)
            self.renderer.remove_annotation(tracker_id)
        self._label_index = 0
        self.logger.info("All trackers reset")

    def _draw_click_hint(self, frame: np.ndarray):
        """Draw hint circle at mouse position."""
        # Get current mouse position (approximation via last frame)
        pass  # Would need additional callback

    def run(self):
        """Run the demo."""
        self.logger.info("Starting HoloRay Demo...")

        # Initialize video capture
        self.video = ThreadedVideoCapture(
            source=self.source,
            resolution=self.resolution
        )

        if not self.video.start():
            self.logger.error("Failed to start video capture")
            return

        # Wait for first frame
        while self.video.latest_frame is None:
            time.sleep(0.01)

        # Initialize tracker manager and renderer
        self.tracker_manager = TrackerManager(
            use_gpu=self.use_gpu,
            enable_reid=self.enable_reid
        )
        self.renderer = AnnotationRenderer(default_style=self.current_style)

        # Setup window and mouse callback
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

        self._running = True
        fps_counter = 0
        fps_time = time.perf_counter()
        display_fps = 0.0

        self.logger.info("Demo running. Click to add trackers.")

        try:
            while self._running:
                # Get latest frame
                frame = self.video.latest_frame
                if frame is None:
                    continue

                # Handle clicks
                self._handle_left_click(frame)
                self._handle_right_click()

                # Update all trackers
                tracking_states = self.tracker_manager.update_all(frame)

                # Update annotations with tracking states
                for tracker_id, state in tracking_states.items():
                    self.renderer.update_annotation(tracker_id, state)

                # Render annotations
                output = frame.copy()
                output = self.renderer.render_all(output, tracking_states)

                # Render HUD
                output = self.renderer.render_hud(
                    output,
                    fps=display_fps,
                    active_trackers=self.tracker_manager.active_count
                )

                # Calculate FPS
                fps_counter += 1
                current_time = time.perf_counter()
                if current_time - fps_time >= 1.0:
                    display_fps = fps_counter / (current_time - fps_time)
                    fps_counter = 0
                    fps_time = current_time

                # Display
                cv2.imshow(self.WINDOW_NAME, output)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    self._running = False
                elif key == ord('r'):  # Reset
                    self._reset_all()
                elif key == ord('s'):  # Style
                    self._cycle_style()

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.video.stop()
            cv2.destroyAllWindows()
            self.logger.info("Demo stopped.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HoloRay Tracking & Annotation Engine Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  LEFT CLICK   Add tracker at cursor
  RIGHT CLICK  Remove nearest tracker
  R            Reset all trackers
  S            Cycle annotation styles
  Q/ESC        Quit

Examples:
  python main_demo.py                    # Use webcam 0
  python main_demo.py --source 1         # Use webcam 1
  python main_demo.py --source video.mp4 # Use video file
  python main_demo.py --no-gpu           # Disable GPU
        """
    )

    parser.add_argument(
        "--source", "-s",
        default=0,
        help="Video source: camera index (0, 1, ...) or file path"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Disable re-identification"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=str,
        default=None,
        help="Resolution as WxH (e.g., 1280x720)"
    )
    parser.add_argument(
        "--style",
        choices=["minimal", "standard", "detailed", "gaming"],
        default="standard",
        help="Annotation style"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Parse source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Parse resolution
    resolution = None
    if args.resolution:
        try:
            w, h = args.resolution.lower().split('x')
            resolution = (int(w), int(h))
        except ValueError:
            print(f"Invalid resolution format: {args.resolution}")
            sys.exit(1)

    # Parse style
    style_map = {
        "minimal": AnnotationStyle.MINIMAL,
        "standard": AnnotationStyle.STANDARD,
        "detailed": AnnotationStyle.DETAILED,
        "gaming": AnnotationStyle.GAMING,
    }
    style = style_map[args.style]

    # Print banner
    print("\n" + "=" * 60)
    print("  HoloRay Tracking & Annotation Engine")
    print("  Proof of Concept Demo")
    print("=" * 60)
    print(f"  Source: {source}")
    print(f"  GPU: {'Enabled' if not args.no_gpu else 'Disabled'}")
    print(f"  Re-ID: {'Enabled' if not args.no_reid else 'Disabled'}")
    print(f"  Style: {args.style}")
    print("=" * 60)
    print("\n  Click on objects to track them!")
    print("  Wave your hand in front to test occlusion detection.")
    print("  Move objects out/in of frame to test re-identification.\n")

    # Run demo
    demo = HoloRayDemo(
        source=source,
        use_gpu=not args.no_gpu,
        enable_reid=not args.no_reid,
        resolution=resolution,
        style=style
    )
    demo.run()


if __name__ == "__main__":
    main()
