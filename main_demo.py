#!/usr/bin/env python3
"""
HoloRay Engine - Proof of Concept Demo

Demonstrates the tracking and annotation capabilities:
1. Opens webcam feed
2. User clicks on objects to track them
3. Labels stick to objects through movement
4. Labels fade when objects are occluded (hand in front)
5. Labels reappear when objects re-enter frame
6. AI-powered object identification with OpenAI

Usage:
    python main_demo.py

Controls:
    - LEFT CLICK: Select point to add tracker
    - During label input:
        - Type label manually, OR
        - Press 'I' to auto-identify with AI (OpenAI)
        - Press ENTER to confirm
        - Press ESC to cancel
    - RIGHT CLICK: Remove nearest tracker
    - R: Reset all trackers
    - S: Cycle annotation styles
    - Q/ESC: Quit

Note: The letter 'I' is reserved for AI identification and cannot be typed in labels.

This is NOT the chess game - it's a generic tracking demo
that proves the HoloRay engine works for the VR Chess team.
"""

import sys
import time
import logging
import argparse
import threading
from typing import Optional

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]

# Add src to path
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from holoray.video_pipeline import ThreadedVideoCapture
from holoray.holoray_core import TrackerManager, TrackingStatus, LabelStatus
from holoray.annotation_layer import AnnotationRenderer, AnnotationStyle
from holoray.ai_labeler import AILabeler

# Default labels for demo (simulating chess pieces)
DEMO_LABELS = [
    "Object"
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
        self.ai_labeler = AILabeler()  # OpenAI-powered object identification

        # State
        self._running = False
        self._label_index = 0
        self._right_click_position = None
        self._last_frame: Optional[np.ndarray] = None  # For AI labeling

        # Label input state
        self._awaiting_label = False
        self._label_buffer = ""
        self._pending_place = None
        self._pending_tracker_id = None
        self._pending_original_label = None

        self.logger = logging.getLogger("HoloRayDemo")

    def _get_next_label(self) -> str:
        """Get next label from rotation."""
        label = DEMO_LABELS[self._label_index % len(DEMO_LABELS)]
        self._label_index += 1
        return label

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # If we are already naming a just-placed tracker, ignore new placements.
            if self._awaiting_label:
                return

            # Request placement; actual creation happens in the main loop using a valid frame.
            self._pending_place = (x, y)
            self._awaiting_label = True
            self._label_buffer = ""
            self._pending_tracker_id = None
            self._pending_original_label = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Ignore and clear right-clicks while typing/naming to avoid stale deletes.
            if self._awaiting_label:
                self._right_click_position = None
                return
            self._right_click_position = (x, y)

    # Naming trackers:

    def _commit_label(self, frame: np.ndarray):
        if not self._awaiting_label:
            return

        # If the tracker hasn't been created yet (very fast Enter), just ignore.
        if self._pending_tracker_id is None:
            return

        label = self._label_buffer.strip()
        if not label:
            label = self._pending_original_label or self._get_next_label()

        # Update tracker label (backend)
        tracker = self.tracker_manager.get_tracker(self._pending_tracker_id)
        if tracker is not None:
            tracker.label = label

        # Update annotation label (frontend)
        ann = self.renderer._annotations.get(self._pending_tracker_id)
        if ann is not None:
            ann.label_text = label

        self.logger.info(f"Renamed tracker {self._pending_tracker_id} to '{label}'")

        # Exit naming mode
        self._awaiting_label = False
        self._pending_tracker_id = None
        self._pending_original_label = None
        self._label_buffer = ""

    def _cancel_label(self):
        # If a tracker was already created for this naming session, remove it.
        if self._pending_tracker_id is not None:
            self.tracker_manager.remove_tracker(self._pending_tracker_id)
            self.renderer.remove_annotation(self._pending_tracker_id)
            self.logger.info(f"Canceled label entry — removed tracker {self._pending_tracker_id}")

        # Also cancel any pending placement request not yet created
        self._pending_place = None

        self._awaiting_label = False
        self._pending_tracker_id = None
        self._pending_original_label = None
        self._label_buffer = ""

    def _handle_key(self, key: int, frame: np.ndarray):
        if key == -1 or key == 255:
            return

        if self._awaiting_label:
            if key in (13, 10):  # Enter
                self._commit_label(frame)
            elif key in (27,):  # ESC
                self._cancel_label()
            elif key == ord('i') or key == ord('I'):  # 'I' = AI Identify
                self._identify_pending_tracker()
            elif key in (8, 127):  # Backspace / Delete
                self._label_buffer = self._label_buffer[:-1]
            elif 32 <= key <= 126 and len(self._label_buffer) < 50:
                # Allow all printable chars EXCEPT 'i' and 'I' (reserved for AI)
                if chr(key).lower() != 'i':
                    self._label_buffer += chr(key)
            return

        if key == ord('q') or key == 27:
            self._running = False
        elif key == ord('r'):
            self._reset_all()
        elif key == ord('s'):
            self._cycle_style()
        elif key == ord('i'):
            self._identify_nearest_tracker()

    def _handle_right_click(self):
        """Handle right click - remove nearest tracker."""
        if self._awaiting_label:
            self._right_click_position = None
            return

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

    def _identify_pending_tracker(self):
        """
        Use AI (OpenAI) to identify the tracker being created (during label input).
        
        Called when user presses 'I' while typing a label.
        """
        if self._pending_tracker_id is None:
            print("[AI] ⚠️ No pending tracker to identify")
            return
        
        if self._last_frame is None:
            print("[AI] ⚠️ No frame available")
            return
        
        tracker = self.tracker_manager.get_tracker(self._pending_tracker_id)
        if tracker is None:
            print("[AI] ⚠️ Tracker not found")
            return
        
        # Check if AI is available
        if not self.ai_labeler.is_available():
            print("[AI] ❌ AI not available - check OPENAI_API_KEY")
            self.logger.warning("AI labeling not available (check OPENAI_API_KEY)")
            return
        
        print(f"[AI] 🤖 Calling OpenAI to identify object at ({tracker.position[0]:.0f}, {tracker.position[1]:.0f})...")
        
        # Update UI to show thinking
        self._label_buffer = "..."
        ann = self.renderer._annotations.get(self._pending_tracker_id)
        if ann:
            ann.label_text = "..."
        
        # Mark tracker as thinking
        tracker.start_thinking()
        
        # Get position and frame copy
        tx, ty = tracker.position
        frame_copy = self._last_frame.copy()
        tracker_id = self._pending_tracker_id
        
        def worker():
            """Background thread worker for AI identification."""
            try:
                print(f"[AI] 📤 Sending image crop to OpenAI...")
                label = self.ai_labeler.identify_object(
                    frame_copy, 
                    int(tx), 
                    int(ty),
                    crop_size=200
                )
                
                print(f"[AI] ✅ OpenAI response: \"{label}\"")
                
                # Update tracker
                tracker.update_label(label)
                
                # Update label buffer so user sees the result
                self._label_buffer = label
                
                # Update annotation
                ann = self.renderer._annotations.get(tracker_id)
                if ann:
                    ann.label_text = label
                
                self.logger.info(f"AI Identified: {label}")
                
            except Exception as e:
                print(f"[AI] ❌ Error: {e}")
                self.logger.error(f"AI identification failed: {e}")
                tracker.set_label_error()
                self._label_buffer = "Error"
        
        # Launch daemon thread
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _identify_nearest_tracker(self):
        """
        Use AI (OpenAI) to identify the nearest tracked object.
        
        Runs in a background thread to avoid blocking the video feed.
        """
        if self._awaiting_label:
            return
        
        if self._last_frame is None:
            return
        
        # Find nearest active tracker
        nearest_id = None
        min_dist = float('inf')
        
        for tracker_id in self.tracker_manager.tracker_ids:
            tracker = self.tracker_manager.get_tracker(tracker_id)
            if tracker and tracker.status == TrackingStatus.TRACKING:
                # Check if not already thinking
                if tracker.label_status == LabelStatus.THINKING:
                    continue
                tx, ty = tracker.position
                # Use center of frame as reference
                h, w = self._last_frame.shape[:2]
                dist = ((tx - w/2) ** 2 + (ty - h/2) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_id = tracker_id
        
        if nearest_id is None:
            self.logger.info("No active tracker to identify")
            return
        
        tracker = self.tracker_manager.get_tracker(nearest_id)
        if tracker is None:
            return
        
        # Check if AI is available
        if not self.ai_labeler.is_available():
            self.logger.warning("AI labeling not available (check OPENAI_API_KEY)")
            return
        
        # Mark as thinking
        tracker.start_thinking()
        self.logger.info(f"Identifying tracker {nearest_id}...")
        
        # Update annotation to show thinking indicator
        ann = self.renderer._annotations.get(nearest_id)
        if ann:
            ann.label_text = "..."
        
        # Get position and frame copy
        tx, ty = tracker.position
        frame_copy = self._last_frame.copy()
        
        def worker():
            """Background thread worker for AI identification."""
            try:
                label = self.ai_labeler.identify_object(
                    frame_copy, 
                    int(tx), 
                    int(ty),
                    crop_size=200
                )
                tracker.update_label(label)
                
                # Also update annotation
                ann = self.renderer._annotations.get(nearest_id)
                if ann:
                    ann.label_text = label
                
                self.logger.info(f"Identified: {label}")
                
            except Exception as e:
                self.logger.error(f"AI identification failed: {e}")
                tracker.set_label_error()
                # Restore original label on error
                ann = self.renderer._annotations.get(nearest_id)
                if ann:
                    ann.label_text = tracker.label or "Unknown"
        
        # Launch daemon thread
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _maybe_place_tracker(self, frame: np.ndarray):
        """
        If the user clicked to place a tracker, create it now using the current valid frame.
        This is intentionally done in the main loop (not in the mouse callback) to avoid
        thread/frame races.
        """
        if self._pending_place is None:
            return

        if frame is None:
            # If frame is invalid, defer placement.
            return

        x, y = self._pending_place
        self._pending_place = None

        # Create tracker immediately with a default label
        default_label = self._get_next_label()

        tracker_id = self.tracker_manager.create_tracker(
            frame, x, y, label=default_label
        )

        self.renderer.create_annotation(
            tracker_id=tracker_id,
            label=default_label,
            x=x, y=y,
            style=self.current_style
        )

        self._pending_tracker_id = tracker_id
        self._pending_original_label = default_label

        self.logger.info(f"Placed tracker '{default_label}' at ({x}, {y}) — awaiting rename")

    def _draw_label_prompt(self, frame: np.ndarray):
        if not self._awaiting_label:
            return frame

        prompt = ("Label: " + (self._label_buffer or "_"))[:60]

        h, w = frame.shape[:2]
        base_y = h - 60

        # Optional: dark box behind text for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, base_y - 30), (min(w - 10, 650), base_y + 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(
            frame,
            "Type label | I = AI identify | Enter = OK | ESC = cancel",
            (20, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            prompt,
            (20, base_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        return frame

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

                # Store for AI labeling (used by background thread)
                self._last_frame = frame

                # Handle pending tracker placement
                self._maybe_place_tracker(frame)

                # Handle right-click removal (disabled while typing)
                self._handle_right_click()

                # Update all trackers
                tracking_states = self.tracker_manager.update_all(frame)

                # Update annotations with tracking states
                for tracker_id, state in tracking_states.items():
                    self.renderer.update_annotation(tracker_id, state)

                # Start rendering
                output = frame.copy()

                # Render annotations
                output = self.renderer.render_all(output, tracking_states)

                # Render HUD (FPS, tracker count)
                output = self.renderer.render_hud(
                    output,
                    fps=display_fps,
                    active_trackers=self.tracker_manager.active_count
                )

                # Draw label prompt if user is typing
                output = self._draw_label_prompt(output)

                # FPS calculation
                fps_counter += 1
                current_time = time.perf_counter()
                if current_time - fps_time >= 1.0:
                    display_fps = fps_counter / (current_time - fps_time)
                    fps_counter = 0
                    fps_time = current_time

                # Display frame
                cv2.imshow(self.WINDOW_NAME, output)

                # Centralized keyboard handling
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key, frame)

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
  LEFT CLICK   Add tracker at cursor, then:
               - Type label manually, OR
               - Press 'I' for AI identification (OpenAI)
               - Press ENTER to confirm
               - Press ESC to cancel
  RIGHT CLICK  Remove nearest tracker
  R            Reset all trackers
  S            Cycle annotation styles
  Q/ESC        Quit

AI Setup:
  Set OPENAI_API_KEY environment variable:
    export OPENAI_API_KEY=your_key_here
  Get key at: https://makersuite.google.com/app/apikey

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
    print("  When label prompt appears:")
    print("    - Type a label manually, OR")
    print("    - Press 'I' to auto-identify with OpenAI AI")
    print("    - Press ENTER to confirm")
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