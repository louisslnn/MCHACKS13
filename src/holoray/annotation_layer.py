"""
HoloRay Annotation Layer - Visual Overlay System

Renders clean, anti-aliased annotations that respond to tracking state:
- Full opacity when actively tracking
- Ghost mode (faded) when occluded
- Hidden when lost
"""

import time
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import cv2

from .holoray_core import TrackingStatus, TrackingState


class AnnotationStyle(Enum):
    """Predefined annotation styles."""
    MINIMAL = "minimal"        # Just a dot and label
    STANDARD = "standard"      # Crosshair + label + confidence
    DETAILED = "detailed"      # Full box, arrow, metrics
    GAMING = "gaming"          # Colorful, animated


@dataclass
class ColorScheme:
    """Color scheme for annotations."""
    primary: Tuple[int, int, int] = (0, 255, 128)     # Green
    secondary: Tuple[int, int, int] = (255, 255, 0)   # Cyan
    warning: Tuple[int, int, int] = (0, 165, 255)     # Orange
    error: Tuple[int, int, int] = (0, 0, 255)         # Red
    text: Tuple[int, int, int] = (255, 255, 255)      # White
    background: Tuple[int, int, int] = (0, 0, 0)      # Black


@dataclass
class Annotation:
    """
    Visual annotation attached to a tracked object.

    Attributes:
        label_text: Text to display
        tracker_id: Associated tracker ID
        coordinates: Current (x, y) position
        opacity: Current opacity (0.0 to 1.0)
        target_opacity: Target opacity for smooth transitions
        style: Visual style
        color_scheme: Colors to use
    """
    label_text: str
    tracker_id: str
    coordinates: Tuple[float, float] = (0.0, 0.0)
    opacity: float = 1.0
    target_opacity: float = 1.0
    style: AnnotationStyle = AnnotationStyle.STANDARD
    color_scheme: ColorScheme = field(default_factory=ColorScheme)

    # Animation state
    _pulse_phase: float = 0.0
    _smooth_x: float = 0.0
    _smooth_y: float = 0.0
    _initialized: bool = False

    def update_position(self, x: float, y: float, smoothing: float = 0.3):
        """Update position with optional smoothing."""
        if not self._initialized:
            self._smooth_x = x
            self._smooth_y = y
            self._initialized = True
        else:
            self._smooth_x += (x - self._smooth_x) * smoothing
            self._smooth_y += (y - self._smooth_y) * smoothing
        self.coordinates = (self._smooth_x, self._smooth_y)

    def update_opacity(self, status: TrackingStatus, transition_speed: float = 0.15):
        """
        Update opacity based on tracking status.

        Status -> Opacity:
        - TRACKING: 1.0 (fully visible)
        - OCCLUDED: 0.3 (ghost mode)
        - LOST/SEARCHING: 0.0 (hidden)
        """
        if status == TrackingStatus.TRACKING:
            self.target_opacity = 1.0
        elif status == TrackingStatus.OCCLUDED:
            self.target_opacity = 0.3
        else:  # LOST, SEARCHING, INACTIVE
            self.target_opacity = 0.0

        # Smooth transition
        self.opacity += (self.target_opacity - self.opacity) * transition_speed
        self.opacity = max(0.0, min(1.0, self.opacity))

    def tick_animation(self, dt: float = 0.033):
        """Update animation state."""
        self._pulse_phase += dt * 3.0  # Pulse frequency
        if self._pulse_phase > 2 * math.pi:
            self._pulse_phase -= 2 * math.pi


class AnnotationRenderer:
    """
    Renders annotations onto video frames.

    Features:
    - Anti-aliased drawing
    - Smooth opacity transitions
    - Multiple visual styles
    - Performance metrics overlay
    """

    def __init__(
        self,
        default_style: AnnotationStyle = AnnotationStyle.STANDARD,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.6,
        thickness: int = 2
    ):
        self.default_style = default_style
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

        self._annotations: Dict[str, Annotation] = {}
        self._last_time = time.perf_counter()

    def create_annotation(
        self,
        tracker_id: str,
        label: str,
        x: float,
        y: float,
        style: Optional[AnnotationStyle] = None,
        color_scheme: Optional[ColorScheme] = None
    ) -> Annotation:
        """Create and register a new annotation."""
        ann = Annotation(
            label_text=label,
            tracker_id=tracker_id,
            coordinates=(x, y),
            style=style or self.default_style,
            color_scheme=color_scheme or ColorScheme()
        )
        ann._smooth_x = x
        ann._smooth_y = y
        ann._initialized = True
        self._annotations[tracker_id] = ann
        return ann

    def update_annotation(
        self,
        tracker_id: str,
        state: TrackingState
    ):
        """Update annotation from tracking state."""
        if tracker_id not in self._annotations:
            return

        ann = self._annotations[tracker_id]
        ann.update_position(state.x, state.y)
        ann.update_opacity(state.status)

    def remove_annotation(self, tracker_id: str):
        """Remove an annotation."""
        if tracker_id in self._annotations:
            del self._annotations[tracker_id]

    def _blend_color(
        self,
        color: Tuple[int, int, int],
        opacity: float
    ) -> Tuple[int, int, int]:
        """Adjust color intensity based on opacity."""
        return tuple(int(c * opacity) for c in color)

    def _draw_crosshair(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        size: int,
        color: Tuple[int, int, int],
        thickness: int = 2
    ):
        """Draw a crosshair marker."""
        # Horizontal line
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
        # Vertical line
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)

    def _draw_circle_marker(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        radius: int,
        color: Tuple[int, int, int],
        pulse_phase: float = 0.0,
        filled: bool = False
    ):
        """Draw a circle marker with optional pulse animation."""
        # Pulsing radius
        pulse_offset = int(3 * math.sin(pulse_phase))
        actual_radius = radius + pulse_offset

        if filled:
            cv2.circle(frame, (x, y), actual_radius, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), actual_radius, color, 2, cv2.LINE_AA)

    def _draw_label_box(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        text: str,
        color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int],
        opacity: float,
        offset: Tuple[int, int] = (15, -15)
    ):
        """Draw a label with background box."""
        label_x = x + offset[0]
        label_y = y + offset[1]

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.thickness
        )

        # Clamp to frame bounds
        h, w = frame.shape[:2]
        label_x = max(5, min(label_x, w - text_w - 15))
        label_y = max(text_h + 10, min(label_y, h - 10))

        # Draw background with opacity
        padding = 5
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (label_x - padding, label_y - text_h - padding),
            (label_x + text_w + padding, label_y + padding),
            bg_color,
            -1
        )
        cv2.rectangle(
            overlay,
            (label_x - padding, label_y - text_h - padding),
            (label_x + text_w + padding, label_y + padding),
            color,
            2,
            cv2.LINE_AA
        )

        # Blend overlay with original
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        # Draw text (always at target opacity for readability)
        text_color = self._blend_color((255, 255, 255), max(0.5, opacity))
        cv2.putText(
            frame,
            text,
            (label_x, label_y),
            self.font,
            self.font_scale,
            text_color,
            self.thickness,
            cv2.LINE_AA
        )

        return (label_x, label_y)

    def _draw_arrowed_connection(
        self,
        frame: np.ndarray,
        from_pt: Tuple[int, int],
        to_pt: Tuple[int, int],
        color: Tuple[int, int, int]
    ):
        """Draw an arrow connecting two points."""
        cv2.arrowedLine(
            frame,
            from_pt,
            to_pt,
            color,
            1,
            cv2.LINE_AA,
            tipLength=0.3
        )

    def _draw_confidence_bar(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        confidence: float,
        width: int = 50,
        height: int = 6
    ):
        """Draw a confidence indicator bar."""
        bar_x = x - width // 2
        bar_y = y + 25

        # Background
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + width, bar_y + height),
            (50, 50, 50),
            -1
        )

        # Confidence fill
        fill_width = int(width * confidence)
        if confidence > 0.7:
            fill_color = (0, 255, 0)
        elif confidence > 0.4:
            fill_color = (0, 255, 255)
        else:
            fill_color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + fill_width, bar_y + height),
            fill_color,
            -1
        )

        # Border
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + width, bar_y + height),
            (200, 200, 200),
            1
        )

    def render_annotation(
        self,
        frame: np.ndarray,
        annotation: Annotation,
        state: Optional[TrackingState] = None
    ):
        """Render a single annotation onto the frame."""
        if annotation.opacity < 0.05:
            return  # Too faded to render

        x, y = int(annotation.coordinates[0]), int(annotation.coordinates[1])
        colors = annotation.color_scheme
        opacity = annotation.opacity

        # Determine marker color based on status
        if state:
            if state.status == TrackingStatus.TRACKING:
                marker_color = colors.primary
            elif state.status == TrackingStatus.OCCLUDED:
                marker_color = colors.warning
            else:
                marker_color = colors.error
        else:
            marker_color = colors.primary

        # Apply opacity to colors
        marker_color = self._blend_color(marker_color, opacity)

        # Render based on style
        if annotation.style == AnnotationStyle.MINIMAL:
            # Just dot and label
            cv2.circle(frame, (x, y), 5, marker_color, -1, cv2.LINE_AA)
            cv2.putText(
                frame, annotation.label_text, (x + 10, y - 10),
                self.font, self.font_scale * 0.8,
                self._blend_color(colors.text, opacity),
                1, cv2.LINE_AA
            )

        elif annotation.style == AnnotationStyle.STANDARD:
            # Crosshair + label box
            self._draw_crosshair(frame, x, y, 12, marker_color)
            self._draw_circle_marker(
                frame, x, y, 8, marker_color,
                pulse_phase=annotation._pulse_phase
            )
            label_pos = self._draw_label_box(
                frame, x, y, annotation.label_text,
                marker_color, colors.background, opacity
            )
            cv2.line(frame, (x, y), label_pos, marker_color, 1, cv2.LINE_AA)

        elif annotation.style == AnnotationStyle.DETAILED:
            # Full visualization with confidence
            self._draw_crosshair(frame, x, y, 15, marker_color)
            self._draw_circle_marker(frame, x, y, 10, marker_color, annotation._pulse_phase)

            # Label with status
            status_text = f"{annotation.label_text}"
            if state:
                status_text += f" [{state.status.value}]"
            self._draw_label_box(
                frame, x, y, status_text,
                marker_color, colors.background, opacity
            )

            # Confidence bar
            if state:
                self._draw_confidence_bar(frame, x, y, state.confidence)

        elif annotation.style == AnnotationStyle.GAMING:
            # Animated gaming style
            pulse_size = int(15 + 5 * math.sin(annotation._pulse_phase))

            # Outer ring
            cv2.circle(frame, (x, y), pulse_size + 5, marker_color, 2, cv2.LINE_AA)
            # Inner filled
            cv2.circle(frame, (x, y), 5, marker_color, -1, cv2.LINE_AA)

            # Diamond markers
            pts = np.array([
                [x, y - pulse_size],
                [x + pulse_size, y],
                [x, y + pulse_size],
                [x - pulse_size, y]
            ], np.int32)
            cv2.polylines(frame, [pts], True, marker_color, 2, cv2.LINE_AA)

            # Label
            cv2.putText(
                frame, annotation.label_text.upper(), (x + 20, y),
                self.font, self.font_scale,
                self._blend_color(colors.secondary, opacity),
                self.thickness, cv2.LINE_AA
            )

    def render_all(
        self,
        frame: np.ndarray,
        tracking_states: Dict[str, TrackingState]
    ) -> np.ndarray:
        """
        Render all annotations onto frame.

        Args:
            frame: Video frame to draw on
            tracking_states: Dict of tracker_id -> TrackingState

        Returns:
            Frame with annotations rendered
        """
        # Update animation time
        current_time = time.perf_counter()
        dt = current_time - self._last_time
        self._last_time = current_time

        # Update and render each annotation
        for tracker_id, annotation in self._annotations.items():
            state = tracking_states.get(tracker_id)

            if state:
                # Update from tracking state
                annotation.update_position(state.x, state.y)
                annotation.update_opacity(state.status)

            # Tick animation
            annotation.tick_animation(dt)

            # Render
            self.render_annotation(frame, annotation, state)

        return frame

    def render_hud(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
        active_trackers: int = 0,
        show_instructions: bool = True
    ) -> np.ndarray:
        """Render heads-up display with metrics and instructions."""
        h, w = frame.shape[:2]

        # Semi-transparent HUD background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # FPS
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 28),
            self.font, 0.6, (0, 255, 0), 1, cv2.LINE_AA
        )

        # Active trackers
        cv2.putText(
            frame, f"Trackers: {active_trackers}", (120, 28),
            self.font, 0.6, (255, 255, 0), 1, cv2.LINE_AA
        )

        # Instructions
        if show_instructions:
            instructions = "Click to track | 'R' reset | 'Q' quit"
            text_size = cv2.getTextSize(instructions, self.font, 0.5, 1)[0]
            cv2.putText(
                frame, instructions, (w - text_size[0] - 10, 28),
                self.font, 0.5, (200, 200, 200), 1, cv2.LINE_AA
            )

        return frame


# Convenience function for quick single-annotation rendering
def draw_tracking_annotation(
    frame: np.ndarray,
    x: float,
    y: float,
    label: str,
    status: TrackingStatus,
    confidence: float = 1.0,
    style: AnnotationStyle = AnnotationStyle.STANDARD
) -> np.ndarray:
    """
    Quick function to draw a single tracking annotation.

    Args:
        frame: Frame to draw on
        x, y: Position
        label: Label text
        status: Tracking status
        confidence: Tracking confidence
        style: Visual style

    Returns:
        Frame with annotation
    """
    renderer = AnnotationRenderer(default_style=style)
    ann = Annotation(
        label_text=label,
        tracker_id="temp",
        coordinates=(x, y),
        style=style
    )

    # Set opacity based on status
    if status == TrackingStatus.TRACKING:
        ann.opacity = 1.0
    elif status == TrackingStatus.OCCLUDED:
        ann.opacity = 0.3
    else:
        ann.opacity = 0.0

    state = TrackingState(
        status=status,
        x=x, y=y,
        confidence=confidence,
        is_occluded=(status == TrackingStatus.OCCLUDED)
    )

    renderer.render_annotation(frame, ann, state)
    return frame


if __name__ == "__main__":
    # Demo: Test annotation rendering
    print("Testing AnnotationRenderer...")

    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # Dark gray background

    renderer = AnnotationRenderer()

    # Create test annotations with different styles
    styles = [
        (AnnotationStyle.MINIMAL, (100, 150), "Minimal"),
        (AnnotationStyle.STANDARD, (320, 150), "Standard"),
        (AnnotationStyle.DETAILED, (540, 150), "Detailed"),
        (AnnotationStyle.GAMING, (320, 350), "Gaming"),
    ]

    for style, pos, label in styles:
        renderer.create_annotation(
            tracker_id=label.lower(),
            label=label,
            x=pos[0],
            y=pos[1],
            style=style
        )

    # Simulate tracking states
    states = {
        "minimal": TrackingState(
            status=TrackingStatus.TRACKING,
            x=100, y=150, confidence=0.95, is_occluded=False
        ),
        "standard": TrackingState(
            status=TrackingStatus.OCCLUDED,
            x=320, y=150, confidence=0.5, is_occluded=True
        ),
        "detailed": TrackingState(
            status=TrackingStatus.TRACKING,
            x=540, y=150, confidence=0.75, is_occluded=False
        ),
        "gaming": TrackingState(
            status=TrackingStatus.TRACKING,
            x=320, y=350, confidence=1.0, is_occluded=False
        ),
    }

    # Render
    frame = renderer.render_all(frame, states)
    frame = renderer.render_hud(frame, fps=60.0, active_trackers=4)

    # Display
    cv2.imshow("Annotation Layer Test", frame)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Test complete.")
