from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:  # TensorFlow is imported lazily; it is slow to load.
    import tensorflow as tf

LOGGER = logging.getLogger("snake_vision")
SCRIPT_DIR = Path(__file__).resolve().parent

Color = tuple[int, int, int]  # BGR
Point = tuple[int, int]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    """Runtime settings for the application."""

    model_path: Path = SCRIPT_DIR / "keras_model.h5"
    labels_path: Path = SCRIPT_DIR / "labels.txt"
    camera_index: int = 0
    image_size: tuple[int, int] = (224, 224)
    preview_size: tuple[int, int] = (1280, 720)
    side_panel_width: int = 340
    window_name: str = "Snake Vision | Live Camera"
    window_size: tuple[int, int] = (1280, 720)
    scan_cycle_seconds: float = 3.6
    ui_fps: float = 60.0
    mirror: bool = True
    inference_interval_seconds: float = 0.12
    top_k: int = 4
    confidence_high: float = 0.85
    confidence_medium: float = 0.60


@dataclass(frozen=True)
class Theme:
    """Colour palette (BGR)."""

    background: Color = (19, 14, 10)
    panel: Color = (37, 29, 22)
    card: Color = (50, 40, 31)
    divider: Color = (70, 60, 48)
    track: Color = (74, 64, 52)
    text: Color = (246, 244, 240)
    muted: Color = (166, 155, 140)
    accent: Color = (240, 190, 90)
    success: Color = (94, 197, 34)
    warning: Color = (40, 180, 250)
    danger: Color = (67, 76, 242)


THEME = Theme()


class Prediction(NamedTuple):
    label: str
    confidence: float


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def load_class_names(labels_path: Path) -> list[str]:
    """Parse a Teachable Machine ``labels.txt`` ("0 Name" per line)."""
    indexed: dict[int, str] = {}
    with labels_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            try:
                indexed[int(parts[0])] = parts[1]
            except ValueError:
                continue

    if not indexed:
        raise ValueError(f"No indexed class names found in {labels_path}")
    if sorted(indexed) != list(range(len(indexed))):
        raise ValueError(f"Class indexes in {labels_path} must start at 0 with no gaps.")
    return [indexed[i] for i in range(len(indexed))]


def load_model(config: Config) -> tuple["tf.keras.Model", list[str]]:
    """Load and validate the model against its label file."""
    for path, kind in ((config.model_path, "model"), (config.labels_path, "labels")):
        if not path.is_file():
            raise FileNotFoundError(f"Teachable Machine {kind} file not found: {path}")

    import tensorflow as tf  # lazy import

    model = tf.keras.models.load_model(config.model_path, compile=False)
    class_names = load_class_names(config.labels_path)

    output_count = model.output_shape[-1]
    if output_count is None:
        raise ValueError("The model must have a fixed number of class outputs.")
    if output_count != len(class_names):
        raise ValueError(
            f"Model has {output_count} outputs, but {config.labels_path.name} "
            f"contains {len(class_names)} class names."
        )
    LOGGER.info("Loaded model with %d classes.", len(class_names))
    return model, class_names


def preprocess(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """BGR frame -> normalised RGB batch of one, shaped for the model."""
    image = cv2.resize(frame, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (image.astype(np.float32) / 255.0)[np.newaxis, ...]


def rank_predictions(
    probabilities: np.ndarray, class_names: Sequence[str], top_k: int
) -> list[Prediction]:
    """Return the ``top_k`` finite scores, highest first."""
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    valid = [
        i for i, score in enumerate(scores) if i < len(class_names) and np.isfinite(score)
    ]
    valid.sort(key=lambda i: scores[i], reverse=True)
    return [Prediction(class_names[i], float(scores[i])) for i in valid[:top_k]]


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera, using DirectShow on Windows for faster start-up."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open webcam {index}. Close other camera apps, allow camera "
            "access in your system privacy settings, or try a different --camera index."
        )
    return camera


def configure_camera(camera: cv2.VideoCapture, size: tuple[int, int]) -> Optional[float]:
    """Request the highest FPS the backend confirms; return it if known."""
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    for requested in (240, 144, 120, 90, 60, 50, 30, 25):
        try:
            accepted = camera.set(cv2.CAP_PROP_FPS, requested)
            reported = camera.get(cv2.CAP_PROP_FPS)
        except cv2.error:
            continue
        if accepted and np.isfinite(reported) and 1 <= reported <= requested * 1.05:
            return float(reported)

    try:
        camera.set(cv2.CAP_PROP_FPS, 30)
        reported = camera.get(cv2.CAP_PROP_FPS)
    except cv2.error:
        return None
    return float(reported) if np.isfinite(reported) and reported > 0 else None


class _Worker(threading.Thread):
    """Daemon thread with a shared stop event and a thread-safe error slot."""

    def __init__(self, name: str, stop_event: threading.Event) -> None:
        super().__init__(name=name, daemon=True)
        self._stop_event = stop_event
        self._lock = threading.Lock()
        self._error: Optional[str] = None

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def _fail(self, message: str) -> None:
        LOGGER.error(message)
        with self._lock:
            self._error = message
        self._stop_event.set()


class CameraStream(_Worker):
    """Continuously reads frames and exposes the most recent one."""

    FPS_WINDOW_SECONDS = 0.5

    def __init__(self, camera: cv2.VideoCapture, stop_event: threading.Event) -> None:
        super().__init__("camera-capture", stop_event)
        self._camera = camera
        self._frame: Optional[np.ndarray] = None
        self._sequence = 0
        self._fps = 0.0

    def snapshot(self) -> tuple[Optional[np.ndarray], int, float]:
        """Return ``(latest_frame, sequence_number, measured_fps)``. Treat frame as read-only."""
        with self._lock:
            return self._frame, self._sequence, self._fps

    def run(self) -> None:
        frames = 0
        window_start = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                success, frame = self._camera.read()
                if not success:
                    self._fail("Webcam opened, but a frame could not be read.")
                    return

                frames += 1
                now = time.perf_counter()
                elapsed = now - window_start
                with self._lock:
                    self._frame = frame
                    self._sequence += 1
                    if elapsed >= self.FPS_WINDOW_SECONDS:
                        self._fps = frames / elapsed
                        frames, window_start = 0, now
        except Exception as exc:  # noqa: BLE001 - surface any backend failure
            self._fail(f"Camera capture failed: {exc}")


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class InferenceWorker(_Worker):
    """Runs the model on the newest frame at a fixed maximum rate."""

    def __init__(
        self,
        model: "tf.keras.Model",
        class_names: Sequence[str],
        camera: CameraStream,
        config: Config,
        stop_event: threading.Event,
    ) -> None:
        super().__init__("model-inference", stop_event)
        self._model = model
        self._class_names = class_names
        self._camera = camera
        self._config = config
        self._predictions: list[Prediction] = []
        self._latency_ms = 0.0

    @property
    def predictions(self) -> list[Prediction]:
        with self._lock:
            return list(self._predictions)

    @property
    def latency_ms(self) -> float:
        with self._lock:
            return self._latency_ms

    def run(self) -> None:
        last_sequence = -1
        try:
            while not self._stop_event.wait(self._config.inference_interval_seconds):
                frame, sequence, _ = self._camera.snapshot()
                if frame is None or sequence == last_sequence:
                    continue
                last_sequence = sequence

                started = time.perf_counter()
                batch = preprocess(frame, self._config.image_size)
                probabilities = self._model.predict(batch, verbose=0)[0]
                ranked = rank_predictions(probabilities, self._class_names, self._config.top_k)
                latency = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self._predictions = ranked
                    self._latency_ms = latency
        except Exception as exc:  # noqa: BLE001
            self._fail(f"Model inference failed: {exc}")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1024)
def _glyph_alpha(
    text: str, font: int, scale: float, thickness: int, factor: int, weight: float
) -> tuple[np.ndarray, int, int]:
    """Rasterise ``text`` once into a supersampled, antialiased alpha mask.

    Returns ``(alpha, origin_x, origin_y)`` where the origin is the text baseline
    position inside the mask. Cached, so repeated labels cost almost nothing.
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 3
    width, height = text_w + 2 * pad, text_h + 2 * baseline + 2 * pad
    origin_y = text_h + baseline + pad

    mask = np.zeros((height * factor, width * factor), dtype=np.uint8)
    cv2.putText(
        mask, text, (pad * factor, origin_y * factor), font, scale * factor, 255,
        max(1, round(thickness * factor * weight)), cv2.LINE_AA,
    )
    alpha = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    alpha = alpha.astype(np.float32)[..., np.newaxis] / 255.0
    alpha.setflags(write=False)
    return alpha, pad, origin_y


class PreviewRenderer:
    """Composes the dashboard: header bar, camera view and a results side panel."""

    HEADER_HEIGHT = 64
    FOOTER_HEIGHT = 30
    PANEL_PADDING = 28
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SUPERSAMPLE = 4
    TEXT_WEIGHT = 1.35  # >1 thickens OpenCV's thin Hershey strokes

    # Scan effect tuning (pixels unless noted).
    SCAN_MARGIN = 24
    SCAN_EDGE_FADE = 90
    SCAN_TRAIL = 120        # maximum length of the fading trail
    SCAN_GLOW_SIGMA = 9.0   # softness of the bloom around the line
    SCAN_CORE_SIGMA = 0.8   # thickness of the bright core

    def __init__(self, config: Config, theme: Theme = THEME) -> None:
        self._config = config
        self._theme = theme
        width, height = config.preview_size
        self._view_box = (
            0,
            self.HEADER_HEIGHT,
            width - config.side_panel_width,
            height - self.FOOTER_HEIGHT,
        )
        self._view_buffer: Optional[np.ndarray] = None
        view_width = self._view_box[2] - self._view_box[0]
        self._scan_fade = self._horizontal_fade(view_width, self.SCAN_MARGIN, self.SCAN_EDGE_FADE)
        accent = np.asarray(theme.accent, dtype=np.float32) / 255.0
        self._scan_glow_color = accent
        self._scan_core_color = accent + (1.0 - accent) * 0.75  # lighter, near-white core

    # -- public ----------------------------------------------------------- #
    def render(
        self,
        frame: np.ndarray,
        predictions: Sequence[Prediction],
        elapsed: float,
        fps: float,
        latency_ms: float = 0.0,
        mirror: bool = False,
    ) -> np.ndarray:
        width, height = self._config.preview_size
        image = np.empty((height, width, 3), dtype=np.uint8)  # fully overwritten below

        x1, y1, x2, y2 = self._view_box
        view = self._fit_frame(frame, (x2 - x1, y2 - y1))
        if mirror:  # flip only the picture; overlays and text stay readable
            cv2.flip(view, 1, dst=view)
        self._draw_view_overlay(view, elapsed)
        image[y1:y2, x1:x2] = view

        self._draw_header(image, elapsed, fps, latency_ms)
        self._draw_side_panel(image, predictions)
        self._draw_footer(image)
        return image

    # -- primitives ------------------------------------------------------- #
    def _fit_frame(self, frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Centre-crop to the target aspect, then scale into a reusable buffer."""
        target_w, target_h = size
        height, width = frame.shape[:2]
        aspect = target_w / target_h
        if width / height > aspect:
            crop_w = int(round(height * aspect))
            left = (width - crop_w) // 2
            crop = frame[:, left : left + crop_w]
        else:
            crop_h = int(round(width / aspect))
            top = (height - crop_h) // 2
            crop = frame[top : top + crop_h, :]

        downscale = crop.shape[1] / target_w
        interpolation = cv2.INTER_AREA if downscale >= 1.5 else cv2.INTER_LINEAR
        buffer = self._view_buffer
        if buffer is None or buffer.shape[:2] != (target_h, target_w):
            buffer = self._view_buffer = np.empty((target_h, target_w, 3), dtype=np.uint8)
        cv2.resize(crop, (target_w, target_h), dst=buffer, interpolation=interpolation)
        return buffer

    @staticmethod
    def _rounded_rect(
        image: np.ndarray, x1: int, y1: int, x2: int, y2: int, radius: int, color: Color
    ) -> None:
        r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        cv2.rectangle(image, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(image, (cx, cy), r, color, -1, cv2.LINE_AA)

    def _text_width(self, text: str, scale: float, thickness: int = 1) -> int:
        return cv2.getTextSize(text, self.FONT, scale, thickness)[0][0]

    def _fit_label(self, text: str, max_width: int, scale: float, thickness: int) -> str:
        """Truncate ``text`` with '...' so it fits within ``max_width`` pixels."""
        if self._text_width(text, scale, thickness) <= max_width:
            return text
        while len(text) > 1 and self._text_width(text + "...", scale, thickness) > max_width:
            text = text[:-1]
        return text + "..."

    def _text(
        self,
        image: np.ndarray,
        text: str,
        origin: Point,
        scale: float,
        color: Color,
        thickness: int = 1,
    ) -> None:
        """Blend cached, supersampled antialiased text onto ``image``."""
        alpha, offset_x, offset_y = _glyph_alpha(
            text, self.FONT, scale, thickness, self.TEXT_SUPERSAMPLE, self.TEXT_WEIGHT
        )
        mask_h, mask_w = alpha.shape[:2]
        x0, y0 = origin[0] - offset_x, origin[1] - offset_y

        height, width = image.shape[:2]
        left, top = max(0, x0), max(0, y0)
        right, bottom = min(width, x0 + mask_w), min(height, y0 + mask_h)
        if left >= right or top >= bottom:
            return

        coverage = alpha[top - y0 : bottom - y0, left - x0 : right - x0]
        region = image[top:bottom, left:right].astype(np.float32)
        blended = region + (np.asarray(color, dtype=np.float32) - region) * coverage
        image[top:bottom, left:right] = blended.astype(np.uint8)

    def _text_centered(
        self,
        image: np.ndarray,
        text: str,
        center_x: int,
        baseline_y: int,
        scale: float,
        color: Color,
        thickness: int = 1,
    ) -> None:
        width = self._text_width(text, scale, thickness)
        self._text(image, text, (center_x - width // 2, baseline_y), scale, color, thickness)

    def _chip_width(self, text: str, with_dot: bool) -> int:
        return self._text_width(text, 0.4) + 24 + (16 if with_dot else 0)

    def _chip(
        self,
        image: np.ndarray,
        x: int,
        center_y: int,
        text: str,
        dot_color: Optional[Color] = None,
    ) -> int:
        """Draw a pill-shaped chip with optional status dot; return its width."""
        t = self._theme
        width = self._chip_width(text, dot_color is not None)
        self._rounded_rect(image, x, center_y - 14, x + width, center_y + 14, 14, t.card)
        text_x = x + 12
        if dot_color is not None:
            cv2.circle(image, (x + 17, center_y), 4, dot_color, -1, cv2.LINE_AA)
            text_x = x + 28
        self._text(image, text, (text_x, center_y + 5), 0.4, t.text)
        return width

    # -- sections --------------------------------------------------------- #
    def _draw_header(self, image: np.ndarray, elapsed: float, fps: float, latency_ms: float) -> None:
        t = self._theme
        width = image.shape[1]
        height = self.HEADER_HEIGHT

        image[0:height] = t.panel
        cv2.line(image, (0, height - 1), (width, height - 1), t.divider, 1)

        # Logo mark and title.
        cv2.circle(image, (40, 32), 14, t.accent, 2, cv2.LINE_AA)
        cv2.circle(image, (40, 32), 5, t.accent, -1, cv2.LINE_AA)
        self._text(image, "SNAKE VISION", (66, 31), 0.62, t.text, 2)
        self._text(image, "REAL-TIME FIELD CLASSIFIER", (66, 50), 0.34, t.muted)

        # Status chips, laid out right to left.
        pulse = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(elapsed * 2.0 * np.pi * 1.2))
        live_dot = tuple(int(c * pulse) for c in t.danger)
        latency = f"{latency_ms:3.0f} MS" if latency_ms > 0 else "-- MS"
        chips = (
            ("LIVE", live_dot),
            (f"{fps:4.1f} FPS", None),
            (f"INFERENCE {latency}", None),
        )
        x = width - 24
        for text, dot in reversed(chips):
            x -= self._chip_width(text, dot is not None)
            self._chip(image, x, 32, text, dot)
            x -= 10

    @staticmethod
    def _horizontal_fade(width: int, margin: int, fade: int) -> np.ndarray:
        """Smoothstep fade-in/out along x, zero outside ``margin``."""
        x = np.arange(width, dtype=np.float32)
        ramp_in = np.clip((x - margin) / fade, 0.0, 1.0)
        ramp_out = np.clip((width - 1 - margin - x) / fade, 0.0, 1.0)
        ramp = np.minimum(ramp_in, ramp_out)
        return ramp * ramp * (3.0 - 2.0 * ramp)

    def _draw_scan_effect(self, view: np.ndarray, elapsed: float) -> None:
        """A sweeping scanner bar with bloom and a motion-aware fading trail.

        The position is computed in floating point from the clock (sub-pixel
        accurate), the trail lengthens with speed and always lags behind the
        direction of travel, and the layers are added as light so the bar looks
        luminous instead of painted on.
        """
        height = view.shape[0]
        margin = self.SCAN_MARGIN
        cycle = (elapsed % self._config.scan_cycle_seconds) / self._config.scan_cycle_seconds
        phase = 2.0 * np.pi * cycle

        eased = 0.5 - 0.5 * np.cos(phase)            # smooth ease in/out, 0..1
        direction = 1.0 if np.sin(phase) >= 0 else -1.0  # +1 = moving down
        speed = abs(np.sin(phase))                    # 0 at the turn-arounds
        y = margin + eased * (height - 2 * margin)

        trail_length = self.SCAN_TRAIL * (0.3 + 0.7 * speed)
        reach = int(max(trail_length, 4 * self.SCAN_GLOW_SIGMA)) + 2
        top, bottom = max(0, int(y) - reach), min(height, int(y) + reach + 1)
        if top >= bottom:
            return

        rows = np.arange(top, bottom, dtype=np.float32)
        delta = rows - np.float32(y)
        behind = -delta * direction  # >0 for rows the bar has already passed

        core = np.exp(-(delta**2) / (2.0 * self.SCAN_CORE_SIGMA**2))
        glow = np.exp(-(delta**2) / (2.0 * self.SCAN_GLOW_SIGMA**2))
        trail = np.clip(1.0 - behind / max(trail_length, 1.0), 0.0, 1.0) ** 2
        trail = np.where(behind > 0, trail, 0.0)

        fade = self._scan_fade[np.newaxis, :]
        glow_2d = (0.34 * glow + 0.30 * trail).astype(np.float32)[:, np.newaxis] * fade
        core_2d = (0.95 * core).astype(np.float32)[:, np.newaxis] * fade

        light = cv2.merge([
            cv2.convertScaleAbs(
                glow_2d * self._scan_glow_color[c] + core_2d * self._scan_core_color[c],
                alpha=255.0,
            )
            for c in range(3)
        ])
        region = view[top:bottom]
        cv2.add(region, light, dst=region)  # saturating additive "light" blend

    def _draw_view_overlay(self, view: np.ndarray, elapsed: float) -> None:
        """Camera tag, corner guides and a soft scan line, drawn on the camera view."""
        t = self._theme
        height, width = view.shape[:2]

        self._draw_scan_effect(view, elapsed)

        # Corner guides.
        length, inset = 36, 20
        for x, y0, dx, dy in (
            (inset, inset, 1, 1),
            (width - inset, inset, -1, 1),
            (inset, height - inset, 1, -1),
            (width - inset, height - inset, -1, -1),
        ):
            cv2.line(view, (x, y0), (x + dx * length, y0), t.text, 2, cv2.LINE_AA)
            cv2.line(view, (x, y0), (x, y0 + dy * length), t.text, 2, cv2.LINE_AA)

        # Camera tag.
        cam_w, cam_h = self._config.preview_size
        self._chip(view, 36, 48, f"CAM 01   {cam_w}x{cam_h}", t.success)

    def _status(self, prediction: Optional[Prediction]) -> tuple[Color, str]:
        t, c = self._theme, self._config
        if prediction is None:
            return t.muted, "ANALYZING..."
        if prediction.confidence >= c.confidence_high:
            return t.success, "HIGH CONFIDENCE"
        if prediction.confidence >= c.confidence_medium:
            return t.warning, "MODERATE CONFIDENCE"
        return t.danger, "LOW CONFIDENCE"

    def _draw_side_panel(self, image: np.ndarray, predictions: Sequence[Prediction]) -> None:
        t = self._theme
        height, width = image.shape[:2]
        x1, y1, y2 = self._view_box[2], self.HEADER_HEIGHT, height - self.FOOTER_HEIGHT
        pad = self.PANEL_PADDING
        left, panel_w = x1 + pad, width - x1 - 2 * pad
        center_x = x1 + (width - x1) // 2

        image[y1:y2, x1:width] = t.panel
        cv2.line(image, (x1, y1), (x1, y2), t.divider, 1)

        # Top match: circular confidence gauge.
        self._text(image, "TOP MATCH", (left, y1 + 38), 0.4, t.muted)
        top = predictions[0] if predictions else None
        confidence = float(np.clip(top.confidence, 0.0, 1.0)) if top else 0.0
        status_color, status_text = self._status(top)

        gauge_center, radius, thickness = (center_x, y1 + 128), 64, 9
        cv2.circle(image, gauge_center, radius, t.track, thickness, cv2.LINE_AA)
        if confidence > 0:
            cv2.ellipse(
                image, gauge_center, (radius, radius), 0,
                -90, -90 + 360 * confidence, status_color, thickness, cv2.LINE_AA,
            )
        self._text_centered(
            image, f"{confidence:.0%}" if top else "--",
            center_x, gauge_center[1] + 12, 1.0, t.text, 2,
        )

        label_y = gauge_center[1] + radius + 46
        label = self._fit_label(top.label if top else "No result yet", panel_w, 0.72, 2)
        self._text_centered(image, label, center_x, label_y, 0.72, t.text, 2)

        pill_w = self._text_width(status_text, 0.4) + 32
        pill_cy = label_y + 30
        self._rounded_rect(
            image, center_x - pill_w // 2, pill_cy - 13, center_x + pill_w // 2, pill_cy + 13,
            13, t.card,
        )
        self._text_centered(image, status_text, center_x, pill_cy + 5, 0.4, status_color)

        # Runner-up predictions.
        divider_y = pill_cy + 38
        cv2.line(image, (left, divider_y), (left + panel_w, divider_y), t.divider, 1)
        self._text(image, "OTHER CANDIDATES", (left, divider_y + 32), 0.4, t.muted)

        for index, prediction in enumerate(predictions[1:]):
            row_y = divider_y + 72 + index * 58
            value = float(np.clip(prediction.confidence, 0.0, 1.0))
            percent = f"{value:.0%}"
            percent_w = self._text_width(percent, 0.46)
            name = self._fit_label(prediction.label, panel_w - percent_w - 16, 0.46, 1)

            self._text(image, name, (left, row_y), 0.46, t.text)
            self._text(image, percent, (left + panel_w - percent_w, row_y), 0.46, t.muted)
            bar_top = row_y + 12
            self._rounded_rect(image, left, bar_top, left + panel_w, bar_top + 6, 3, t.track)
            fill = int(panel_w * value)
            if fill >= 6:
                self._rounded_rect(image, left, bar_top, left + fill, bar_top + 6, 3, t.accent)

    def _draw_footer(self, image: np.ndarray) -> None:
        t = self._theme
        height, width = image.shape[:2]
        top = height - self.FOOTER_HEIGHT

        image[top:height] = t.panel
        cv2.line(image, (0, top), (width, top), t.divider, 1)
        self._text(image, "Q / ESC   CLOSE      M   MIRROR", (24, height - 10), 0.36, t.muted)

        in_w, in_h = self._config.image_size
        info = f"MODEL INPUT {in_w}x{in_h}   |   TEACHABLE MACHINE"
        self._text(image, info, (width - self._text_width(info, 0.36) - 24, height - 10), 0.36, t.muted)


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
def enable_dpi_awareness() -> None:
    """Stop Windows from stretching the window on scaled (125%/150%) displays."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def is_window_open(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def run(config: Config) -> None:
    enable_dpi_awareness()
    model, class_names = load_model(config)

    camera = open_camera(config.camera_index)
    requested_fps = configure_camera(camera, config.preview_size)
    if requested_fps is None:
        LOGGER.info("Camera FPS cannot be queried; using the camera's default mode.")
    else:
        LOGGER.info("Camera accepted a request for %.0f FPS.", requested_fps)

    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.window_name, *config.window_size)
    LOGGER.info("Live preview started. Close the window, or press Q or Esc to quit.")

    stop_event = threading.Event()
    stream = CameraStream(camera, stop_event)
    inference = InferenceWorker(model, class_names, stream, config, stop_event)
    renderer = PreviewRenderer(config)
    start_time = time.perf_counter()
    frame_interval = 1.0 / config.ui_fps
    mirror = config.mirror
    next_tick = start_time

    stream.start()
    inference.start()
    try:
        while not stop_event.is_set():
            error = stream.error or inference.error
            if error:
                raise RuntimeError(error)

            frame, _, fps = stream.snapshot()
            if frame is None:
                cv2.waitKey(1)
                if not is_window_open(config.window_name):
                    break
                continue

            preview = renderer.render(
                frame,
                inference.predictions,
                time.perf_counter() - start_time,
                fps,
                inference.latency_ms,
                mirror,
            )
            cv2.imshow(config.window_name, preview)

            # Fixed-rate UI clock: animation stays smooth even if the camera is slower.
            next_tick += frame_interval
            now = time.perf_counter()
            if next_tick < now:  # running behind: resync instead of spiralling
                next_tick = now
            key = cv2.waitKey(max(1, int((next_tick - now) * 1000))) & 0xFF
            if key in (ord("m"), ord("M")):
                mirror = not mirror
            if key in (ord("q"), 27) or not is_window_open(config.window_name):
                break

        error = stream.error or inference.error
        if error:
            raise RuntimeError(error)
    finally:
        stop_event.set()
        camera.release()
        stream.join(timeout=2)
        inference.join(timeout=2)
        cv2.destroyAllWindows()


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Real-time snake classifier (Teachable Machine).")
    parser.add_argument("--model", type=Path, default=defaults.model_path, help="Path to keras_model.h5")
    parser.add_argument("--labels", type=Path, default=defaults.labels_path, help="Path to labels.txt")
    parser.add_argument("--camera", type=int, default=defaults.camera_index, help="Webcam index")
    parser.add_argument("--no-mirror", action="store_true", help="Start with the camera un-mirrored (press M to toggle)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    return Config(
        model_path=args.model,
        labels_path=args.labels,
        camera_index=args.camera,
        mirror=not args.no_mirror,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    config = parse_args(argv)
    try:
        run(config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Interrupted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())