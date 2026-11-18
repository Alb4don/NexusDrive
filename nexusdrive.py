from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import logging
import logging.handlers
import math
import os
import re
import sqlite3
import threading
import time
import uuid
import warnings
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from scipy.optimize import linear_sum_assignment

APP_NAME = "NexusDrive"
APP_VERSION = "2.0.0"
LOGGER_NAME = "nexusdrive"

VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv", ".ts"})
MODEL_EXTENSIONS = frozenset({".pt", ".onnx", ".engine", ".torchscript"})
DOWNLOADABLE_MODELS = frozenset({"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"})
STREAM_SCHEMES = frozenset({"rtsp", "rtsps", "http", "https"})

TARGET_CLASSES: dict[int, str] = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_WIDTH_M: dict[int, tuple[float, float]] = {
    0: (0.55, 0.30),
    1: (0.60, 0.25),
    2: (1.80, 0.12),
    3: (0.80, 0.25),
    5: (2.55, 0.08),
    7: (2.45, 0.12),
}
CLASS_GROUP: dict[int, int] = {0: 0, 1: 1, 3: 1, 2: 2, 5: 2, 7: 2}
VULNERABLE_CLASSES = frozenset({0, 1, 3})

THREAT_LOW = 0
THREAT_WARNING = 1
THREAT_CRITICAL = 2
THREAT_NAMES = {THREAT_LOW: "LOW", THREAT_WARNING: "WARNING", THREAT_CRITICAL: "CRITICAL"}
THREAT_COLORS_BGR = {THREAT_LOW: (90, 200, 90), THREAT_WARNING: (0, 170, 255), THREAT_CRITICAL: (40, 40, 235)}

CHI2_4DOF_99 = 13.2767


class NexusError(Exception):
    public_message = "Operation failed"

    def __init__(self, public_message: str | None = None) -> None:
        super().__init__(public_message or self.public_message)
        if public_message:
            self.public_message = public_message


class SourceError(NexusError):
    public_message = "Invalid video source"


class ModelError(NexusError):
    public_message = "Detection model unavailable"


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, lo), hi)


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    return int(round(_env_float(name, float(default), float(lo), float(hi))))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    width: int = 1280
    height: int = 720
    lane_scale: float = 0.5
    model_path: str = "yolov8n.pt"
    model_sha256: str = ""
    device: str = "auto"
    det_conf: float = 0.40
    det_iou: float = 0.50
    imgsz: int = 640
    hfov_deg: float = 70.0
    camera_height_m: float = 1.30
    horizon_ratio: float = 0.6125
    lane_width_m: float = 3.7
    lane_lookahead_m: float = 30.0
    ttc_critical_s: float = 2.5
    ttc_warning_s: float = 4.5
    warning_distance_m: float = 12.0
    jpeg_quality: int = 80
    stream_fps: float = 30.0
    max_upload_mb: int = 2048
    allow_network_streams: bool = False
    data_dir: Path = field(default_factory=lambda: Path.home() / ".nexusdrive")
    debug: bool = False

    @property
    def focal_px(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @classmethod
    def from_env(cls) -> "Settings":
        base = cls()
        return cls(
            width=_env_int("NEXUSDRIVE_WIDTH", base.width, 320, 3840) // 2 * 2,
            height=_env_int("NEXUSDRIVE_HEIGHT", base.height, 240, 2160) // 2 * 2,
            lane_scale=_env_float("NEXUSDRIVE_LANE_SCALE", base.lane_scale, 0.25, 1.0),
            model_path=_env_str("NEXUSDRIVE_MODEL", base.model_path),
            model_sha256=_env_str("NEXUSDRIVE_MODEL_SHA256", "").strip().lower(),
            device=_env_str("NEXUSDRIVE_DEVICE", base.device).strip().lower(),
            det_conf=_env_float("NEXUSDRIVE_DET_CONF", base.det_conf, 0.05, 0.95),
            det_iou=_env_float("NEXUSDRIVE_DET_IOU", base.det_iou, 0.1, 0.9),
            imgsz=_env_int("NEXUSDRIVE_IMGSZ", base.imgsz, 160, 1920) // 32 * 32,
            hfov_deg=_env_float("NEXUSDRIVE_HFOV", base.hfov_deg, 20.0, 150.0),
            camera_height_m=_env_float("NEXUSDRIVE_CAMERA_HEIGHT", base.camera_height_m, 0.3, 4.0),
            horizon_ratio=_env_float("NEXUSDRIVE_HORIZON", base.horizon_ratio, 0.3, 0.85),
            ttc_critical_s=_env_float("NEXUSDRIVE_TTC_CRITICAL", base.ttc_critical_s, 0.5, 10.0),
            ttc_warning_s=_env_float("NEXUSDRIVE_TTC_WARNING", base.ttc_warning_s, 1.0, 15.0),
            warning_distance_m=_env_float("NEXUSDRIVE_WARN_DISTANCE", base.warning_distance_m, 2.0, 60.0),
            jpeg_quality=_env_int("NEXUSDRIVE_JPEG_QUALITY", base.jpeg_quality, 30, 95),
            stream_fps=_env_float("NEXUSDRIVE_STREAM_FPS", base.stream_fps, 5.0, 60.0),
            max_upload_mb=_env_int("NEXUSDRIVE_MAX_UPLOAD_MB", base.max_upload_mb, 1, 16384),
            allow_network_streams=_env_bool("NEXUSDRIVE_ALLOW_STREAMS", base.allow_network_streams),
            data_dir=Path(_env_str("NEXUSDRIVE_HOME", str(base.data_dir))).expanduser(),
            debug=_env_bool("NEXUSDRIVE_DEBUG", base.debug),
        )


class RedactingFilter(logging.Filter):
    _patterns: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/@]+@"), r"\1***@"),
        (re.compile(r"(?i)\b(password|passwd|pwd|token|secret|api[_-]?key|auth)=([^\s&;]+)"), r"\1=***"),
        (re.compile(r"[\r\n\t\x00-\x08\x0b\x0c\x0e-\x1f]+"), " "),
    )

    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        home = str(Path.home())
        self._home = home if len(home) > 1 else None
        self._debug = debug

    def redact(self, text: str) -> str:
        for pattern, repl in self._patterns:
            text = pattern.sub(repl, text)
        if self._home:
            text = text.replace(self._home, "~")
        return text[:2000]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = self.redact(message)
        record.args = None
        if not self._debug:
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True


def _secure_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def setup_logging(settings: Settings) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(logger, "_nexus_configured", False):
        return logger
    logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    logger.propagate = False
    redactor = RedactingFilter(settings.debug)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.addFilter(redactor)
    console.setFormatter(formatter)
    logger.addHandler(console)
    try:
        log_dir = settings.data_dir / "logs"
        _secure_mkdir(log_dir)
        log_file = log_dir / "nexusdrive.log"
        handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.addFilter(redactor)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        try:
            os.chmod(log_file, 0o600)
        except OSError:
            pass
    except OSError:
        logger.warning("file logging unavailable")
    logger._nexus_configured = True
    return logger


log = logging.getLogger(LOGGER_NAME)


def preflight(settings: Settings) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": True, "modules": {}, "data_dir_writable": False}
    for module in ("cv2", "numpy", "scipy", "torch", "ultralytics", "nicegui"):
        found = importlib.util.find_spec(module) is not None
        report["modules"][module] = found
        if not found and module in {"cv2", "numpy", "scipy"}:
            report["ok"] = False
    try:
        _secure_mkdir(settings.data_dir)
        probe = settings.data_dir / f".probe-{uuid.uuid4().hex}"
        probe.write_bytes(b"1")
        probe.unlink()
        report["data_dir_writable"] = True
    except OSError:
        report["ok"] = False
    report["opencv_ffmpeg"] = "FFMPEG" in cv2.getBuildInformation().upper()
    return report


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    target: int | str
    key: str
    label: str


def _file_fingerprint(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    chunk = 1 << 20
    with path.open("rb") as handle:
        digest.update(handle.read(chunk))
        if size > 2 * chunk:
            handle.seek(size - chunk)
            digest.update(handle.read(chunk))
    return "file:" + digest.hexdigest()[:32]


def resolve_source(raw: Any, settings: Settings) -> SourceSpec:
    if isinstance(raw, bool):
        raise SourceError()
    if isinstance(raw, int):
        raw = str(raw)
    if not isinstance(raw, str):
        raise SourceError()
    text = raw.strip().strip("\"'").strip()
    if not text or len(text) > 4096 or "\x00" in text or any(ord(ch) < 32 for ch in text):
        raise SourceError()
    if re.fullmatch(r"\d{1,2}", text):
        index = int(text)
        if index > 15:
            raise SourceError("Camera index out of range")
        return SourceSpec("camera", index, f"camera:{index}", f"Camera {index}")
    if "://" in text:
        if not settings.allow_network_streams:
            raise SourceError("Network streams are disabled")
        try:
            parts = urlsplit(text)
            port = parts.port
        except ValueError as exc:
            raise SourceError() from exc
        scheme = parts.scheme.lower()
        if scheme not in STREAM_SCHEMES or not parts.hostname:
            raise SourceError("Unsupported stream protocol")
        identity = f"{scheme}://{parts.hostname}:{port or ''}{parts.path}"
        key = "stream:" + hashlib.sha256(identity.encode()).hexdigest()[:32]
        return SourceSpec("stream", text, key, f"{scheme}://{parts.hostname}")
    try:
        resolved = Path(text).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SourceError("File not found") from exc
    if not resolved.is_file():
        raise SourceError("Not a regular file")
    if resolved.suffix.lower() not in VIDEO_EXTENSIONS:
        raise SourceError("Unsupported file type")
    size = resolved.stat().st_size
    if size <= 0:
        raise SourceError("Empty file")
    if size > settings.max_upload_mb * (1 << 20) * 4:
        raise SourceError("File too large")
    try:
        key = _file_fingerprint(resolved, size)
    except OSError as exc:
        raise SourceError("File not readable") from exc
    return SourceSpec("file", str(resolved), key, resolved.name[:120])


def normalize_frame(frame: Any, size: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
        return None
    if frame.dtype != np.uint8:
        if frame.dtype == np.uint16:
            frame = (frame >> 8).astype(np.uint8)
        elif np.issubdtype(frame.dtype, np.floating):
            data = np.nan_to_num(frame.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
            peak = float(data.max()) if data.size else 0.0
            data = np.clip(data, 0.0, 1.0) * 255.0 if peak <= 1.0 else np.clip(data, 0.0, 255.0)
            frame = data.astype(np.uint8)
        elif np.issubdtype(frame.dtype, np.integer) or frame.dtype == np.bool_:
            frame = np.clip(frame.astype(np.int64), 0, 255).astype(np.uint8)
        else:
            return None
    if frame.ndim == 3:
        channels = frame.shape[2]
        if channels == 1:
            frame = frame[:, :, 0]
        elif channels == 4:
            frame = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGRA2BGR)
        elif channels != 3:
            return None
    if frame.ndim == 2:
        frame = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_GRAY2BGR)
    height, width = frame.shape[:2]
    if height < 16 or width < 16:
        return None
    target_w, target_h = size
    target_aspect = target_w / target_h
    aspect = width / height
    if abs(aspect - target_aspect) / target_aspect > 0.02:
        if aspect > target_aspect:
            new_w = max(1, int(round(height * target_aspect)))
            x0 = (width - new_w) // 2
            frame = frame[:, x0:x0 + new_w]
        else:
            new_h = max(1, int(round(width / target_aspect)))
            y0 = (height - new_h) // 2
            frame = frame[y0:y0 + new_h]
        height, width = frame.shape[:2]
    if (width, height) != (target_w, target_h):
        interp = cv2.INTER_AREA if width > target_w else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (target_w, target_h), interpolation=interp)
    return np.ascontiguousarray(frame)


@dataclass
class LaneState:
    valid: bool = False
    confidence: float = 0.0
    radius_m: float | None = None
    offset_m: float | None = None
    left_img: np.ndarray | None = None
    right_img: np.ndarray | None = None
    vanishing_y: float | None = None

    def corridor_at(self, y: float) -> tuple[float, float] | None:
        if not self.valid or self.left_img is None or self.right_img is None:
            return None
        ly, lx = self.left_img[:, 1], self.left_img[:, 0]
        ry, rx = self.right_img[:, 1], self.right_img[:, 0]
        lo = np.argsort(ly)
        ro = np.argsort(ry)
        xl = float(np.interp(y, ly[lo], lx[lo]))
        xr = float(np.interp(y, ry[ro], rx[ro]))
        if not (math.isfinite(xl) and math.isfinite(xr)) or xr <= xl:
            return None
        return xl, xr


def _robust_polyfit(y: np.ndarray, x: np.ndarray, min_span: float) -> np.ndarray | None:
    if y.size < 40 or float(np.ptp(y)) < min_span:
        return None
    if y.size > 6000:
        idx = np.linspace(0, y.size - 1, 6000).astype(np.int64)
        y, x = y[idx], x[idx]
    yf = y.astype(np.float64)
    xf = x.astype(np.float64)
    weights = np.ones_like(yf)
    coef = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(4):
            try:
                coef = np.polyfit(yf, xf, 2, w=np.sqrt(weights))
            except (np.linalg.LinAlgError, ValueError):
                return None
            residual = xf - np.polyval(coef, yf)
            mad = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826 + 1e-6
            k = 1.345 * mad
            magnitude = np.abs(residual)
            weights = np.where(magnitude <= k, 1.0, k / np.maximum(magnitude, 1e-9))
    if coef is None or not np.all(np.isfinite(coef)):
        return None
    return coef


class LaneAnalyzer:
    def __init__(self, width: int, height: int, scale_to_full: float, lane_width_m: float = 3.7,
                 lookahead_m: float = 30.0, top_ratio: float = 0.65) -> None:
        self.w = int(width)
        self.h = int(height)
        self.scale_to_full = float(scale_to_full)
        w, h = float(self.w), float(self.h)
        src = np.float32([[w * 0.45, h * top_ratio], [w * 0.55, h * top_ratio], [w * 0.95, h * 0.95], [w * 0.05, h * 0.95]])
        dst = np.float32([[w * 0.2, 0], [w * 0.8, 0], [w * 0.8, h], [w * 0.2, h]])
        self.matrix = cv2.getPerspectiveTransform(src, dst)
        self.inv_matrix = cv2.getPerspectiveTransform(dst, src)
        self.xm_per_px = lane_width_m / (0.6 * w)
        self.ym_per_px = lookahead_m / h
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        kernel_w = max(9, int(round(w * 0.035)) | 1)
        self.tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        self.ploty = np.linspace(0, h - 1, 40)
        gauss = np.exp(-0.5 * (np.arange(-15, 16) / 6.0) ** 2)
        self.hist_kernel = gauss / gauss.sum()
        self.left: np.ndarray | None = None
        self.right: np.ndarray | None = None
        self.misses = 0
        self.confidence = 0.0
        self.max_misses = 12

    def reset(self) -> None:
        self.left = None
        self.right = None
        self.misses = 0
        self.confidence = 0.0

    def export(self) -> dict[str, list[float]] | None:
        if self.left is None or self.right is None or self.confidence < 0.5:
            return None
        return {"left": [float(v) for v in self.left], "right": [float(v) for v in self.right]}

    def seed(self, prior: dict[str, Any] | None) -> None:
        self.reset()
        if not isinstance(prior, dict):
            return
        try:
            left = np.asarray(prior.get("left"), dtype=np.float64)
            right = np.asarray(prior.get("right"), dtype=np.float64)
        except (TypeError, ValueError):
            return
        if left.shape == (3,) and right.shape == (3,) and np.all(np.isfinite(left)) and np.all(np.isfinite(right)):
            if self._pair_quality(left, right) > 0.0:
                self.left, self.right = left, right
                self.confidence = 0.3
                self.misses = self.max_misses // 2

    def binary_mask(self, bgr: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
        hue = hls[:, :, 0]
        light = self.clahe.apply(hls[:, :, 1])
        sat = hls[:, :, 2]
        tophat = cv2.morphologyEx(light, cv2.MORPH_TOPHAT, self.tophat_kernel)
        sobel = np.abs(cv2.Sobel(light, cv2.CV_32F, 1, 0, ksize=3))
        peak = float(sobel.max())
        if peak > 1e-6:
            scaled = (sobel * (255.0 / peak)).astype(np.uint8)
        else:
            scaled = np.zeros_like(light)
        edges = (scaled >= 25) & (tophat >= 12)
        yellow = (hue >= 15) & (hue <= 35) & (sat >= 80) & (light >= 70) & (tophat >= 8)
        bright = tophat >= 32
        return (edges | yellow | bright).astype(np.uint8)

    def _sliding_window(self, xs: np.ndarray, ys: np.ndarray, warped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, h = self.w, self.h
        hist = warped[h // 2:, :].sum(axis=0).astype(np.float64)
        hist = np.convolve(hist, self.hist_kernel, mode="same")
        mid = w // 2
        lo, hi = int(w * 0.05), int(w * 0.95)
        left_base = lo + int(np.argmax(hist[lo:mid]))
        right_base = mid + int(np.argmax(hist[mid:hi]))
        windows = 9
        win_h = max(1, h // windows)
        margin = max(8, int(w * 0.08))
        minpix = max(10, int(margin * win_h * 0.01))
        lx, rx = left_base, right_base
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        for i in range(windows):
            y_low = h - (i + 1) * win_h
            y_high = h - i * win_h
            in_row = (ys >= y_low) & (ys < y_high)
            good_l = np.flatnonzero(in_row & (xs >= lx - margin) & (xs < lx + margin))
            good_r = np.flatnonzero(in_row & (xs >= rx - margin) & (xs < rx + margin))
            left_parts.append(good_l)
            right_parts.append(good_r)
            shift_l = int(np.mean(xs[good_l])) - lx if good_l.size > minpix else None
            shift_r = int(np.mean(xs[good_r])) - rx if good_r.size > minpix else None
            if shift_l is None and shift_r is not None:
                shift_l = shift_r
            if shift_r is None and shift_l is not None:
                shift_r = shift_l
            lx += shift_l or 0
            rx += shift_r or 0
        return np.concatenate(left_parts), np.concatenate(right_parts)

    def _around_prior(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        margin = max(8, int(self.w * 0.06))
        assert self.left is not None and self.right is not None
        lc = np.polyval(self.left, ys)
        rc = np.polyval(self.right, ys)
        return np.flatnonzero(np.abs(xs - lc) < margin), np.flatnonzero(np.abs(xs - rc) < margin)

    def _pair_quality(self, left: np.ndarray, right: np.ndarray) -> float:
        samples = np.linspace(0, self.h - 1, 12)
        widths = (np.polyval(right, samples) - np.polyval(left, samples)) * self.xm_per_px
        if not np.all(np.isfinite(widths)):
            return 0.0
        if widths.min() < 2.3 or widths.max() > 5.4:
            return 0.0
        spread = float(np.std(widths))
        if spread > 0.6:
            return 0.0
        return float(np.clip(1.0 - spread / 0.6, 0.0, 1.0))

    def analyze(self, bgr_small: np.ndarray) -> LaneState:
        binary = self.binary_mask(bgr_small)
        warped = cv2.warpPerspective(binary, self.matrix, (self.w, self.h), flags=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(warped)
        if self.left is not None and self.right is not None and self.misses < 3:
            li, ri = self._around_prior(xs, ys)
            if li.size < 40 or ri.size < 40:
                li, ri = self._sliding_window(xs, ys, warped)
        else:
            li, ri = self._sliding_window(xs, ys, warped)
        min_span = self.h * 0.3
        left_fit = _robust_polyfit(ys[li], xs[li], min_span)
        right_fit = _robust_polyfit(ys[ri], xs[ri], min_span)
        quality = 0.0
        if left_fit is not None and right_fit is not None:
            quality = self._pair_quality(left_fit, right_fit)
        if quality <= 0.0 and self.left is not None and self.right is not None:
            prior_offset = self.right - self.left
            if left_fit is not None and right_fit is None:
                candidate = (left_fit, left_fit + prior_offset)
            elif right_fit is not None and left_fit is None:
                candidate = (right_fit - prior_offset, right_fit)
            else:
                candidate = None
            if candidate is not None:
                q = self._pair_quality(*candidate)
                if q > 0.0:
                    left_fit, right_fit = candidate
                    quality = 0.6 * q
        if quality > 0.0 and left_fit is not None and right_fit is not None:
            if self.left is None or self.right is None:
                self.left, self.right = left_fit, right_fit
            else:
                alpha = 0.25 + 0.35 * quality
                self.left = (1 - alpha) * self.left + alpha * left_fit
                self.right = (1 - alpha) * self.right + alpha * right_fit
            self.misses = 0
            self.confidence = float(min(1.0, 0.7 * self.confidence + 0.3 * (0.4 + 0.6 * quality) + 0.05))
        else:
            self.misses += 1
            self.confidence *= 0.85
            if self.misses > self.max_misses:
                self.reset()
        return self._state()

    def _state(self) -> LaneState:
        if self.left is None or self.right is None:
            return LaneState()
        center = (self.left + self.right) / 2.0
        a, b = center[0], center[1]
        am = a * self.xm_per_px / (self.ym_per_px ** 2)
        bm = b * self.xm_per_px / self.ym_per_px
        y_eval = (self.h - 1) * self.ym_per_px
        radius = math.inf if abs(am) < 1e-7 else ((1 + (2 * am * y_eval + bm) ** 2) ** 1.5) / abs(2 * am)
        lane_center_x = float(np.polyval(center, self.h - 1))
        offset = (self.w / 2.0 - lane_center_x) * self.xm_per_px
        left_w = np.stack([np.polyval(self.left, self.ploty), self.ploty], axis=1).astype(np.float32)
        right_w = np.stack([np.polyval(self.right, self.ploty), self.ploty], axis=1).astype(np.float32)
        left_img = cv2.perspectiveTransform(left_w.reshape(-1, 1, 2), self.inv_matrix).reshape(-1, 2)
        right_img = cv2.perspectiveTransform(right_w.reshape(-1, 1, 2), self.inv_matrix).reshape(-1, 2)
        left_img = left_img * self.scale_to_full
        right_img = right_img * self.scale_to_full
        vanishing = None
        if self.confidence >= 0.6 and self.misses == 0 and radius > 900.0:
            vanishing = self._vanishing_y(left_img, right_img)
        valid = self.confidence >= 0.35 and bool(np.all(np.isfinite(left_img))) and bool(np.all(np.isfinite(right_img)))
        return LaneState(
            valid=valid,
            confidence=float(self.confidence),
            radius_m=float(radius) if math.isfinite(radius) else None,
            offset_m=float(offset) if math.isfinite(offset) else None,
            left_img=left_img,
            right_img=right_img,
            vanishing_y=vanishing,
        )

    @staticmethod
    def _vanishing_y(left: np.ndarray, right: np.ndarray) -> float | None:
        n = left.shape[0]
        sel = slice(int(n * 0.55), n)
        try:
            al, bl = np.polyfit(left[sel, 1], left[sel, 0], 1)
            ar, br = np.polyfit(right[sel, 1], right[sel, 0], 1)
        except (np.linalg.LinAlgError, ValueError):
            return None
        if abs(al - ar) < 1e-6:
            return None
        y = (br - bl) / (al - ar)
        return float(y) if math.isfinite(y) else None


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


def sanitize_detections(raw: Sequence[Detection], width: int, height: int) -> list[Detection]:
    clean: list[Detection] = []
    for det in raw:
        values = (det.x1, det.y1, det.x2, det.y2, det.conf)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            continue
        if det.cls not in TARGET_CLASSES:
            continue
        x1 = min(max(det.x1, 0.0), width - 1.0)
        x2 = min(max(det.x2, 0.0), width - 1.0)
        y1 = min(max(det.y1, 0.0), height - 1.0)
        y2 = min(max(det.y2, 0.0), height - 1.0)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        ratio = (x2 - x1) / (y2 - y1)
        if not 0.12 <= ratio <= 8.0:
            continue
        clean.append(Detection(x1, y1, x2, y2, float(min(max(det.conf, 0.0), 1.0)), int(det.cls)))
    return clean


class DetectorProtocol(Protocol):
    def detect(self, frame: np.ndarray, conf: float) -> list[Detection]:
        ...


def verify_weights(path: Path, expected_sha256: str) -> None:
    if not expected_sha256:
        return
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ModelError("Invalid model checksum configuration")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelError("Model file not readable") from exc
    if digest.hexdigest() != expected_sha256:
        raise ModelError("Model checksum mismatch")


def resolve_weights(settings: Settings) -> Path:
    candidate = Path(settings.model_path).expanduser()
    if candidate.suffix.lower() not in MODEL_EXTENSIONS:
        raise ModelError("Unsupported model format")
    if candidate.is_file():
        return candidate.resolve()
    if candidate.name in DOWNLOADABLE_MODELS and len(candidate.parts) == 1:
        models_dir = settings.data_dir / "models"
        _secure_mkdir(models_dir)
        return models_dir / candidate.name
    raise ModelError("Model file not found")


def _describe_error(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text[:300]}" if text else type(exc).__name__


def candidate_devices(preference: str, torch_module: Any) -> list[str]:
    available: list[str] = []
    try:
        if torch_module.cuda.is_available():
            available.extend(f"cuda:{i}" for i in range(torch_module.cuda.device_count()))
    except Exception as exc:
        log.warning("cuda probe failed: %s", _describe_error(exc))
    try:
        mps = getattr(torch_module.backends, "mps", None)
        if mps is not None and mps.is_available():
            available.append("mps")
    except Exception as exc:
        log.warning("mps probe failed: %s", _describe_error(exc))
    preference = (preference or "auto").strip().lower()
    if preference == "cpu":
        return ["cpu"]
    if preference == "cuda":
        preference = "cuda:0"
    ordered = [preference] if preference in available else []
    if preference not in {"auto", *available}:
        log.warning("requested device unavailable, using automatic selection")
    ordered.extend(d for d in available if d not in ordered)
    ordered.append("cpu")
    return ordered


class YoloDetector:
    def __init__(self, settings: Settings) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelError("Detection backend not installed") from exc
        weights = resolve_weights(settings)
        self.settings = settings
        self._torch = torch
        if weights.is_file():
            verify_weights(weights, settings.model_sha256)
        try:
            self.model = YOLO(str(weights), task="detect")
        except Exception as exc:
            log.error("model load failed: %s", _describe_error(exc), exc_info=True)
            raise ModelError("Model could not be loaded") from exc
        verify_weights(weights, settings.model_sha256)
        names = getattr(self.model, "names", {}) or {}
        for cls_id, cls_name in TARGET_CLASSES.items():
            if str(names.get(cls_id, "")).lower() != cls_name:
                raise ModelError("Model class map is incompatible")
        self.classes = sorted(TARGET_CLASSES)
        self._accelerator_failures = 0
        probe = np.zeros((settings.height, settings.width, 3), dtype=np.uint8)
        last_error: BaseException | None = None
        for device in candidate_devices(settings.device, torch):
            self._select_device(device)
            try:
                self.detect(probe, settings.det_conf)
            except Exception as exc:
                last_error = exc
                log.warning("inference warm-up failed on %s: %s", device, _describe_error(exc), exc_info=True)
                continue
            log.info("inference device: %s", device)
            return
        raise ModelError("Inference failed on every available device") from last_error

    def _select_device(self, device: str) -> None:
        self.device = device
        self.precision: dict[str, Any] = {}
        if device.startswith("cuda"):
            try:
                from ultralytics.utils import DEFAULT_CFG_DICT
                supports_quantize = "quantize" in DEFAULT_CFG_DICT
            except ImportError:
                supports_quantize = False
            self.precision = {"quantize": 16} if supports_quantize else {"half": True}
        elif device == "cpu":
            try:
                self._torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
            except RuntimeError:
                pass

    def detect(self, frame: np.ndarray, conf: float) -> list[Detection]:
        try:
            results = self._predict(frame, conf)
        except Exception as exc:
            if self.device == "cpu":
                raise
            self._accelerator_failures += 1
            log.warning("inference failed on %s: %s", self.device, _describe_error(exc))
            if self._accelerator_failures < 3:
                raise
            log.warning("switching inference to cpu after repeated accelerator failures")
            self._select_device("cpu")
            results = self._predict(frame, conf)
        else:
            self._accelerator_failures = 0
        return self._parse(results)

    def _predict(self, frame: np.ndarray, conf: float) -> Any:
        return self.model.predict(
            frame,
            conf=float(conf),
            iou=self.settings.det_iou,
            classes=self.classes,
            imgsz=self.settings.imgsz,
            device=self.device,
            agnostic_nms=True,
            max_det=64,
            verbose=False,
            **self.precision,
        )

    @staticmethod
    def _parse(results: Any) -> list[Detection]:
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64)
        confs = boxes.conf.detach().cpu().numpy().astype(np.float64)
        classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
        return [Detection(*map(float, xyxy[i]), float(confs[i]), int(classes[i])) for i in range(xyxy.shape[0])]


class KalmanTrack:
    _H = np.hstack([np.eye(4), np.zeros((4, 4))])

    def __init__(self, track_id: int, det: Detection, t: float, frame_size: tuple[int, int] = (1280, 720)) -> None:
        self.id = track_id
        self.frame_w, self.frame_h = frame_size
        w, h = max(det.w, 1.0), max(det.h, 1.0)
        self.x = np.array([(det.x1 + det.x2) / 2, (det.y1 + det.y2) / 2, math.log(w), math.log(h), 0, 0, 0, 0], dtype=np.float64)
        self.P = np.diag([(0.1 * w) ** 2, (0.1 * h) ** 2, 0.1 ** 2, 0.1 ** 2, (1.0 * w) ** 2, (1.0 * h) ** 2, 0.6 ** 2, 0.6 ** 2])
        self.t = t
        self.born = t
        self.last_update = t
        self.hits = 1
        self.misses = 0
        self.cls_votes: Counter[int] = Counter({det.cls: det.conf})
        self.conf = det.conf
        self.threat = THREAT_LOW
        self.up_count = 0
        self.down_count = 0
        self.event_id: int | None = None
        self.event_onset = 0.0
        self.event_peak = THREAT_LOW

    @property
    def cls(self) -> int:
        return self.cls_votes.most_common(1)[0][0]

    def box(self) -> tuple[float, float, float, float]:
        cx, cy = self.x[0], self.x[1]
        w = math.exp(min(self.x[2], 12.0))
        h = math.exp(min(self.x[3], 12.0))
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def predict(self, t: float) -> None:
        dt = min(max(t - self.t, 1e-3), 1.0)
        F = np.eye(8)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = dt
        w = math.exp(min(self.x[2], 12.0))
        sigma = np.array([1.5 * w, 1.5 * w, 0.5, 0.5])
        g = np.array([dt * dt / 2.0, dt])
        Q = np.zeros((8, 8))
        for i in range(4):
            block = np.outer(g, g) * sigma[i] ** 2
            Q[i, i], Q[i, i + 4], Q[i + 4, i], Q[i + 4, i + 4] = block[0, 0], block[0, 1], block[1, 0], block[1, 1]
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def _measurement(self, det: Detection) -> tuple[np.ndarray, np.ndarray]:
        w, h = max(det.w, 1.0), max(det.h, 1.0)
        z = np.array([(det.x1 + det.x2) / 2, (det.y1 + det.y2) / 2, math.log(w), math.log(h)])
        scale = 1.0 / max(det.conf, 0.2)
        variances = np.array([(0.04 * w) ** 2, (0.04 * h) ** 2, 0.045 ** 2, 0.045 ** 2]) * scale
        if det.x1 <= 2 or det.x2 >= self.frame_w - 3:
            variances[0] *= 400.0
            variances[2] = 25.0
        if det.y1 <= 2 or det.y2 >= self.frame_h - 3:
            variances[1] *= 400.0
            variances[3] = 25.0
        return z, np.diag(variances)

    def mahalanobis(self, det: Detection) -> float:
        z, R = self._measurement(det)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + R
        try:
            return float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            return math.inf

    def update(self, det: Detection, t: float) -> None:
        z, R = self._measurement(det)
        H = self._H
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ y
        I_KH = np.eye(8) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2.0
        self.hits += 1
        self.misses = 0
        self.last_update = t
        self.cls_votes[det.cls] += det.conf
        self.conf = 0.7 * self.conf + 0.3 * det.conf

    def scale_rate(self) -> tuple[float, float]:
        return float(self.x[6]), float(math.sqrt(max(self.P[6, 6], 0.0)))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


class MultiObjectTracker:
    def __init__(self, min_hits: int = 3, max_coast_s: float = 0.6, on_drop: Callable[[KalmanTrack], None] | None = None,
                 frame_size: tuple[int, int] = (1280, 720), max_report_misses: int = 2) -> None:
        self.tracks: list[KalmanTrack] = []
        self.frame_size = frame_size
        self.max_report_misses = max_report_misses
        self.next_id = 1
        self.min_hits = min_hits
        self.max_coast_s = max_coast_s
        self.on_drop = on_drop

    def reset(self) -> None:
        if self.on_drop:
            for track in self.tracks:
                self.on_drop(track)
        self.tracks = []

    def update(self, detections: Sequence[Detection], t: float) -> list[KalmanTrack]:
        for track in self.tracks:
            track.predict(t)
        n_t, n_d = len(self.tracks), len(detections)
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        if n_t and n_d:
            tb = np.array([trk.box() for trk in self.tracks])
            db = np.array([(d.x1, d.y1, d.x2, d.y2) for d in detections])
            ious = iou_matrix(tb, db)
            cost = np.full((n_t, n_d), 1e6)
            for i, track in enumerate(self.tracks):
                group = CLASS_GROUP.get(track.cls, -1)
                for j, det in enumerate(detections):
                    if CLASS_GROUP.get(det.cls, -2) != group:
                        continue
                    d2 = track.mahalanobis(det)
                    if ious[i, j] < 0.05 and d2 > CHI2_4DOF_99:
                        continue
                    cost[i, j] = 0.7 * (1.0 - ious[i, j]) + 0.3 * min(d2 / CHI2_4DOF_99, 1.0)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols, strict=True):
                if cost[i, j] < 1e5:
                    self.tracks[i].update(detections[j], t)
                    matched_t.add(i)
                    matched_d.add(j)
        for i, track in enumerate(self.tracks):
            if i not in matched_t:
                track.misses += 1
        for j, det in enumerate(detections):
            if j not in matched_d:
                self.tracks.append(KalmanTrack(self.next_id, det, t, self.frame_size))
                self.next_id += 1
        survivors: list[KalmanTrack] = []
        for track in self.tracks:
            coasting = t - track.last_update
            tentative_dead = track.hits < self.min_hits and track.misses > 1
            if coasting > self.max_coast_s or tentative_dead or not np.all(np.isfinite(track.x)):
                if self.on_drop:
                    self.on_drop(track)
                continue
            survivors.append(track)
        self.tracks = survivors
        return [trk for trk in self.tracks if trk.hits >= self.min_hits and trk.misses <= self.max_report_misses]


@dataclass
class RiskParams:
    det_conf: float
    ttc_critical_s: float
    ttc_warning_s: float
    warning_distance_m: float
    raise_frames: int = 3
    clear_frames: int = 6
    show_lanes: bool = True
    show_boxes: bool = True


@dataclass
class TrackReport:
    id: int
    cls: int
    box: tuple[int, int, int, int]
    distance_m: float | None
    ttc_s: float | None
    closing_mps: float | None
    in_path: bool
    threat: int
    conf: float
    coasting: bool = False


@dataclass
class ThreatEvent:
    kind: str
    track_id: int
    cls: int
    threat: int
    t: float
    distance_m: float | None
    ttc_s: float | None
    event_id: int | None = None
    duration_s: float = 0.0
    track: KalmanTrack | None = None


class RiskAssessor:
    def __init__(self, settings: Settings) -> None:
        self.w = settings.width
        self.h = settings.height
        self.focal = settings.focal_px
        self.camera_height = settings.camera_height_m
        self.horizon_y = settings.horizon_ratio * self.h
        self.horizon_calibrated = False
        src_top = 0.65
        self.default_left = np.array([[0.45 * self.w, src_top * self.h], [0.05 * self.w, 0.95 * self.h]])
        self.default_right = np.array([[0.55 * self.w, src_top * self.h], [0.95 * self.w, 0.95 * self.h]])

    def set_horizon(self, y: float, calibrated: bool) -> None:
        if math.isfinite(y):
            self.horizon_y = float(min(max(y, 0.3 * self.h), 0.85 * self.h))
            self.horizon_calibrated = calibrated

    def _truncated(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        return x1 <= 2 or x2 >= self.w - 3 or y2 >= self.h - 3

    def plausible(self, det: Detection) -> bool:
        tolerance = 0.03 * self.h if self.horizon_calibrated else 0.10 * self.h
        if det.y2 < self.horizon_y - tolerance:
            return False
        dy = det.y2 - self.horizon_y
        if dy < 0.04 * self.h or self._truncated(det.x1, det.y1, det.x2, det.y2):
            return True
        physical_w = det.w * self.camera_height / dy
        mean_w, _ = CLASS_WIDTH_M[det.cls]
        ratio = physical_w / mean_w
        lo, hi = (0.35, 2.8) if self.horizon_calibrated else (0.2, 5.0)
        if det.cls == 0:
            lo, hi = lo * 0.7, hi * 1.4
        return lo <= ratio <= hi

    def distance(self, cls: int, box: tuple[float, float, float, float]) -> float | None:
        x1, y1, x2, y2 = box
        estimates: list[tuple[float, float]] = []
        dy = y2 - self.horizon_y
        if dy > 2.0:
            d_ground = self.focal * self.camera_height / dy
            sigma_y = 3.0 if self.horizon_calibrated else 3.0 + 0.02 * self.h
            sigma = d_ground * d_ground / (self.focal * self.camera_height) * sigma_y
            estimates.append((d_ground, max(sigma, 0.05)))
        if not self._truncated(x1, y1, x2, y2) and x2 - x1 > 2.0:
            mean_w, rel = CLASS_WIDTH_M.get(cls, (1.8, 0.3))
            d_width = self.focal * mean_w / (x2 - x1)
            estimates.append((d_width, max(rel * d_width, 0.05)))
        if not estimates:
            return None
        weights = np.array([1.0 / (s * s) for _, s in estimates])
        values = np.array([d for d, _ in estimates])
        fused = float((weights * values).sum() / weights.sum())
        return fused if math.isfinite(fused) and fused > 0 else None

    def _corridor(self, lane: LaneState | None, y: float) -> tuple[float, float]:
        if lane is not None:
            span = lane.corridor_at(y)
            if span is not None:
                return span
        top_y, bot_y = self.default_left[0, 1], self.default_left[1, 1]
        frac = float(np.clip((y - top_y) / (bot_y - top_y), 0.0, 1.0))
        xl = self.default_left[0, 0] + frac * (self.default_left[1, 0] - self.default_left[0, 0])
        xr = self.default_right[0, 0] + frac * (self.default_right[1, 0] - self.default_right[0, 0])
        return xl, xr

    def in_path(self, track: KalmanTrack, lane: LaneState | None, ttc: float | None) -> bool:
        x1, _, x2, y2 = track.box()
        xl, xr = self._corridor(lane, y2)
        margin = 0.08 * (xr - xl)
        xl, xr = xl - margin, xr + margin
        width = max(x2 - x1, 1.0)
        overlap = max(0.0, min(x2, xr) - max(x1, xl)) / min(width, xr - xl)
        if overlap >= 0.3:
            return True
        horizon = min(ttc if ttc is not None else 1.5, 2.0)
        future_cx = track.x[0] + track.x[4] * horizon
        future_y2 = y2 + track.x[5] * horizon
        fxl, fxr = self._corridor(lane, future_y2)
        return fxl <= future_cx <= fxr

    def assess(self, tracks: Sequence[KalmanTrack], lane: LaneState | None, t: float, params: RiskParams) -> tuple[list[TrackReport], list[ThreatEvent]]:
        reports: list[TrackReport] = []
        events: list[ThreatEvent] = []
        for track in tracks:
            cls = track.cls
            box = track.box()
            distance = self.distance(cls, box)
            rate, sigma = track.scale_rate()
            ttc = 1.0 / rate if rate > 1e-3 else None
            conservative_rate = rate - 1.0 * sigma
            ttc_cons = 1.0 / conservative_rate if conservative_rate > 1e-3 else None
            closing = distance * rate if distance is not None and rate > 0 else None
            on_path = self.in_path(track, lane, ttc_cons)
            candidate = THREAT_LOW
            if on_path:
                if ttc_cons is not None and ttc_cons < params.ttc_critical_s and (distance is None or distance < 80.0):
                    candidate = THREAT_CRITICAL
                elif (ttc_cons is not None and ttc_cons < params.ttc_warning_s) or (distance is not None and distance < params.warning_distance_m) or (cls in VULNERABLE_CLASSES and distance is not None and distance < 2.0 * params.warning_distance_m):
                    candidate = THREAT_WARNING
            previous = track.threat
            if track.misses > 0:
                if candidate < track.threat:
                    track.down_count += 1
                    if track.down_count >= params.clear_frames:
                        track.threat = candidate
                        track.down_count = 0
            elif candidate > track.threat:
                track.up_count += 1
                track.down_count = 0
                if track.up_count >= params.raise_frames:
                    track.threat = candidate
                    track.up_count = 0
            elif candidate < track.threat:
                track.down_count += 1
                track.up_count = 0
                if track.down_count >= params.clear_frames:
                    track.threat = candidate
                    track.down_count = 0
            else:
                track.up_count = 0
                track.down_count = 0
            if track.threat != previous:
                if previous == THREAT_LOW:
                    track.event_onset = t
                    track.event_peak = track.threat
                    events.append(ThreatEvent("start", track.id, cls, track.threat, t, distance, ttc, track=track))
                elif track.threat == THREAT_LOW:
                    events.append(ThreatEvent("end", track.id, cls, track.event_peak, t, distance, ttc, track.event_id, t - track.event_onset, track))
                else:
                    track.event_peak = max(track.event_peak, track.threat)
                    events.append(ThreatEvent("escalate", track.id, cls, track.threat, t, distance, ttc, track.event_id, track=track))
            ib = tuple(int(round(v)) for v in box)
            reports.append(TrackReport(track.id, cls, ib, distance, ttc, closing, on_path, track.threat, track.conf, track.misses > 0))
        return reports, events


class EpisodicMemory:
    SCHEMA = (
        "CREATE TABLE IF NOT EXISTS sources (key TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL, created REAL NOT NULL, last_seen REAL NOT NULL, sessions INTEGER NOT NULL DEFAULT 0, horizon_ratio REAL, horizon_samples INTEGER NOT NULL DEFAULT 0, lane_prior TEXT, flicker_rate REAL NOT NULL DEFAULT 0.0, alerts_total INTEGER NOT NULL DEFAULT 0)",
        "CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY AUTOINCREMENT, source_key TEXT NOT NULL REFERENCES sources(key) ON DELETE CASCADE, started REAL NOT NULL, ended REAL, frames INTEGER NOT NULL DEFAULT 0, warnings INTEGER NOT NULL DEFAULT 0, criticals INTEGER NOT NULL DEFAULT 0, mean_latency_ms REAL)",
        "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE, wall REAL NOT NULL, media_t REAL NOT NULL, track_id INTEGER NOT NULL, cls INTEGER NOT NULL, threat INTEGER NOT NULL, distance_m REAL, ttc_s REAL, duration_s REAL)",
        "CREATE INDEX IF NOT EXISTS idx_events_episode ON events(episode_id)",
        "CREATE INDEX IF NOT EXISTS idx_events_wall ON events(wall)",
        "CREATE INDEX IF NOT EXISTS idx_episodes_source ON episodes(source_key)",
    )

    def __init__(self, path: Path, retention_days: float = 30.0) -> None:
        self.path = path
        self.retention_s = retention_days * 86400.0
        self._lock = threading.Lock()
        _secure_mkdir(path.parent)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5.0, isolation_level=None)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            for statement in self.SCHEMA:
                self._conn.execute(statement)
            self._prune_locked()

    def _prune_locked(self) -> None:
        cutoff = time.time() - self.retention_s
        self._conn.execute("DELETE FROM episodes WHERE started < ?", (cutoff,))
        self._conn.execute("DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT 200000)")

    def begin_episode(self, spec: SourceSpec) -> tuple[int, dict[str, Any]]:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sources(key, kind, label, created, last_seen, sessions) VALUES(?,?,?,?,?,1) "
                "ON CONFLICT(key) DO UPDATE SET last_seen=excluded.last_seen, label=excluded.label, sessions=sessions+1",
                (spec.key, spec.kind, spec.label, now, now),
            )
            row = self._conn.execute(
                "SELECT horizon_ratio, horizon_samples, lane_prior, flicker_rate, sessions, alerts_total FROM sources WHERE key=?",
                (spec.key,),
            ).fetchone()
            cursor = self._conn.execute("INSERT INTO episodes(source_key, started) VALUES(?,?)", (spec.key, now))
            episode_id = int(cursor.lastrowid)
        prior: dict[str, Any] = {"horizon_ratio": None, "horizon_samples": 0, "lane_prior": None, "flicker_rate": 0.0, "sessions": 1, "alerts_total": 0}
        if row:
            lane_prior = None
            if row[2]:
                try:
                    lane_prior = json.loads(row[2])
                except (ValueError, TypeError):
                    lane_prior = None
            prior.update(horizon_ratio=row[0], horizon_samples=int(row[1] or 0), lane_prior=lane_prior, flicker_rate=float(row[3] or 0.0), sessions=int(row[4] or 1), alerts_total=int(row[5] or 0))
        return episode_id, prior

    def update_source(self, key: str, horizon_ratio: float | None, horizon_samples: int, lane_prior: dict[str, Any] | None, flicker_rate: float) -> None:
        lane_json = json.dumps(lane_prior) if lane_prior else None
        with self._lock:
            self._conn.execute(
                "UPDATE sources SET horizon_ratio=COALESCE(?, horizon_ratio), horizon_samples=?, lane_prior=COALESCE(?, lane_prior), flicker_rate=?, last_seen=? WHERE key=?",
                (horizon_ratio, int(horizon_samples), lane_json, float(flicker_rate), time.time(), key),
            )

    def end_episode(self, episode_id: int, frames: int, warnings_count: int, criticals: int, mean_latency_ms: float | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE episodes SET ended=?, frames=?, warnings=?, criticals=?, mean_latency_ms=? WHERE id=?",
                (time.time(), int(frames), int(warnings_count), int(criticals), mean_latency_ms, int(episode_id)),
            )

    def log_event(self, episode_id: int, event: ThreatEvent) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO events(episode_id, wall, media_t, track_id, cls, threat, distance_m, ttc_s) VALUES(?,?,?,?,?,?,?,?)",
                (int(episode_id), time.time(), float(event.t), int(event.track_id), int(event.cls), int(event.threat), event.distance_m, event.ttc_s),
            )
            self._conn.execute("UPDATE sources SET alerts_total=alerts_total+1 WHERE key=(SELECT source_key FROM episodes WHERE id=?)", (int(episode_id),))
            return int(cursor.lastrowid)

    def update_event(self, event_id: int, threat: int | None = None, duration_s: float | None = None) -> None:
        with self._lock:
            if threat is not None:
                self._conn.execute("UPDATE events SET threat=MAX(threat, ?) WHERE id=?", (int(threat), int(event_id)))
            if duration_s is not None:
                self._conn.execute("UPDATE events SET duration_s=? WHERE id=?", (float(duration_s), int(event_id)))

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = int(min(max(limit, 1), 500))
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.id, e.wall, e.track_id, e.cls, e.threat, e.distance_m, e.ttc_s, e.duration_s, s.label FROM events e "
                "JOIN episodes p ON p.id=e.episode_id JOIN sources s ON s.key=p.source_key ORDER BY e.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "time": time.strftime("%H:%M:%S", time.localtime(r[1])),
                "track": r[2],
                "class": TARGET_CLASSES.get(r[3], "object"),
                "threat": THREAT_NAMES.get(r[4], "LOW"),
                "distance": f"{r[5]:.1f} m" if r[5] is not None else "-",
                "ttc": f"{r[6]:.1f} s" if r[6] is not None else "-",
                "duration": f"{r[7]:.1f} s" if r[7] is not None else "active",
                "source": r[8],
            }
            for r in rows
        ]

    def stats(self) -> dict[str, int]:
        with self._lock:
            sources = self._conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            episodes = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            events = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {"sources": int(sources), "episodes": int(episodes), "events": int(events)}

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


@dataclass
class FramePacket:
    frame: np.ndarray
    t: float
    seq: int
    generation: int


class FrameSource:
    def __init__(self, spec: SourceSpec, settings: Settings, sink: Callable[[FramePacket], None], on_failure: Callable[[str], None]) -> None:
        self.spec = spec
        self.settings = settings
        self.sink = sink
        self.on_failure = on_failure
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self.opened = threading.Event()
        self.failed = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    def _open(self) -> cv2.VideoCapture | None:
        if self.spec.kind == "camera":
            capture = cv2.VideoCapture(int(self.spec.target))
        elif self.spec.kind == "stream":
            capture = cv2.VideoCapture(str(self.spec.target), cv2.CAP_FFMPEG)
        else:
            capture = cv2.VideoCapture(str(self.spec.target))
        if not capture.isOpened():
            capture.release()
            return None
        if self.spec.kind != "file":
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self) -> None:
        capture = None
        try:
            capture = self._open()
            if capture is None:
                self.failed.set()
                self.on_failure("Unable to open video source")
                return
            ok, first = capture.read()
            frame = normalize_frame(first, (self.settings.width, self.settings.height)) if ok else None
            if frame is None:
                self.failed.set()
                self.on_failure("Video source produced no readable frames")
                return
            self.opened.set()
            is_file = self.spec.kind == "file"
            fps = capture.get(cv2.CAP_PROP_FPS) if is_file else 0.0
            fps = fps if math.isfinite(fps) and 1.0 <= fps <= 240.0 else 30.0
            seq = 0
            generation = 0
            media_index = 0
            loop_base = 0.0
            wall_start = time.monotonic()
            failures = 0
            while not self._stop.is_set():
                if frame is not None:
                    if is_file:
                        media_t = loop_base + media_index / fps
                        delay = (wall_start + media_t - loop_base) - time.monotonic()
                        if delay > 0:
                            if self._stop.wait(min(delay, 0.5)):
                                break
                            if delay > 0.5:
                                continue
                        stamp = media_t
                        media_index += 1
                    else:
                        stamp = time.monotonic()
                    seq += 1
                    self.sink(FramePacket(frame, stamp, seq, generation))
                    frame = None
                ok, raw = capture.read()
                if not ok:
                    if is_file and failures == 0:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        generation += 1
                        loop_base = loop_base + media_index / fps + 1.0
                        media_index = 0
                        wall_start = time.monotonic()
                        failures += 1
                        continue
                    failures += 1
                    if failures > (3 if is_file else 60):
                        self.on_failure("Video source stopped delivering frames")
                        return
                    self._stop.wait(0.02)
                    continue
                frame = normalize_frame(raw, (self.settings.width, self.settings.height))
                if frame is None:
                    failures += 1
                    if failures > 60:
                        self.on_failure("Video source delivered invalid frames")
                        return
                    continue
                failures = 0
        except Exception as exc:
            log.warning("capture loop failure: %s", type(exc).__name__)
            self.on_failure("Video source failed")
        finally:
            if capture is not None:
                capture.release()


def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buffer.tobytes() if ok else b""


def _placeholder(settings: Settings, text: str) -> bytes:
    canvas = np.full((settings.height, settings.width, 3), 18, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (settings.width - 1, settings.height - 1), (60, 60, 60), 2)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    org = ((settings.width - size[0]) // 2, (settings.height + size[1]) // 2)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (170, 170, 170), 2, cv2.LINE_AA)
    return _encode_jpeg(canvas, settings.jpeg_quality)


class Renderer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def draw(self, frame: np.ndarray, lane: LaneState, reports: Sequence[TrackReport], params: RiskParams, critical: bool) -> np.ndarray:
        canvas = frame
        if params.show_lanes and lane.valid and lane.left_img is not None and lane.right_img is not None:
            poly = np.vstack([lane.left_img, lane.right_img[::-1]]).astype(np.int32)
            layer = canvas.copy()
            cv2.fillPoly(layer, [poly], (60, 200, 60))
            cv2.addWeighted(layer, 0.28, canvas, 0.72, 0, dst=canvas)
            cv2.polylines(canvas, [lane.left_img.astype(np.int32)], False, (255, 220, 0), 3, cv2.LINE_AA)
            cv2.polylines(canvas, [lane.right_img.astype(np.int32)], False, (255, 220, 0), 3, cv2.LINE_AA)
        if params.show_boxes:
            for rep in reports:
                color = THREAT_COLORS_BGR[rep.threat]
                x1, y1, x2, y2 = rep.box
                thickness = 1 if rep.coasting else (3 if rep.threat else 2)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                parts = [f"#{rep.id} {TARGET_CLASSES.get(rep.cls, '?')}"]
                if rep.distance_m is not None:
                    parts.append(f"{rep.distance_m:.1f}m")
                if rep.ttc_s is not None and rep.ttc_s < 20:
                    parts.append(f"TTC {rep.ttc_s:.1f}s")
                text = " ".join(parts)
                (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(y1 - 6, th + 4)
                cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 6, ty + base - 2), color, -1)
                cv2.putText(canvas, text, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
        if critical:
            h, w = canvas.shape[:2]
            cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), THREAT_COLORS_BGR[THREAT_CRITICAL], 12)
            text = "COLLISION WARNING"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)
            x = (w - tw) // 2
            cv2.rectangle(canvas, (x - 16, 24), (x + tw + 16, 40 + th + 16), (20, 20, 160), -1)
            cv2.putText(canvas, text, (x, 40 + th), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
        return canvas


class Engine:
    def __init__(self, settings: Settings, memory: EpisodicMemory | None = None, detector_factory: Callable[[Settings], DetectorProtocol] | None = None) -> None:
        self.settings = settings
        self.memory = memory
        self.detector_factory = detector_factory or YoloDetector
        self.detector: DetectorProtocol | None = None
        lane_w = int(settings.width * settings.lane_scale) // 2 * 2
        lane_h = int(settings.height * settings.lane_scale) // 2 * 2
        self.lane_size = (lane_w, lane_h)
        self.lanes = LaneAnalyzer(lane_w, lane_h, settings.width / lane_w, settings.lane_width_m, settings.lane_lookahead_m)
        self.risk = RiskAssessor(settings)
        self.tracker = MultiObjectTracker(on_drop=self._on_track_drop, frame_size=(settings.width, settings.height))
        self.renderer = Renderer(settings)
        self.params = RiskParams(settings.det_conf, settings.ttc_critical_s, settings.ttc_warning_s, settings.warning_distance_m)
        self._analysis_lock = threading.RLock()
        self._lock = threading.RLock()
        self._cond = threading.Condition()
        self._packet: FramePacket | None = None
        self._dropped = 0
        self._shutdown = threading.Event()
        self._worker: threading.Thread | None = None
        self._source: FrameSource | None = None
        self._spec: SourceSpec | None = None
        self._episode_id: int | None = None
        self._jpeg = _placeholder(settings, "No video source")
        self._jpeg_seq = 0
        self._status: dict[str, Any] = {"state": "idle", "message": "Select a video source", "model": "loading"}
        self._metrics: dict[str, Any] = self._empty_metrics()
        self._horizon_ratio = settings.horizon_ratio
        self._horizon_samples = 0
        self._flicker_rate = 0.0
        self._last_persist = 0.0
        self._episode_frames = 0
        self._episode_warnings = 0
        self._episode_criticals = 0
        self._latency_sum = 0.0
        self._consecutive_errors = 0
        self._source_generation = 0
        self._bind_lock = threading.Lock()

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        return {
            "fps": 0.0,
            "latency_ms": 0.0,
            "objects": 0,
            "threat": "LOW",
            "nearest_m": None,
            "min_ttc_s": None,
            "lane_confidence": 0.0,
            "radius_m": None,
            "offset_m": None,
            "horizon_ratio": None,
            "horizon_calibrated": False,
            "frames": 0,
        }

    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._process_loop, name="processing", daemon=True)
        self._worker.start()
        threading.Thread(target=self._load_model, name="model-loader", daemon=True).start()

    def _load_model(self) -> None:
        try:
            detector = self.detector_factory(self.settings)
        except NexusError as exc:
            cause = exc.__cause__
            log.error("detector unavailable: %s%s", exc.public_message, f" ({_describe_error(cause)})" if cause else "", exc_info=True)
            with self._lock:
                self._status["model"] = "unavailable"
                self._status["model_message"] = exc.public_message
            return
        except Exception as exc:
            log.error("detector initialisation failed: %s", _describe_error(exc), exc_info=True)
            with self._lock:
                self._status["model"] = "unavailable"
                self._status["model_message"] = ModelError.public_message
            return
        with self._lock:
            self.detector = detector
            self._status["model"] = "ready"
            self._status.pop("model_message", None)
        log.info("detector ready")

    def update_params(self, **changes: Any) -> None:
        allowed = {"det_conf": (0.05, 0.95), "ttc_critical_s": (0.5, 10.0), "ttc_warning_s": (1.0, 15.0), "warning_distance_m": (2.0, 60.0)}
        with self._lock:
            for name, value in changes.items():
                if name in allowed:
                    if isinstance(value, bool):
                        continue
                    try:
                        number = float(value)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if math.isfinite(number):
                        lo, hi = allowed[name]
                        setattr(self.params, name, min(max(number, lo), hi))
                elif name in {"show_lanes", "show_boxes"}:
                    setattr(self.params, name, bool(value))
            if self.params.ttc_warning_s < self.params.ttc_critical_s:
                self.params.ttc_warning_s = self.params.ttc_critical_s

    def bind(self, raw: Any) -> tuple[bool, str]:
        try:
            spec = resolve_source(raw, self.settings)
        except SourceError as exc:
            return False, exc.public_message
        except Exception as exc:
            log.warning("source resolution failure: %s", type(exc).__name__)
            return False, SourceError.public_message
        with self._bind_lock:
            return self._bind_spec(spec)

    def _bind_spec(self, spec: SourceSpec) -> tuple[bool, str]:
        self.stop_source()
        prior: dict[str, Any] = {}
        episode_id = None
        if self.memory is not None:
            try:
                episode_id, prior = self.memory.begin_episode(spec)
            except sqlite3.Error as exc:
                log.warning("memory unavailable: %s", type(exc).__name__)
        with self._analysis_lock:
            with self._lock:
                self._source_generation += 1
                generation = self._source_generation
                self._spec = spec
                self._episode_id = episode_id
                self._reset_analysis(prior)
                self._status.update(state="connecting", message=f"Opening {spec.label}")

        def sink(packet: FramePacket) -> None:
            with self._cond:
                if generation != self._source_generation:
                    return
                if self._packet is not None:
                    self._dropped += 1
                self._packet = packet
                self._cond.notify()

        def failure(message: str) -> None:
            with self._lock:
                if generation == self._source_generation:
                    self._status.update(state="error", message=message)
                    self._jpeg = _placeholder(self.settings, message)
                    self._jpeg_seq += 1
            log.warning("source failure: %s", message)

        source = FrameSource(spec, self.settings, sink, failure)
        with self._lock:
            self._source = source
        source.start()
        deadline = time.monotonic() + (10.0 if spec.kind == "stream" else 5.0)
        while time.monotonic() < deadline and not source.opened.is_set() and not source.failed.is_set():
            time.sleep(0.02)
        if source.opened.is_set():
            with self._lock:
                if generation == self._source_generation and self._status.get("state") == "connecting":
                    self._status.update(state="running", message=f"Streaming {spec.label}")
            log.info("source bound kind=%s key=%s", spec.kind, spec.key[:20])
            return True, f"Streaming {spec.label}"
        source.stop()
        with self._analysis_lock:
            with self._lock:
                if self._source is source:
                    self._source = None
                self._source_generation += 1
                self._finish_episode()
                self._spec = None
                self._status.update(state="error", message="Unable to open video source")
                self._jpeg = _placeholder(self.settings, "Unable to open video source")
                self._jpeg_seq += 1
        return False, "Unable to open video source"

    def _reset_analysis(self, prior: dict[str, Any]) -> None:
        self.tracker.reset()
        self.lanes.seed(prior.get("lane_prior"))
        horizon = prior.get("horizon_ratio")
        samples = int(prior.get("horizon_samples") or 0)
        if isinstance(horizon, (int, float)) and math.isfinite(horizon) and 0.3 <= horizon <= 0.85:
            self._horizon_ratio = float(horizon)
            self._horizon_samples = samples
        else:
            self._horizon_ratio = self.settings.horizon_ratio
            self._horizon_samples = 0
        self.risk.set_horizon(self._horizon_ratio * self.settings.height, self._horizon_samples >= 30)
        flicker = prior.get("flicker_rate")
        self._flicker_rate = float(flicker) if isinstance(flicker, (int, float)) and 0.0 <= flicker <= 1.0 else 0.0
        self.params.raise_frames = int(min(max(3 + round(4 * self._flicker_rate), 3), 6))
        self._metrics = self._empty_metrics()
        self._episode_frames = 0
        self._episode_warnings = 0
        self._episode_criticals = 0
        self._latency_sum = 0.0
        self._consecutive_errors = 0
        self._last_persist = time.monotonic()
        with self._cond:
            self._packet = None
            self._dropped = 0

    def stop_source(self) -> None:
        with self._lock:
            source = self._source
            self._source = None
            self._source_generation += 1
        if source is not None:
            source.stop()
        with self._analysis_lock:
            with self._lock:
                self.tracker.reset()
                self._finish_episode()
                self._spec = None
                self._status.update(state="idle", message="Select a video source")
                self._jpeg = _placeholder(self.settings, "No video source")
                self._jpeg_seq += 1
        with self._cond:
            self._packet = None

    def _finish_episode(self) -> None:
        if self.memory is None or self._episode_id is None:
            self._episode_id = None
            return
        try:
            self._persist_source(force=True)
            mean_latency = self._latency_sum / self._episode_frames if self._episode_frames else None
            self.memory.end_episode(self._episode_id, self._episode_frames, self._episode_warnings, self._episode_criticals, mean_latency)
        except sqlite3.Error as exc:
            log.warning("episode close failed: %s", type(exc).__name__)
        self._episode_id = None

    def _persist_source(self, force: bool = False) -> None:
        if self.memory is None or self._spec is None:
            return
        now = time.monotonic()
        if not force and now - self._last_persist < 10.0:
            return
        self._last_persist = now
        horizon = self._horizon_ratio if self._horizon_samples > 0 else None
        self.memory.update_source(self._spec.key, horizon, self._horizon_samples, self.lanes.export(), self._flicker_rate)

    def _on_track_drop(self, track: KalmanTrack) -> None:
        if track.threat != THREAT_LOW and track.event_id is not None:
            self._close_event(track.event_id, track.t - track.event_onset)
        track.threat = THREAT_LOW
        track.event_id = None

    def _close_event(self, event_id: int, duration: float) -> None:
        duration = max(0.0, float(duration)) if math.isfinite(duration) else 0.0
        self._flicker_rate = 0.95 * self._flicker_rate + 0.05 * (1.0 if duration < 0.4 else 0.0)
        with self._lock:
            self.params.raise_frames = int(min(max(3 + round(4 * self._flicker_rate), 3), 6))
        if self.memory is not None:
            try:
                self.memory.update_event(event_id, duration_s=duration)
            except sqlite3.Error as exc:
                log.warning("event update failed: %s", type(exc).__name__)

    def _handle_events(self, events: Sequence[ThreatEvent]) -> None:
        for event in events:
            track = event.track
            if event.kind == "start":
                if event.threat == THREAT_CRITICAL:
                    self._episode_criticals += 1
                else:
                    self._episode_warnings += 1
                if self.memory is not None and self._episode_id is not None and track is not None:
                    try:
                        track.event_id = self.memory.log_event(self._episode_id, event)
                    except sqlite3.Error as exc:
                        log.warning("event log failed: %s", type(exc).__name__)
            elif event.kind == "escalate":
                if event.threat == THREAT_CRITICAL:
                    self._episode_criticals += 1
                if self.memory is not None and event.event_id is not None:
                    try:
                        self.memory.update_event(event.event_id, threat=event.threat)
                    except sqlite3.Error as exc:
                        log.warning("event update failed: %s", type(exc).__name__)
            elif event.kind == "end":
                if event.event_id is not None:
                    self._close_event(event.event_id, event.duration_s)
                if track is not None:
                    track.event_id = None

    def _update_horizon(self, lane: LaneState) -> None:
        if lane.vanishing_y is None:
            return
        ratio = lane.vanishing_y / self.settings.height
        if not 0.3 <= ratio <= 0.85:
            return
        if self._horizon_samples >= 10 and abs(ratio - self._horizon_ratio) > 0.08:
            return
        n = min(self._horizon_samples + 1, 300)
        self._horizon_ratio += (ratio - self._horizon_ratio) / n
        self._horizon_samples += 1
        self.risk.set_horizon(self._horizon_ratio * self.settings.height, self._horizon_samples >= 30)

    def process_frame(self, frame: np.ndarray, t: float) -> tuple[np.ndarray, dict[str, Any]]:
        with self._lock:
            params = replace(self.params)
            detector = self.detector
        small = cv2.resize(frame, self.lane_size, interpolation=cv2.INTER_AREA)
        lane = self.lanes.analyze(small)
        self._update_horizon(lane)
        detections: list[Detection] = []
        if detector is not None:
            detections = sanitize_detections(detector.detect(frame, params.det_conf), self.settings.width, self.settings.height)
            detections = [d for d in detections if self.risk.plausible(d)]
        tracks = self.tracker.update(detections, t)
        reports, events = self.risk.assess(tracks, lane, t, params)
        self._handle_events(events)
        critical = any(r.threat == THREAT_CRITICAL for r in reports)
        annotated = self.renderer.draw(frame.copy(), lane, reports, params, critical)
        in_path = [r for r in reports if r.in_path]
        nearest = min((r.distance_m for r in in_path if r.distance_m is not None), default=None)
        min_ttc = min((r.ttc_s for r in in_path if r.ttc_s is not None), default=None)
        level = max((r.threat for r in reports), default=THREAT_LOW)
        metrics = {
            "objects": len(reports),
            "threat": THREAT_NAMES[level],
            "nearest_m": nearest,
            "min_ttc_s": min_ttc,
            "lane_confidence": lane.confidence,
            "radius_m": lane.radius_m,
            "offset_m": lane.offset_m if lane.valid else None,
            "horizon_ratio": self._horizon_ratio,
            "horizon_calibrated": self.risk.horizon_calibrated,
            "reports": reports,
        }
        return annotated, metrics

    def _process_loop(self) -> None:
        last_generation = -1
        fps_ema = 0.0
        last_done = time.perf_counter()
        while not self._shutdown.is_set():
            with self._cond:
                if self._packet is None:
                    self._cond.wait(0.25)
                packet = self._packet
                self._packet = None
            if packet is None:
                continue
            started = time.perf_counter()
            try:
                with self._analysis_lock:
                    with self._lock:
                        source_generation = self._source_generation
                        active = self._source is not None
                    if not active:
                        continue
                    if packet.generation != last_generation:
                        if last_generation != -1 and packet.generation > last_generation:
                            self.tracker.reset()
                        last_generation = packet.generation
                    annotated, metrics = self.process_frame(packet.frame, packet.t)
                jpeg = _encode_jpeg(annotated, self.settings.jpeg_quality)
            except Exception as exc:
                self._consecutive_errors += 1
                log.warning("frame processing failure: %s", _describe_error(exc), exc_info=self._consecutive_errors == 1)
                if self._consecutive_errors >= 30:
                    with self._lock:
                        self._status.update(state="error", message="Processing halted after repeated failures")
                    self._consecutive_errors = 0
                continue
            self._consecutive_errors = 0
            finished = time.perf_counter()
            latency = (finished - started) * 1000.0
            interval = finished - last_done
            last_done = finished
            if interval > 0:
                fps_ema = (1.0 / interval) if fps_ema == 0 else 0.9 * fps_ema + 0.1 / interval
            metrics.pop("reports", None)
            with self._analysis_lock:
                with self._lock:
                    if source_generation != self._source_generation:
                        continue
                    self._jpeg = jpeg
                    self._jpeg_seq += 1
                    self._episode_frames += 1
                    self._latency_sum += latency
                    self._metrics.update(metrics)
                    self._metrics.update(fps=fps_ema, latency_ms=latency, frames=self._episode_frames)
                    if self._status.get("state") == "connecting":
                        self._status.update(state="running")
                try:
                    self._persist_source()
                except sqlite3.Error as exc:
                    log.warning("memory persist failed: %s", type(exc).__name__)

    def process_packet_sync(self, frame: np.ndarray, t: float) -> tuple[np.ndarray, dict[str, Any]]:
        with self._analysis_lock:
            return self.process_frame(frame, t)

    def jpeg(self) -> tuple[int, bytes]:
        with self._lock:
            return self._jpeg_seq, self._jpeg

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self._metrics)
            data.update(self._status)
            data["source"] = self._spec.label if self._spec else None
            data["dropped"] = self._dropped
            data["params"] = {
                "det_conf": self.params.det_conf,
                "ttc_critical_s": self.params.ttc_critical_s,
                "ttc_warning_s": self.params.ttc_warning_s,
                "warning_distance_m": self.params.warning_distance_m,
                "raise_frames": self.params.raise_frames,
                "show_lanes": self.params.show_lanes,
                "show_boxes": self.params.show_boxes,
            }
            return data

    def close(self) -> None:
        self.stop_source()
        self._shutdown.set()
        with self._cond:
            self._cond.notify_all()
        if self._worker is not None:
            self._worker.join(2.0)
        if self.memory is not None:
            self.memory.close()


def _fmt(value: Any, unit: str, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}{unit}"


def prune_uploads(directory: Path, keep: int = 5, max_age_s: float = 7 * 86400.0) -> int:
    removed = 0
    try:
        files = sorted((p for p in directory.iterdir() if p.is_file() and not p.is_symlink()), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return 0
    now = time.time()
    for index, path in enumerate(files):
        try:
            if index >= keep or now - path.stat().st_mtime > max_age_s:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"} or host.startswith("127.")


def build_app(engine: Engine, settings: Settings) -> None:
    from nicegui import app, run, ui

    boundary = "nexusframe"

    @app.get("/video/stream")
    async def video_stream(request: Request) -> StreamingResponse:
        async def generator():
            last = -1
            interval = 1.0 / settings.stream_fps
            while True:
                if await request.is_disconnected():
                    break
                seq, payload = engine.jpeg()
                if seq != last and payload:
                    last = seq
                    yield (f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(payload)}\r\n\r\n").encode() + payload + b"\r\n"
                await asyncio.sleep(interval)

        headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
        return StreamingResponse(generator(), media_type=f"multipart/x-mixed-replace; boundary={boundary}", headers=headers)

    @app.get("/video/frame")
    async def video_frame() -> Response:
        _, payload = engine.jpeg()
        return Response(content=payload, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        data = engine.snapshot()
        return JSONResponse({k: v for k, v in data.items() if isinstance(v, (str, int, float, bool, dict, type(None)))})

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled request failure: %s", type(exc).__name__)
        return JSONResponse({"error": "Internal error"}, status_code=500)

    upload_dir = settings.data_dir / "uploads"
    _secure_mkdir(upload_dir)
    prune_uploads(upload_dir)

    @ui.page("/")
    def index() -> None:
        ui.dark_mode(True)
        ui.add_head_html(
            "<style>body{background:#0b0f14}.nx-card{background:#121821;border:1px solid #1f2a37;border-radius:14px}"
            ".nx-metric{font-variant-numeric:tabular-nums}.nx-feed{width:100%;aspect-ratio:16/9;object-fit:contain;background:#05070a;border-radius:12px;border:1px solid #1f2a37}</style>"
        )
        with ui.header().classes("items-center justify-between bg-[#0f141b] border-b border-[#1f2a37] px-6 py-2"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("radar", size="28px").classes("text-cyan-400")
                ui.label(APP_NAME).classes("text-xl font-semibold tracking-wide")
                ui.label("Spatial driver assistance").classes("text-sm text-gray-400")
            with ui.row().classes("items-center gap-2"):
                model_badge = ui.badge("model: loading", color="grey-8").classes("px-3 py-1")
                state_badge = ui.badge("idle", color="grey-8").classes("px-3 py-1")

        with ui.row().classes("w-full no-wrap gap-4 p-4 items-start"):
            with ui.column().classes("grow gap-3").style("min-width:0"):
                ui.element("img").props('src="/video/stream" alt="Processed video feed"').classes("nx-feed")
                with ui.row().classes("w-full gap-3"):
                    tiles: dict[str, ui.label] = {}
                    for key, title in (("threat", "Threat"), ("nearest", "Nearest in path"), ("ttc", "Min TTC"), ("fps", "Throughput"), ("latency", "Latency"), ("objects", "Tracked")):
                        with ui.card().classes("nx-card grow p-3 gap-0").style("min-width:130px"):
                            ui.label(title).classes("text-xs uppercase text-gray-400")
                            tiles[key] = ui.label("-").classes("nx-metric text-2xl font-semibold")
                with ui.card().classes("nx-card w-full p-3"):
                    ui.label("Alert history").classes("text-sm font-semibold text-gray-300")
                    columns = [
                        {"name": "time", "label": "Time", "field": "time", "align": "left"},
                        {"name": "source", "label": "Source", "field": "source", "align": "left"},
                        {"name": "class", "label": "Class", "field": "class", "align": "left"},
                        {"name": "threat", "label": "Level", "field": "threat", "align": "left"},
                        {"name": "distance", "label": "Distance", "field": "distance"},
                        {"name": "ttc", "label": "TTC", "field": "ttc"},
                        {"name": "duration", "label": "Duration", "field": "duration"},
                    ]
                    table = ui.table(columns=columns, rows=[], row_key="id", pagination=8).classes("w-full").props("dense flat dark")

            with ui.column().classes("gap-3").style("width:360px;flex-shrink:0"):
                with ui.card().classes("nx-card w-full p-4 gap-3"):
                    ui.label("Video source").classes("text-sm font-semibold text-gray-300")
                    source_input = ui.input(placeholder="File path or camera index").props("dense outlined dark clearable").classes("w-full").mark("source-input")

                    async def bind(value: Any) -> None:
                        ok, message = await run.io_bound(engine.bind, value)
                        ui.notify(message, type="positive" if ok else "negative")

                    async def bind_from_input() -> None:
                        await bind(source_input.value or "")

                    source_input.on("keydown.enter", bind_from_input)
                    with ui.row().classes("w-full gap-2"):
                        ui.button("Open", icon="play_arrow", on_click=bind_from_input).props("unelevated color=cyan-8").classes("grow")
                        ui.button("Webcam", icon="videocam", on_click=lambda: bind(0)).props("outline color=cyan-5").classes("grow")

                        async def stop() -> None:
                            await run.io_bound(engine.stop_source)
                            ui.notify("Source stopped")

                        ui.button(icon="stop", on_click=stop).props("outline color=red-5").tooltip("Stop").mark("stop-source")

                    async def handle_upload(event: Any) -> None:
                        name = str(getattr(event.file, "name", "") or "")
                        suffix = Path(name).suffix.lower()
                        if suffix not in VIDEO_EXTENSIONS:
                            ui.notify("Unsupported file type", type="negative")
                            return
                        target = upload_dir / f"{uuid.uuid4().hex}{suffix}"
                        try:
                            await event.file.save(target)
                            os.chmod(target, 0o600)
                            await run.io_bound(prune_uploads, upload_dir)
                        except OSError as exc:
                            log.warning("upload save failed: %s", type(exc).__name__)
                            ui.notify("Upload failed", type="negative")
                            return
                        await bind(str(target))

                    ui.upload(on_upload=handle_upload, auto_upload=True, max_file_size=settings.max_upload_mb * (1 << 20), label="Upload video",
                              on_rejected=lambda: ui.notify("File rejected", type="negative")).props(f'accept="{",".join(sorted(VIDEO_EXTENSIONS))}" dark flat bordered').classes("w-full")
                    source_label = ui.label("No source").classes("text-xs text-gray-400")

                with ui.card().classes("nx-card w-full p-4 gap-1"):
                    ui.label("Lane & calibration").classes("text-sm font-semibold text-gray-300")
                    lane_rows: dict[str, ui.label] = {}
                    for key, title in (("lane", "Lane confidence"), ("radius", "Curve radius"), ("offset", "Lateral offset"), ("horizon", "Horizon"), ("persist", "Alert persistence")):
                        with ui.row().classes("w-full justify-between"):
                            ui.label(title).classes("text-sm text-gray-400")
                            lane_rows[key] = ui.label("-").classes("nx-metric text-sm")

                with ui.card().classes("nx-card w-full p-4 gap-2"):
                    ui.label("Sensitivity").classes("text-sm font-semibold text-gray-300")
                    p = engine.snapshot()["params"]

                    def slider(title: str, key: str, lo: float, hi: float, step: float, unit: str) -> None:
                        caption = ui.label(f"{title}: {p[key]:.2f}{unit}").classes("text-xs text-gray-400")

                        def changed(event: Any) -> None:
                            engine.update_params(**{key: event.value})
                            caption.set_text(f"{title}: {float(event.value):.2f}{unit}")

                        ui.slider(min=lo, max=hi, step=step, value=p[key], on_change=changed).props("dark color=cyan-6 dense")

                    slider("Detection confidence", "det_conf", 0.1, 0.9, 0.05, "")
                    slider("Critical TTC", "ttc_critical_s", 0.5, 6.0, 0.1, " s")
                    slider("Warning TTC", "ttc_warning_s", 1.0, 10.0, 0.1, " s")
                    slider("Warning distance", "warning_distance_m", 2.0, 40.0, 0.5, " m")
                    with ui.row().classes("w-full justify-between"):
                        ui.switch("Lanes", value=p["show_lanes"], on_change=lambda e: engine.update_params(show_lanes=e.value)).props("dark color=cyan-6")
                        ui.switch("Boxes", value=p["show_boxes"], on_change=lambda e: engine.update_params(show_boxes=e.value)).props("dark color=cyan-6")

                ui.label("Advisory tool only. Not a certified safety system.").classes("text-xs text-gray-500 px-1")

        threat_colors = {"LOW": "text-green-400", "WARNING": "text-amber-400", "CRITICAL": "text-red-500"}
        state_colors = {"running": "positive", "connecting": "info", "error": "negative", "idle": "grey-8"}

        def refresh() -> None:
            data = engine.snapshot()
            level = data.get("threat", "LOW") if data.get("state") == "running" else "LOW"
            tiles["threat"].set_text(level)
            tiles["threat"].classes(replace=f"nx-metric text-2xl font-semibold {threat_colors.get(level, '')}")
            tiles["nearest"].set_text(_fmt(data.get("nearest_m"), " m"))
            tiles["ttc"].set_text(_fmt(data.get("min_ttc_s"), " s"))
            tiles["fps"].set_text(_fmt(data.get("fps"), " fps"))
            tiles["latency"].set_text(_fmt(data.get("latency_ms"), " ms", 0))
            tiles["objects"].set_text(str(data.get("objects", 0)))
            lane_rows["lane"].set_text(f"{100 * float(data.get('lane_confidence') or 0):.0f}%")
            radius = data.get("radius_m")
            if float(data.get("lane_confidence") or 0.0) < 0.35:
                lane_rows["radius"].set_text("-")
            elif radius is None or radius > 3000:
                lane_rows["radius"].set_text("straight")
            else:
                lane_rows["radius"].set_text(f"{radius:.0f} m")
            lane_rows["offset"].set_text(_fmt(data.get("offset_m"), " m", 2))
            horizon = data.get("horizon_ratio")
            calibrated = " (calibrated)" if data.get("horizon_calibrated") else " (prior)"
            lane_rows["horizon"].set_text("-" if horizon is None else f"{100 * horizon:.1f}%{calibrated}")
            lane_rows["persist"].set_text(f"{data['params']['raise_frames']} frames")
            state = str(data.get("state", "idle"))
            state_badge.set_text(state)
            state_badge.props(f"color={state_colors.get(state, 'grey-8')}")
            model = str(data.get("model", "loading"))
            model_badge.set_text(f"model: {model}")
            model_badge.props(f"color={'positive' if model == 'ready' else ('negative' if model == 'unavailable' else 'grey-8')}")
            source_label.set_text(str(data.get("message") or ""))

        def refresh_events() -> None:
            if engine.memory is None:
                return
            try:
                table.rows = engine.memory.recent_events(50)
                table.update()
            except sqlite3.Error as exc:
                log.warning("event query failed: %s", type(exc).__name__)

        ui.timer(0.25, refresh)
        ui.timer(2.0, refresh_events)


def diagnose(settings: Settings) -> int:
    redactor = RedactingFilter(False)
    lines = [f"{APP_NAME} {APP_VERSION}"]
    report = preflight(settings)
    lines.append("modules: " + ", ".join(f"{k}={'ok' if v else 'missing'}" for k, v in report["modules"].items()))
    lines.append(f"data dir writable: {report['data_dir_writable']}  opencv ffmpeg: {report['opencv_ffmpeg']}")
    status = 0
    try:
        import torch
        import ultralytics
        lines.append(f"torch {torch.__version__} (cuda build {torch.version.cuda})  ultralytics {ultralytics.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                lines.append(f"cuda:{i} {torch.cuda.get_device_name(i)} sm_{major}{minor}")
        lines.append(f"torch supported archs: {', '.join(torch.cuda.get_arch_list()) or 'none'}")
        devices = candidate_devices(settings.device, torch)
    except Exception as exc:
        lines.append(f"backend probe failed: {_describe_error(exc)}")
        devices = []
        status = 1
    try:
        weights = resolve_weights(settings)
        from ultralytics import YOLO
        model = YOLO(str(weights), task="detect")
        probe = np.zeros((settings.height, settings.width, 3), dtype=np.uint8)
        for device in devices:
            try:
                started = time.perf_counter()
                model.predict(probe, device=device, imgsz=settings.imgsz, verbose=False)
                lines.append(f"inference {device}: ok ({(time.perf_counter() - started) * 1000:.0f} ms)")
            except Exception as exc:
                lines.append(f"inference {device}: FAILED {_describe_error(exc)}")
    except Exception as exc:
        lines.append(f"model load failed: {_describe_error(exc)}")
        status = 1
    print("\n".join(redactor.redact(line) for line in lines))
    return status


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nexusdrive", description=f"{APP_NAME} spatial driver assistance")
    parser.add_argument("--host", default=os.environ.get("NEXUSDRIVE_HOSTNAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("NEXUSDRIVE_PORT", 8080, 1, 65535))
    parser.add_argument("--source", default=None)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    settings = Settings.from_env()
    setup_logging(settings)
    if args.diagnose:
        raise SystemExit(diagnose(settings))
    if not _is_loopback(args.host):
        log.warning("server bound to a non-loopback interface; video and local files become reachable from the network")
    report = preflight(settings)
    if not report["ok"]:
        log.error("preflight failed")
        raise SystemExit("Startup checks failed. See log for details.")
    for module, found in report["modules"].items():
        if not found:
            log.warning("optional module missing: %s", module)
    memory = None
    try:
        memory = EpisodicMemory(settings.data_dir / "memory.sqlite3")
    except (sqlite3.Error, OSError) as exc:
        log.warning("episodic memory disabled: %s", type(exc).__name__)
    engine = Engine(settings, memory)
    from nicegui import app, ui

    app.on_startup(engine.start)
    app.on_shutdown(engine.close)
    if args.source is not None:
        def autostart() -> None:
            threading.Thread(target=engine.bind, args=(args.source,), name="autostart", daemon=True).start()
        app.on_startup(autostart)
    build_app(engine, settings)
    ui.run(
        host=args.host,
        port=args.port,
        title=APP_NAME,
        dark=True,
        reload=False,
        show=not args.no_browser and not args.native,
        native=args.native,
        favicon="🛰",
        show_welcome_message=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
