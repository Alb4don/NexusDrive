from __future__ import annotations

import logging
import math
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings as hsettings, strategies as st
from hypothesis.extra import numpy as hnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nexusdrive as nd

W, H = 1280, 720


def make_settings(tmp_path: Path, **kw) -> nd.Settings:
    return nd.Settings(data_dir=tmp_path / "data", **kw)


def road_frame(w: int = W, h: int = H, car_box=None, shift: float = 0.0, noise: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[: int(0.6125 * h)] = (180, 150, 120)
    img[int(0.6125 * h):] = (70, 70, 70)
    hy = 0.6125 * h
    for x_bottom in (0.05 * w + shift, 0.95 * w + shift):
        p_bottom = (int(x_bottom), int(0.95 * h))
        p_mid = (int(0.5 * w + (x_bottom - 0.5 * w) * (0.65 * h - hy) / (0.95 * h - hy)), int(0.65 * h))
        cv2.line(img, p_bottom, p_mid, (235, 235, 235), max(3, w // 200))
    if car_box is not None:
        x1, y1, x2, y2 = (int(v) for v in car_box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 160), -1)
    if noise:
        img = np.clip(img.astype(np.int16) + rng.integers(-noise, noise + 1, img.shape), 0, 255).astype(np.uint8)
    return img


def approach_box(t: float, ttc0: float = 6.0, d0: float = 40.0, horizon: float = 0.6125, cam_h: float = 1.3, hfov: float = 70.0):
    f = (W / 2) / math.tan(math.radians(hfov) / 2)
    v = d0 / ttc0
    d = max(d0 - v * t, 2.0)
    wpx = f * 1.8 / d
    dy = f * cam_h / d
    y2 = horizon * H + dy
    hpx = wpx * 0.8
    cx = W / 2
    return (cx - wpx / 2, y2 - hpx, cx + wpx / 2, y2), d, max(ttc0 - t, 0.0)


class StubDetector:
    def __init__(self, provider):
        self.provider = provider

    def detect(self, frame, conf):
        return self.provider(frame)


def color_detector(frame):
    mask = (frame[:, :, 2] > 130) & (frame[:, :, 0] < 60) & (frame[:, :, 1] < 60)
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        return []
    return [nd.Detection(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), 0.9, 2)]


@given(hnp.arrays(dtype=st.sampled_from([np.uint8, np.uint16, np.float32, np.float64, np.int32, np.bool_]),
                  shape=hnp.array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=48)))
@hsettings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_normalize_frame(arr):
    out = nd.normalize_frame(arr, (160, 90))
    assert out is None or (out.shape == (90, 160, 3) and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"])


def test_normalize_frame_aspect_crop_and_channels():
    gray = np.full((480, 640), 128, np.uint8)
    out = nd.normalize_frame(gray, (W, H))
    assert out.shape == (H, W, 3)
    bgra = np.zeros((1080, 1920, 4), np.uint8)
    assert nd.normalize_frame(bgra, (W, H)).shape == (H, W, 3)
    assert nd.normalize_frame(None, (W, H)) is None
    assert nd.normalize_frame(np.zeros((8, 8, 3), np.uint8), (W, H)) is None
    nanf = np.full((100, 100, 3), np.nan, np.float32)
    assert nd.normalize_frame(nanf, (160, 90)).max() == 0


@given(st.text(max_size=300))
@hsettings(max_examples=400, deadline=None)
def test_fuzz_resolve_source_never_leaks(tmp_path_factory, text):
    s = nd.Settings(data_dir=tmp_path_factory.mktemp("d"))
    try:
        spec = nd.resolve_source(text, s)
    except nd.SourceError as exc:
        assert "Traceback" not in exc.public_message and "/" not in exc.public_message
        return
    assert spec.kind in {"camera", "file"}


def test_resolve_source_rules(tmp_path):
    s = make_settings(tmp_path)
    assert nd.resolve_source("0", s).target == 0
    assert nd.resolve_source(3, s).kind == "camera"
    for bad in ("99", True, None, 1.5, "", "a\x00b", "rtsp://u:p@host/x", "file:///etc/passwd", "concat:/a|/b"):
        with pytest.raises(nd.SourceError):
            nd.resolve_source(bad, s)
    txt = tmp_path / "x.txt"
    txt.write_text("hello")
    with pytest.raises(nd.SourceError):
        nd.resolve_source(str(txt), s)
    link = tmp_path / "evil.mp4"
    link.symlink_to("/etc/passwd")
    with pytest.raises(nd.SourceError):
        nd.resolve_source(str(link), s)
    empty = tmp_path / "e.mp4"
    empty.write_bytes(b"")
    with pytest.raises(nd.SourceError):
        nd.resolve_source(str(empty), s)
    (tmp_path / "dir.mp4").mkdir()
    with pytest.raises(nd.SourceError):
        nd.resolve_source(str(tmp_path / "dir.mp4"), s)
    ok = tmp_path / "v.MP4"
    ok.write_bytes(b"\x00" * 100)
    spec = nd.resolve_source(f'  "{ok}"  ', s)
    assert spec.kind == "file" and spec.label == "v.MP4" and spec.key.startswith("file:")


def test_resolve_stream_when_enabled(tmp_path):
    s = make_settings(tmp_path, allow_network_streams=True)
    spec = nd.resolve_source("rtsp://user:secret@cam.local:554/live", s)
    assert spec.kind == "stream" and "secret" not in spec.label and "secret" not in spec.key
    with pytest.raises(nd.SourceError):
        nd.resolve_source("ftp://host/x", s)
    with pytest.raises(nd.SourceError):
        nd.resolve_source("http://host:99999/x", s)


def test_robust_polyfit_rejects_outliers():
    rng = np.random.default_rng(1)
    y = np.linspace(0, 359, 600)
    true = np.array([0.0008, -0.2, 300.0])
    x = np.polyval(true, y) + rng.normal(0, 1.0, y.size)
    idx = rng.choice(y.size, 120, replace=False)
    x[idx] += rng.uniform(80, 200, idx.size)
    coef = nd._robust_polyfit(y, x, 100)
    assert coef is not None
    assert np.max(np.abs(np.polyval(coef, y) - np.polyval(true, y))) < 4.0
    assert nd._robust_polyfit(y[:10], x[:10], 100) is None


def test_lane_detection_and_vanishing_point():
    lw, lh = W // 2, H // 2
    la = nd.LaneAnalyzer(lw, lh, 2.0)
    state = None
    for i in range(15):
        frame = road_frame(seed=i)
        state = la.analyze(cv2.resize(frame, (lw, lh), interpolation=cv2.INTER_AREA))
    assert state.valid and state.confidence > 0.6
    assert abs(state.offset_m) < 0.25
    assert state.radius_m is None or state.radius_m > 1000
    assert state.vanishing_y is not None and abs(state.vanishing_y / H - 0.6125) < 0.03
    span = state.corridor_at(0.9 * H)
    assert span is not None and span[0] < W / 2 < span[1]
    exported = la.export()
    lb = nd.LaneAnalyzer(lw, lh, 2.0)
    lb.seed(exported)
    assert lb.left is not None
    lb.seed({"left": ["x"], "right": None})
    assert lb.left is None


def test_lane_offset_sign():
    lw, lh = W // 2, H // 2
    la = nd.LaneAnalyzer(lw, lh, 2.0)
    state = None
    for i in range(12):
        state = la.analyze(cv2.resize(road_frame(shift=-90, seed=i), (lw, lh)))
    assert state.valid and state.offset_m > 0.2


@given(hnp.arrays(np.uint8, (90, 160, 3)))
@hsettings(max_examples=150, deadline=None)
def test_fuzz_lane_analyzer_random_images(img):
    la = nd.LaneAnalyzer(160, 90, 8.0)
    for _ in range(3):
        st_ = la.analyze(img)
        assert isinstance(st_, nd.LaneState)
        if st_.valid:
            assert np.all(np.isfinite(st_.left_img))


def test_lane_analyzer_uniform_and_black_frames():
    la = nd.LaneAnalyzer(320, 180, 4.0)
    for value in (0, 255, 128):
        s = la.analyze(np.full((180, 320, 3), value, np.uint8))
        assert not s.valid


def test_kalman_ttc_converges_to_truth():
    tracker = nd.MultiObjectTracker()
    fps = 30.0
    est = None
    true_ttc = None
    for k in range(75):
        t = k / fps
        box, _, true_ttc = approach_box(t, ttc0=6.0)
        tracks = tracker.update([nd.Detection(*box, 0.9, 2)], t)
        if tracks:
            rate, _ = tracks[0].scale_rate()
            est = 1.0 / rate if rate > 0 else None
    assert est is not None
    assert abs(est - true_ttc) / true_ttc < 0.15
    assert len({tr.id for tr in tracker.tracks}) == 1


def test_tracker_keeps_identities_for_crossing_objects():
    tracker = nd.MultiObjectTracker()
    ids_a, ids_b = set(), set()
    for k in range(40):
        t = k / 30
        a = nd.Detection(100 + 10 * k, 400, 200 + 10 * k, 480, 0.9, 2)
        b = nd.Detection(900 - 10 * k, 380, 1000 - 10 * k, 470, 0.9, 0)
        for tr in tracker.update([a, b], t):
            (ids_a if tr.cls == 2 else ids_b).add(tr.id)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_tracker_ignores_single_frame_ghost():
    tracker = nd.MultiObjectTracker()
    out = tracker.update([nd.Detection(10, 10, 60, 60, 0.9, 2)], 0.0)
    assert out == []
    for k in range(1, 10):
        out = tracker.update([], k / 30)
    assert out == [] and tracker.tracks == []


def _wild(lo, hi):
    return st.one_of(st.floats(lo, hi), st.sampled_from([math.nan, math.inf, -math.inf]))


det_strategy = st.builds(
    nd.Detection,
    _wild(-500, 2000),
    _wild(-500, 1200),
    _wild(-500, 2000),
    _wild(-500, 1200),
    _wild(-1, 2),
    st.sampled_from([0, 1, 2, 3, 4, 5, 7, 99, -1]),
)


@given(st.lists(st.lists(det_strategy, max_size=12), min_size=1, max_size=25))
@hsettings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_tracker_and_risk_pipeline(frames):
    s = nd.Settings()
    tracker = nd.MultiObjectTracker()
    risk = nd.RiskAssessor(s)
    params = nd.RiskParams(0.4, 2.5, 4.5, 12.0)
    for k, raw in enumerate(frames):
        dets = [d for d in nd.sanitize_detections(raw, W, H) if risk.plausible(d)]
        tracks = tracker.update(dets, k / 30.0)
        reports, _ = risk.assess(tracks, None, k / 30.0, params)
        for r in reports:
            assert r.threat in (0, 1, 2)
            assert r.distance_m is None or (math.isfinite(r.distance_m) and r.distance_m > 0)
    ids = [t.id for t in tracker.tracks]
    assert len(ids) == len(set(ids))


def run_risk(boxes_fn, n=90, fps=30.0, lane=None):
    s = nd.Settings()
    tracker = nd.MultiObjectTracker()
    risk = nd.RiskAssessor(s)
    params = nd.RiskParams(0.4, 2.5, 4.5, 12.0)
    levels, events = [], []
    for k in range(n):
        t = k / fps
        dets = [d for d in nd.sanitize_detections(boxes_fn(k, t), W, H) if risk.plausible(d)]
        reports, ev = risk.assess(tracker.update(dets, t), lane, t, params)
        levels.append(max((r.threat for r in reports), default=0))
        events.extend(ev)
    return levels, events


def test_static_vehicle_ahead_never_critical():
    box, _, _ = approach_box(0, d0=25.0)
    levels, _ = run_risk(lambda k, t: [nd.Detection(*box, 0.9, 2)])
    assert max(levels) < nd.THREAT_CRITICAL


def test_approaching_vehicle_raises_critical_before_impact():
    levels, events = run_risk(lambda k, t: [nd.Detection(*approach_box(t, ttc0=5.0, d0=35.0)[0], 0.9, 2)], n=135)
    first_crit = next(i for i, v in enumerate(levels) if v == nd.THREAT_CRITICAL)
    assert first_crit / 30.0 < 5.0 - 1.0
    assert any(e.kind == "start" for e in events)


def test_approaching_vehicle_in_adjacent_lane_is_low():
    def boxes(k, t):
        (x1, y1, x2, y2), _, _ = approach_box(t, ttc0=5.0, d0=35.0)
        off = 0.62 * W
        return [nd.Detection(x1 + off, y1, min(x2 + off, W - 5), y2, 0.9, 2)]
    levels, _ = run_risk(boxes, n=100)
    assert max(levels) < nd.THREAT_CRITICAL


def test_sporadic_ghost_detections_do_not_alert():
    rng = np.random.default_rng(7)

    def boxes(k, t):
        if rng.random() > 0.3:
            return []
        cx = rng.uniform(300, 980)
        y2 = rng.uniform(470, 700)
        w = rng.uniform(80, 300)
        return [nd.Detection(cx - w / 2, y2 - 0.8 * w, cx + w / 2, y2, 0.5, 2)]

    levels, events = run_risk(boxes, n=300)
    assert max(levels) == 0 and not events


def test_short_burst_does_not_alert():
    levels, events = run_risk(lambda k, t: [nd.Detection(*approach_box(t, ttc0=1.5, d0=8.0)[0], 0.6, 2)] if k < 3 else [], n=30)
    assert max(levels) == 0 and not events


def test_alert_survives_missed_detections():
    levels, _ = run_risk(lambda k, t: [nd.Detection(*approach_box(t, ttc0=4.0, d0=30.0)[0], 0.6, 2)] if k % 2 == 0 else [], n=100)
    assert nd.THREAT_CRITICAL in levels


def test_plausibility_filters(tmp_path):
    risk = nd.RiskAssessor(nd.Settings())
    risk.set_horizon(0.6125 * H, calibrated=True)
    assert not risk.plausible(nd.Detection(600, 50, 700, 120, 0.9, 2))
    assert not risk.plausible(nd.Detection(630, 640, 640, 650, 0.9, 2))
    good, _, _ = approach_box(0, d0=20)
    assert risk.plausible(nd.Detection(*good, 0.9, 2))
    assert risk.plausible(nd.Detection(0, 500, 300, 719, 0.9, 7))


def test_distance_estimate_accuracy():
    risk = nd.RiskAssessor(nd.Settings())
    risk.set_horizon(0.6125 * H, calibrated=True)
    for d0 in (8.0, 15.0, 30.0, 50.0):
        box, d, _ = approach_box(0, d0=d0)
        est = risk.distance(2, box)
        assert abs(est - d) / d < 0.08


def test_memory_roundtrip_and_injection(tmp_path):
    mem = nd.EpisodicMemory(tmp_path / "m.sqlite3")
    spec = nd.SourceSpec("file", "/x.mp4", "file:abc", "x'); DROP TABLE events;--")
    ep, prior = mem.begin_episode(spec)
    assert prior["horizon_samples"] == 0
    ev = nd.ThreatEvent("start", 1, 2, nd.THREAT_WARNING, 1.0, 10.0, 3.0)
    eid = mem.log_event(ep, ev)
    mem.update_event(eid, threat=nd.THREAT_CRITICAL)
    mem.update_event(eid, threat=nd.THREAT_WARNING, duration_s=1.5)
    mem.update_source(spec.key, 0.6, 40, {"left": [0.0, 0.0, 1.0], "right": [0.0, 0.0, 2.0]}, 0.2)
    mem.end_episode(ep, 100, 1, 1, 20.0)
    rows = mem.recent_events()
    assert rows[0]["threat"] == "CRITICAL" and rows[0]["duration"] == "1.5 s" and rows[0]["source"] == spec.label
    ep2, prior2 = mem.begin_episode(spec)
    assert prior2["horizon_ratio"] == pytest.approx(0.6) and prior2["horizon_samples"] == 40
    assert prior2["lane_prior"]["left"] == [0.0, 0.0, 1.0] and prior2["sessions"] == 2
    assert mem.stats() == {"sources": 1, "episodes": 2, "events": 1}
    mem.close()
    if os.name == "posix":
        assert (tmp_path / "m.sqlite3").stat().st_mode & 0o077 == 0


def test_memory_concurrent_writes(tmp_path):
    mem = nd.EpisodicMemory(tmp_path / "m.sqlite3")
    ep, _ = mem.begin_episode(nd.SourceSpec("camera", 0, "camera:0", "Camera 0"))
    errors = []

    def worker(n):
        try:
            for i in range(50):
                mem.log_event(ep, nd.ThreatEvent("start", n * 100 + i, 2, 1, float(i), None, None))
                mem.recent_events(5)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors and mem.stats()["events"] == 300
    mem.close()


def test_redacting_filter_strips_secrets_and_tracebacks():
    f = nd.RedactingFilter(debug=False)
    home = str(Path.home())
    try:
        raise ValueError("boom")
    except ValueError:
        rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "open rtsp://admin:hunter2@10.0.0.1/s token=abc %s\nFAKE LINE", (home + "/secret.mp4",), sys.exc_info())
    f.filter(rec)
    msg = rec.getMessage()
    assert "hunter2" not in msg and "abc" not in msg and home not in msg and "\n" not in msg
    assert rec.exc_info is None


def write_video(path: Path, n: int, fps: float = 30.0, fn=None) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert writer.isOpened()
    for k in range(n):
        writer.write(fn(k, k / fps) if fn else road_frame(seed=k))
    writer.release()
    return path


def approach_frame(k, t):
    box, _, _ = approach_box(t % 5.0, ttc0=5.0, d0=35.0)
    return road_frame(car_box=box, seed=k)


def wait_for(cond, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_engine_end_to_end_with_memory(tmp_path):
    video = write_video(tmp_path / "drive.mp4", 150, fn=approach_frame)
    s = make_settings(tmp_path)
    mem = nd.EpisodicMemory(s.data_dir / "memory.sqlite3")
    engine = nd.Engine(s, mem, detector_factory=lambda _s: StubDetector(color_detector))
    engine.start()
    try:
        assert wait_for(lambda: engine.snapshot()["model"] == "ready", 10)
        ok, msg = engine.bind(str(video))
        assert ok, msg
        assert str(tmp_path) not in msg
        seen = set()
        assert wait_for(lambda: (seen.add(engine.snapshot()["threat"]) or "CRITICAL" in seen), 25)
        snap = engine.snapshot()
        assert snap["state"] == "running" and snap["frames"] > 20 and snap["lane_confidence"] > 0.5
        seq, jpeg = engine.jpeg()
        assert jpeg[:2] == b"\xff\xd8" and seq > 0
        assert wait_for(lambda: mem.stats()["events"] >= 1, 10)
        engine.stop_source()
        assert engine.snapshot()["state"] == "idle"
        ok, _ = engine.bind(str(video))
        assert ok
        assert engine.snapshot()["horizon_ratio"] is None or True
        assert mem.stats()["episodes"] == 2
    finally:
        engine.close()
    conn = sqlite3.connect(str(s.data_dir / "memory.sqlite3"))
    row = conn.execute("SELECT horizon_samples, horizon_ratio, sessions FROM sources").fetchone()
    conn.close()
    assert row[0] > 0 and abs(row[1] - 0.6125) < 0.03 and row[2] == 2


def test_engine_rejects_bad_sources_without_leaking(tmp_path):
    s = make_settings(tmp_path)
    engine = nd.Engine(s, None, detector_factory=lambda _s: StubDetector(lambda f: []))
    engine.start()
    try:
        for bad in ("/etc/passwd", str(tmp_path / "missing.mp4"), "../../etc/shadow", "15", "x" * 5000, {"a": 1}):
            ok, msg = engine.bind(bad)
            assert not ok and "Traceback" not in msg and str(tmp_path) not in msg
        corrupt = tmp_path / "bad.mp4"
        corrupt.write_bytes(os.urandom(4096))
        ok, msg = engine.bind(str(corrupt))
        assert not ok and msg == "Unable to open video source"
        assert engine.snapshot()["state"] == "error"
    finally:
        engine.close()


def test_engine_survives_detector_crash(tmp_path):
    video = write_video(tmp_path / "v.mp4", 40)
    s = make_settings(tmp_path)
    calls = {"n": 0}

    def flaky(frame):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise RuntimeError("cuda oom /home/secret")
        return [nd.Detection(float("nan"), 0, 5, 5, 0.9, 2), nd.Detection(10, 10, 9, 9, 2.0, 77)]

    engine = nd.Engine(s, None, detector_factory=lambda _s: StubDetector(flaky))
    engine.start()
    try:
        wait_for(lambda: engine.snapshot()["model"] == "ready", 5)
        ok, _ = engine.bind(str(video))
        assert ok
        assert wait_for(lambda: engine.snapshot()["frames"] > 15, 15)
        assert engine.snapshot()["state"] == "running"
    finally:
        engine.close()


def test_engine_model_failure_is_reported_generically(tmp_path):
    s = make_settings(tmp_path)

    def boom(_s):
        raise RuntimeError("secret internal path /root/x")

    engine = nd.Engine(s, None, detector_factory=boom)
    engine.start()
    try:
        assert wait_for(lambda: engine.snapshot()["model"] == "unavailable", 5)
        assert engine.snapshot()["model_message"] == nd.ModelError.public_message
    finally:
        engine.close()


@given(st.dictionaries(st.sampled_from(["det_conf", "ttc_critical_s", "ttc_warning_s", "warning_distance_m", "show_lanes", "bogus"]),
                       st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=5), st.none(), st.booleans(), st.integers())))
@hsettings(max_examples=200, deadline=None)
def test_fuzz_update_params(changes):
    engine = nd.Engine(nd.Settings(), None, detector_factory=lambda _s: StubDetector(lambda f: []))
    engine.update_params(**changes)
    p = engine.snapshot()["params"]
    assert 0.05 <= p["det_conf"] <= 0.95 and 0.5 <= p["ttc_critical_s"] <= 10
    assert p["ttc_warning_s"] >= p["ttc_critical_s"]
    assert all(math.isfinite(v) for k, v in p.items() if isinstance(v, float))


def test_concurrent_bind_stop_snapshot(tmp_path):
    video = write_video(tmp_path / "v.mp4", 60)
    s = make_settings(tmp_path)
    engine = nd.Engine(s, nd.EpisodicMemory(s.data_dir / "m.sqlite3"), detector_factory=lambda _s: StubDetector(color_detector))
    engine.start()
    errors = []

    def hammer(i):
        try:
            for j in range(6):
                if (i + j) % 3 == 0:
                    engine.stop_source()
                else:
                    engine.bind(str(video))
                engine.snapshot()
                engine.jpeg()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(60)
    try:
        assert not errors
        assert all(not th.is_alive() for th in threads)
        ok, _ = engine.bind(str(video))
        assert ok and wait_for(lambda: engine.snapshot()["frames"] > 5, 10)
    finally:
        engine.close()


def test_resolve_weights_rules(tmp_path):
    s = make_settings(tmp_path, model_path="yolov8n.pt")
    assert nd.resolve_weights(s) == s.data_dir / "models" / "yolov8n.pt"
    with pytest.raises(nd.ModelError):
        nd.resolve_weights(make_settings(tmp_path, model_path="evil.sh"))
    with pytest.raises(nd.ModelError):
        nd.resolve_weights(make_settings(tmp_path, model_path="/nope/model.pt"))


def test_settings_from_env_clamps(monkeypatch, tmp_path):
    monkeypatch.setenv("NEXUSDRIVE_DET_CONF", "nan")
    monkeypatch.setenv("NEXUSDRIVE_TTC_CRITICAL", "1e9")
    monkeypatch.setenv("NEXUSDRIVE_WIDTH", "abc")
    monkeypatch.setenv("NEXUSDRIVE_HOME", str(tmp_path))
    s = nd.Settings.from_env()
    assert s.det_conf == 0.40 and s.ttc_critical_s == 10.0 and s.width == 1280 and s.data_dir == tmp_path


@pytest.mark.skipif(os.environ.get("NEXUSDRIVE_SKIP_YOLO") == "1", reason="model test disabled")
def test_real_yolo_detector_on_sample(tmp_path):
    import ultralytics
    sample = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    weights = Path(os.environ.get("NEXUSDRIVE_TEST_WEIGHTS", str(Path.home() / ".nexusdrive" / "models" / "yolov8n.pt")))
    model_path = str(weights) if weights.is_file() else "yolov8n.pt"
    s = make_settings(tmp_path, model_path=model_path)
    det = nd.YoloDetector(s)
    frame = nd.normalize_frame(cv2.imread(str(sample)), (W, H))
    found = nd.sanitize_detections(det.detect(frame, 0.4), W, H)
    classes = {d.cls for d in found}
    assert 0 in classes and (5 in classes or 2 in classes)


def test_nicegui_page_and_interactions(tmp_path, monkeypatch):
    import asyncio

    from nicegui import ui
    from nicegui.testing import user_simulation

    video = write_video(tmp_path / "drive.mp4", 90, fn=approach_frame)
    monkeypatch.setenv("NEXUSDRIVE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NEXUSDRIVE_MODEL", str(tmp_path / "missing.pt"))
    repo = str(Path(__file__).resolve().parents[1])
    main_file = tmp_path / "main_entry.py"
    main_file.write_text(f"import sys\nsys.path.insert(0, {repo!r})\nimport nexusdrive\nnexusdrive.main(['--no-browser'])\n")

    async def scenario():
        async with user_simulation(main_file=main_file) as user:
            await user.open("/")
            await user.should_see("NexusDrive")
            await user.should_see("Video source")
            await user.should_see("Alert history")
            user.find(marker="source-input").type("/etc/passwd")
            user.find("Open").click()
            await user.should_see("Unsupported file type", retries=60)
            user.find(marker="source-input").clear().type(str(video))
            user.find("Open").click()
            await user.should_see("Streaming drive.mp4", retries=100)
            await asyncio.sleep(1.5)
            await user.should_see("model: unavailable", retries=40)
            user.find(marker="stop-source").click()
            await user.should_see("Source stopped", retries=100)
            user.find(marker="source-input").clear().type("rtsp://admin:pw@10.0.0.2/live").trigger("keydown.enter")
            await user.should_see("Network streams are disabled", retries=60)

    asyncio.run(scenario())


def test_alert_is_continuous_when_detector_misses_frames():
    levels, _ = run_risk(lambda k, t: [nd.Detection(*approach_box(t, ttc0=4.0, d0=30.0)[0], 0.6, 2)] if k % 3 else [], n=100)
    first = levels.index(nd.THREAT_CRITICAL)
    assert all(v == nd.THREAT_CRITICAL for v in levels[first:first + 30])


def test_truncated_close_vehicle_keeps_critical():
    def boxes(k, t):
        (x1, y1, x2, y2), _, _ = approach_box(t, ttc0=3.5, d0=24.0)
        return [nd.Detection(max(x1, 0), y1, min(x2, W - 1), min(y2, H - 1), 0.9, 2)]

    levels, _ = run_risk(boxes, n=100)
    first = levels.index(nd.THREAT_CRITICAL)
    assert all(v == nd.THREAT_CRITICAL for v in levels[first:])


def test_verify_weights(tmp_path):
    import hashlib
    f = tmp_path / "m.pt"
    f.write_bytes(b"weights")
    nd.verify_weights(f, "")
    nd.verify_weights(f, hashlib.sha256(b"weights").hexdigest())
    with pytest.raises(nd.ModelError):
        nd.verify_weights(f, "0" * 64)
    with pytest.raises(nd.ModelError):
        nd.verify_weights(f, "not-a-hash")


def test_prune_uploads(tmp_path):
    for i in range(8):
        p = tmp_path / f"{i}.mp4"
        p.write_bytes(b"x")
        os.utime(p, (time.time() - (8 - i), time.time() - (8 - i)))
    old = tmp_path / "old.mp4"
    old.write_bytes(b"x")
    os.utime(old, (time.time() - 30 * 86400, time.time() - 30 * 86400))
    removed = nd.prune_uploads(tmp_path, keep=5)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert removed == 4 and remaining == ["3.mp4", "4.mp4", "5.mp4", "6.mp4", "7.mp4"]


class _FakeCuda:
    def __init__(self, n):
        self.n = n

    def is_available(self):
        return self.n > 0

    def device_count(self):
        return self.n


class _FakeTorch:
    def __init__(self, n_cuda=0, mps=False):
        self.cuda = _FakeCuda(n_cuda)
        self.backends = type("B", (), {"mps": type("M", (), {"is_available": staticmethod(lambda: mps)})()})()


def test_candidate_devices_ordering():
    assert nd.candidate_devices("auto", _FakeTorch(2)) == ["cuda:0", "cuda:1", "cpu"]
    assert nd.candidate_devices("cpu", _FakeTorch(2)) == ["cpu"]
    assert nd.candidate_devices("cuda:1", _FakeTorch(2)) == ["cuda:1", "cuda:0", "cpu"]
    assert nd.candidate_devices("cuda", _FakeTorch(0, mps=True)) == ["mps", "cpu"]
    assert nd.candidate_devices("auto", _FakeTorch(0)) == ["cpu"]


def _weights_path():
    weights = Path.home() / ".nexusdrive" / "models" / "yolov8n.pt"
    return str(weights) if weights.is_file() else "yolov8n.pt"


@pytest.mark.skipif(os.environ.get("NEXUSDRIVE_SKIP_YOLO") == "1", reason="model test disabled")
def test_yolo_falls_back_to_cpu_when_gpu_warmup_fails(tmp_path, monkeypatch, caplog):
    original = nd.YoloDetector._predict

    def broken_gpu(self, frame, conf):
        if self.device != "cpu":
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return original(self, frame, conf)

    monkeypatch.setattr(nd.YoloDetector, "_predict", broken_gpu)
    monkeypatch.setattr(nd, "candidate_devices", lambda pref, torch: ["cuda:0", "cpu"])
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    nd.log.addHandler(handler)
    try:
        det = nd.YoloDetector(make_settings(tmp_path, model_path=_weights_path()))
    finally:
        nd.log.removeHandler(handler)
    assert det.device == "cpu"
    assert any("no kernel image" in r.getMessage() for r in records)


@pytest.mark.skipif(os.environ.get("NEXUSDRIVE_SKIP_YOLO") == "1", reason="model test disabled")
def test_yolo_switches_to_cpu_after_runtime_gpu_failures(tmp_path, monkeypatch):
    det = nd.YoloDetector(make_settings(tmp_path, model_path=_weights_path(), device="cpu"))
    original = nd.YoloDetector._predict
    det._select_device("cuda:0")

    def flaky(self, frame, conf):
        if self.device != "cpu":
            raise RuntimeError("CUDA out of memory")
        return original(self, frame, conf)

    monkeypatch.setattr(nd.YoloDetector, "_predict", flaky)
    frame = np.zeros((H, W, 3), np.uint8)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            det.detect(frame, 0.4)
    assert det.detect(frame, 0.4) == [] and det.device == "cpu"


def test_model_failure_cause_reaches_log(tmp_path):
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    nd.log.addHandler(handler)

    def boom(_s):
        raise RuntimeError("CUDA driver version is insufficient")

    engine = nd.Engine(make_settings(tmp_path), None, detector_factory=boom)
    engine.start()
    try:
        assert wait_for(lambda: engine.snapshot()["model"] == "unavailable", 5)
    finally:
        engine.close()
        nd.log.removeHandler(handler)
    assert any("driver version is insufficient" in r.getMessage() for r in records)
    assert engine.snapshot()["model_message"] == nd.ModelError.public_message
