from __future__ import annotations

import csv
import math
import os
import tempfile
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.optimize import curve_fit


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_RADIUS = 5
MAX_RADIUS = 60
RELEASE_DX = 15
RELEASE_CONSECUTIVE_FRAMES = 3
GRAVITY_CM_S2 = 981.0
BASEBALL_DIAMETER_CM = 7.4
SOFTBALL_DIAMETER_CM = 9.7
SAVE_CSV = os.getenv("SAVE_CSV", "0") == "1"

DISTANCE_PRESETS = {
    "硬式・軟式野球": 18.44,
    "中学野球": 16.00,
    "小学生野球": 14.00,
    "ソフトボール（女子）": 13.11,
    "キャッチボール近": 10.00,
    "キャッチボール中": 15.00,
    "キャッチボール遠": 20.00,
}

app = FastAPI(title="Pitch Analyzer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def detect_ball(frame: np.ndarray, fgmask: np.ndarray) -> tuple[int, int, float] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 180]),
        np.array([180, 70, 255]),
    )
    combined = cv2.bitwise_and(fgmask, color_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[int, int, float] | None = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < math.pi * MIN_RADIUS**2:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if not MIN_RADIUS <= radius <= MAX_RADIUS:
            continue
        score = area / (math.pi * radius**2)
        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), float(radius))

    return best


def find_release_frame(trajectory: list[list[float]]) -> int | None:
    streak = 0

    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1]
        cur = trajectory[i]
        frame_gap = int(cur[0] - prev[0])
        dx = abs(cur[2] - prev[2])

        if frame_gap == 1 and dx >= RELEASE_DX:
            streak += 1
            if streak >= RELEASE_CONSECUTIVE_FRAMES:
                return i - RELEASE_CONSECUTIVE_FRAMES + 1
        else:
            streak = 0

    return None


def calc_speed_kmh(start_frame: int, end_frame: int, fps: float, distance_m: float) -> float:
    elapsed_frames = max(0, end_frame - start_frame)
    if elapsed_frames <= 0 or fps <= 0:
        return 0.0
    return (distance_m / (elapsed_frames / fps)) * 3.6


def infer_ball_diameter_cm(distance_m: float) -> float:
    if abs(distance_m - DISTANCE_PRESETS["ソフトボール（女子）"]) < 0.01:
        return SOFTBALL_DIAMETER_CM
    return BASEBALL_DIAMETER_CM


def estimate_cm_per_px(trajectory_slice: list[list[float]], ball_diameter_cm: float) -> float | None:
    radii = [float(row[4]) for row in trajectory_slice if len(row) > 4 and float(row[4]) > 0]
    if not radii:
        return None
    median_radius = float(np.median(radii))
    if median_radius <= 0:
        return None
    return ball_diameter_cm / (median_radius * 2)


def calc_break(
    trajectory_slice: list[list[float]],
    fps: float,
    cm_per_px: float | None,
    ball_diameter_cm: float,
) -> dict[str, float | str | None]:
    if len(trajectory_slice) < 4:
        return {}

    ts = np.array([row[1] for row in trajectory_slice], dtype=float) - float(trajectory_slice[0][1])
    xs = np.array([row[2] for row in trajectory_slice], dtype=float)
    ys = np.array([row[3] for row in trajectory_slice], dtype=float)

    try:
        x_fit = np.polyfit(ts, xs, 1)
        horiz_err = np.abs(xs - np.polyval(x_fit, ts))
    except Exception:
        horiz_err = np.zeros_like(xs)

    effective_cm_per_px = cm_per_px if cm_per_px and cm_per_px > 0 else estimate_cm_per_px(
        trajectory_slice,
        ball_diameter_cm,
    )
    scale_source = "manual" if cm_per_px and cm_per_px > 0 else "ball_estimate"

    if effective_cm_per_px and effective_cm_per_px > 0:
        g_px = GRAVITY_CM_S2 / effective_cm_per_px
    else:
        g_px = None
        scale_source = None

    if g_px is not None:
        def gravity_fixed(t: np.ndarray, y0: float, vy0: float) -> np.ndarray:
            return y0 + vy0 * t + 0.5 * g_px * t**2

        try:
            popt, _ = curve_fit(gravity_fixed, ts, ys, p0=[float(ys[0]), 0.0], maxfev=5000)
            vert_err = np.abs(ys - gravity_fixed(ts, *popt))
        except Exception:
            vert_err = np.zeros_like(ys)
    else:
        try:
            y_fit = np.polyfit(ts, ys, 2)
            vert_err = np.abs(ys - np.polyval(y_fit, ts))
        except Exception:
            vert_err = np.zeros_like(ys)

    max_v_px = float(np.max(vert_err))
    max_h_px = float(np.max(horiz_err))

    return {
        "max_vert_px": max_v_px,
        "max_horiz_px": max_h_px,
        "max_vert_cm": max_v_px * effective_cm_per_px if effective_cm_per_px else None,
        "max_horiz_cm": max_h_px * effective_cm_per_px if effective_cm_per_px else None,
        "cm_per_px": effective_cm_per_px,
        "scale_source": scale_source,
        "gravity_fixed": g_px is not None,
    }


def save_csv(trajectory: list[list[float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_no", "timestamp_sec", "cx", "cy", "radius_px"])
        writer.writerows(trajectory)


def analyze_video(path: Path, fps_override: float | None, distance_m: float, cm_per_px: float | None) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="動画を開けませんでした。")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    fps = fps_override if fps_override and fps_override > 0 else source_fps
    if fps <= 0:
        fps = 60.0

    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=200,
        varThreshold=50,
        detectShadows=False,
    )
    trajectory: list[list[float]] = []
    trail: deque[tuple[int, int]] = deque(maxlen=60)
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = back_sub.apply(frame)
        detected = detect_ball(frame, fgmask)
        if detected:
            cx, cy, radius = detected
            timestamp = frame_no / fps
            trajectory.append([frame_no, round(timestamp, 4), cx, cy, round(radius, 1)])
            trail.append((cx, cy))

        frame_no += 1

    cap.release()

    release_idx = find_release_frame(trajectory)
    result: dict[str, Any] = {
        "fps": fps,
        "source_fps": source_fps or None,
        "distance_m": distance_m,
        "total_frames": frame_no,
        "detections": len(trajectory),
        "release_found": release_idx is not None,
        "release_frame": None,
        "flight_frames": None,
        "speed_kmh": None,
        "break": {},
        "trajectory": trajectory,
    }

    if len(trajectory) < 4:
        return result

    ball_diameter_cm = infer_ball_diameter_cm(distance_m)

    if release_idx is None:
        start_frame = int(trajectory[0][0])
        end_frame = int(trajectory[-1][0])
        result["flight_frames"] = max(0, end_frame - start_frame)
        result["speed_kmh"] = calc_speed_kmh(start_frame, end_frame, fps, distance_m)
        result["break"] = calc_break(trajectory, fps, cm_per_px, ball_diameter_cm)
        return result

    flight = trajectory[release_idx:]
    start_frame = int(flight[0][0])
    end_frame = int(flight[-1][0])
    result["release_frame"] = start_frame
    result["flight_frames"] = max(0, end_frame - start_frame)
    result["speed_kmh"] = calc_speed_kmh(start_frame, end_frame, fps, distance_m)
    result["break"] = calc_break(flight, fps, cm_per_px, ball_diameter_cm)
    return result


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    distance_m: float = Form(18.44),
    fps: float | None = Form(None),
    cm_per_px: float | None = Form(None),
) -> dict[str, Any]:
    if distance_m <= 0:
        raise HTTPException(status_code=400, detail="投球距離は正の数値にしてください。")

    suffix = Path(video.filename or "pitch.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await video.read())

    try:
        result = analyze_video(tmp_path, fps, distance_m, cm_per_px)
    finally:
        tmp_path.unlink(missing_ok=True)

    if SAVE_CSV:
        csv_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.csv"
        save_csv(result["trajectory"], csv_path)
        result["csv_path"] = str(csv_path.name)
    return result
