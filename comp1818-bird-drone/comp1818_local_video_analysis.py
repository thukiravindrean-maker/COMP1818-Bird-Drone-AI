# ============================================================
# COMP1818 - BIRD VS DRONE AI
# YOLO11n + ByteTrack / BoT-SORT
# CPU VIDEO ANALYSIS
# ============================================================

import os

for var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS"
):
    os.environ[var] = "1"

import argparse
import csv
import json
import time

from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import cv2
from ultralytics import YOLO


# ------------------------------------------------------------
# COMMAND LINE SETTINGS
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--model", default="models/best_50epochs.pt")
parser.add_argument("--video", required=True)
parser.add_argument("--tracker", default="bytetrack")
parser.add_argument("--imgsz", type=int, default=320)
parser.add_argument("--width", type=int, default=320)
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--stride", type=int, default=2)
parser.add_argument("--max-frames", type=int, default=80)

args = parser.parse_args()

if args.tracker not in ("bytetrack", "botsort"):
    raise ValueError("Use bytetrack or botsort")

if args.stride < 1:
    raise ValueError("Stride must be at least 1")

# ------------------------------------------------------------
# CHECK MODEL AND VIDEO
# ------------------------------------------------------------

model_path = Path(args.model)
video_path = Path(args.video)

if not model_path.exists():
    raise FileNotFoundError(f"Model missing: {model_path}")

if not video_path.exists():
    raise FileNotFoundError(f"Video missing: {video_path}")

print("Loading model:", model_path)

model = YOLO(str(model_path))

print("Classes:", model.names)

# ------------------------------------------------------------
# CREATE OUTPUT FOLDER
# ------------------------------------------------------------

run_name = (
    video_path.stem + "_" +
    args.tracker + "_" +
    datetime.now().strftime("%Y%m%d_%H%M%S")
)

output = Path("results/video_analysis") / run_name
output.mkdir(parents=True, exist_ok=True)

csv_path = output / "predictions.csv"

# ------------------------------------------------------------
# OPEN VIDEO
# ------------------------------------------------------------

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if source_width <= 0 or source_height <= 0:
    cap.release()
    raise RuntimeError("Invalid video dimensions")

scale = min(1.0, args.width / source_width)

width = max(2, int(source_width * scale) // 2 * 2)
height = max(2, int(source_height * scale) // 2 * 2)

video_output = output / "annotated.avi"

writer = cv2.VideoWriter(
    str(video_output),
    cv2.VideoWriter_fourcc(*"MJPG"),
    max(1, fps / args.stride),
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video")

print("\nStarting CPU analysis...")
print("Tracker:", args.tracker)
print("Input video:", video_path)
print("Processing resolution:", width, "x", height)

# ------------------------------------------------------------
# TRACKING ANALYSIS
# ------------------------------------------------------------

detections = Counter()
track_classes = defaultdict(Counter)
track_history = defaultdict(list)

processed = 0
frame_number = 0
processing_time = 0.0

with open(csv_path, "w", newline="") as file:

    csv_writer = csv.writer(file)

    csv_writer.writerow([
        "frame",
        "time_sec",
        "track_id",
        "class",
        "confidence",
        "x1", "y1", "x2", "y2"
    ])

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            if (frame_number - 1) % args.stride != 0:
                continue

            frame = cv2.resize(frame, (width, height))

            start = time.perf_counter()

            result = model.track(
                frame,
                tracker=args.tracker + ".yaml",
                persist=True,
                conf=args.conf,
                imgsz=args.imgsz,
                device="cpu",
                verbose=False
            )[0]

            processing_time += time.perf_counter() - start

            processed += 1

            for box in result.boxes:

                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                confidence = float(box.conf.item())

                track_id = (
                    int(box.id.item())
                    if box.id is not None
                    else None
                )

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections[class_name] += 1

                if track_id is not None:
                    track_classes[track_id][class_name] += 1
                    track_history[track_id].append(
                        (frame_number, class_name)
                    )

                csv_writer.writerow([
                    frame_number,
                    round((frame_number - 1) / fps, 3),
                    track_id,
                    class_name,
                    round(confidence, 4),
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ])

            annotated = result.plot()
            writer.write(annotated)

            print(
                f"Processed frame {processed}",
                end="\r"
            )

            if args.max_frames and processed >= args.max_frames:
                break

    finally:
        cap.release()
        writer.release()

# ------------------------------------------------------------
# CLASS CONSISTENCY ANALYSIS
# ------------------------------------------------------------

track_details = {}

for track_id, classes in track_classes.items():

    history = track_history[track_id]

    changes = sum(
        history[i][1] != history[i-1][1]
        for i in range(1, len(history))
    )

    track_details[str(track_id)] = {
        "predictions": dict(classes),
        "majority_class": classes.most_common(1)[0][0],
        "class_transitions": changes
    }

# ------------------------------------------------------------
# PERFORMANCE RESULTS
# ------------------------------------------------------------

mean_ms = (
    processing_time * 1000 / processed
    if processed else 0
)

summary = {
    "model": str(model_path),
    "video": str(video_path),
    "tracker": args.tracker,
    "device": "CPU",
    "image_size": args.imgsz,
    "confidence": args.conf,
    "stride": args.stride,
    "processed_frames": processed,
    "detection_events": dict(detections),
    "unique_track_ids": len(track_classes),
    "mean_processing_ms": round(mean_ms, 2),
    "track_details": track_details,
    "limitations": [
        "Detection events are not physical object counts.",
        "Tracking IDs may fragment or be reused.",
        "Class transitions are not tracking ID switches.",
        "Counting accuracy requires manual ground truth."
    ]
}

with open(output / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n\n===================================")
print("VIDEO ANALYSIS COMPLETE")
print("===================================")

print("Processed frames:", processed)
print("Detection events:", dict(detections))
print("Unique tracking IDs:", len(track_classes))
print("CPU processing time:", round(mean_ms, 2), "ms/frame")

print("\nResults:", output)
print("Annotated video:", video_output)
print("Predictions:", csv_path)
print("Summary:", output / "summary.json")