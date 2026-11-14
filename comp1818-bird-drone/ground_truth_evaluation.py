# ============================================================
# COMP1818 - BIRD VS DRONE AI
# GROUND-TRUTH COUNT EVALUATION
# ============================================================

import cv2
import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean

# EXISTING EXPERIMENT
RUN = Path(
    "results/video_analysis/"
    "birds_bytetrack_20261109_174505"
)

VIDEO = Path("clips/birds.mp4")
PREDICTIONS = RUN / "predictions.csv"

OUTPUT = RUN / "ground_truth_review"
RAW_DIR = OUTPUT / "original_frames"
PRED_DIR = OUTPUT / "predicted_frames"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

if not VIDEO.exists():
    raise FileNotFoundError(f"Video missing: {VIDEO}")

if not PREDICTIONS.exists():
    raise FileNotFoundError(
        f"Predictions missing: {PREDICTIONS}"
    )

# LOAD EXISTING YOLO PREDICTIONS
detections = defaultdict(list)

with open(PREDICTIONS, newline="") as f:
    for row in csv.DictReader(f):
        detections[int(row["frame"])].append(row)

# SELECT 20 FRAMES FROM THE EXPERIMENT
frames = sorted(detections.keys())
selected = frames[::4][:20]

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Could not open video")

source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

scale = min(1.0, 320 / source_width)

width = max(2, int(source_width * scale) // 2 * 2)
height = max(2, int(source_height * scale) // 2 * 2)

template_rows = []

print("Preparing ground-truth review frames...")

for frame_num in selected:

    # Frame numbers in predictions.csv start at 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)

    success, frame = cap.read()

    if not success:
        print("Could not read frame:", frame_num)
        continue

    frame = cv2.resize(frame, (width, height))

    # Save image without model predictions
    cv2.imwrite(
        str(RAW_DIR / f"frame_{frame_num:04d}.jpg"),
        frame
    )

    annotated = frame.copy()

    bird_count = 0

    for row in detections[frame_num]:

        if row["class"] != "bird":
            continue

        bird_count += 1

        x1 = int(float(row["x1"]))
        y1 = int(float(row["y1"]))
        x2 = int(float(row["x2"]))
        y2 = int(float(row["y2"]))

        track_id = row["track_id"]

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f"Bird ID {track_id}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

    # Save frame with YOLO predictions
    cv2.imwrite(
        str(PRED_DIR / f"frame_{frame_num:04d}.jpg"),
        annotated
    )

    template_rows.append({
        "frame": frame_num,
        "predicted_birds": bird_count,
        "ground_truth_birds": "",
        "notes": ""
    })

cap.release()

# CREATE MANUAL COUNTING TEMPLATE
GT_FILE = OUTPUT / "ground_truth_counts.csv"

if not GT_FILE.exists():

    with open(GT_FILE, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "predicted_birds",
                "ground_truth_birds",
                "notes"
            ]
        )

        writer.writeheader()
        writer.writerows(template_rows)

    print("\nGround-truth template created.")

else:

    # Preserve existing manual annotations
    with open(GT_FILE, newline="") as f:
        completed = [
            row for row in csv.DictReader(f)
            if row["ground_truth_birds"].strip()
        ]

    if completed:

        errors = [
            abs(
                int(row["predicted_birds"])
                - int(row["ground_truth_birds"])
            )
            for row in completed
        ]

        print("\nGROUND-TRUTH RESULTS")
        print("Reviewed frames:", len(completed))
        print("Mean absolute count error:", round(mean(errors), 3))

print("\n===================================")
print("GROUND-TRUTH PREPARATION COMPLETE")
print("===================================")

print("Original frames:", RAW_DIR)
print("Predicted frames:", PRED_DIR)
print("Manual counting file:", GT_FILE)