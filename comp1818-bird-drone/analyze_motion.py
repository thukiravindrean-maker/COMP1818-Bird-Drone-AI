# ============================================================
# COMP1818 - BIRD VS DRONE AI
# MOTION TRAJECTORY ANALYSIS
# YOLO11n + ByteTrack / BoT-SORT
# ============================================================

import csv
import math
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib.pyplot as plt


ROOT = Path("results/video_analysis")

csv_files = sorted(ROOT.glob("*/predictions.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No predictions.csv files found in results/video_analysis"
    )

print("=" * 55)
print("COMP1818 - MOTION TRAJECTORY ANALYSIS")
print("=" * 55)

for csv_path in csv_files:

    print("\nAnalysing:", csv_path.parent.name)

    tracks = defaultdict(list)

    # LOAD PREDICTIONS
    with open(csv_path, newline="") as file:

        reader = csv.DictReader(file)

        for row in reader:

            track_id = str(row["track_id"]).strip()

            if track_id.lower() in ("", "none", "null"):
                continue

            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            tracks[track_id].append({
                "frame": int(row["frame"]),
                "time": float(row["time_sec"]),
                "x": cx,
                "y": cy,
                "class": row["class"]
            })

    print("Tracking IDs:", len(tracks))

    # CREATE MOTION GRAPH
    fig, ax = plt.subplots(figsize=(10, 7))

    motion_results = []

    for track_id, observations in tracks.items():

        observations.sort(key=lambda r: r["frame"])

        if len(observations) < 2:
            continue

        x = [p["x"] for p in observations]
        y = [p["y"] for p in observations]

        labels = Counter(
            p["class"] for p in observations
        )

        majority_class = labels.most_common(1)[0][0]

        # TOTAL IMAGE-PLANE MOVEMENT
        distance = 0.0

        for i in range(1, len(observations)):

            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]

            distance += math.hypot(dx, dy)

        # TRACK DURATION
        duration = (
            observations[-1]["time"]
            - observations[0]["time"]
        )

        # AVERAGE IMAGE-PLANE SPEED
        average_speed = (
            distance / duration if duration > 0 else 0
        )

        # COUNT CLASS CHANGES
        transitions = sum(
            observations[i]["class"]
            != observations[i - 1]["class"]
            for i in range(1, len(observations))
        )

        motion_results.append({
            "track_id": track_id,
            "class": majority_class,
            "observations": len(observations),
            "duration_sec": round(duration, 3),
            "distance_pixels": round(distance, 2),
            "average_speed_px_sec": round(average_speed, 2),
            "class_transitions": transitions
        })

        # DRAW TRAJECTORY
        ax.plot(
            x,
            y,
            marker=".",
            markersize=3,
            linewidth=1,
            label=f"ID {track_id}: {majority_class}"
        )

        ax.text(
            x[0],
            y[0],
            str(track_id),
            fontsize=8
        )

    # GRAPH FORMATTING
    ax.set_title(
        "Bird vs Drone - Motion Trajectory Analysis\n"
        + csv_path.parent.name
    )

    ax.set_xlabel("X coordinate (pixels)")
    ax.set_ylabel("Y coordinate (pixels)")

    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    if motion_results:
        ax.legend(
            fontsize=7,
            loc="best",
            ncol=2
        )

    plt.tight_layout()

    # SAVE GRAPH
    graph_path = csv_path.parent / "motion_trajectories.png"

    plt.savefig(graph_path, dpi=200)
    plt.close(fig)

    # SAVE MOTION METRICS
    metrics_path = csv_path.parent / "motion_metrics.csv"

    with open(metrics_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "track_id",
                "class",
                "observations",
                "duration_sec",
                "distance_pixels",
                "average_speed_px_sec",
                "class_transitions"
            ]
        )

        writer.writeheader()
        writer.writerows(motion_results)

    print("Motion graph saved:", graph_path)
    print("Motion metrics saved:", metrics_path)

print("\n========================================")
print("MOTION ANALYSIS COMPLETE!")
print("========================================")