import csv
from pathlib import Path

files = {
    "ByteTrack": Path(
        "results/video_analysis/"
        "birds_bytetrack_20261109_174505/predictions.csv"
    ),
    "BoT-SORT": Path(
        "results/video_analysis/"
        "birds_botsort_20261109_174752/predictions.csv"
    ),
}

print("\nBIRD TRACKING COMPARISON")
print("=" * 55)

for tracker, path in files.items():

    if not path.exists():
        print(f"\n{tracker}: File not found: {path}")
        continue

    with open(path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    ids = {
        row["track_id"]
        for row in rows
        if row.get("track_id", "").strip()
    }

    frames = {
        row["frame"]
        for row in rows
        if row.get("frame", "").strip()
    }

    print(f"\nTracker: {tracker}")
    print(f"Detection events: {len(rows)}")
    print(f"Unique track IDs: {len(ids)}")
    print(f"Frames with detections: {len(frames)}")

print("\nComparison completed.")