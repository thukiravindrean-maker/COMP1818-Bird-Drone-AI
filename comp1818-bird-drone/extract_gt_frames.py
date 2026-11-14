import cv2
import csv
from pathlib import Path

video_path = Path("clips/birds.mp4")
csv_path = Path(
    "results/video_analysis/"
    "birds_bytetrack_20261109_174505/predictions.csv"
)

output_dir = Path("labels_gt/birds/frames")
output_dir.mkdir(parents=True, exist_ok=True)

with open(csv_path, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

frame_numbers = sorted({
    int(row["frame"])
    for row in rows
    if row.get("frame")
})

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Cannot open birds.mp4")

saved = 0

for frame_number in frame_numbers:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    ok, frame = cap.read()

    if not ok:
        print(f"Could not read frame {frame_number}")
        continue

    output_path = output_dir / f"frame_{frame_number:06d}.jpg"

    cv2.imwrite(str(output_path), frame)
    saved += 1

cap.release()

print("Frames requested:", len(frame_numbers))
print("Frames saved:", saved)
print("Output:", output_dir.resolve())