# COMP1818 — Bird vs Drone Detection and Tracking

**Research question:** How reliably can YOLO11n and tracking distinguish birds and drones on a CPU laptop, and when does the system fail?

This project implements transfer learning, test-set object detection evaluation, matched baseline comparison, per-video tracking, identity evaluation and exploratory motion features. **No real experiment results are included in this skeleton: run scripts on your own files and cite actual logs.**

## Required inputs (not included)
- `models/best.pt`: your VERIFIED fine-tuned checkpoint. Check `args.yaml` and `results.csv` to confirm actual completed epochs.
- `datasets/AOD-4/`: original images and YOLO-format labels with train/val/test partitions. Edit `data/aod4_local.yaml` (laptop) and `data/aod4_colab.yaml` (Colab) to the real ABSOLUTE paths and **verify class ID ordering**, which is not established by this ZIP.
- `clips/bird.mp4`, `clips/drone.mp4`, or your own recordings with usage rights.
- `labels_gt/<clip-name>/gt/gt.txt`: independent hand-labelled MOT-style identity ground truth (MOT frames 1-based).
- `yolo11n.pt` original pretrained checkpoint (downloaded by Ultralytics if network access is available).

## Install (Windows)
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```
Colab training (T4 GPU; dataset unzipped to /content/datasets/AOD-4; run folder saved to Drive):
```bash
python train.py --data data/aod4_colab.yaml --epochs 50 --imgsz 640 --batch 16 --device 0 --name T1_e50_640 --project /content/drive/MyDrive/comp1818_runs
# if Colab disconnects:
yolo train resume model=/content/drive/MyDrive/comp1818_runs/T1_e50_640/weights/last.pt
```
Do **not** claim the requested epochs completed - confirm them in `args.yaml` and `results.csv`.

## Evaluation commands (run from project root)
```bash
python evaluate.py --data data/aod4_local.yaml --model models/best.pt --imgsz 640 --conf 0.25
python track.py --clip clips/bird.mp4 --tracker bytetrack.yaml --imgsz 640 --conf 0.25
python track.py --clip clips/bird.mp4 --tracker botsort.yaml --imgsz 640 --conf 0.25
python track.py --clip clips/drone.mp4 --tracker bytetrack.yaml --imgsz 640 --conf 0.25
python track.py --clip clips/drone.mp4 --tracker botsort.yaml --imgsz 640 --conf 0.25
python motion.py --tracks results/tracks/bird_bytetrack_s640_c0.25_v1.csv --fps 30
python eval_tracking.py --gt labels_gt/bird/gt/gt.txt --tracks results/tracks/bird_bytetrack_s640_c0.25_v1.csv --class-map labels_gt/class_map.json
```
Use the clip's real fps (printed in the tracks `.json` file). `labels_gt/class_map.json` maps CVAT class IDs to model class names, e.g. `{"1": "bird", "2": "drone"}` - check the IDs in your CVAT export first. The example map is NOT a verified annotation schema.

### OC-SORT
Stock Ultralytics installations generally ship `bytetrack.yaml` and `botsort.yaml`, **not OC-SORT**. OC-SORT is deliberately **not presented as working** in this package. To add it, install a verified compatible third-party implementation and document its exact version, tracking settings and adapter. Renaming a YAML file to `ocsort.yaml` does not implement OC-SORT.

## Outputs
- `results/detection_test.csv`: class P/R/AP50/AP50-95 from held-out test data.
- `results/confusion_matrix.png`: test-set confusion matrix.
- `results/baselines.csv`: majority-class classification baseline (not a detection score).
- `results/matched_pr_baselines.csv`: matched confidence/IoU precision and recall for pretrained COCO and fine-tuned weights.
- `results/tracks/*.csv`: 1-based original-frame tracking boxes with IDs.
- `results/tracks/*.timing.csv`: timing **including frames with no boxes**, one record per processed frame.
- `results/tracks/*.json`: exact run settings and source metadata.
- `results/motion/*.csv`: px/s, turn and low-motion proportion; **not calibrated speed**.
- `results/tracking_eval.csv`: actual GT-associated ID switches, track fragments and count error.

## Fair comparison
Hold checkpoint, source clip, sampled frame indices, confidence, image size, IoU rule, CPU and video decoding method fixed. Start at 640px, confidence 0.25 and stride 1. Count each real object once; a tracker ID change for one GT object is an ID switch; matched–unmatched–matched is a fragment. Count error = `abs(predicted_distinct_ids - GT_distinct_objects) / GT_distinct_objects * 100`, undefined for zero GT. Count separately class errors, missed detections, false tracks and time to prevent misleading conclusions. The timing includes full model+tracker calls but excludes loading, CSV export and decoding. A 640/960 or confidence sensitivity run changes just one setting from baseline.

## Known constraints
- Earlier reduced video tests and pilot CSV figures must not be relabelled as new controlled test evidence.
- Motion features in image coordinates depend on camera motion, object distance, and missing frames.
- Tracking errors may originate with object detection and classification; review video frames and the detection confusion matrix.
- The user's assessment brief requires links to implementation and dataset, the Moodle report template and an honest AI-use declaration. Add working shared URLs and your own research analysis before submission.
- AI helped produce this code implementation; disclose this usage accurately under your university's rules. The official brief says AI must not produce substantive assessed analysis or arguments.

## AI-assisted fixes (declare these in the AI-use reflection)
Every changed line is marked `# AI-assisted fix` in the code.
- `eval_tracking.py`: clip name read from the correct folder; predicted objects counted by distinct track ID (a class flip no longer doubles the count); median ms after 5 warm-up frames; re-runs replace their old row; `--class-map` accepts a `.json` file.
- `evaluate.py`: default image size 640; prints the COCO class mapping and skips the COCO baseline with a warning if nothing matched.
- `train.py`: defaults 50 epochs, 640px, batch 16, Colab YAML.
- `track.py`: default stride 1; adds median ms after warm-up to the run `.json`.
- `motion.py`: turn angles only counted between two moving steps.
- `data/`: separate absolute-path YAMLs for laptop and Colab (Ultralytics and `evaluate.py` resolve relative paths differently).
