# Experiment ledger - no measurements invented
One row per run. Change ONE setting from E1 at a time. Every number in the report must come
from a row here. For each run also record: command, code commit, tracker YAML, library
versions, GT frame range, clip fps and any errors. Do not fill gaps with the earlier pilot metrics.

## Environment
| Item | Value |
|---|---|
| Python | |
| Ultralytics | |
| Training hardware | Colab GPU: |
| Testing hardware | Ryzen 5 7530U, 8 GB RAM, CPU only |

## Training
| Run | Start weights | Epochs requested | Epochs completed | imgsz | batch | seed | Run folder | Status |
|---|---|---|---|---:|---:|---:|---|---|
| T0 | yolo11n.pt | 10? | ? | ? | ? | ? | ? | Old checkpoint - confirm from args.yaml / results.csv |
| T1 | yolo11n.pt | 50 | | 640 | 16 | 42 | T1_e50_640 | Pending |

## Detection - TEST split (evaluate.py)
mAP does not depend on the conf threshold; matched precision/recall at a set conf does.
| Run | Model | imgsz | conf | Bird P | Bird R | Drone P | Drone R | mAP50 | mAP50-95 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B0 | Most frequent class | - | - | - | - | - | - | - | - | Pending |
| B1 | yolo11n.pt (COCO, not fine-tuned) | 640 | 0.25 | | | | | - | - | Pending |
| E1 | best.pt (T1) | 640 | 0.25 | | | | | | | Pending |

## Tracking - hand-labelled clips (track.py + eval_tracking.py)
| Run | Clip | Tracker | imgsz | conf | stride | ID switches | Fragments | Count error % | Median ms/frame | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| E1 | bird | ByteTrack | 640 | 0.25 | 1 | | | | | Pending |
| E2 | bird | BoT-SORT | 640 | 0.25 | 1 | | | | | Pending |
| E3 | drone | ByteTrack | 640 | 0.25 | 1 | | | | | Pending |
| E4 | drone | BoT-SORT | 640 | 0.25 | 1 | | | | | Pending |
| E5 | drone | ByteTrack | 640 | 0.40 | 1 | | | | | Pending (conf sensitivity) |
| E6 | drone | ByteTrack | 640 | 0.60 | 1 | | | | | Pending (conf sensitivity) |
