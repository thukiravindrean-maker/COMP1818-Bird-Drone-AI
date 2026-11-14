Everything the report quotes must be reproducible from files in here.
  training/          args.yaml + results.csv from the training run
  detection_test.csv per-class P, R, AP50, AP50-95 on the TEST split
  baselines.csv      B0 (most frequent class) and B1 (COCO YOLO11n, not fine-tuned)
  confusion_matrix.png
  tracks/            one CSV per clip x tracker x conf
  motion/            motion features per track
  tracking_eval.csv  ID switches, fragments, count error, ms/frame
  trackers/          copies of the exact tracker yaml files used
