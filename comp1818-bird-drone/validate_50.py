import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
from ultralytics import YOLO

torch.set_num_threads(1)

print("Loading 50-epoch model...")

model = YOLO("models/best_50epochs.pt")

print("Starting low-memory validation...")

results = model.val(
    data="data.yaml",
    imgsz=416,
    batch=1,
    workers=0,
    device="cpu",
    plots=False,
    save_json=False,
    project="final_evidence_checks",
    name="validation_50_safe"
)

print("\nVALIDATION COMPLETED")

print("Precision:", results.box.mp)
print("Recall:", results.box.mr)
print("mAP50:", results.box.map50)
print("mAP50-95:", results.box.map)

print("\nPER-CLASS RESULTS")

for i, name in model.names.items():
    p, r, ap50, ap = results.box.class_result(i)
    print(
        f"{name}: P={p:.4f}, R={r:.4f}, "
        f"mAP50={ap50:.4f}, mAP50-95={ap:.4f}"
    )