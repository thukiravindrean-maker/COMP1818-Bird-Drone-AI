"""COMP1818 Bird/Drone interactive local evaluation dashboard.

Run from the comp1818-bird-drone project directory:
    python -m streamlit run dashboard.py

Requires a trained model at models/best_50epochs.pt.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import csv
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
CLIPS = ROOT / "clips"
RESULTS = ROOT / "results"
VIDEO_RESULTS = RESULTS / "video_analysis"
DASHBOARD_RESULTS = RESULTS / "dashboard_runs"
UPLOADS = RESULTS / "dashboard_uploads"

st.set_page_config(page_title="Bird vs Drone | COMP1818", page_icon="🛰️", layout="wide")
st.title("🛰️ Bird vs Drone · YOLO11n Research Dashboard")
st.caption("COMP1818 | Local CPU experiments · Detection, tracking, motion and evidence")
st.warning("Detection events are NOT distinct object counts. Track IDs are estimates and may fragment. ")

models = sorted(MODELS.glob("*.pt")) if MODELS.exists() else []
if not models:
    st.error("No .pt model in models/. Add best_50epochs.pt to your existing project, then refresh.")
    st.stop()
preferred = next((i for i, p in enumerate(models) if p.name == "best_50epochs.pt"), 0)

with st.sidebar:
    st.header("Experiment settings")
    model_path = st.selectbox("Model checkpoint", models, index=preferred, format_func=lambda p: p.name)
    tracker = st.selectbox("Tracker", ["bytetrack", "botsort"])
    imgsz = st.selectbox("Processing width & YOLO image size", [320, 416, 640], index=0)
    confidence = st.selectbox("Confidence threshold", [0.25, 0.4, 0.6])
    stride = st.selectbox("Process every Nth frame", [1, 2, 3], index=1)
    max_frames = st.number_input("Maximum processed frames", min_value=10, max_value=1000, value=80, step=10)
    st.caption("CPU only. Start with 320 px / 80 frames on your 8 GB laptop.")


def run_analysis(source: Path, source_name: str) -> dict:
    DASHBOARD_RESULTS.mkdir(parents=True, exist_ok=True)
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(source_name).stem)[:55]
    run = DASHBOARD_RESULTS / f"{clean}_{tracker}_{imgsz}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run.mkdir(parents=True, exist_ok=False)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not read video: {source_name}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or orig_w <= 0 or orig_h <= 0:
        cap.release()
        raise RuntimeError("Input video has missing or invalid FPS/dimensions")

    scale = min(1.0, imgsz / orig_w)
    width = max(2, (int(orig_w * scale) // 2) * 2)
    height = max(2, (int(orig_h * scale) // 2) * 2)
    out_path = run / "annotated.avi"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), max(1, fps / stride), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV could not create the annotated AVI output")

    model = YOLO(str(model_path))  # fresh instance: never reuse track state across videos
    progress = st.progress(0, text="Analysing frames on CPU...")
    image_spot = st.empty()
    counts = Counter()
    ids = set()
    per_track_classes = defaultdict(list)
    rows = []
    sampled = 0
    frame_index = 0
    inference_seconds = 0.0
    total_start = time.perf_counter()
    max_visible = 0
    preview = None

    try:
        while sampled < int(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % stride:
                continue
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            start = time.perf_counter()
            result = model.track(
                frame, device="cpu", imgsz=imgsz, conf=confidence,
                tracker=f"{tracker}.yaml", persist=True, verbose=False,
            )[0]
            inference_seconds += time.perf_counter() - start
            sampled += 1
            frame_count = 0

            for box in result.boxes:
                cl = model.names[int(box.cls.item())]
                tid = int(box.id.item()) if box.id is not None else None
                x1, y1, x2, y2 = (float(x) for x in box.xyxy[0].tolist())
                counts[cl] += 1
                frame_count += 1
                if tid is not None:
                    ids.add(tid)
                    per_track_classes[tid].append(cl)
                rows.append({
                    "frame": frame_index, "time_sec": round((frame_index - 1) / fps, 4),
                    "track_id": tid, "class": cl, "confidence": round(float(box.conf.item()), 5),
                    "x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2),
                })
            max_visible = max(max_visible, frame_count)
            annotated = result.plot()
            writer.write(annotated)
            preview = annotated
            if sampled == 1 or sampled % 10 == 0:
                image_spot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Annotated processed frame {sampled}")
                progress.progress(min(1.0, sampled / max_frames), text=f"Processed {sampled} frames")
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - total_start
    transitions = sum(sum(a != b for a, b in zip(seq, seq[1:])) for seq in per_track_classes.values())
    summary = {
        "model": str(model_path.relative_to(ROOT)), "video": source_name,
        "tracker": tracker, "device": "CPU", "image_size": imgsz,
        "confidence": confidence, "stride": stride, "processed_frames": sampled,
        "detection_events": dict(counts), "unique_track_ids": len(ids),
        "max_simultaneous_detections": max_visible,
        "class_transitions": transitions,
        "mean_processing_ms": round(1000 * inference_seconds / sampled, 2) if sampled else None,
        "processing_only_fps": round(sampled / inference_seconds, 2) if inference_seconds else None,
        "end_to_end_fps": round(sampled / elapsed, 2) if elapsed else None,
        "identity_switches": None, "count_error": None,
        "identity_ground_truth_status": "not annotated",
        "limitations": [
            "Event totals count detections across frames, not physical individuals.",
            "Unique tracking IDs are not verified object counts.",
            "Class transitions are not identity switches.",
            "CPU timings are from one run; repeat for reliable benchmarking.",
        ],
    }
    summary_path = run / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = run / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=["frame", "time_sec", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"])
        out.writeheader()
        out.writerows(rows)

    if preview is not None:
        preview_path = run / "preview.jpg"
        cv2.imwrite(str(preview_path), preview)
    else:
        preview_path = None
    progress.progress(1.0, text=f"Complete: {sampled} sampled frames")
    return {"run_dir": str(run), "summary": summary, "video": str(out_path), "csv": str(csv_path), "json": str(summary_path), "preview": str(preview_path) if preview_path else ""}


def show_result(result: dict):
    data = result["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed frames", data["processed_frames"])
    c2.metric("Detection events", sum(data["detection_events"].values()))
    c3.metric("Distinct track IDs (estimate)", data["unique_track_ids"])
    c4.metric("CPU ms / processed frame", data["mean_processing_ms"] or "—")
    st.caption(f"Class transitions: {data['class_transitions']} | Sampled throughput: {data['processing_only_fps']} FPS | End-to-end throughput: {data['end_to_end_fps']} FPS")
    if data["detection_events"]:
        st.subheader("Predicted classes · detection events")
        st.bar_chart(pd.DataFrame({"detections": data["detection_events"]}).sort_index())
    if result.get("preview") and Path(result["preview"]).exists():
        st.image(str(result["preview"]), caption="Last annotated frame · not a ground-truth label")
    st.caption("Annotated video is an AVI file for local playback; some browsers cannot preview AVI directly.")
    cols = st.columns(3)
    for col, label, key, mime in [
        (cols[0], "Download predictions CSV", "csv", "text/csv"),
        (cols[1], "Download summary JSON", "json", "application/json"),
        (cols[2], "Download annotated AVI", "video", "video/x-msvideo"),
    ]:
        path = Path(result[key])
        if path.exists():
            with path.open("rb") as f:
                col.download_button(label, f, file_name=path.name, mime=mime, key=f"dl_{key}_{result['run_dir']}")
    with st.expander("Where files are saved"):
        st.code(result["run_dir"])


new_tab, past_tab, evidence_tab = st.tabs(["🎬 Analyse a video", "📊 Existing experiments", "✅ Ground-truth evidence"])

with new_tab:
    clips = sorted([p for p in CLIPS.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]) if CLIPS.exists() else []
    source_choice = st.radio("Video input", ["Existing clip", "Upload a new clip"], horizontal=True)
    chosen = None
    uploaded = None
    if source_choice == "Existing clip":
        if clips:
            chosen = st.selectbox("Existing clips", clips, format_func=lambda p: p.name)
        else:
            st.info("No video found in clips/. Choose upload instead.")
    else:
        uploaded = st.file_uploader("Upload MP4 / AVI / MOV / MKV", type=["mp4", "avi", "mov", "mkv"])
    st.caption("Processing runs locally on CPU. A maximum of 80 sampled frames is recommended for the first test.")

    if st.button("▶ Analyse video", type="primary", disabled=(chosen is None and uploaded is None)):
        if uploaded is not None:
            UPLOADS.mkdir(parents=True, exist_ok=True)
            filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(uploaded.name).name)
            source_file = UPLOADS / (f"{datetime.now():%Y%m%d_%H%M%S_%f}_" + filename)
            with source_file.open("wb") as fh:
                uploaded.seek(0)
                while True:
                    chunk = uploaded.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            display_name = uploaded.name
        else:
            source_file = chosen
            display_name = chosen.name
        try:
            result = run_analysis(source_file, display_name)
            st.session_state["dashboard_last"] = result
            st.success("Video analysis completed. Results were saved to your project's results/dashboard_runs directory.")
        except Exception as ex:
            st.error(f"Analysis failed: {ex}")
            st.info("Check that the model is readable and required tracker dependency 'lap' is installed.")
    if "dashboard_last" in st.session_state:
        st.subheader("Latest dashboard result")
        show_result(st.session_state["dashboard_last"])

with past_tab:
    sources = []
    for root in [VIDEO_RESULTS, DASHBOARD_RESULTS]:
        if root.exists():
            sources.extend(root.glob("*/summary.json"))
    table = []
    for file in sorted(sources):
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
            table.append({
                "Run": file.parent.name, "Video": record.get("video", "—"),
                "Tracker": record.get("tracker", "—"), "Image size": record.get("image_size", "—"),
                "Frames": record.get("processed_frames", "—"), "IDs": record.get("unique_track_ids", "—"),
                "Class transitions": record.get("class_transitions", sum(x.get("class_transitions", 0) for x in record.get("track_details", {}).values())),
                "CPU ms/frame": record.get("mean_processing_ms", "—"),
                "Evidence type": "external-video CPU inference",
            })
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    if table:
        st.dataframe(pd.DataFrame(table), hide_index=True)
        st.caption("Metrics shown are observed CPU experiment outputs, NOT test-set accuracy or verified identity-switch rates.")
        chosen_run = st.selectbox("Open a run's saved figures", [p.parent for p in sources], format_func=lambda p: p.name)
        for figure in ["motion_trajectories.png", "confusion_matrix.png", "preview.jpg"]:
            if (chosen_run / figure).exists():
                st.image(str(chosen_run / figure), caption=f"{chosen_run.name}: {figure}")
    else:
        st.info("No saved experiment summaries yet. Your previous runs should appear under results/video_analysis.")

with evidence_tab:
    st.subheader("Independent AOD-4 test evidence")
    st.info("Reported from your completed 50-epoch Colab test output; the evaluation JSON is not automatically copied to this Windows project.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", "93.63%")
    c2.metric("Recall", "90.84%")
    c3.metric("mAP@50", "95.01%")
    c4.metric("mAP@50–95", "63.70%")
    st.caption("AOD-4 held-out test: 2,241 images and 3,171 annotated instances. These are NOT external-video classification accuracies.")
    st.subheader("Manual per-frame counting evaluation")
    gt_files = sorted(VIDEO_RESULTS.glob("*/ground_truth_review/ground_truth_counts.csv")) if VIDEO_RESULTS.exists() else []
    if not gt_files:
        st.warning("Ground-truth CSV not found. Run ground_truth_evaluation.py and complete manual bird counts.")
    else:
        for gt in gt_files:
            gt_df = pd.read_csv(gt)
            if not {"ground_truth_birds", "predicted_birds"}.issubset(gt_df.columns):
                continue
            verified = gt_df.dropna(subset=["ground_truth_birds", "predicted_birds"]).copy()
            for col in ["ground_truth_birds", "predicted_birds"]:
                verified[col] = pd.to_numeric(verified[col], errors="coerce")
            verified = verified.dropna(subset=["ground_truth_birds", "predicted_birds"])
            st.write(f"**{gt.parent.parent.name}** — {len(verified)} of {len(gt_df)} sampled frames manually counted")
            if len(verified):
                mae = (verified["predicted_birds"] - verified["ground_truth_birds"]).abs().mean()
                st.metric("Mean absolute per-frame count error (reviewed frames only)", f"{mae:.3f}")
            else:
                st.caption("No manual counts yet — count error remains unknown, not zero.")
    st.warning("Actual ID switches / tracking fragmentation require manual object-identity annotations; they have not been measured by this dashboard.")
# ====================================================
# COMP1818 - EXTERNAL VIDEO EXPERIMENT COMPARISON
# ====================================================

import streamlit as st
import pandas as pd
from pathlib import Path

st.header("📊 External Video Experiment Results")

csv_file = Path("results/all_video_results.csv")

if csv_file.exists():

    df = pd.read_csv(csv_file)

    st.success(
        f"Loaded {len(df)} video experiments successfully!"
    )

    # DISPLAY COMPLETE TABLE
    st.subheader("Experiment Comparison")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # CLASSIFICATION COUNTS
    st.subheader("Predicted Object Classes")

    class_columns = [
        "Bird",
        "Drone",
        "Airplane",
        "Helicopter"
    ]

    chart_data = df.set_index("Video")[class_columns]

    st.bar_chart(chart_data)

    # CPU PERFORMANCE
    st.subheader("CPU Processing Performance")

    performance = df.set_index("Video")[["CPU_ms"]]

    st.bar_chart(performance)

    # EXPERIMENT SUMMARY
    st.subheader("Experiment Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Videos Tested", len(df))

    with col2:
        st.metric(
            "Total Detection Events",
            int(df[class_columns].sum().sum())
        )

    with col3:
        st.metric(
            "Average CPU Time",
            f"{df['CPU_ms'].mean():.2f} ms"
        )

    # DOWNLOAD RESULTS
    st.download_button(
        "📥 Download Experiment Results",
        data=df.to_csv(index=False),
        file_name="COMP1818_Experiment_Results.csv",
        mime="text/csv"
    )

    st.info(
        "Detection events and tracking IDs are estimates. "
        "They do not represent verified object counts "
        "or ground-truth classification accuracy."
    )

else:

    st.warning("Experiment CSV file not found.")