# Bird vs Drone Detection and Tracking Using YOLO11n

### COMP1818 — Artificial Intelligence Applications

**University:** University of Greenwich  
**Student:** Thulakshan Ravindrean  
**Academic Year:** 2026–2027

## 1. About This Project

This project investigates whether a computer vision model can distinguish birds from drones and other flying objects in aerial images and videos.

The main research question is:

**How reliably can YOLO11n and video tracking distinguish birds from drones on a CPU laptop, and under which recorded conditions do predictions become unreliable?**

The project uses YOLO11n for object detection and ByteTrack and BoT-SORT for video tracking.

The main aim was not to develop a completely new detection algorithm, but to evaluate how an existing model performs on labelled images and unfamiliar videos.

## 2. Why I Chose This Topic

Small drones can be difficult to distinguish from birds, especially when they are far away from the camera.

This matters in surveillance and monitoring applications because an incorrect prediction could cause a false alert or prevent a drone from being identified correctly.

I wanted to investigate whether an object detection model could provide useful results on an ordinary laptop without requiring expensive hardware for inference.

The research also examines the difference between high dataset evaluation scores and performance on external videos.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO11n | Object detection |
| Ultralytics | Model training and inference |
| PyTorch | Machine learning framework |
| Google Colab | GPU-based model training |
| OpenCV | Video processing |
| ByteTrack | Object tracking |
| BoT-SORT | Alternative tracking method |
| Streamlit | Project dashboard |
| CSV and JSON | Saving experiment results |

Training was completed using Google Colab GPU resources.

The main video experiments were carried out using an AMD Ryzen 5 7530U laptop with 8 GB RAM and CPU-only inference.

## 4. Dataset

The project uses the AOD-4 dataset.

AOD-4 contains images of four object classes:

1. Airplane
2. Bird
3. Drone
4. Helicopter

The dataset uses YOLO-format bounding-box annotations.

**Dataset:**  
https://data.mendeley.com/datasets/cd5z895tr2/1

**Dataset DOI:**  
https://doi.org/10.17632/cd5z895tr2.1

The complete dataset is not included in this GitHub repository.

The locally configured evaluation folder contained 2,241 images. The independence of these images from the training data was not separately audited.

## 5. Model Training

The project started with the pretrained YOLO11n model.

Three training stages were recorded:

| Training stage | Completed epochs | Image size | Batch size |
|---|---:|---:|---:|
| Pilot | 1 | 416 | 8 |
| Second run | 10 | 416 | 8 |
| Final run | 50 | 416 | 8 |

The final 50-epoch model was selected for the main experiments.

The 50-epoch run used seed 42 and recorded `optimizer=auto`.

The 10-epoch run used seed 0. The difference between these runs should not be interpreted as an epoch-only comparison.

The selected local checkpoint is:

`models/best_50epochs.pt`

## 6. Detection Results

The final checkpoint was evaluated on the locally configured 2,241-image AOD-4 evaluation folder.

### Overall evaluation results

| Metric | Score |
|---|---:|
| Precision | 0.9363 |
| Recall | 0.9084 |
| mAP50 | 0.9500 |
| mAP50-95 | 0.6370 |

### Per-class results

| Object | Precision | Recall | mAP50 |
|---|---:|---:|---:|
| Airplane | 0.9475 | 0.9168 | 0.9674 |
| Bird | 0.9223 | 0.9086 | 0.9447 |
| Drone | 0.9428 | 0.8982 | 0.9359 |
| Helicopter | 0.9326 | 0.9098 | 0.9522 |

The evaluation used an image size of 416 on the CPU.

The saved CPU evaluation output can be found at:

`results/validation_50_cpu.txt`

These results describe the configured labelled-image evaluation. They do not establish accuracy on external drone videos.

## 7. Video Tracking

The project tested two object tracking algorithms:

- ByteTrack
- BoT-SORT

Both were tested using the same bird video with image size 320, confidence 0.25 and frame stride 2.

Each run processed 80 sampled frames.

### Tracking comparison

| Metric | ByteTrack | BoT-SORT |
|---|---:|---:|
| Detection events | 492 | 491 |
| Assigned track IDs | 24 | 26 |
| Mean processing time | 82.16 ms | 95.61 ms |

ByteTrack recorded a lower mean processing time in this single comparison.

The assigned track IDs cannot be treated as the actual number of birds because tracking identities were not checked against manually labelled ground truth.

## 8. External Video Testing

The project also tested the model using external drone videos.

Some predictions were labelled as airplanes or helicopters rather than drones.

### Recorded results

| Video | Airplane events | Bird events | Drone events | Helicopter events |
|---|---:|---:|---:|---:|
| drone_1.mp4 | 78 | 0 | 0 | 3 |
| drone_2.mp4 | 75 | 1 | 0 | 2 |
| drone_tracking_safe.mp4 | 55 | 0 | 12 | 10 |

These figures represent detection events across sampled frames, not unique objects or independently confirmed classification errors.

The video results demonstrate why a model should be evaluated using different types of footage rather than relying only on labelled-image metrics.

## 9. Running the Project

### Requirements

Use Python and install the dependencies required by the relevant scripts.

From the `comp1818-bird-drone` directory, create a virtual environment:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project's listed dependencies:

```powershell
python -m pip install -r requirements.txt
```

To use the dashboard, Streamlit may also need to be installed:

```powershell
python -m pip install streamlit matplotlib
```

### Run the CPU Evaluation

Make sure the following are available locally:

- `models/best_50epochs.pt`
- `data.yaml`
- The AOD-4 evaluation images and annotations

Update `data.yaml` so its paths match the dataset's location on your computer.

Then run:

```powershell
python validate_50.py
```

The script loads the selected checkpoint and runs CPU evaluation at image size 416.

**Important:** The local `data.yaml` must point to the intended evaluation partition. It should not be used for model training if its training entry points to test images.

### Run the Dashboard

From the project directory:

```powershell
python -m streamlit run dashboard.py
```

The dashboard provides a way to inspect project outputs and video analysis.

Its interface was developed as part of the project, but no formal usability study was completed.

### Other Project Scripts

The repository also contains scripts for training, object detection, tracking and motion analysis.

Some scripts support additional experiments beyond the completed coursework runs.

Their presence does not mean every experiment has been completed or evaluated.

See `EXPERIMENTS.md` for the completed experiment settings and recorded results.

## 10. Project Files

Important files include:

| File or folder | Purpose |
|---|---|
| `README.md` | Project overview and instructions |
| `EXPERIMENTS.md` | Experiment records and limitations |
| `train.py` | Training script |
| `validate_50.py` | Final checkpoint CPU evaluation |
| `evaluate.py` | Additional detection evaluation utility |
| `track.py` | Video tracking utility |
| `eval_tracking.py` | Tracking evaluation utility requiring ground truth |
| `dashboard.py` | Streamlit dashboard |
| `analyze_motion.py` | Motion analysis |
| `requirements.txt` | Core project dependencies |
| `models/` | Model files |
| `results/` | Saved experiment outputs |

The repository does not necessarily contain all original videos, dataset files or Google Colab training archives.

## 11. Research Limitations

The main limitations identified during the project were:

- High labelled-image scores did not guarantee reliable predictions on external videos.
- Some small drone objects were assigned airplane or helicopter labels.
- The laptop encountered a memory limitation when processing a larger video.
- Downscaling made some video processing feasible but may reduce visual detail.
- Video predictions were not independently labelled to measure classification accuracy.
- Tracking IDs were not verified against real object identities.
- Dataset overlap was not independently audited.

The results should therefore be interpreted as an experimental evaluation, not proof that the system is reliable enough for operational surveillance.

## 12. Possible Future Improvements

Future work could include evaluating the model at different image sizes, testing confidence thresholds, collecting more independently labelled drone videos and measuring tracking identity errors.

A controlled comparison with the pretrained model or a simple baseline would also strengthen the evaluation.

These are proposed improvements, not completed experimental results.

## 13. AI Assistance

ChatGPT was used during the project to support Python coding, debugging, technical explanations, project organisation, documentation and report drafting.

The training and local experiments were carried out using Google Colab and the student's computer. AI-generated suggestions and explanations were reviewed against the available experiment logs.

The final assessed analysis and conclusions must be checked and written in accordance with the university's academic integrity requirements.

## 14. References

**Dataset:**

Soni, V. et al. (2024). *Introducing AOD 4: A dataset for air borne object detection.* Data in Brief, 56, 110801.

https://doi.org/10.1016/j.dib.2024.110801

**YOLO11 documentation:**

https://docs.ultralytics.com/models/yolo11/

**ByteTrack:**

Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.*

**BoT-SORT:**

Aharon, N., Orfaig, R. and Bobrovsky, B.-Z. (2022). *BoT-SORT: Robust Associations Multi-Pedestrian Tracking.*

## 15. Conclusion

This project investigated bird and drone detection using YOLO11n and video tracking on a CPU laptop.

The final checkpoint achieved strong results on the configured AOD-4 evaluation images. However, the external-video experiments showed that predictions could vary considerably between different clips.

The research highlights the importance of testing machine learning models beyond their original datasets and understanding their limitations before considering practical deployment.
