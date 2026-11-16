# COMP1818 — Experiment Records

## Bird and Drone Detection Using YOLO11n

This file records the experiments completed for my COMP1818 Artificial Intelligence Applications project.

The purpose of this research was to investigate how reliably YOLO11n could distinguish birds from drones and other flying objects, particularly when processing videos on a CPU-only laptop.

The results below come from saved Google Colab training records and local Python experiment outputs. Planned experiments are recorded separately. Results have not been added for tests that were not completed.

## 1. Experimental Environment

| Component | Details |
|---|---|
| Model | YOLO11n |
| Dataset | AOD-4 |
| Dataset classes | Airplane, Bird, Drone, Helicopter |
| Training environment | Google Colab GPU |
| Local computer | AMD Ryzen 5 7530U |
| Memory | 8 GB RAM |
| Testing device | CPU |
| Operating system | Windows |
| Python | 3.13.16 in the recorded local environment |
| Ultralytics | 8.4.203 in the recorded local environment |
| Tracking methods | ByteTrack and BoT-SORT |

The exact Google Colab GPU should be checked against the saved training-session records.

## 2. Model Training

YOLO11n was fine-tuned using the AOD-4 dataset. The training process included an initial one-epoch pilot, followed by completed 10-epoch and 50-epoch training runs.

| Run | Starting weights | Completed epochs | Image size | Batch size | Seed | Status |
|---|---|---:|---:|---:|---:|---|
| T0 | yolo11n.pt | 1 | 416 | 8 | Not confirmed | Completed pilot |
| T1 | yolo11n.pt | 10 | 416 | 8 | 0 | Completed |
| T2 | yolo11n.pt | 50 | 416 | 8 | 42 | Completed |

Both the 10-epoch and 50-epoch configurations recorded `optimizer=auto` and two data-loader workers.

The selected 50-epoch training run was saved in:

`bird_drone_50epochs_20261109_125356`

Its saved `best.pt` checkpoint was used for the final CPU evaluation.

The 10-epoch and 50-epoch runs used different random seeds. Their results should therefore not be treated as a perfectly controlled comparison of epoch count alone.

## 3. Training and Validation Results

The following results were recorded in the saved Colab training CSV files.

| Metric | 1 Epoch | 10 Epochs | 50 Epochs |
|---|---:|---:|---:|
| Precision | 0.5732 | 0.9117 | 0.9391 |
| Recall | 0.5421 | 0.8709 | 0.9186 |
| mAP50 | 0.5738 | 0.9187 | 0.9518 |
| mAP50-95 | 0.2933 | 0.5748 | 0.6326 |

These are training-run validation metrics, not independently verified test results.

The 50-epoch run achieved the highest recorded validation scores among these three runs.

## 4. CPU Detection Evaluation

The selected 50-epoch checkpoint was evaluated using the locally configured AOD-4 evaluation folder.

### Evaluation settings

| Setting | Value |
|---|---|
| Checkpoint | `models/best_50epochs.pt` |
| Evaluation images | 2,241 |
| Image size | 416 |
| Device | CPU |
| Batch size | 1 |
| Workers | 0 |
| Evaluation script | `validate_50.py` |
| Saved output | `results/validation_50_cpu.txt` |

The evaluation was recorded on 15 November 2026.

### Overall results

| Metric | Result |
|---|---:|
| Precision | 0.9363 |
| Recall | 0.9084 |
| mAP50 | 0.9500 |
| mAP50-95 | 0.6370 |

### Results by class

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Airplane | 0.9475 | 0.9168 | 0.9674 | 0.6927 |
| Bird | 0.9223 | 0.9086 | 0.9447 | 0.6489 |
| Drone | 0.9428 | 0.8982 | 0.9359 | 0.5997 |
| Helicopter | 0.9326 | 0.9098 | 0.9522 | 0.6066 |

The local evaluation folder was named `test`, but its separation from the training data was not independently audited. The original AOD-4 publication also contains inconsistent descriptions of the dataset splits.

Therefore, these results are described as performance on the locally configured evaluation split, rather than proof of performance on a completely independent dataset.

## 5. Bird Video Tracking

Two tracking methods were tested using the same bird video and inference settings.

### Settings

| Parameter | Value |
|---|---|
| Video | `birds.mp4` |
| Image size | 320 |
| Confidence | 0.25 |
| Frame stride | 2 |
| Processed frames | 80 |
| Device | CPU |

### Recorded results

| Metric | ByteTrack | BoT-SORT |
|---|---:|---:|
| Detection events | 492 | 491 |
| Assigned track IDs | 24 | 26 |
| Mean processing time (ms) | 82.16 | 95.61 |

ByteTrack recorded a lower mean processing time in this experiment.

However, an assigned track ID is not necessarily a separate bird. The actual bird count and identity-switch accuracy were not established using independent ground-truth annotations.

The timing values are means from the saved runs, not medians.

## 6. External Drone Video Testing

The 50-epoch model was also tested on external drone videos.

These experiments used video inference at image size 320, confidence 0.25 and stride 2.

### Recorded prediction events

| Video | Airplane | Bird | Drone | Helicopter |
|---|---:|---:|---:|---:|
| drone_1.mp4 | 78 | 0 | 0 | 3 |
| drone_2.mp4 | 75 | 1 | 0 | 2 |
| drone_tracking_safe.mp4 | 55 | 0 | 12 | 10 |

The first two videos produced no drone-labelled events under their recorded settings.

The third video produced some drone predictions, showing that the model's output varied between clips.

These values count repeated prediction events, not unique objects or independently verified classification errors.

Without hand-labelled video frames, it would be incorrect to convert these figures into an external-video accuracy percentage.

## 7. CPU Performance and Memory Limitations

The project was tested on a laptop with 8 GB RAM and no dedicated training GPU.

An initial attempt to track the original drone video failed because of insufficient memory.

A reduced video version using a resolution of 320 × 180 pixels and 10 frames per second allowed tracking to complete.

This showed a practical resource limitation when processing larger videos.

It does not prove that CPU processing itself caused incorrect class predictions.

## 8. Limitations

The main limitations were:

- No independent audit of training and evaluation image overlap.
- A limited number of external videos.
- No manually labelled tracking ground truth.
- No verified ID-switch, HOTA or IDF1 results.
- No confirmed physical bird-count accuracy.
- No controlled experiment isolating the effect of object distance, motion blur or input resolution.
- CPU timing results were obtained from individual recorded runs.

The project therefore provides evidence about detector behaviour and practical CPU deployment, but it does not establish full real-world surveillance reliability.

## 9. Future Experiments — Not Completed

The following experiments were proposed but are not part of the reported measurements:

| Experiment | Purpose | Status |
|---|---|---|
| 640-pixel detection evaluation | Compare higher image resolution | Not completed |
| Confidence 0.40 and 0.60 | Test threshold sensitivity | Not completed |
| Hand-labelled tracking evaluation | Measure ID switches and count error | Not completed |
| Pretrained YOLO11n comparison | Establish an additional baseline | Not completed |
| Majority-class baseline | Provide a simple comparison | Not completed |
| Repeated CPU timing runs | Assess timing variability | Not completed |

No results have been invented for these experiments.

## 10. Supporting Evidence

The project contains or references:

- Google Colab training logs.
- Saved `args.yaml` configurations.
- Training `results.csv` files.
- Saved model checkpoints.
- CPU validation output.
- Class-level detection metrics.
- Confusion matrix.
- Tracking JSON/CSV records.
- External-video prediction results.

Not all training archives or original video files are stored in the public GitHub repository.

## 11. Research Conclusion

The 50-epoch YOLO11n checkpoint achieved an mAP50 of approximately 0.9500 on the configured AOD-4 evaluation split.

However, its predictions were less consistent on some external drone videos, where airplane labels appeared frequently.

The experiments suggest that strong performance on labelled images does not automatically establish reliable classification in unfamiliar videos.

Further testing with independently annotated videos would be needed before using this system for real-world surveillance.
