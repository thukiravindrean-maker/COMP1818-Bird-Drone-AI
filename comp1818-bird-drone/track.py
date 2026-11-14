"""CPU video tracking with frame-level timing and traceable settings."""
import argparse, csv, json, shutil, statistics, time
from pathlib import Path
import cv2
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_yaml

FIELDS=['frame','track_id','class_id','class_name','confidence','x1','y1','x2','y2','ms']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--clip',required=True)
    p.add_argument('--model',default='models/best.pt')
    p.add_argument('--tracker',default='bytetrack.yaml')
    p.add_argument('--conf',type=float,default=0.25)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--stride',type=int,default=1)  # AI-assisted fix: trackers expect consecutive frames
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',default=None)
    a=p.parse_args()
    for file in (a.clip,a.model):
        if not Path(file).is_file(): p.error(f'Missing input: {file}')
    if a.stride<1:p.error('--stride must be >= 1')
    cap=cv2.VideoCapture(a.clip)
    if not cap.isOpened():p.error('Cannot open video')
    fps=cap.get(cv2.CAP_PROP_FPS)
    frames_total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model=YOLO(a.model)
    name=Path(a.clip).stem
    tracker_name=Path(a.tracker).stem
    stem=f'{name}_{tracker_name}_s{a.imgsz}_c{a.conf:g}_v{a.stride}'
    out=Path(a.output or f'results/tracks/{stem}.csv')
    out.parent.mkdir(parents=True,exist_ok=True)
    logs=[]; rows=[]; processed=0; frame_index=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frame_index+=1  # MOT convention: original frame numbers start at 1
            if (frame_index-1)%a.stride:continue
            start=time.perf_counter()
            res=model.track(source=frame,persist=True,tracker=a.tracker,
                            conf=a.conf,imgsz=a.imgsz,device=a.device,verbose=False)[0]
            elapsed=(time.perf_counter()-start)*1000
            processed+=1
            # Separate timing log, including empty frames (CSV only contains box rows).
            logs.append((frame_index,elapsed))
            if res.boxes is not None:
                for box in res.boxes:
                    cls=int(box.cls.item())
                    tid=int(box.id.item()) if box.id is not None else ''
                    xyxy=[round(float(v),3) for v in box.xyxy[0].tolist()]
                    rows.append([frame_index,tid,cls,str(model.names[cls]),round(float(box.conf.item()),5),*xyxy,round(elapsed,3)])
    finally:cap.release()
    with out.open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(FIELDS);writer.writerows(rows)
    times=out.with_suffix('.timing.csv')
    with times.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['frame','ms']);w.writerows(logs)
    metadata={'clip':str(Path(a.clip).resolve()),'model':str(Path(a.model).resolve()),
              'tracker':a.tracker,'imgsz':a.imgsz,'conf':a.conf,'stride':a.stride,
              'video_fps':fps,'video_frame_count':frames_total,'processed_frames':processed,
              'mean_ms':sum(x[1] for x in logs)/processed if processed else None,
              # AI-assisted fix: median after 5 warm-up frames is the fairer speed figure
              'median_ms_after_warmup':statistics.median([x[1] for x in logs[5:]] or [x[1] for x in logs]) if logs else None,
              'processed_fps':1000*processed/sum(x[1] for x in logs) if logs else None,
              'frame_convention':'1-based original video frame index'}
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    track_cfg=Path('results/trackers');track_cfg.mkdir(parents=True,exist_ok=True)
    # Resolve stock Ultralytics YAML, or a supplied custom YAML. OC-SORT is not
    # shipped in all Ultralytics versions: require an actual compatible config.
    try:
        src=Path(check_yaml(a.tracker))
        shutil.copy2(src,track_cfg/(stem+'_'+src.name))
    except Exception as exc:
        print('WARNING: tracker config could not be archived:',exc)
    print('Wrote',out,'and',times,'processed',processed,'mean ms',metadata['mean_ms'])

if __name__=='__main__':
    main()
