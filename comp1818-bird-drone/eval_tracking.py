"""IoU>=0.5 class-aware MOT comparison; run once per clip/tracker CSV."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from utils import match

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gt',required=True,help='MOT gt.txt; 1-based frame and IDs')
    p.add_argument('--tracks',required=True)
    p.add_argument('--output',default='results/tracking_eval.csv')
    p.add_argument('--iou',type=float,default=0.5)
    p.add_argument('--class-map',default=None,help='Optional JSON mapping MOT class id -> model class name; required for class-aware scoring')
    a=p.parse_args()
    if a.class_map is None:p.error('--class-map required: JSON map e.g. {"1":"bird","2":"drone"}; MOT class IDs depend on your annotation export')
    # AI-assisted fix: accept a .json file as well (Windows PowerShell strips quotes from inline JSON)
    raw=Path(a.class_map).read_text(encoding='utf8') if a.class_map.lower().endswith('.json') else a.class_map
    class_map={int(k):str(v).lower() for k,v in json.loads(raw).items()}
    cols=['frame','id','x','y','w','h','valid','class_id','visibility']
    gt=pd.read_csv(a.gt,header=None,comment='#')
    if len(gt.columns)<7:p.error('Expected MOT ground truth >=7 columns')
    gt.columns=cols[:len(gt.columns)]
    if 'class_id' not in gt: p.error('Ground truth needs a class id in column 8')
    gt=gt[(gt.valid>0)&gt.class_id.isin(class_map)]
    gt=gt.copy();gt['class_name']=gt.class_id.map(class_map)
    tracks=pd.read_csv(a.tracks)
    if not tracks.empty:tracks=tracks.dropna(subset=['track_id'])
    timing_path=Path(a.tracks).with_suffix('.timing.csv')
    times=pd.read_csv(timing_path) if timing_path.exists() else pd.DataFrame()
    if times.empty: p.error('Missing frame timing CSV; it is needed to identify processed frames with zero detections')
    # Evaluate only the identical sampled original frame indices.
    gt=gt[gt.frame.astype(int).isin(set(times.frame.astype(int)))].copy()
    matched_previously={};active={}; switches=0;fragments=0
    gt_keys={(int(z.class_id),int(z.id)) for z in gt.itertuples()}
    # AI-assisted fix: count distinct tracker IDs only (a class flip is still one track)
    pred_ids=set(tracks.track_id.astype(int)) if not tracks.empty else set()
    # Count fragments only on an absence between successive successful matches.
    frames=sorted(set(times.frame.astype(int)))
    for frame in frames:
        g=gt[gt.frame==frame].reset_index(drop=True)
        pr=tracks[tracks.frame==frame].reset_index(drop=True)
        pairs=[]
        for label in sorted(set(g.class_name)|set(pr.class_name)):
            g_idx=[i for i in range(len(g)) if str(g.iloc[i].class_name).lower()==label]
            p_idx=[j for j in range(len(pr)) if str(pr.iloc[j].class_name).lower()==label]
            gb=[[g.iloc[i].x,g.iloc[i].y,g.iloc[i].x+g.iloc[i].w,g.iloc[i].y+g.iloc[i].h] for i in g_idx]
            pb=[[pr.iloc[j].x1,pr.iloc[j].y1,pr.iloc[j].x2,pr.iloc[j].y2] for j in p_idx]
            pairs.extend((g_idx[i],p_idx[j]) for i,j in match(gb,pb,a.iou))
        this_matched=set()
        for i,j in pairs:
            key=(int(g.iloc[i].class_id),int(g.iloc[i].id));track_id=int(pr.iloc[j].track_id)
            if key in matched_previously and matched_previously[key]!=track_id:switches+=1
            if key in active and not active[key]:fragments+=1
            matched_previously[key]=track_id;active[key]=True;this_matched.add(key)
        for z in g.itertuples():
            key=(int(z.class_id),int(z.id))
            if key in active and key not in this_matched:active[key]=False
    gt_count=len(gt_keys);pred_count=len(pred_ids)
    meta_path=Path(a.tracks).with_suffix('.json')
    metadata=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    # AI-assisted fix: median ms after skipping 5 warm-up frames (first calls are much slower)
    median_ms=float(times.ms.iloc[5:].median()) if len(times)>5 else float(times.ms.median())
    # AI-assisted fix: clip name is the folder above gt/ (labels_gt/<clip>/gt/gt.txt)
    result={'clip':Path(a.gt).parents[1].name,
            'tracks_file':Path(a.tracks).name,'tracker':metadata.get('tracker','unknown'),
            'imgsz':metadata.get('imgsz',np.nan),'conf':metadata.get('conf',np.nan),
            'stride':metadata.get('stride',np.nan),'gt_objects':gt_count,'pred_ids':pred_count,
            'id_switches':switches,'fragments':fragments,
            'count_error_pct':abs(pred_count-gt_count)*100/gt_count if gt_count else np.nan,
            'median_ms':median_ms,'timed_frames':len(times),'iou_threshold':a.iou}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    # AI-assisted fix: re-running the same clip + tracks file replaces its old row instead of duplicating it
    table=pd.DataFrame([result])
    if output.exists():
        old=pd.read_csv(output)
        if {'clip','tracks_file'}<=set(old.columns):
            old=old[~((old['clip']==result['clip'])&(old['tracks_file']==result['tracks_file']))]
        table=pd.concat([old,table],ignore_index=True)
    table.to_csv(output,index=False)
    print(result)

if __name__=='__main__':main()
