"""Exploratory pixel-motion summaries; not physical speed or class predictions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tracks',required=True)
    p.add_argument('--fps',type=float,required=True,help='Original video FPS')
    p.add_argument('--slow-threshold',type=float,default=10.0,help='px/second; sensitivity-dependent')
    p.add_argument('--output',default=None)
    a=p.parse_args()
    if a.fps<=0:p.error('--fps must be positive')
    df=pd.read_csv(a.tracks)
    records=[]
    if not df.empty:
        df=df.dropna(subset=['track_id'])
        for tid,group in df.groupby('track_id'):
            group=group.sort_values('frame').drop_duplicates('frame')
            t=group.frame.to_numpy(dtype=float)
            x=(group.x1.to_numpy()+group.x2.to_numpy())/2
            y=(group.y1.to_numpy()+group.y2.to_numpy())/2
            dt=np.diff(t)/a.fps
            dx,dy=np.diff(x),np.diff(y)
            valid=dt>0
            vx=dx[valid]/dt[valid];vy=dy[valid]/dt[valid]
            speeds=np.hypot(vx,vy)
            headings=np.arctan2(vy,vx)
            # AI-assisted fix: only count turns between two moving steps (direction of a near-still box is jitter)
            moving=speeds>=a.slow_threshold
            turns=np.abs(np.rad2deg(np.angle(np.exp(1j*np.diff(headings)))))[moving[:-1]&moving[1:]]
            records.append({'track_id':int(tid),'class_name':group.class_name.mode().iloc[0],
                            'n_frames':len(group),'mean_speed_px_s':float(np.mean(speeds)) if len(speeds) else np.nan,
                            'max_speed_px_s':float(np.max(speeds)) if len(speeds) else np.nan,
                            'mean_abs_turn_deg':float(np.mean(turns)) if len(turns) else np.nan,
                            'low_motion_fraction':float(np.mean(speeds<a.slow_threshold)) if len(speeds) else np.nan})
    output=Path(a.output or f'results/motion/{Path(a.tracks).stem}_motion.csv')
    output.parent.mkdir(parents=True,exist_ok=True)
    cols=['track_id','class_name','n_frames','mean_speed_px_s','max_speed_px_s','mean_abs_turn_deg','low_motion_fraction']
    pd.DataFrame(records,columns=cols).to_csv(output,index=False)
    print('Wrote',output,'tracks',len(records))

if __name__=='__main__':main()
