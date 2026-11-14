"""Final reproducibility checks for COMP1818 (run locally, with TRUSTED YOLO checkpoints).

Example Windows PowerShell:
 python run_final_checks.py --data data.yaml --weights models/best_10epochs.pt models/best_50epochs.pt --videos clips/drone_1.mp4 clips/drone_2.mp4

Run from project root. Videos should be original unsampled video clips, not annotated output.
Requires: pip install ultralytics opencv-python torch
"""
import argparse,csv,json,time,sys
from pathlib import Path
from collections import Counter

FIELDS = ['checkpoint','class_id','class_name','precision','recall','mAP50','mAP50_95']

def save_csv(path,rows,fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def checkpoint_metadata(path,torch):
    d={'file':str(path),'checkpoint_best_epoch_index':None,'best_checkpoint_epoch_number':None,'configured_epochs':None,'optimizer':None,'imgsz':None,'batch':None,'training_args':{}}
    try:
        # weights_only=False can execute arbitrary code; only inspect your own trusted weights.
        ckpt=torch.load(str(path),map_location='cpu',weights_only=False)
        if isinstance(ckpt,dict):
            epoch=ckpt.get('epoch'); d['checkpoint_best_epoch_index']=epoch
            if isinstance(epoch,int) and epoch>=0: d['best_checkpoint_epoch_number']=epoch+1
            args=ckpt.get('train_args') or ckpt.get('args') or {}
            if hasattr(args,'__dict__'): args=vars(args)
            if isinstance(args,dict):
                d['training_args']={str(k):str(v) for k,v in args.items()}
                d['configured_epochs']=args.get('epochs')
                d['optimizer']=args.get('optimizer')
                d['imgsz']=args.get('imgsz');d['batch']=args.get('batch')
            if 'optimizer' in ckpt and d['optimizer'] is None: d['optimizer']='checkpoint holds optimiser state; read training args'
    except Exception as e: d['metadata_error']=str(e)
    return d

def validate(model,weight,data,output):
    print(f'\nVALIDATE: {weight} at default validation confidence')
    metrics=model.val(data=str(data),split='test',imgsz=416,device='cpu',batch=1,plots=True,project=str(output),name='val_'+weight.stem,exist_ok=True)
    names=model.names; box=metrics.box
    ids=[int(i) for i in box.ap_class_index]
    rows=[]
    for j,i in enumerate(ids):
        rows.append(dict(checkpoint=weight.name,class_id=i,class_name=str(names[i]),precision=float(box.p[j]),recall=float(box.r[j]),mAP50=float(box.ap50[j]),mAP50_95=float(box.ap[j])))
    overall=dict(checkpoint=weight.name,precision=float(box.mp),recall=float(box.mr),mAP50=float(box.map50),mAP50_95=float(box.map))
    return rows,overall

def video_run(model,weight,video,imgsz,output,cv2):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open original video {video}')
    # Restart tracker from scratch, matching every 2nd original frame, up to 80 processed frames.
    # Using a separate fresh YOLO object avoids carrying state across videos.
    frame_index=0; sampled=0; frame_hits=0; drone_hits=0; events=[]; timings=[]
    while sampled<80:
        ok,frame=cap.read()
        if not ok: break
        frame_index+=1
        if frame_index%2==0: continue
        sampled+=1
        t=time.perf_counter()
        res=model.track(frame,persist=True,tracker='bytetrack.yaml',imgsz=imgsz,conf=.25,device='cpu',verbose=False)[0]
        timings.append((time.perf_counter()-t)*1000)
        boxes=res.boxes
        if boxes is None or len(boxes)==0: continue
        frame_hits+=1
        names=res.names
        labels=[str(names[int(x)]) for x in boxes.cls.cpu().tolist()]
        if 'drone' in labels: drone_hits+=1
        for j,(cls,cf,coords) in enumerate(zip(labels,boxes.conf.cpu().tolist(),boxes.xyxy.cpu().tolist())):
            track_id=None
            if boxes.id is not None: track_id=int(boxes.id[j].item())
            events.append(dict(checkpoint=weight.name,video=video.name,imgsz=imgsz,frame=frame_index,sampled_frame=sampled,track_id=track_id,pred_class=cls,confidence=round(float(cf),5),x1=round(coords[0],2),y1=round(coords[1],2),x2=round(coords[2],2),y2=round(coords[3],2)))
    cap.release()
    counts=Counter(e['pred_class'] for e in events)
    summary={'checkpoint':weight.name,'video':video.name,'imgsz':imgsz,'confidence':.25,'stride':2,'sampled_frames':sampled,'frames_with_any_prediction':frame_hits,'frames_with_drone_prediction':drone_hits,'prediction_events':len(events),'prediction_classes':dict(counts),'mean_processing_ms':round(sum(timings)/len(timings),2) if timings else None,'NOTE':'Prediction coverage is not ground truth object detection recall. Manual visual checking is still needed.'}
    return events,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True,help='AOD-4 data.yaml with test path')
    p.add_argument('--weights',type=Path,nargs='+',required=True)
    p.add_argument('--videos',type=Path,nargs='*',default=[],help='Original drone_1 and drone_2 video paths')
    p.add_argument('--output',type=Path,default=Path('final_evidence_checks'))
    p.add_argument('--skip-validation',action='store_true')
    p.add_argument('--skip-video',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import torch,cv2
    from ultralytics import YOLO
    meta=[];classrows=[];overall=[];events=[];summaries=[]
    for weight in a.weights:
        if not weight.is_file(): print('Missing model:',weight,file=sys.stderr);continue
        print('CHECKPOINT:',weight)
        item=checkpoint_metadata(weight,torch); meta.append(item)
        print(' Best checkpoint epoch:',item['best_checkpoint_epoch_number'],' configured epochs:',item['configured_epochs'],' optimiser:',item['optimizer'])
        if not a.skip_validation:
            if not a.data.exists(): print('Data YAML missing:',a.data,file=sys.stderr)
            else:
                try:
                    model=YOLO(str(weight)); rows,summary=validate(model,weight,a.data,a.output);classrows+=rows;overall.append(summary)
                except Exception as e: print('Validation failed:',weight,e,file=sys.stderr)
        if not a.skip_video and weight==a.weights[-1]:
            for video in a.videos:
                if not video.is_file(): print('Missing video:',video,file=sys.stderr);continue
                try:
                    model=YOLO(str(weight)) # reset tracker for every video
                    ev,summary=video_run(model,weight,video,416,a.output,cv2);events+=ev;summaries.append(summary)
                    print('VIDEO 416px:',json.dumps(summary,indent=2))
                except Exception as e: print('Video failed:',video,e,file=sys.stderr)
    (a.output/'checkpoint_metadata.json').write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
    (a.output/'validation_overall.json').write_text(json.dumps(overall,indent=2),encoding='utf-8')
    (a.output/'video_416_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    save_csv(a.output/'validation_four_classes.csv',classrows,FIELDS)
    save_csv(a.output/'video_416_predictions.csv',events,['checkpoint','video','imgsz','frame','sampled_frame','track_id','pred_class','confidence','x1','y1','x2','y2'])
    print('\nDone. Output directory:',a.output.resolve())
    print('NOTE: best checkpoint epoch is NOT proof of last completed training epoch; check last.pt/results.csv or train logs.')

if __name__=='__main__': main()
