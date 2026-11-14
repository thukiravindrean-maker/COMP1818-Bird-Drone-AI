"""Held-out test evaluation, COCO pretrained baseline and majority-class baseline.
Requires actual AOD-4 test labels/images and fine-tuned weights; writes no fabricated scores.
"""
import argparse,csv,shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
from utils import match,yolo_to_xyxy

def resolved_data(yaml_file):
    cfg=yaml.safe_load(Path(yaml_file).read_text(encoding='utf8'))
    root=Path(cfg.get('path','.'))
    if not root.is_absolute():root=(Path(yaml_file).resolve().parent/root).resolve()
    names=cfg['names']
    if isinstance(names,list):names={i:x for i,x in enumerate(names)}
    names={int(i):str(x) for i,x in names.items()}
    test=root/cfg['test']
    if not test.exists():raise FileNotFoundError(f'Missing test images: {test}')
    images=sorted(x for x in test.rglob('*') if x.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'))
    if not images:raise RuntimeError(f'No test images found under {test}')
    return names,images,root,cfg

def label_path(image):
    parts=list(image.parts)
    locations=[i for i,p in enumerate(parts) if p.lower()=='images']
    if not locations:raise ValueError(f'Expected an images/ directory: {image}')
    parts[locations[-1]]='labels'
    return Path(*parts).with_suffix('.txt')

def gt_for(image):
    h,w=cv2.imread(str(image)).shape[:2]
    file=label_path(image)
    if not file.exists():raise FileNotFoundError(f'Missing ground truth label file: {file}')
    result=[]
    for line in file.read_text().splitlines():
        if not line.strip():continue
        cells=line.split()
        if len(cells)<5:raise ValueError(f'Bad label: {file}: {line}')
        result.append((int(float(cells[0])),yolo_to_xyxy(*map(float,cells[1:5]),w,h)))
    return result

def baseline_counts(model,images,names,conf,imgsz,remap=None):
    ids=sorted(names);tp={i:0 for i in ids}; fp={i:0 for i in ids};fn={i:0 for i in ids}
    for image in images:
        gt=gt_for(image)
        res=model.predict(source=str(image),conf=conf,imgsz=imgsz,device='cpu',verbose=False)[0]
        pred=[]
        for box in res.boxes:
            cls=int(box.cls.item())
            if remap is not None:cls=remap.get(cls,-1)
            if cls in names:pred.append((cls,tuple(box.xyxy[0].tolist())))
        for cls in ids:
            gb=[xy for c,xy in gt if c==cls]
            pb=[xy for c,xy in pred if c==cls]
            hits=len(match(gb,pb,0.5))
            tp[cls]+=hits;fp[cls]+=len(pb)-hits;fn[cls]+=len(gb)-hits
    return [{'class_id':i,'class_name':names[i],
             'precision':tp[i]/(tp[i]+fp[i]) if tp[i]+fp[i] else 0.0,
             'recall':tp[i]/(tp[i]+fn[i]) if tp[i]+fn[i] else 0.0,
             'tp':tp[i],'fp':fp[i],'fn':fn[i]} for i in ids]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/aod4_local.yaml')  # AI-assisted fix: absolute-path YAML
    p.add_argument('--model',default='models/best.pt')
    p.add_argument('--imgsz',type=int,default=640)  # AI-assisted fix: same size as training and tracking
    p.add_argument('--conf',type=float,default=0.25)
    p.add_argument('--pretrained',default='yolo11n.pt')
    p.add_argument('--skip-pretrained',action='store_true')
    a=p.parse_args()
    if not Path(a.model).is_file():p.error(f'Missing fine-tuned checkpoint: {a.model}')
    names,images,root,cfg=resolved_data(a.data)
    out=Path('results');out.mkdir(exist_ok=True)
    counts={i:0 for i in names}
    for image in images:
        for cls,_ in gt_for(image):counts[cls]+=1
    total=sum(counts.values());majority=max(counts,key=counts.get)
    pd.DataFrame([{'baseline':'B0_majority_class','majority_class':names[majority],
                   'object_class_accuracy':counts[majority]/total if total else np.nan,
                   'total_gt_objects':total,'note':'Classification-only prior; not detection mAP or tracking'}]).to_csv(out/'baselines.csv',index=False)
    model=YOLO(a.model)
    # Ultralytics test-split AP scores (its internal confidence defaults suit AP computation).
    metrics=model.val(data=a.data,split='test',imgsz=a.imgsz,device='cpu',plots=True,
                      project=str(out),name='test_validation',exist_ok=True)
    rows=[]
    for i,cls in enumerate(metrics.box.ap_class_index):
        cls=int(cls)
        rows.append({'class_id':cls,'class_name':names[cls],
                     'precision':float(metrics.box.p[i]),'recall':float(metrics.box.r[i]),
                     'ap50':float(metrics.box.ap50[i]),'ap50_95':float(metrics.box.ap[i])})
    pd.DataFrame(rows).to_csv(out/'detection_test.csv',index=False)
    matrix=out/'test_validation'/'confusion_matrix.png'
    if matrix.exists():shutil.copy2(matrix,out/'confusion_matrix.png')
    # Matched, threshold-specific P/R for a fair COCO vs fine-tuned comparison.
    results=[]
    for label,net,remap in [('finetuned',model,None)]:
        for entry in baseline_counts(net,images,names,a.conf,a.imgsz,remap):
            results.append({'baseline':label,'conf':a.conf,'imgsz':a.imgsz,**entry})
    if not a.skip_pretrained:
        mapping={}
        for coco_id,coco_name in YOLO(a.pretrained).names.items():
            if str(coco_name).lower() in ('bird','airplane'):
                for class_id,name in names.items():
                    if name.lower()==str(coco_name).lower():mapping[int(coco_id)]=class_id
        # AI-assisted fix: show the mapping and skip (not silently zero) if no class names matched
        print('COCO -> dataset class mapping:',mapping)
        if not mapping:
            print('WARNING: no COCO class matched your dataset names (expected "bird"/"airplane"). COCO baseline skipped - check spelling in the YAML names.')
        else:
            for entry in baseline_counts(YOLO(a.pretrained),images,names,a.conf,a.imgsz,mapping):
                results.append({'baseline':'COCO_unfinetuned','conf':a.conf,'imgsz':a.imgsz,**entry})
    pd.DataFrame(results).to_csv(out/'matched_pr_baselines.csv',index=False)
    print('Evaluation complete:',len(images),'images. Outputs in results/.')
    print('COCO pretrained lacks drone/helicopter categories; interpret zero recall accordingly.')

if __name__=='__main__':main()
