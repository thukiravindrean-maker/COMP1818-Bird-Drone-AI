"""Fine-tune YOLO11n; never overwrite an existing checkpoint implicitly."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data', default='data/aod4_colab.yaml')  # AI-assisted fix: absolute-path YAML for Colab
    p.add_argument('--epochs',type=int,default=50)  # AI-assisted fix: longer than the old 10-epoch run
    p.add_argument('--imgsz',type=int,default=640)  # AI-assisted fix: same size as evaluation and tracking
    p.add_argument('--batch',type=int,default=16)  # AI-assisted fix: fits a Colab T4 GPU
    p.add_argument('--device',default='0',help='GPU 0 in Colab; cpu if necessary')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--name',default='T1_e50_640')
    p.add_argument('--project',default='runs/detect')
    a=p.parse_args()
    if not Path(a.data).exists(): p.error(f'Dataset YAML not found: {a.data}')
    model=YOLO('yolo11n.pt')
    model.train(data=a.data,epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,
                device=a.device,seed=a.seed,patience=a.patience,
                name=a.name,project=a.project,exist_ok=False,plots=True)
    directory=Path(model.trainer.save_dir)
    print('Run folder:',directory)
    print('Best checkpoint:',directory/'weights'/'best.pt')
    print('Inspect args.yaml and results.csv to verify completed epochs before reporting.')

if __name__=='__main__': main()
