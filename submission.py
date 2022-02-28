#import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from imutils.paths import list_images
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
import polars as pd

from module import LitModel,make_trsf,padding


def inference(check_point,hparam):
    trainer=pl.Trainer(devices='1',
        accelerator="auto",
        callbacks=[RichProgressBar(),])
    model=LitModel.load_from_checkpoint(
        checkpoint_path=check_point,
        hparams_file=hparam,
        map_location=None,
        )
    img_files=list(list_images('data/test/'))
    # cpu_number=os.cpu_count()
    # n=min(2,int(cpu_number/2))
    # print(f'using cpu cores:{n} of {cpu_number}')
    _,trsf=make_trsf(imgsize=(420,540))
    model.eval()
    with torch.inference_mode():
        for x in tqdm(img_files):
            single_predict(x,model,trsf)
    return 

def single_predict(x,model,trsf):
    image0=cv2.imread(x,1)/255
    image=padding(image0,(420,540))
    x_tensor=trsf(image=image)['image'][None].float()
    y=model(x_tensor)[0]
    y=y.permute(1,2,0).detach().numpy().squeeze()
    stem=Path(x).stem
    h0,w0=image0.shape[:2]
    #y=cv2.resize(y,(w0,h0));
    y=y[:h0,:w0]
    if h0==540 and w0==540:
        raise RuntimeError
    with open(f'data/test_cleaned/{stem}.npy','wb+') as f:
        np.save(f,y)
 
def check_length(npys):
    length=0
    for npy in npys:
        imgid=Path(npy).stem
        with open(npy,'rb') as f:
            array=np.load(f)
        h,w=array.shape
        length+=h*w
    if length== 14230080:
        print('length of result ok')
    else:
        raise RuntimeError(length)
    
def submission(banngo):
    ids=[];vals=[]
    df=pd.read_csv('data/sampleSubmission.csv')
    npys=tqdm(list(Path('data/test_cleaned/').glob('*.npy')))
    check_length(npys)
    for npy in npys:
        imgid=Path(npy).stem
        with open(npy,'rb') as f:
            array=np.load(f)
        h,w=array.shape
        #print(h,w,npy);exit()
        for r in range(h):
            for c in range(w):
                ids.append(str(imgid)+'_'+str(r + 1)+'_'+str(c + 1))
                vals.append(round(float(array[r, c]),5))
    
    df=pd.DataFrame({'id': ids, 'value': vals})
    try:
        df.to_csv(f'data/submission_{banngo}.csv',index=False)
    except:
        df.to_csv(f'data/submission_{banngo}.csv')
    print('Results saved to submission.csv!')

def pick_history(banngo):
    log_vnm=f'lightning_logs/version_{banngo}'
    check_point=list(Path(f'{log_vnm}/checkpoints').glob('*.ckpt'))[0]
    hparam=f'{log_vnm}/hparams.yaml'
    inference(check_point,hparam)
    
if __name__=='__main__':
    banngo=input('v_num:')
    pick_history(banngo)   
    submission(banngo)
    
    
    
