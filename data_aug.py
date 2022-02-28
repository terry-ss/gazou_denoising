import cv2
from imutils.paths import list_images
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from pathlib import Path

def data_flip(path:str):
    if path.endswith('_cleaned'):
        parent=path.removesuffix('_cleaned')+'_flip_cleaned'
    else:
        parent=path+'_flip'
    Path(parent).mkdir(exist_ok=True)
    img_files=list(list_images(path))
    cpu_number=os.cpu_count()
    n=min(1,int(cpu_number/2))
    _ = Parallel(n_jobs=n)(delayed(flips)(x,parent) for x in tqdm(img_files))
    
def flips(x,parent):
    img0=cv2.imread(x)
    
    stem=Path(x).stem;
    suffix=Path(x).suffix
    for i in range(-1,2):
        img=cv2.flip(img0,i)
        save=f'{parent}/{stem}_{i+1}{suffix}'
        cv2.imwrite(save,img)
        
if __name__=='__main__':
    data_flip('data/train')
    data_flip('data/train_cleaned')