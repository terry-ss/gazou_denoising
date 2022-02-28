# console print 
import pretty_errors
pretty_errors.configure(
    filename_display    = pretty_errors.FILENAME_EXTENDED,
    line_number_first   = True,
    lines_before        = 5,
    lines_after         = 2,
    line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
    code_color          = '  ' + pretty_errors.default_config.line_color,
    truncate_code       = True,
    display_locals      = True
)


#common
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from imutils.paths import list_images
import sys,os
from tqdm import tqdm

import sqlite3

#deep learning
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from torchsummary import summary
from pytorch_lightning.core.lightning import LightningModule
import albumentations as A
from albumentations.pytorch import ToTensorV2

    
class Noizu(Dataset):
    def __init__(self,args,stage):
        self.stage=stage
        self.image_size=args.image_size
        self.get_imgs(args.sql_path,args.table_name)
        self.if_cache=args.if_cache
        if self.if_cache:
            self.load_imgs()
        self.train_aug,self.val_aug=make_trsf(args)
        if self.stage=='train':
            self.aug_on_img0=self.train_aug
        else:
            self.aug_on_img0=self.val_aug
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self,index):
        if self.if_cache:
            img0=self.imgs[index]
            imgc=self.imgs_clean[index]
        else:
            img0=cv2.imread(self.img_files[index])
            img0=padding(img0,self.image_size)
            imgc=cv2.imread(self.img_clean_files[index],cv2.IMREAD_GRAYSCALE)/255
            imgc=padding(imgc,self.image_size)
            #imgc=cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
        img_tensor0=self.aug_on_img0(image=img0)['image'].float()
        
        img_tensorc=self.val_aug(image=imgc)['image'].float()
        return img_tensor0,img_tensorc
    
    def get_imgs(self,sql_path,table_name):
        conn = sqlite3.connect(sql_path)
        c = conn.cursor()
        cursor =c.execute(f"SELECT * FROM '{table_name}' WHERE STAGE='{self.stage}'")
        self.img_files=[]
        self.img_clean_files=[]
        for res in cursor:
            ID,PATH,STAGE=res
            self.img_files.append(PATH)
            img_file=Path(PATH)
            data_path=img_file.parent.parent
            parent=img_file.parts[-2]+'_cleaned'
            name=img_file.name
            img_clean_file=str(data_path/parent/name)
            self.img_clean_files.append(img_clean_file)
        
    def load_imgs(self):
        self.imgs=[]
        self.imgs_clean=[]
        print('loading all images...')
        for img1,img2 in tqdm(zip(self.img_files,self.img_clean_files)):
            img1=cv2.imread(img1);
            img1=padding(img1,self.image_size)
            img2=cv2.imread(img2,cv2.IMREAD_GRAYSCALE)/255
            img2=padding(img2,self.image_size)
            if img1 is None or img2 is None:
                raise FileNotFoundError(img1+' '+img2)
            #img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            self.imgs.append(img1)
            self.imgs_clean.append(img2)
            
def padding(img,image_size):
    shape=img.shape
    ww,hh=image_size
    if img.dtype==np.uint8:
        c=255
    elif img.dtype==float:
        c=1
    else:
        raise RuntimeError(img.dtype)
    if len(shape)==2:
        h,w=shape
        hc=max(hh-h,0);wc=max(ww-w,0)
        img=np.pad(img,((0,hc),(0,wc)),constant_values=c)
    elif len(shape)==3:
        h,w,_=shape
        hc=max(hh-h,0);wc=max(ww-w,0)
        img=np.pad(img,((0,hc),(0,wc),(0,0)),constant_values=c)
    else:
        raise RuntimeError
    return img
            
def make_trsf(args=None,imgsize=None):
    if imgsize:
        w,h=imgsize
    else:
        w,h=args.image_size
    train_aug = A.Compose([
        A.RandomShadow(num_shadows_upper=1,),
        A.ISONoise(),
        A.ImageCompression(quality_lower=90, quality_upper=100),
        #A.ToGray(),
        A.CoarseDropout(max_height=int(h * 0.01), max_width=int(w * 0.01), max_holes=20, p=0.5),
        ToTensorV2()
    ])

    val_aug = A.Compose([
            ToTensorV2()
        ])
    return train_aug,val_aug
   
  
class Denoising(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.m=nn.Sequential(
            nn.Conv2d(3,96,3,padding='same'),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Conv2d(96,96,3,padding='same'),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Conv2d(96,96,3,padding='same'),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Conv2d(96,96,3,padding='same'),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Conv2d(96,96,3,padding='same'),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Conv2d(96,1,3,padding='same'),
            )
            
    def forward(self,x):
        y=self.m(x)
        return y
        
    def init_weight(self,checkpoint):
        checkpoint=torch.load(checkpoint,map_location='cpu')
        state_dict=checkpoint['state_dict']
        _dict={}
        for key in state_dict.keys():
            _dict[key.removeprefix('model.')]=state_dict[key]
        #self.to('cpu')
        self.load_state_dict(_dict)
        print('pretrained model loaded')
    
    @staticmethod
    def rename_key(key,new_key):
        pass
 
def build_model(args,config):
    model = Denoising(config)
    if args.pretrained:
        if not Path(args.pretrained).is_file():
            raise FileNotFoundError(args.pretrained)
        model.init_weight(args.pretrained)
    
    summary(model,(3,*args.image_size), device="cpu")   
    
    return model

class LitModel(LightningModule):
    def __init__(self,args,config):
        super().__init__()
        self.save_hyperparameters()
        self.lr=args.lr
        self.batch_size=args.bs
        self.model=build_model(args,config)
        self.args=args
        self.best_acc={'train':0,'val':0,'test':0}
        self.metric=Meassure()
        self.loss=nn.SmoothL1Loss()
        self.score=nn.L1Loss()
        
    def forward(self,x):
        y=self.model(x)
        return y 
    
    def training_step(self, batch, batch_idx):    
        img0,img1=list(batch)
        batch_size = img0.size(0)
        y0=self(img0/255);
        loss=self.loss(y0,img1)
        res = {'loss': loss,}
        #self.metric.update_loss(loss,pbatch_sizeair)
        self.log('train_step_loss',loss,batch_size=batch_size)
        return res
        
    
    def validation_step(self, batch, batch_idx):
        img0,img1=list(batch)
        batch_size = img0.size(0)
        y0=self(img0/255);
        loss=self.loss(y0,img1)
        score=self.score(y0,img1)
        res = {'loss': loss,'score':score}
        self.metric.update_loss(loss,batch_size)
        self.metric.update_score(score,batch_size)
        self.log('val_step_loss',loss,batch_size=batch_size)
        return res
    
    def validation_epoch_end(self,res):
        res=self.metric.compute()
        epoch_loss=res['loss']
        score=res['score']
        self.log('val_epoch_loss',epoch_loss)
        self.log('val_score',score)
        self.metric.reset() 
           
    def configure_optimizers(self):
        parameters=filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.optim=='SGD':
            optimizer = getattr(optim,self.args.optim)(parameters,
                                      lr=self.lr,momentum=0.9)
        else:
            optimizer = getattr(optim,self.args.optim)(parameters,
                                      lr=self.lr)
        #optimizer=optim.Adam(self.model.parameters(),lr=self.lr)
        if self.args.scheduler=='CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.T_max)
            saiteki={'optimizer':optimizer,'lr_scheduler':scheduler}
        elif self.args.scheduler=='ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            saiteki={'optimizer':optimizer,'lr_scheduler':scheduler,'monitor':'val_epoch_loss'}
        elif self.args.scheduler=='ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=self.args.gamma)
            saiteki={'optimizer':optimizer,'lr_scheduler':scheduler}
        else:
            saiteki={'optimizer':optimizer}
        return saiteki
        
        
    def prepare_data(self):
        print('')
        
    def train_dataloader(self):
        train_dataset=Noizu(self.args,'train')
        train_loader = DataLoader(train_dataset,
           batch_size=self.batch_size, 
           num_workers=4, 
           pin_memory=False, 
           shuffle=True,)
                                                 
        return train_loader
        
    def val_dataloader(self):
        val_dataset=Noizu(self.args,'val')
        val_loader = DataLoader(val_dataset,
           batch_size=self.batch_size, 
           num_workers=4, 
           pin_memory=False, 
           shuffle=False,)
        return val_loader 
  
  
class Meassure():
    def __init__(self):
        super().__init__()
        self.loss=0
        self.score=0
        self.batch=0

    def update_loss(self, loss,batch_size): 
        self.loss+=loss*batch_size
        self.batch+=batch_size

    def update_score(self,score,batch_size):
        self.score+=score*batch_size
        self.batch+=batch_size
        
    def compute(self):
        self.loss/=self.batch
        self.score/=self.batch
        return {'loss':self.loss, 'score':self.score}
    
    def reset(self):
        self.__init__()

