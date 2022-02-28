from rich import print,console
console = console.Console()
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint,RichProgressBar
from easydict import EasyDict as edict
import gc
import torch

from module import LitModel,prepare_data

def get_args():
    args=edict()
    args.image_size=(540, 420) # w,h
    args.bs=4
    args.pretrained=None #'lightning_logs/version_*/checkpoints/*.ckpt'
    args.checkpoint='None' 
    args.lr=1e-3 #0.0083
    args.auto_lr=True
    args.epoch=1000
    args.sql_path='data/train_source.db'
    args.table_name='origin_flip'
    args.if_cache=True
    args.scheduler=None #'ExponentialLR',CosineAnnealingLR
    args.gamma=0.1
    args.optim='AdamW'
    args.T_max=300
    return args
   
def get_config():
    config=edict()
    config.c1=8
    config.c2=64
    config.k=5
  
    return config
    
def hunt(args):  
    config=get_config()
    model=LitModel(args,config)
    
    checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min",
        monitor="val_score",save_last=False,save_top_k=1)

    if args.checkpoint.lower()=='none':
        check_path=None
    else:
        check_path=args.checkpoint

    trainer=pl.Trainer(gpus=[1],
        auto_select_gpus=True,
        #accumulate_grad_batches=8,
        strategy=None,
        benchmark=False,
        num_processes=2,
        amp_backend="native",
        max_epochs=args.epoch,
        auto_lr_find=args.auto_lr,
        auto_scale_batch_size=False,
        precision=32,
        deterministic=True,
        enable_model_summary=False,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            RichProgressBar(),
            
        ],  
        )
    train_loader,val_loader=prepare_data(args)
    if args.auto_lr:
        trainer.tune(model,train_loader,val_loader)
    #gc.collect()
    torch.cuda.empty_cache()
    trainer.fit(model,train_loader,val_loader,ckpt_path=check_path)
    print(checkpoint_callback.best_model_path)
    
if __name__=='__main__':
    seed=42   
    pl.seed_everything(seed, workers=True)
    args=get_args()
    print(args)
    
    hunt(args)

