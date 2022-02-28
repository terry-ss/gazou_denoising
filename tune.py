from rich import print,console
console = console.Console()
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint,RichProgressBar
from easydict import EasyDict as edict
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
    
from module import LitModel,prepare_data

def get_args():
    args=edict()
    args.image_size=(540, 420) # w,h
    args.bs=4
    #args.mname='efficientnet_b0'
    #args.num_classes=1000
    args.pretrained=None #'lightning_logs/version_41/checkpoints/epoch=942-step=122589.ckpt'
    args.checkpoint='None' #,'lightning_logs/version_55/checkpoints/epoch=90-step=11829.ckpt'
    args.lr=6.91e-3 #0.0083
    args.auto_lr=False
    args.epoch=300
    args.sql_path='data/train_source.db'
    args.table_name='2021_12_29_3'
    #args.image_path=
    args.if_cache=True
    args.scheduler='CosineAnnealingLR'
    args.gamma=0.1
    args.optim='Adam'
    return args
 
def get_config():
    config=edict()
    config.c1=tune.sample_from(lambda _:tune.randint(8,64))
    config.c2=tune.sample_from(lambda _:tune.randint(8,64))
    config.k=tune.choice([3,5,7])
    # config={
        # 'c1':tune.sample_from(lambda _:2**np.random.int(3,6)),
        # 'c2':tune.sample_from(lambda _:2**np.random.int(3,6)),
        # 'k':tune.choice([3,5,7])}
    return config
 
def hunt_tune(config,args,train_loader,val_loader):   
    model=LitModel(args,config)

    if args.checkpoint.lower()=='none':
        check_path=None
    else:
        check_path=args.checkpoint
    checkpoint_callback=TuneReportCallback(
        {
            "loss": "val_score",
            "score": "val_epoch_loss"
        },
        on="validation_end")
    trainer=pl.Trainer(gpus=1,
        auto_select_gpus=True,
        
        strategy=None,
        benchmark=False,
        num_processes=2,
        amp_backend="native",
        max_epochs=args.epoch,
        auto_lr_find=args.auto_lr,
        precision=32,
        deterministic=True,
        callbacks=[
            checkpoint_callback,
            
            RichProgressBar(),
        ],  
        )
    
    if args.auto_lr:
        trainer.tune(model,train_loader,val_loader)
    trainer.fit(model,train_loader,val_loader,ckpt_path=check_path)
    #print(checkpoint_callback.best_model_path)
    
def tune_with_ray(args):
    config=get_config()
    scheduler = ASHAScheduler(
        max_t=args.epoch,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["c1", "c2", "k"],
        metric_columns=["score", "loss", "training_iteration"])
    train_loader,val_loader=prepare_data(args)
    tune_fn_with_paremeters=tune.with_parameters(hunt_tune,
        args=args,
        train_loader=train_loader,val_loader=val_loader)
        
    analysis = tune.run(
        tune_fn_with_paremeters,
        
        resources_per_trial={
            "cpu": 1,
            "gpu": 1
        },
        metric="score",
        mode="max",
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_ray")

    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis

if __name__=='__main__':
    seed=42   
    pl.seed_everything(seed, workers=True)
    args=get_args()
    print(args)
    
    analysis=tune_with_ray(args)

