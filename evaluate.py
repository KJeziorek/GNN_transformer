from data.ncaltech101 import NCaltech101
from data.gen1 import Gen1

from models.detection import LNDetection

from lightning.pytorch.loggers.wandb import WandbLogger
import lightning as L
import argparse
import multiprocessing as mp
from lightning.pytorch.callbacks import LearningRateMonitor


def main(args):
    if args.dataset == 'ncaltech101':
        dm = NCaltech101(data_dir='dataset', batch_size=args.batch_size, radius=args.radius)
    elif args.dataset == 'gen1':
        dm = Gen1(data_dir='dataset', batch_size=args.batch_size, radius=args.radius)
    else:
        raise ValueError(f'Dataset {args.dataset} not supported')
    
    dm.setup()

    model = LNDetection(lr=1e-2, weight_decay=1e-5, num_classes=dm.num_classes, batch_size=args.batch_size, conf_thre=0.001)
    wandb_logger = WandbLogger(project='event_detection_test', name=f'{args.dataset}_{args.radius}')
    trainer = L.Trainer(max_steps=200000, 
                        log_every_n_steps=1, 
                        check_val_every_n_epoch=20,
                        gradient_clip_val=0.1, 
                        logger=wandb_logger)
    
    trainer.test(model, dm, ckpt_path='event_detection/odh8ru18/checkpoints/best.ckpt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ncaltech101')
    parser.add_argument('--radius', type=int, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    mp.set_start_method('spawn', force=True)
    args = parser.parse_args()
    L.seed_everything(777)
    main(args)