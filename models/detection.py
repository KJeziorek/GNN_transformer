"""Partly copied from rpg-asynet paper: https://github.com/uzh-rpg/rpg_asynet"""
import collections
import logging
import numpy as np
import math
import wandb
import torch
import torch_geometric
import lightning as L
import matplotlib.pyplot as plt

from copy import deepcopy
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from models.networks.full_model import GNNModel
from utils.convert_bbox import convert_to_training_format

from data.base.class_dict import ncaltech_dict

class LNDetection(L.LightningModule):

    def __init__(self, lr, weight_decay, num_classes, batch_size, input_dimension=256, conf_thre=0.001):
        super(LNDetection, self).__init__()

        # Define the YOLO detection grid as the model's outputs.
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # Define network architecture by name.
        self.model = GNNModel(conf_thre=conf_thre)
        self.ema_module = ModelEMA(self.model, decay=0.9999)
        self.model_ema = self.ema_module.ema

        # Based on AEGNN metric
        self.map = MeanAveragePrecision(iou_thresholds=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], class_metrics=True)

        self.lr = lr
        self.weight_decay = weight_decay

        self.val_pred = None
        self.test_pred = None

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.bbox = data.bbox.float()                          
        target = convert_to_training_format(data.bbox, data.batch_bbox, self.batch_size)

        out = self.model(data, target)
        return out
    
    def forward_ema(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.bbox = data.bbox.float()                          
        target = convert_to_training_format(data.bbox, data.batch_bbox, self.batch_size)

        ema_model = self.model_ema.to(data.x.device)
        out = ema_model(data, target)
        targets = []
        for d in data.to_data_list():
            bbox = d.bbox.clone()
            bbox[:,2:4] += bbox[:,:2]
            targets.append({
                "boxes": bbox[:,:4].cpu(),
                "labels": bbox[:, 4].cpu().long() # class 0 is background class
            })
        
        return out, targets
        
    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.ema_module:
            self.ema_module.update(self.model)
            self.model_ema = deepcopy(self.ema_module.ema)
    
    ###############################################################################################
    # Steps #######################################################################################
    ###############################################################################################
    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch.clone())

        total_loss, iou_loss, l1_loss, conf_loss, cls_loss = outputs["total_loss"], outputs["iou_loss"], outputs["l1_loss"], outputs["conf_loss"], outputs["cls_loss"]
        self.log("train/total_loss", total_loss)
        self.log("train/iou_loss", iou_loss)
        self.log("train/l1_loss", l1_loss)
        self.log("train/conf_loss", conf_loss)
        self.log("train/cls_loss", cls_loss)

        return total_loss

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        preds, gts = self.forward_ema(data=batch)
        self.map.update(preds, gts)

        self.val_pred = {'batch': batch,
                         'gts': gts, 
                         'preds': preds}

    def on_validation_epoch_end(self) -> None:
        maps = self.map.compute()
        self.log("val/mAP", maps['map'])
        self.log("val/mAP50", maps['map_50'])
        self.log("val/mAP75", maps['map_75'])
        self.map.reset()

        # Visualize the detection results.
        if self.val_pred:
            self.log_detections(self.val_pred)
            self.val_pred = None

    def test_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        preds, gts = self.forward_ema(data=batch)
        self.map.update(preds, gts)

        self.test_pred = {'batch': batch,
                         'gts': gts, 
                         'preds': preds}
    
    def on_test_epoch_end(self) -> None:
        maps = self.map.compute()
        self.log("test/mAP", maps['map'])
        self.log("test/mAP50", maps['map_50'])
        self.log("test/mAP75", maps['map_75'])
        self.map.reset()

        print(maps)
        # Visualize the detection results.
        if self.test_pred:
            self.log_detections(self.test_pred)
            self.test_pred = None

    def log_detections(self, val_pred: dict) -> None:
        # Create 2d image from pos
        batch = val_pred['batch']

        gts = val_pred['gts']
        preds = val_pred['preds']
        ev_img = batch.ev_img

        class_id_to_label = {int(v): k for k, v in ncaltech_dict.items()}

        images = []

        # Iterate over the first four batches
        for i in range(batch.batch_bbox.max().item() + 1):
            bbs = preds[i]['boxes'].cpu().numpy()
            labels = preds[i]['labels'].cpu().numpy()
            scores = preds[i]['scores'].cpu().numpy()

            pred_all_boxes = []
            gt_all_boxes = []
            for bb, label, score in zip(bbs, labels, scores):
                x1, y1, x2, y2 = bb
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                box_data = {
                    'position': {'minX': x1, 'minY': y1, 'maxX': x2, 'maxY': y2},
                    'class_id': int(label),
                    'box_caption': f'{label} {score:.2f}',
                    'scores': {'score': float(score)},
                    'domain': 'pixel'
                }
                pred_all_boxes.append(box_data)

            for bb, label in zip(gts[i]['boxes'], gts[i]['labels']):
                x1, y1, x2, y2 = bb
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                box_data = {
                    'position': {'minX': x1, 'minY': y1, 'maxX': x2, 'maxY': y2},
                    'class_id': int(label.cpu().item()),
                    'box_caption': f'{label.cpu().item()}',
                    'scores': {'score': 1.0},
                    'domain': 'pixel'
                }
                gt_all_boxes.append(box_data)

            # Add the image with bounding boxes to the list
            images.append(
                wandb.Image(ev_img[i], boxes={
                    'predictions': {
                        'box_data': pred_all_boxes,
                        'class_labels': class_id_to_label
                    },
                    'ground_truth': {
                        'box_data': gt_all_boxes,
                        'class_labels': class_id_to_label
                    }
                })
            )

        # Log all images under a single log key
        self.logger.experiment.log({'val/predictions': images})

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
    
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.lr,
            total_steps=150000,
            pct_start=0.005,
            div_factor=10,
            final_div_factor=1000,
            anneal_strategy='cos',
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model).eval()

        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()