import torch
import lightning as L

from torchmetrics import Accuracy
from torchmetrics.classification import ConfusionMatrix

from typing import Dict, Tuple
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import StepLR

# from models.networks.spline_multistage import MultiStage

from models.networks.multistage import MultiStage
from models.networks.aegnn import AEGNN
from models.networks.aegnn_original import GraphRes

import wandb
import numpy as np
import matplotlib.pyplot as plt


class LNRecognition(L.LightningModule):
    def __init__(self, lr, weight_decay, num_classes, batch_size, input_dimension=256):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        self.batch_size = batch_size
        self.num_classes = num_classes

        self.input_dimension = input_dimension

        model_input_shape = torch.tensor((240, 180) + (3, ), device='cuda')
        self.model = GraphRes(model_input_shape, num_outputs=100)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        if num_classes > 3:
            self.accuracy_top_3 = Accuracy(task="multiclass", num_classes=num_classes, top_k=3).to(self.device)

        self.save_hyperparameters()

        self.val_pred = None
        self.train_pred = None
        self.pred = []
        self.target = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

    def forward(self, data):
        x = self.model(data)
        return x

    def training_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, target=batch['y'].long())
        
        y_prediction = torch.argmax(outputs, dim=-1)
        accuracy = self.accuracy(preds=y_prediction, target=batch['y'])

        self.log('train_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('train_acc', accuracy, on_epoch=True, logger=True, batch_size=self.batch_size)

        if self.num_classes > 3:
            pred = softmax(outputs, dim=-1)
            top_3 = self.accuracy_top_3(preds=pred, target=batch['y'])
            self.log('train_acc_top_3', top_3, on_epoch=True, logger=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, target=batch['y'].long())

        y_prediction = torch.argmax(outputs, dim=-1)
        accuracy = self.accuracy(preds=y_prediction, target=batch['y'])

        self.log('val_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('val_acc', accuracy, on_epoch=True, logger=True, batch_size=self.batch_size)

        if self.num_classes > 3:
            pred = softmax(outputs, dim=-1)
            top_3 = self.accuracy_top_3(preds=pred, target=batch['y'])
            self.log('val_acc_top_3', top_3, on_epoch=True, logger=True, batch_size=self.batch_size)