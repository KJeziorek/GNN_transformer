from models.networks.head import GNNHead
from models.networks.eagr import EAGR

import torch
import torch.nn as nn
import torchvision


class GNNModel(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self,
                 conf_thre=0.001,):
        super().__init__()
        backbone = EAGR()
        head = GNNHead()

        self.backbone = backbone
        self.head = head
        self.conf_thre = conf_thre

    def forward(self, x, targets=None):
        backbone_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                backbone_outs, targets)
            
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(backbone_outs)
            outputs = self.postprocess_network_output(outputs, 100, height=180, width=240, filtering=True)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)

    def postprocess_network_output(self, prediction, num_classes, conf_thre=0.001, nms_thre=0.65, height=640, width=640, filtering=True):
        prediction[..., :2] -= prediction[...,2:4] / 2 # cxcywh->xywh
        prediction[..., 2:4] += prediction[...,:2]

        # print(prediction[:,4])
        output = []
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if len(image_pred) == 0:
                output.append({
                    "boxes": torch.zeros(0, 4, dtype=torch.float32, device='cpu'),
                    "scores": torch.zeros(0, dtype=torch.float, device='cpu'),
                    "labels": torch.zeros(0, dtype=torch.long, device='cpu')
                })
                continue
            
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            image_pred_copy = image_pred.clone()
            image_pred_copy[:, 4:5] *= class_conf

            conf_mask = (image_pred_copy[:, 4] * class_conf.squeeze() >= self.conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred_copy[:, :5], class_pred), 1)

            if filtering:
                detections = detections[conf_mask]

            if len(detections) == 0:
                output.append({
                    "boxes": torch.zeros(0, 4, dtype=torch.float32, device='cpu'),
                    "scores": torch.zeros(0, dtype=torch.float, device='cpu'),
                    "labels": torch.zeros(0, dtype=torch.long, device='cpu')
                })
                continue

            nms_out_index = self.batched_nms_coordinate_trick(detections[:, :4], detections[:, 4], detections[:, 5],
                                                        nms_thre, width=width, height=height)

            if filtering:
                detections = detections[nms_out_index]

            # print(detections[:,4])
            output.append({
                "boxes": detections[:, :4].to('cpu'),
                "scores": detections[:, 4].to('cpu'),
                "labels": detections[:, -1].long().to('cpu')
            })

        return output

    def batched_nms_coordinate_trick(self, boxes, scores, idxs, iou_threshold, width, height):
        # adopted from torchvision nms, but faster
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        max_dim = max([width, height])
        offsets = idxs * float(max_dim + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)
        return keep