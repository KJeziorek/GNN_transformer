import torch
import torch.nn as nn

from YOLOX.yolox.models.yolo_head import YOLOXHead
from YOLOX.yolox.models.losses import IOUloss

from models.networks.layers.head_layer import DetectionHead

class GNNHead(YOLOXHead):
    def __init__(
            self,
            num_classes=101,
            strides=[15, 30],
            in_channels=[64, 64],
            act="silu",
            depthwise=False,
            args=None,
            width=1.0,
    ):
        YOLOXHead.__init__(self, num_classes, width, strides, in_channels, act, depthwise)

        self.num_classes = num_classes
        self.decode_in_inference = True
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        self.heads = nn.ModuleList()
        for i in range(len(in_channels)):
            head = DetectionHead(in_channels[i], num_classes, strides[i])
            self.heads.append(head)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (head, stride_this_level, x) in enumerate(zip(self.heads, self.strides, xin)):
            cls_output, reg_output, obj_output = head(x)


            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].x.type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0].x)
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].x.dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].x.type())
            else:
                return outputs