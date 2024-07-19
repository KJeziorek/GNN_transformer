import torch
import torch.nn as nn

from torch.nn import Linear
from torch.nn.functional import elu, relu
from torch_geometric.nn import SplineConv, BatchNorm, PointNetConv, GPSConv, LGConv                    
from torch_geometric.data import Data

class DetectionHead(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 num_classes,
                 stride=15,
                 height=180,
                 width=240,):
        super().__init__()

        self.num_classes = num_classes
        self.stride = stride
        self.height = height
        self.width = width

        self.conv_lg = LGConv()
        self.conv_gps = GPSConv(channels=in_channels, conv=self.conv_lg, heads=8, dropout=0.1)
        self.node_emb = Linear(in_channels, in_channels-10, bias=False)
        self.pos_emb = Linear(2, in_channels, bias=False)
        # self.linear11 = Linear(in_channels+2, in_channels, bias=False)
        # self.linear12 = Linear(in_channels, in_channels, bias=False)
        # self.conv1 = PointNetConv(local_nn=self.linear11, global_nn=self.linear12)
        self.conv1 = SplineConv(in_channels=in_channels, out_channels=in_channels, dim=2, kernel_size=5, bias=False)
        self.norm1 = BatchNorm(in_channels=in_channels)

        # self.linear21 = Linear(in_channels+2, in_channels, bias=False)
        # self.linear22 = Linear(in_channels, in_channels, bias=False)
        # self.conv2 = PointNetConv(local_nn=self.linear21, global_nn=self.linear22)
        self.conv2 = SplineConv(in_channels=in_channels, out_channels=in_channels, dim=2, kernel_size=5, bias=False)
        self.norm2 = BatchNorm(in_channels=in_channels)

        # self.linear31 = Linear(in_channels+2, in_channels, bias=False)
        # self.linear32 = Linear(in_channels, in_channels, bias=False)
        # self.conv3 = PointNetConv(local_nn=self.linear31, global_nn=self.linear32)
        self.conv3 = SplineConv(in_channels=in_channels, out_channels=in_channels, dim=2, kernel_size=5, bias=False)
        self.norm3 = BatchNorm(in_channels=in_channels)

        # self.lregr1 = Linear(in_channels+2, in_channels, bias=True)
        # self.lregr = Linear(in_channels, 4, bias=True)

        # self.lcls1 = Linear(in_channels+2, in_channels, bias=True)
        # self.lcls = Linear(in_channels, num_classes, bias=True)

        # self.lobj1 = Linear(in_channels+2, in_channels, bias=True)
        # self.lobj = Linear(in_channels, 1, bias=True)

        # self.regr = PointNetConv(local_nn=self.lregr1, global_nn=self.lregr)
        # self.cls = PointNetConv(local_nn=self.lcls1, global_nn=self.lcls)
        # self.obj = PointNetConv(local_nn=self.lobj1, global_nn=self.lobj)

        self.regr = SplineConv(in_channels=in_channels, out_channels=4, dim=2, kernel_size=5, bias=True)
        self.cls = SplineConv(in_channels=in_channels, out_channels=num_classes, dim=2, kernel_size=5, bias=True)
        self.obj = SplineConv(in_channels=in_channels, out_channels=1, dim=2, kernel_size=5, bias=True)

    def forward(self, data):
        
        data.edge_attr = data.edge_attr[:,:2]


        # data.x = torch.cat((self.node_emb(data.x), self.pos_emb(data.pos[:,:2])),1)
        # data.x = data.x + self.pos_emb(data.pos[:,:2])
        # data.x = self.conv_gps(data.x, data.edge_index, data.batch) # , edge_attr=data.edge_attr


        x1 = self.conv1(data.x, data.edge_index, data.edge_attr)
        # x1 = self.conv1(data.x, data.pos[:,:2], data.edge_index)
        x1 = self.norm1(x1)
        x1 = relu(x1)

        x1_copy = x1.clone()
        x2 = self.conv2(x1_copy, data.edge_index, data.edge_attr)
        # x2 = self.conv2(x1, data.pos[:,:2], data.edge_index)
        x2 = self.norm2(x2)
        x2 = relu(x2)

        x3 = self.conv3(x1, data.edge_index, data.edge_attr)
        # x3 = self.conv3(x1_copy, data.pos[:,:2], data.edge_index)
        x3 = self.norm3(x3)
        x3 = relu(x3)

        x3_copy = x3.clone()
        reg = self.regr(x2, data.edge_index, data.edge_attr)
        cls = self.cls(x3_copy, data.edge_index, data.edge_attr)
        obj = self.obj(x3, data.edge_index, data.edge_attr)

        # pointnet
        # reg = self.regr(x2, data.pos[:,:2], data.edge_index)
        # cls = self.cls(x3_copy, data.pos[:,:2], data.edge_index)
        # obj = self.obj(x3, data.pos[:,:2], data.edge_index)

        # reg = self.lregr1(x2)
        # cls = self.lcls1(x3)
        # obj = self.lobj1(x3_copy)

        batch_size = data.batch.max().item() + 1
        cls_out = torch.zeros(batch_size, self.num_classes, self.height//self.stride, self.width//self.stride, device=data.x.device)
        reg_out = torch.zeros(batch_size, 4, self.height//self.stride, self.width//self.stride, device=data.x.device)
        obj_out = torch.zeros(batch_size, 1, self.height//self.stride, self.width//self.stride, device=data.x.device)

        x_idx = ((data.pos[:, 0]*self.width) // self.stride).long()
        y_idx = ((data.pos[:, 1]*self.height) // self.stride).long()

        cls_out[data.batch, :, y_idx, x_idx] = cls
        reg_out[data.batch, :, y_idx, x_idx] = reg
        obj_out[data.batch, :, y_idx, x_idx] = obj
        
        return (cls_out, reg_out, obj_out)
    

def shallow_copy(data):
    out =  Data(x=data.x.clone(), edge_index=data.edge_index, edge_attr=data.edge_attr, pos=data.pos, batch=data.batch)
    return out