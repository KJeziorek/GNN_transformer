import torch
import torch_geometric

from torch.nn import Sequential, Linear, Dropout
from torch.nn.functional import elu, relu
from torch_geometric.nn.conv import PointNetConv, GPSConv, LGConv, SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.transforms import Cartesian

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool, max_pool_x, voxel_grid
from typing import Callable, List, Optional, Tuple, Union


class MaxPooling(torch.nn.Module):

    def __init__(self, size: List[int], effective_radius: float):
        super(MaxPooling, self).__init__()
        self.voxel_size = list(size)

        self.effective_radius = effective_radius

    def forward(self, data):
        cluster = voxel_grid(data.pos, batch=data.batch, size=self.voxel_size)
        data = max_pool(cluster, data=data, transform=Cartesian(cat=False, norm=True, max_value=self.effective_radius))
        return data

class PointNetLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear11 = Linear(in_channels+2, out_channels, bias=False)
        self.linear12 = Linear(out_channels, out_channels, bias=False)
        self.conv1 = PointNetConv(local_nn=self.linear11, global_nn=self.linear12)
        self.norm1 = BatchNorm(in_channels=out_channels)

        self.linear21 = Linear(out_channels+2, out_channels, bias=False)
        self.linear22 = Linear(out_channels, out_channels, bias=False)
        self.conv2 = PointNetConv(local_nn=self.linear21, global_nn=self.linear22)
        self.norm2 = BatchNorm(in_channels=out_channels)

    def forward(self, data):
        # Skip connection
        if self.in_channels == self.out_channels:
            data_skip = shallow_copy(data)

        # Main path
        data.x = self.conv1(data.x, data.pos[:,:2], data.edge_index)
        data.x = self.norm1(data.x)
        data.x = relu(data.x)

        if self.in_channels != self.out_channels:
            data_skip = shallow_copy(data)

        data.x = self.conv2(data.x, data.pos[:,:2], data.edge_index)
        data.x = self.norm2(data.x)

        # Skip connection
        data.x = data.x + data_skip.x
        data.x = relu(data.x)
        return data
    
class SplineLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = SplineConv(in_channels=in_channels+2, out_channels=out_channels, dim=2, kernel_size=5, bias=False)
        self.norm1 = BatchNorm(in_channels=out_channels)

        self.conv2 = SplineConv(in_channels=out_channels, out_channels=out_channels, dim=2, kernel_size=5, bias=False)
        self.norm2 = BatchNorm(in_channels=out_channels)

        self.linear = Linear(in_channels+2, out_channels, bias=False)
        self.norm3 = BatchNorm(in_channels=out_channels)

    def forward(self, data):
        data.edge_attr = data.edge_attr[:,:2]

        data.x = torch.cat([data.x, data.pos[:,:2].clone()], dim=1)
        # Skip connection
        data_skip = shallow_copy(data)
        data_skip.x = self.linear(data_skip.x)
        data_skip.x = self.norm3(data_skip.x)

        # Main path
        data.x = self.conv1(data.x, data.edge_index, data.edge_attr)
        data.x = self.norm1(data.x)
        data.x = relu(data.x)
        data.x = self.conv2(data.x, data.edge_index, data.edge_attr)
        data.x = self.norm2(data.x)

        # Skip connection
        data.x = data.x + data_skip.x
        data.x = relu(data.x)
        return data

class EAGR(torch.nn.Module):
    def __init__(self):
        super().__init__()

        effective_radius = 2*float(int(0.01* 240 + 2) / 240)

        # self.block1 = SplineLayer(1, 16)
        self.block1 = PointNetLayer(1, 16)
        self.maxpool1 = MaxPooling(size=(1/64, 1/48, 1), effective_radius=2*effective_radius)
        # self.block2 = SplineLayer(16, 64)
        self.block2 = PointNetLayer(16, 64)
        self.maxpool2 = MaxPooling(size=(1/32, 1/24, 1), effective_radius=2*(1/24))
        # self.block3 = SplineLayer(64, 64)
        self.block3 = PointNetLayer(64, 64)
        self.maxpool3 = MaxPooling(size=(1/16, 1/12, 1), effective_radius=2*(1/12))
        # self.block4 = SplineLayer(64, 64)
        self.block4 = PointNetLayer(64, 64)
        self.maxpool4 = MaxPooling(size=(1/8, 1/6, 1), effective_radius=2*(1/6))
        # self.block5 = SplineLayer(64, 64)
        self.block5 = PointNetLayer(64, 64)

    def forward(self, data):

        data = self.block1(data)
        data = self.maxpool1(data=data)
        data = self.block2(data)
        data = self.maxpool2(data=data)
        data = self.block3(data)
        data = self.maxpool3(data=data)
        data = self.block4(data)
        data1 = data.clone()
        data = self.maxpool4(data=data)
        data = self.block5(data)
        data2 = data.clone()
        return [data1, data2]
    
def shallow_copy(data):
    out = Data(x=data.x.clone(), edge_index=data.edge_index, edge_attr=data.edge_attr, pos=data.pos, batch=data.batch)
    return out