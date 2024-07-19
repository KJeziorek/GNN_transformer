import torch
import numpy as np
import numba
from torch.utils.data import Dataset
from torch_geometric.transforms import Cartesian
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data

from data.base.augmentation import RandomZoom, RandomCrop, RandomTranslate, Crop

import generate_edges

class DS(Dataset):
    def __init__(self, files, radius=0.01, max_num_neighbors=16, dim=(240,180,50000), augmentation=False):
        self.files = files

        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.dim = dim

        self.augmentation = augmentation
        effective_radius = 2*float(int(0.01* 240 + 2) / 240)

        self.edge_attr = Cartesian(norm=True, cat=False, max_value=effective_radius) #changed

        self.random_crop = RandomCrop([0.75, 0.75], p=0.2, width=240, height=180)
        self.zoom = RandomZoom([1, 1.5], subsample=True, width=240, height=180)
        self.translate = RandomTranslate([0.1, 0.1, 0], width=240, height=180)
        self.crop = Crop([0,0], [1, 1], width=240, height=180)

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int):
        data_file = self.files[index]
        data = torch.load(data_file)
        data.bbox = data.bbox.float()
        
        if self.augmentation:
            data_new = data.clone()
            data_new = self.random_crop(data_new)
            # data_new = self.zoom(data_new)
            data_new = self.translate(data_new)
            data_new = self.crop(data_new)

            if data_new.pos.size(0) > 0:
                data = data_new

        data = normalize(data)
        # data['edge_index'] = radius_graph(data.pos, r=self.radius, max_num_neighbors=self.max_num_neighbors)

        edges = generate_edges.generate_edges(data.pos.cpu().numpy(), self.radius, self.max_num_neighbors, 256)
        data.edge_index = torch.tensor(edges, dtype=torch.long).T
        data = self.edge_attr(data)
        data.edge_attr = torch.clamp(data.edge_attr, min=0, max=1)
        return data

def normalize(data: torch.Tensor, width = 240, height = 180) -> torch.Tensor:
        """Normalize x y and t to range [0, 1]"""
        data.pos[:, 0] = data.pos[:, 0] / width
        data.pos[:, 1] = data.pos[:, 1] / height
        data.pos[:, 2] = data.pos[:, 2] - data.pos[:, 2].min()
        # data.pos[:, 2] = data.pos[:, 2] / data.pos[:, 2].max()
        data.pos[:, 2] = data.pos[:, 2] / 1000000
        return data