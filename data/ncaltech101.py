import os
import glob
import numpy as np
import torch
import lightning as L
import torch_geometric
import numba

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from models.utils.bounding_box import crop_to_frame

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.transforms import FixedPoints

from data.base.ds import DS
from data.base.class_dict import ncaltech_dict

class NCaltech101(L.LightningDataModule):
    def __init__(self, 
                 data_dir, 
                 batch_size=8,
                 radius=0.02):
        super().__init__()

        # Dataset directory and name.
        self.data_dir = data_dir
        self.data_name = 'ncaltech101'

        # Time window, original dimension, normalized dimension and radius.
        self.time_window = 50000  # 50 ms
        self.dim = (240, 180, self.time_window)
        self.radius = radius
        self.max_num_neighbors = 16
        self.num_nodes = 25000

        # Number of workers, batch size and processes for data preparation.
        self.num_workers = 2
        self.batch_size = batch_size
        self.processes = 3

        # Number of classes and class dictionary.
        self.num_classes = 101
        self.class_dict = ncaltech_dict
            
    def process_file(self, data_file) -> None:   
        # Create name for processed file.
        processed_file = data_file.replace(self.data_name, self.data_name + '/processed').replace('bin', 'pt')

        # Check if processed file already exists.
        if os.path.exists(processed_file):
            return

        # Create directory for processed file.
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)

        # Extract events, bounding boxes and class id from raw data.
        data = self.extract_events(data_file)
        bboxes = self.extract_bboxes(data_file)
        class_id = self.class_dict[data_file.split('/')[-2]]
        # data['edge_index'] = radius_graph(data.pos, r=self.radius, max_num_neighbors=self.max_num_neighbors)
        data['bbox'] = torch.tensor(bboxes).long()
        data['bbox'] = data['bbox'].view(-1, 5)
        data['bbox'] = crop_to_frame(data['bbox'], image_shape=(240,180))
        data['y'] = torch.tensor([class_id]).long()

        # Create image from events for visualization.
        ev_img = np.zeros((self.dim[1], self.dim[0]), dtype=np.uint8)
        ev_img[data.pos[:, 1].cpu().long(), data.pos[:, 0].cpu().long()] = 128

        data['ev_img'] = ev_img
        torch.save(data.to('cpu'), processed_file)

    ############################################################################
    # DATA PREPARATION #########################################################
    ############################################################################

    def _prepare_data(self, mode: str) -> None:
        # Prepare data for training and testing using multiprocessing.
        data_files = glob.glob(os.path.join(self.data_dir, self.data_name, mode, '*', '*.bin'))
        process_map(self.process_file, data_files, max_workers=self.processes, chunksize=1, )

    def prepare_data(self, flag=None) -> None:
        if flag == 'prepare':
            # Prepare training and testing data.
            # Split data into training and testing data.
            data_files = glob.glob(os.path.join(self.data_dir, self.data_name, 'events', '*', '*.bin'))
            # split randomly 80% of data for training and 20% for testing and training
            data_files_train = np.random.choice(data_files, int(0.8 * len(data_files)), replace=False)
            data_files_rest = np.setdiff1d(data_files, data_files_train)

            #
            data_files_test = np.random.choice(data_files_rest, int(0.5 * len(data_files_rest)), replace=False)
            data_files_val = np.setdiff1d(data_files_rest, data_files_test)
            
            # Copy files to train, val and test directories - make sure folder exists
            if not os.path.exists(os.path.join(self.data_dir, self.data_name, 'train')):
                for file in tqdm(data_files_train):
                    os.makedirs(file.replace("events", "train").replace(os.path.basename(file), ''), exist_ok=True)
                    os.system(f'cp {file} {file.replace("events", "train")}')
                    
            if not os.path.exists(os.path.join(self.data_dir, self.data_name, 'test')):
                for file in tqdm(data_files_test):
                    os.makedirs(file.replace("events", "test").replace(os.path.basename(file), ''), exist_ok=True)
                    os.system(f'cp {file} {file.replace("events", "test")}')

            if not os.path.exists(os.path.join(self.data_dir, self.data_name, 'val')):
                for file in tqdm(data_files_val):
                    os.makedirs(file.replace("events", "val").replace(os.path.basename(file), ''), exist_ok=True)
                    os.system(f'cp {file} {file.replace("events", "val")}')

            print('Preparing data...')
            for mode in ['train', 'test', 'val']:
                print(f'Loading {mode} data')
                os.makedirs(os.path.join(self.data_dir, self.data_name, 'processed', mode), exist_ok=True)
                self._prepare_data(mode)

    def generate_ds(self, mode: str):
        # Generate dataset from processed files.
        processed_files = glob.glob(os.path.join(self.data_dir, self.data_name, 'processed',  mode, '*', '*.pt'))
        augmentation = True if mode == 'train' else False
        return DS(processed_files, radius=self.radius, max_num_neighbors=self.max_num_neighbors, dim =self.dim, augmentation=augmentation)
    
    def setup(self, stage=None):
        # Load training and testing data.
        self.train_data = self.generate_ds('train')
        self.test_data = self.generate_ds('test')
        self.val_data = self.generate_ds('val')

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn, persistent_workers=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn, persistent_workers=False)
    
    @staticmethod
    def collate_fn(data_list):
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        if hasattr(data_list[0], 'bbox'):
            batch_bbox = sum([[i] * len(data.y) for i, data in enumerate(data_list)], [])
            batch.batch_bbox = torch.tensor(batch_bbox, dtype=torch.long)
        return batch
    
    ############################################################################
    # EXTRACTOR FUNCTIONS ######################################################
    ############################################################################

    def extract_events(self, data_file):
        # Read raw data from binary file and convert it to numpy array.
        f = open(data_file, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()

        # Extract x, y, t and p from raw data.
        # From https://github.com/uzh-rpg/aegnn/blob/master/aegnn/datasets/ncaltech101.py

        raw_data = np.uint32(raw_data)
        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        all_p = all_p.astype(np.float64)
        all_p[all_p == 0] = -1
        events = np.column_stack((all_x, all_y, all_ts, all_p))
        events = torch.from_numpy(events).float().cuda()

        x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
        data = Data(x=x, pos=pos)
        
        # Cut-off window of highest increase of events.
        original_num_nodes = data.num_nodes

        t = data.pos[data.num_nodes // 2, 2]
        index1 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t) - 1, 0, data.num_nodes - 1)
        index0 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t-self.time_window) - 1, 0, data.num_nodes - 1)
        for key, item in data:
            if torch.is_tensor(item) and item.size(0) == original_num_nodes and item.size(0) != 1:
                data[key] = item[index0:index1, :]
        

        # Subsample events.
        sampler = FixedPoints(num=self.num_nodes, allow_duplicates=False, replace=False)
        data = sampler(data)

        # sort by time
        idx = torch.argsort(data.pos[:, 2])
        data['pos'] = data['pos'][idx]
        data['x'] = data['x'][idx]
        return data

    def extract_bboxes(self, data_file):
        # Read annotation file and extract bounding box.
        annotation_file = data_file.replace('train', 'annotations')
        annotation_file = annotation_file.replace('test', 'annotations')
        annotation_file = annotation_file.replace('val', 'annotations')
        annotation_file = annotation_file.replace('image', 'annotation')

        f = open(annotation_file)
        annotations = np.fromfile(f, dtype=np.int16)
        annotations = np.array(annotations[2:10])
        f.close()
        
        # Class id
        class_id = self.class_dict[data_file.split('/')[-2]]

        # # Extract bounding box from annotation file.
        bbox = np.array([
            annotations[0], annotations[1],  # upper-left corner
            annotations[2] - annotations[0],  # width
            annotations[5] - annotations[1],  # height
            class_id
        ])
        bbox[:2] = np.maximum(bbox[:2], 0)
        bbox = bbox.reshape((1, 1, -1))

        return bbox