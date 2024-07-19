import os
import sys
import glob
import numpy as np
import torch
import lightning as L
import torch_geometric

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import FixedPoints

from data.base.ds import DS
from data.base.class_dict import gen1_dict

class Gen1(L.LightningDataModule):
    def __init__(self, 
                 data_dir, 
                 batch_size=8,
                 radius=3,
                 detection=True):
        super().__init__()

        # Dataset directory and name.
        self.data_dir = data_dir
        self.data_name = 'gen1'

        # Initialize train and test dataset.
        self.train_data = None
        self.test_data = None

        # Time window, original dimension, normalized dimension and radius.
        self.time_window = 400000  # 400 ms
        self.dim = (304, 240, self.time_window)
        self.radius = radius
        self.max_num_neighbors = 128

        # Number of workers, batch size and processes for data preparation.
        self.num_workers = 2
        self.batch_size = batch_size
        self.processes = 1

        # Number of classes and class dictionary.
        self.num_classes = 2
        self.class_dict = gen1_dict
            
    def process_file(self, data_file) -> None:   
        # Create name for processed file.

        data_loader = PSEELoader(data_file)
        annotation_file = data_file.replace("_td.dat", "_bbox.npy")
        bounding_boxes = self._read_annotations(annotation_file)
        labels = np.array(self._read_label(bounding_boxes))

        for i, bbox in enumerate(bounding_boxes):
            t_bbox = bbox[0]
            t_start = t_bbox - 100000
            t_end = t_bbox + 300000

            processed_file = data_file.replace(self.data_name, self.data_name + '/processed' + f'_{self.radius}').replace('_td.dat', f'_td_{t_bbox}.pt')

            if os.path.exists(processed_file):
                continue
                
            # Get all bounding boxes within the time window.
            bbox_mask = np.logical_and(t_start < bounding_boxes['ts'], bounding_boxes['ts'] < t_end)
            bb = torch.tensor(bounding_boxes[bbox_mask].tolist())

            # Get all events within the time window.
            idx_start = data_loader.seek_time(t_start)
            events = data_loader.load_delta_t(t_end - t_start)

            if events.size < 4000:
                continue

            data = self._buffer_to_data(events)

            # Fixe the number of events to 25000.
            sampler = FixedPoints(num=25000, allow_duplicates=False, replace=False)
            data = sampler(data)

            # Normalize the time of the events.
            data.pos[:, 2] = data.pos[:, 2] - data.pos[0, 2]
            data.pos[:, 2] = data.pos[:, 2] / (self.time_window) * 100

            data['edge_index'] = radius_graph(data.pos, r=self.radius, max_num_neighbors=self.max_num_neighbors)
            data['bbox'] = bb[:, 1:6].long()
            data['y'] = data.bbox[:, -1]

            torch.save(data.to('cpu'), processed_file)

    ############################################################################
    # DATA PREPARATION #########################################################
    ############################################################################

    def _prepare_data(self, mode: str) -> None:
        # Prepare data for training and testing using multiprocessing.
        data_files = glob.glob(os.path.join(self.data_dir, self.data_name, mode, '*.dat'))
        process_map(self.process_file, data_files, max_workers=self.processes, chunksize=1, )

    def prepare_data(self) -> None:
        # Prepare training and testing data.
        print('Preparing data...')
        for mode in ['train', 'test']:
            print(f'Loading {mode} data')
            os.makedirs(os.path.join(self.data_dir, self.data_name, 'processed' + f'_{self.radius}', mode), exist_ok=True)
            self._prepare_data(mode)

    def generate_ds(self, mode: str):
        # Generate dataset from processed files.
        processed_files = glob.glob(os.path.join(self.data_dir, self.data_name, 'processed' + f'_{self.radius}',  mode, '*.pt'))
        return DS(processed_files, radius=self.radius)
    
    def setup(self, stage=None):
        # Load training and testing data.
        self.train_data = self.generate_ds('train')
        self.test_data = self.generate_ds('test')

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn, persistent_workers=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn, persistent_workers=False)
    
    @staticmethod
    def collate_fn(data_list):
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        if hasattr(data_list[0], 'bbox'):
            batch_bbox = sum([[i] * len(data.y) for i, data in enumerate(data_list)], [])
            batch.batch_bbox = torch.tensor(batch_bbox, dtype=torch.long)
        return batch
    
    ###############################################################################
    # DATA EXTRACTION #############################################################
    ###############################################################################

    @staticmethod
    def _read_annotations(raw_file: str, skip_ts=int(5e5), min_box_diagonal=30, min_box_side=20) -> np.ndarray:
        boxes = np.load(raw_file.replace("_td.dat", "_bbox.npy"))

        # Bounding box filtering to avoid dealing with too small or initial bounding boxes.
        # See https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/io/box_filtering.py
        # Format: (ts, x, y, w, h, class_id, confidence = 1, track_id)
        ts = boxes['ts']
        width = boxes['w']
        height = boxes['h']
        diagonal_square = width**2+height**2
        mask = (ts > skip_ts)*(diagonal_square >= min_box_diagonal**2)*(width >= min_box_side)*(height >= min_box_side)
        return boxes[mask]

    @staticmethod
    def _read_label(bounding_boxes: np.ndarray):
        class_id = bounding_boxes['class_id'].tolist()
        label_dict = {0: "car", 1: "pedestrian"}
        return [label_dict.get(cid, None) for cid in class_id]
    
    @staticmethod
    def _buffer_to_data(buffer: np.ndarray) -> Data:
        x = torch.from_numpy(buffer['x'].astype(np.float32))
        y = torch.from_numpy(buffer['y'].astype(np.float32))
        t = torch.from_numpy(buffer['t'].astype(np.float32))
        p = torch.from_numpy(buffer['p'].astype(np.float32)).view(-1, 1)
        pos = torch.stack([x, y, t], dim=1)
        return Data(x=p, pos=pos)


#########################################################################################################
# Gen1 specific loader function from https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox
#########################################################################################################
EV_TYPE = [('t', 'u4'), ('_', 'i4')]  # Event2D
EV_STRING = 'Event2D'


class PSEELoader(object):
    """
    PSEELoader loads a dat or npy file and stream events
    """

    def __init__(self, datfile):
        """
        ctor
        :param datfile: binary dat or npy file
        """
        self._extension = datfile.split('.')[-1]
        assert self._extension in ["dat", "npy"], 'input file path = {}'.format(datfile)
        self._file = open(datfile, "rb")
        self._start, self.ev_type, self._ev_size, self._size = _parse_header(self._file)
        assert self._ev_size != 0
        self._dtype = EV_TYPE

        self._decode_dtype = []
        for dtype in self._dtype:
            if dtype[0] == '_':
                self._decode_dtype += [('x', 'u2'), ('y', 'u2'), ('p', 'u1')]
            else:
                self._decode_dtype.append(dtype)

        # size
        self._file.seek(0, os.SEEK_END)
        self._end = self._file.tell()
        self._ev_count = (self._end - self._start) // self._ev_size
        self.done = False
        self._file.seek(self._start)
        # If the current time is t, it means that next event that will be loaded has a
        # timestamp superior or equal to t (event with timestamp exactly t is not loaded yet)
        self.current_time = 0
        self.duration_s = self.total_time() * 1e-6

    def reset(self):
        """reset at beginning of file"""
        self._file.seek(self._start)
        self.done = False
        self.current_time = 0

    def event_count(self):
        """
        getter on event_count
        :return:
        """
        return self._ev_count

    def get_size(self):
        """"(height, width) of the imager might be (None, None)"""
        return self._size

    def __repr__(self):
        """
        prints properties
        :return:
        """
        wrd = ''
        wrd += 'PSEELoader:' + '\n'
        wrd += '-----------' + '\n'
        if self._extension == 'dat':
            wrd += 'Event Type: ' + str(EV_STRING) + '\n'
        elif self._extension == 'npy':
            wrd += 'Event Type: numpy array element\n'
        wrd += 'Event Size: ' + str(self._ev_size) + ' bytes\n'
        wrd += 'Event Count: ' + str(self._ev_count) + '\n'
        wrd += 'Duration: ' + str(self.duration_s) + ' s \n'
        wrd += '-----------' + '\n'
        return wrd

    def load_n_events(self, ev_count):
        """
        load batch of n events
        :param ev_count: number of events that will be loaded
        :return: events
        Note that current time will be incremented to reach the timestamp of the first event not loaded yet
        """
        event_buffer = np.empty((ev_count + 1,), dtype=self._decode_dtype)

        pos = self._file.tell()
        count = (self._end - pos) // self._ev_size
        if ev_count >= count:
            self.done = True
            ev_count = count
            _stream_td_data(self._file, event_buffer, self._dtype, ev_count)
            self.current_time = event_buffer['t'][ev_count - 1] + 1
        else:
            _stream_td_data(self._file, event_buffer, self._dtype, ev_count + 1)
            self.current_time = event_buffer['t'][ev_count]
            self._file.seek(pos + ev_count * self._ev_size)

        return event_buffer[:ev_count]

    def load_delta_t(self, delta_t):
        """
        loads a slice of time.
        :param delta_t: (us) slice thickness
        :return: events
        Note that current time will be incremented by delta_t.
        If an event is timestamped at exactly current_time it will not be loaded.
        """
        if delta_t < 1:
            raise ValueError("load_delta_t(): delta_t must be at least 1 micro-second: {}".format(delta_t))

        if self.done or (self._file.tell() >= self._end):
            self.done = True
            return np.empty((0,), dtype=self._decode_dtype)

        final_time = self.current_time + delta_t
        tmp_time = self.current_time
        start = self._file.tell()
        pos = start
        nevs = 0
        batch = 100000
        event_buffer = []
        # data is read by buffers until enough events are read or until the end of the file
        while tmp_time < final_time and pos < self._end:
            count = (min(self._end, pos + batch * self._ev_size) - pos) // self._ev_size
            buffer = np.empty((count,), dtype=self._decode_dtype)
            _stream_td_data(self._file, buffer, self._dtype, count)
            tmp_time = buffer['t'][-1]
            event_buffer.append(buffer)
            nevs += count
            pos = self._file.tell()
        if tmp_time >= final_time:
            self.current_time = final_time
        else:
            self.current_time = tmp_time + 1
        assert len(event_buffer) > 0
        idx = np.searchsorted(event_buffer[-1]['t'], final_time)
        event_buffer[-1] = event_buffer[-1][:idx]
        event_buffer = np.concatenate(event_buffer)
        idx = len(event_buffer)
        self._file.seek(start + idx * self._ev_size)
        self.done = self._file.tell() >= self._end
        return event_buffer

    def seek_event(self, ev_count):
        """
        seek in the file by ev_count events
        :param ev_count: seek in the file after ev_count events
        Note that current time will be set to the timestamp of the next event.
        """
        if ev_count <= 0:
            self._file.seek(self._start)
            self.current_time = 0
        elif ev_count >= self._ev_count:
            # we put the cursor one event before and read the last event
            # which puts the file cursor at the right place
            # current_time is set to the last event timestamp + 1
            self._file.seek(self._start + (self._ev_count - 1) * self._ev_size)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0] + 1
        else:
            # we put the cursor at the *ev_count*nth event
            self._file.seek(self._start + ev_count * self._ev_size)
            # we read the timestamp of the following event (this change the position in the file)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0]
            # this is why we go back at the right position here
            self._file.seek(self._start + ev_count * self._ev_size)
        self.done = self._file.tell() >= self._end

    def seek_time(self, final_time, term_criterion: int = 100000):
        """
        go to the time final_time inside the file. This is implemented using a binary search algorithm
        :param final_time: expected time
        :param term_criterion: (nb event) binary search termination criterion
        it will load those events in a buffer and do a numpy searchsorted so the result is always exact
        """
        if final_time > self.total_time():
            self._file.seek(self._end)
            self.done = True
            self.current_time = self.total_time() + 1
            return

        if final_time <= 0:
            self.reset()
            return

        low = 0
        high = self._ev_count

        # binary search
        while high - low > term_criterion:
            middle = (low + high) // 2

            self.seek_event(middle)
            mid = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0]

            if mid > final_time:
                high = middle
            elif mid < final_time:
                low = middle + 1
            else:
                self.current_time = final_time
                self.done = self._file.tell() >= self._end
                return
        # we now know that it is between low and high
        self.seek_event(low)
        final_buffer = np.fromfile(self._file, dtype=self._dtype, count=high - low)['t']
        final_index = np.searchsorted(final_buffer, final_time)

        self.seek_event(low + final_index)
        self.current_time = final_time
        self.done = self._file.tell() >= self._end
        return low + final_index

    def total_time(self):
        """
        get total duration of video in mus, providing there is no overflow
        :return:
        """
        if not self._ev_count:
            return 0
        # save the state of the class
        pos = self._file.tell()
        current_time = self.current_time
        done = self.done
        # read the last event's timestamp
        self.seek_event(self._ev_count - 1)
        time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0]
        # restore the state
        self._file.seek(pos)
        self.current_time = current_time
        self.done = done

        return time

    def __del__(self):
        self._file.close()


def _parse_header(f):
    """
    Parses the header of a dat file
    Args:
        - f file handle to a dat file
    return :
        - int position of the file cursor after the header
        - int type of event
        - int size of event in bytes
        - size (height, width) tuple of int or None
    """
    f.seek(0, os.SEEK_SET)
    bod = None
    end_of_header = False
    header = []
    num_comment_line = 0
    size = [None, None]
    # parse header
    while not end_of_header:
        bod = f.tell()
        line = f.readline()
        if sys.version_info > (3, 0):
            first_item = line.decode("latin-1")[:2]
        else:
            first_item = line[:2]

        if first_item != '% ':
            end_of_header = True
        else:
            words = line.split()
            if len(words) > 1:
                if words[1] == 'Date':
                    header += ['Date', words[2] + ' ' + words[3]]
                if words[1] == 'Height' or words[1] == b'Height':  # compliant with python 3 (and python2)
                    size[0] = int(words[2])
                    header += ['Height', words[2]]
                if words[1] == 'Width' or words[1] == b'Width':  # compliant with python 3 (and python2)
                    size[1] = int(words[2])
                    header += ['Width', words[2]]
            else:
                header += words[1:3]
            num_comment_line += 1
    # parse data
    f.seek(bod, os.SEEK_SET)

    if num_comment_line > 0:  # Ensure compatibility with previous files.
        # Read event type
        ev_type = np.frombuffer(f.read(1), dtype=np.uint8)[0]
        # Read event size
        ev_size = np.frombuffer(f.read(1), dtype=np.uint8)[0]
    else:
        ev_type = 0
        ev_size = sum([int(n[-1]) for _, n in EV_TYPE])

    bod = f.tell()
    return bod, ev_type, ev_size, size


def _stream_td_data(file_handle, buffer, dtype, ev_count=-1):
    """
    Streams data from opened file_handle
    args :
        - file_handle: file object
        - buffer: pre-allocated buffer to fill with events
        - dtype:  expected fields
        - ev_count: number of events
    """

    dat = np.fromfile(file_handle, dtype=dtype, count=ev_count)
    count = len(dat['t'])
    for name, _ in dtype:
        if name == '_':
            buffer['x'][:count] = np.bitwise_and(dat["_"], 16383)
            buffer['y'][:count] = np.right_shift(np.bitwise_and(dat["_"], 268419072), 14)
            buffer['p'][:count] = np.right_shift(np.bitwise_and(dat["_"], 268435456), 28)
        else:
            buffer[name][:count] = dat[name]