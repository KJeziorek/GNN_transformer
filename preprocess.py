from data.ncaltech101 import NCaltech101
from data.gen1 import Gen1

import argparse
import multiprocessing as mp
import lightning as L


def main(args):

    if args.dataset == 'ncaltech101':
        dm = NCaltech101(data_dir='dataset', batch_size=1)
    elif args.dataset == 'gen1':
        dm = Gen1(data_dir='dataset', batch_size=1)
    else:
        raise ValueError(f'Dataset {args.dataset} not supported')
    
    dm.prepare_data(flag='prepare')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ncaltech101')
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    L.seed_everything(777)
    main(args)