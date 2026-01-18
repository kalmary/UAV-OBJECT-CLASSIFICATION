import pathlib as pth
import numpy as np
import random
import h5py
from typing import Optional, Union

import torch
from torch.utils.data import IterableDataset, get_worker_info

import random

import sys
import os

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils import rotate_points, tilt_points, transform_points
from utils import cloud2sideViews_torch, gaussian_blur


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class Dataset(IterableDataset):

    def __init__(self,
                 path_dir: Union[str, pth.Path],
                 resolution_xy: int,
                 batch_size: int,
                 shuffle: bool = True,
                 weights: Optional[torch.Tensor] = None,
                 buffer: int = 200,
                 device: torch.device = None):

        super(Dataset).__init__()

        self.path = pth.Path(path_dir)
        self.resolution_xy = resolution_xy
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weights = weights
        self.device = device
        

        self.oversample = None
        self.buffer = buffer
        if self.weights is not None:
            self.weights/= self.weights.max()
            self.oversample = (1 - self.weights)*10
            self.oversample = self.oversample.floor().long()


    def _key_streamer(self):
        """
        Generator over all keys. Each worker processes all keys,
        but only its assigned chunks within each key.
        """
        path_list = list(self.path.rglob('*.npy'))
        if self.shuffle:
            random.shuffle(path_list)

        worker_info = get_worker_info()
        if worker_info is None:
            iter_paths = path_list
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            iter_paths = path_list[worker_id::total_workers]

        for path in iter_paths:

            file_name = path.stem
            label = file_name.rsplit('_', 1)[-1]
            label = int(label)

            points = np.load(path)
            # points = torch.from_numpy(points)
            points = torch.from_numpy(points[:, :3]).float()
            label = torch.asarray(label)




            yield  (points, label)


    def _process_cloud(self):
        stream = self._key_streamer()
        for (cloud_tensor, label) in stream:
            cloud_tensor = cloud_tensor.to(self.device)
            if self.shuffle:
                cloud_tensor = transform_points(cloud_tensor, device=self.device)
                cloud_tensor = rotate_points(cloud_tensor, device=self.device)
                cloud_tensor = tilt_points(cloud_tensor, device=self.device)

            cloud_tensor = cloud2sideViews_torch(points=cloud_tensor, resolution_xy=self.resolution_xy)
            if self.shuffle:
                kernel = random.choice([3, 5, 7, 9])
                sigma = random.uniform(1., 3.)
            else:
                kernel = 5
                sigma = 1.5
                
            cloud_tensor = gaussian_blur(cloud_tensor, kernel_size=(kernel, kernel), sigma=sigma, device=self.device)


            cloud_tensor = cloud_tensor.cpu()

            yield cloud_tensor, label

    def __iter__(self):
        stream = self._process_cloud()

        cloud_batch = []
        label_batch = []

        for (cloud, label) in stream:
            # print('SAMPLE', cloud.shape)
            cloud_batch.append(cloud.unsqueeze(0))
            label_batch.append(label)

            if len(label_batch) >= self.batch_size:
                cloud_batch = torch.vstack(cloud_batch).float()
                label_batch = torch.asarray(label_batch).long()
                # print('BATCH', cloud_batch.shape)
                yield cloud_batch, label_batch

                cloud_batch = []
                label_batch = []

        if len(label_batch) > 0:
            cloud_batch = torch.vstack(cloud_batch).float()
            label_batch = torch.asarray(label_batch).long()

            yield cloud_batch, label_batch

