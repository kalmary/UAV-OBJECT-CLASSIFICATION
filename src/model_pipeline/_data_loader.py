import pathlib as pth
import numpy as np
from typing import Optional, Union, Tuple
import cv2

import torch
from torch.utils.data import IterableDataset, get_worker_info

import random

import sys
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class Dataset(IterableDataset):

    def __init__(self,
                 path_dir: Union[str, pth.Path],
                 batch_size: int,
                 resolution_xy: Optional[Tuple[int, int]] = None,
                 shuffle: bool = True,
                 buffer: int = 200,
                 device: torch.device = torch.device('cpu'),
                 min_visibility: float = 0.3):

        super(Dataset).__init__()

        self.path = pth.Path(path_dir)
        self.resolution_xy = resolution_xy if resolution_xy else (640, 640)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.min_visibility = min_visibility

        # Define augmentation pipeline
        if self.shuffle:
            self.transform = A.Compose([
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,  # ← Use 'fill' instead of 'value'
                    p=0.5
                ),
                
                # Crops and scaling
                A.RandomResizedCrop(
                    size=(self.resolution_xy[1], self.resolution_xy[0]),  # (height, width)
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5
                ),
                
                # Color/appearance augmentations
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                ], p=0.3),
                
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                
                A.OneOf([
                    A.RandomBrightnessContrast(p=1.0),
                    A.RandomGamma(p=1.0),
                    A.CLAHE(p=1.0),
                ], p=0.3),
                
                # Noise
                A.OneOf([
                    A.GaussNoise(noise_scale_factor=(10.0, 50.0), p=1.0),
                    A.ISONoise(p=1.0),
                ], p=0.2),
                
                # Other
                A.RandomShadow(p=0.1),
                
                # Convert to tensor
                ToTensorV2(),
                
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_area=100,
                min_visibility=self.min_visibility
            ))
        else:
            # No augmentation, just resize and convert to tensor
            self.transform = A.Compose([
                A.Resize(height=self.resolution_xy[1], width=self.resolution_xy[0]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))

    def _key_streamer(self):
        """
        Generator over all keys. Each worker processes all keys,
        but only its assigned chunks within each key.
        """
        path_list_imgs = list(self.path.rglob('*.jpg'))
        path_list_bbox = list(self.path.rglob('*.txt'))

        if self.shuffle:
            path_indices = list(range(len(path_list_imgs)))
            random.shuffle(path_indices)
            path_list_imgs = [path_list_imgs[i] for i in path_indices]
            path_list_bbox = [path_list_bbox[i] for i in path_indices]

        worker_info = get_worker_info()
        if worker_info is None:
            iter_paths_bbox = path_list_bbox
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            iter_paths_bbox = path_list_bbox[worker_id::total_workers]

        for path in iter_paths_bbox:
            bbox_path = pth.Path(path)
            img_path = bbox_path.parent.parent.joinpath('images', bbox_path.name.replace('.txt', '.jpg'))

            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Convert BGR to RGB (Albumentations expects RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            bbox = np.loadtxt(str(bbox_path), ndmin=2)

            yield (img, bbox)

    def _data_augmentation(self, img: np.ndarray, bbox: np.ndarray):
        """
        Apply augmentation using Albumentations.
        
        Args:
            img: numpy array (H, W, C) in RGB format
            bbox: numpy array (N, 5) where each row is [class, center_x, center_y, width, height]
        
        Returns:
            img: torch.Tensor (C, H, W)
            bbox: torch.Tensor (N, 4) - [center_x, center_y, width, height] without class
        """
        if bbox.shape[0] == 0:
            # No bboxes, apply transform to get tensor
            transformed = self.transform(image=img, bboxes=[], class_labels=[])
            img_tensor = transformed['image']
            return img_tensor, torch.from_numpy(np.zeros((0, 4))).float()

        # Extract class labels and bbox coordinates
        class_labels = bbox[:, 0].astype(int).tolist()
        bboxes = bbox[:, 1:].tolist()  # [center_x, center_y, width, height]

        try:
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            img_tensor = transformed['image']  # Already a tensor from ToTensorV2
            bboxes_transformed = transformed['bboxes']
            
            # Convert back to numpy array (without class labels)
            if len(bboxes_transformed) > 0:
                bbox_array = np.array(bboxes_transformed)
            else:
                # All bboxes were filtered out
                bbox_array = np.zeros((0, 4))
            
        except Exception as e:
            # If augmentation fails, fallback to original with resize
            print(f"Augmentation failed: {e}, using original image")
            img = cv2.resize(img, self.resolution_xy)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            bbox_array = bbox[:, 1:]  # Extract bbox coordinates only

        # img_tensor is already a tensor, bbox needs conversion
        bbox_tensor = torch.from_numpy(bbox_array).float()

        return img_tensor, bbox_tensor

    def _process_cloud(self):
        stream = self._key_streamer()
        for (img, bbox) in stream:
            if img is None:
                continue

            # Apply augmentation
            img_tensor, bbox_tensor = self._data_augmentation(img, bbox)
            
            # Skip if no valid bboxes remain after augmentation
            if bbox_tensor.shape[0] == 0:
                continue

            yield (img_tensor.to(self.device), bbox_tensor.to(self.device))

    def __iter__(self):
        stream = self._process_cloud()

        img_batch = []
        bbox_batch = []

        for (img, bbox) in stream:
            img_batch.append(img.unsqueeze(0))
            bbox_batch.append(bbox)

            if len(bbox_batch) >= self.batch_size:
                img_batch = torch.vstack(img_batch).float()
                yield img_batch, bbox_batch
                img_batch = []
                bbox_batch = []

        if len(bbox_batch) > 0:
            img_batch = torch.vstack(img_batch).float()
            yield img_batch, bbox_batch