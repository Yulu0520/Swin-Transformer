from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def gen_discrete_map(height, width, points, downsample_ratio=8):
    """Scatter unit head counts, then block-sum to the density grid.

    Points must be inside the crop. Rounding near the right/bottom boundary
    is clipped to the last pixel; duplicate points accumulate rather than overwrite.
    """
    if height % downsample_ratio or width % downsample_ratio:
        raise ValueError('Discrete-map dimensions must be divisible by the stride')
    discrete = np.zeros((height, width), dtype=np.float32)
    if len(points):
        pixels = np.rint(points).astype(np.int64)
        xs = np.clip(pixels[:, 0], 0, width - 1)
        ys = np.clip(pixels[:, 1], 0, height - 1)
        np.add.at(discrete, (ys, xs), 1.0)
    discrete = discrete.reshape(height // downsample_ratio, downsample_ratio,
                                width // downsample_ratio, downsample_ratio).sum(axis=(1, 3))
    if discrete.sum() != len(points):
        raise RuntimeError('Discrete map did not preserve the number of head points')
    return discrete[None, ...]


def load_points(path):
    points = np.load(path)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError('Expected an (N, 2+) point array in {}'.format(path))
    points = points[:, :2].astype(np.float32, copy=True)
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite annotation coordinates in {}'.format(path))
    return points


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.root_path = root_path
        self.im_list = sorted(path for path in glob(os.path.join(self.root_path, '*.jpg'))
                              if not os.path.basename(path).startswith('._'))
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        with Image.open(img_path) as image:
            img = image.convert('RGB')
        if self.method == 'train':
            keypoints = load_points(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = load_points(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        """Keep crop-internal head centers as unit targets for DM-Count."""
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.size == 0:
            keypoints = np.empty((0, 2), dtype=np.float32)
        elif keypoints.ndim != 2 or keypoints.shape[1] < 2:
            raise ValueError('Training annotations must have shape (N, 2+)')
        else:
            keypoints = keypoints[:, :2].copy()
        wd, ht = img.size
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        re_size = random.random() * 0.5 + 0.75
        resized_w, resized_h = int(wd * re_size), int(ht * re_size)
        if min(resized_w, resized_h) >= self.c_size:
            img = img.resize((resized_w, resized_h))
            # Match the actual integer image dimensions on each axis.
            keypoints *= [resized_w / wd, resized_h / ht]
            wd, ht = resized_w, resized_h
        if min(wd, ht) < self.c_size:
            raise ValueError('Training images must be at least crop_size on both axes')
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        keypoints -= [j, i]
        # A half-open crop excludes centers on/outside its far boundary.
        inside = ((keypoints[:, 0] >= 0) & (keypoints[:, 0] < w)
                  & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h))
        keypoints = keypoints[inside]
        if random.random() > 0.5:
            img = F.hflip(img)
            # Pixel-center convention, matching PIL's horizontal flip.
            keypoints[:, 0] = np.clip(w - 1 - keypoints[:, 0], 0, w - 1)
        discrete = gen_discrete_map(h, w, keypoints, self.d_ratio)
        return (self.trans(img), torch.from_numpy(keypoints.copy()).float(),
                torch.from_numpy(discrete.copy()).float())
