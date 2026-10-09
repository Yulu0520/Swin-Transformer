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


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area



class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train', pair_root=None, pair_prefix='night_',
                 disable_gray=False):

        self.root_path = root_path
        self.im_list = sorted(path for path in glob(os.path.join(self.root_path, '*.jpg'))
                              if not os.path.basename(path).startswith('._'))
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method
        self.pair_root = pair_root
        self.pair_prefix = pair_prefix
        self.disable_gray = disable_gray
        if pair_root:
            if method != 'train' or is_gray:
                raise ValueError('CLIP pairs require RGB training data')
            self.pair_list = [os.path.join(
                pair_root, (os.path.basename(path)[len(pair_prefix):]
                            if os.path.basename(path).startswith(pair_prefix)
                            else os.path.basename(path)))
                for path in self.im_list]
            missing = [path for path in self.pair_list if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError('Missing paired image: {}'.format(missing[0]))

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
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(os.path.basename(img_path).split('.')[0])
        if self.method == 'train':
            keypoints = np.load(gd_path)
            if self.pair_root:
                with Image.open(self.pair_list[item]) as source:
                    pair = source.convert('RGB')
                return self.train_transform(img, keypoints, pair)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

    def train_transform(self, img, keypoints, pair=None):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        paired = pair is not None
        if paired and pair.size != img.size:
            raise ValueError('Paired images must have the same geometry: {} vs {}'.format(
                img.size, pair.size))
        # assert len(keypoints) > 0
        if not paired and not self.disable_gray and random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        re_size = random.random() * 0.5 + 0.75
        wdd = (int)(wd*re_size)
        htt = (int)(ht*re_size)
        if min(wdd, htt) >= self.c_size:
            original_wd, original_ht = wd, ht
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            if paired:
                pair = pair.resize((wd, ht))
                keypoints = keypoints.copy()
                scale_x, scale_y = wd / original_wd, ht / original_ht
                keypoints[:, 0] *= scale_x
                keypoints[:, 1] *= scale_y
                keypoints[:, 2] *= (scale_x * scale_y) ** 0.5
            else:
                keypoints = keypoints*re_size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if paired:
            pair = F.crop(pair, i, j, h, w)
        if len(keypoints) > 0:
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(j, i, j + w, i + h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)

            target = ratio[mask]
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                if paired:
                    pair = F.hflip(pair)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            target = np.array([])
            if random.random() > 0.5:
                img = F.hflip(img)
                if paired:
                    pair = F.hflip(pair)
        result = (self.trans(img), torch.from_numpy(keypoints.copy()).float(),
                  torch.from_numpy(target.copy()).float(), st_size)
        if paired:
            return result + (F.to_tensor(pair),)
        return result
