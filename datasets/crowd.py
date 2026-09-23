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
                 method='train', name_prefix=None):

        self.root_path = root_path
        self.im_list = sorted(
            path for path in glob(os.path.join(self.root_path, '*.jpg'))
            if (not os.path.basename(path).startswith('._')
                and (name_prefix is None
                     or os.path.basename(path).startswith(name_prefix)))
        )
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
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(os.path.basename(img_path).split('.')[0])
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        # assert len(keypoints) > 0
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        re_size = random.random() * 0.5 + 0.75
        wdd = (int)(wd*re_size)
        htt = (int)(ht*re_size)
        if min(wdd, htt) >= self.c_size:
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            keypoints = keypoints*re_size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
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
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            target = np.array([])
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size


class PairedCrowd(Crowd):
    """Aligned day/night crops used only for feature matching."""

    def __init__(self, day_root, night_root, crop_size, downsample_ratio,
                 is_gray=False):
        super().__init__(night_root, crop_size, downsample_ratio, is_gray,
                         'train', name_prefix='night_')
        self.day_root = day_root
        self.pairs = []
        expected_day_names = set()
        for night_path in self.im_list:
            night_name = os.path.basename(night_path)
            day_path = os.path.join(day_root, night_name.removeprefix('night_'))
            if not os.path.isfile(day_path):
                raise FileNotFoundError(
                    'missing paired day image for {}'.format(night_path)
                )
            with Image.open(day_path) as day, Image.open(night_path) as night:
                if day.size != night.size:
                    raise ValueError(
                        'unaligned pair sizes: {} {}'.format(day_path, night_path)
                    )
            expected_day_names.add(os.path.basename(day_path))
            self.pairs.append((day_path, night_path))
        available_day_names = {
            os.path.basename(path)
            for path in glob(os.path.join(day_root, 'IMG_*.jpg'))
            if not os.path.basename(path).startswith('._')
        }
        if available_day_names != expected_day_names:
            raise ValueError('unpaired day images found in {}'.format(day_root))

    def __getitem__(self, item):
        day_path, night_path = self.pairs[item]
        with Image.open(day_path) as image:
            day = image.convert('RGB')
        with Image.open(night_path) as image:
            night = image.convert('RGB')
        if day.size != night.size:
            raise ValueError('unaligned pair sizes: {} {}'.format(day_path, night_path))
        return self.paired_train_transform(day, night)

    def paired_train_transform(self, day, night):
        width, height = day.size
        resize_scale = random.random() * 0.5 + 0.75
        resized_width = int(width * resize_scale)
        resized_height = int(height * resize_scale)
        if min(resized_width, resized_height) >= self.c_size:
            width, height = resized_width, resized_height
            day = day.resize((width, height))
            night = night.resize((width, height))
        if min(width, height) < self.c_size:
            raise ValueError('paired image is smaller than crop size')

        top, left, crop_h, crop_w = random_crop(
            height, width, self.c_size, self.c_size
        )
        day = F.crop(day, top, left, crop_h, crop_w)
        night = F.crop(night, top, left, crop_h, crop_w)
        flip = random.random() > 0.5
        if flip:
            day = F.hflip(day)
            night = F.hflip(night)
        return self.trans(day), self.trans(night)
