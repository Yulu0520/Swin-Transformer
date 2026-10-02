"""Dataset and transforms for the day/night image router."""

import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def label_from_filename(path):
    """Use the dataset naming convention to generate a training label."""
    return int(os.path.basename(path).lower().startswith('night_'))


def build_router_transform(training=False):
    operations = [transforms.Resize((224, 224))]
    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(operations)


class DayNightDataset(Dataset):
    """Read router images; filenames are labels, never model inputs."""

    def __init__(self, root_path, training=False):
        if not os.path.isdir(root_path):
            raise FileNotFoundError(
                'Router dataset directory not found: {}'.format(root_path))
        paths = []
        for extension in ('*.jpg', '*.jpeg', '*.png'):
            paths.extend(glob(os.path.join(root_path, extension)))
        self.im_list = sorted(
            path for path in paths
            if not os.path.basename(path).startswith('._'))
        if not self.im_list:
            raise ValueError('No router images found in {}'.format(root_path))
        self.transform = build_router_transform(training=training)

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        path = self.im_list[index]
        with Image.open(path) as image:
            image = self.transform(image.convert('RGB'))
        return image, label_from_filename(path), os.path.basename(path)
