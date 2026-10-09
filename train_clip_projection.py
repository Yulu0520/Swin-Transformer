"""Align whole-image Swin features to two fixed CLIP crowd text centres."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from losses.clip_guidance import (
    CLIPTeacher, text_alignment_logits, TEXT_ALIGNMENT_TARGET,
    DEFAULT_NIGHT_PROMPT, DEFAULT_DAY_PROMPT,
)
from models.clip_projection import CLIPProjection, backbone_signature, branch_signature
from models import swin_c_multibatch as models
from utils.clip_model_loading import load_trained_dual


def split_scenes(paths, fraction, seed, prefix='night_'):
    groups = {}
    for path in sorted(paths):
        name = Path(path).name
        scene = name[len(prefix):] if name.startswith(prefix) else name
        groups.setdefault(scene, []).append(path)
    if len(groups) < 2:
        raise ValueError('Calibration requires at least two independent scenes')
    scenes = sorted(groups)
    random.Random(seed).shuffle(scenes)
    count = max(1, min(len(scenes) - 1, round(len(scenes) * fraction)))
    held_out = set(scenes[:count])
    train, val = [], []
    for scene, files in groups.items():
        (val if scene in held_out else train).extend(files)
    return train, val


class CalibrationImages(Dataset):
    def __init__(self, paths, crop_size, training, night_prefix='night_'):
        self.paths, self.crop_size, self.training = paths, crop_size, training
        self.night_prefix = night_prefix
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = source.convert('RGB')
        if min(image.size) < self.crop_size:
            image = TF.resize(image, self.crop_size)
        if self.training:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, (self.crop_size, self.crop_size))
            image = TF.crop(image, i, j, h, w)
            if random.random() < 0.5:
                image = TF.hflip(image)
        else:
            image = TF.center_crop(image, [self.crop_size, self.crop_size])
        rgb = TF.to_tensor(image)
        route_id = int(Path(self.paths[index]).name.startswith(self.night_prefix))
        return self.normalize(rgb), route_id, index


@torch.no_grad()
def extract_calibration_features(model, inputs, route_ids=None):
    if route_ids is None:
        return model.extract_backbone_features(inputs)
    features = None
    for route_id, route in ((0, 'dora1'), (1, 'dora')):
        indices = (route_ids == route_id).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        model.set_adapter_route(route)
        selected = model.extract_backbone_features(inputs[indices])
        if features is None:
            features = selected.new_empty((inputs.size(0),) + selected.shape[1:])
        features[indices] = selected
    return features


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dual-checkpoint', required=True,
                        help='trained day DoRA1 + night DoRA checkpoint')
    parser.add_argument('--dora-rank', type=int, default=4)
    parser.add_argument('--dora-alpha', type=float, default=4.0)
    parser.add_argument('--dora1-rank', type=int, default=4)
    parser.add_argument('--dora1-alpha', type=float, default=4.0)
    parser.add_argument('--model-name', choices=['swin_t_trans', 'swin_l_trans'],
                        default='swin_t_trans')
    parser.add_argument('--train-dirs', nargs='+', required=True,
                        help='day/train and night/train; do not pass benchmark val/test')
    parser.add_argument('--clip-model', default='ViT-B/32')
    parser.add_argument('--clip-cache-dir', default='pre_models/clip')
    parser.add_argument('--clip-negative-prompt', default=DEFAULT_NIGHT_PROMPT)
    parser.add_argument('--clip-positive-prompt', default=DEFAULT_DAY_PROMPT)
    parser.add_argument('--clip-temperature', type=float, default=0.07)
    parser.add_argument('--output', default='model/clip_projection/best_projection.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--holdout-fraction', type=float, default=0.1)
    parser.add_argument('--pair-prefix', default='night_')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not np.isfinite(args.clip_temperature) or args.clip_temperature <= 0:
        parser.error('clip-temperature must be finite and positive')
    if not (0 < args.holdout_fraction < 1):
        parser.error('holdout-fraction must be between zero and one')
    if (min(args.epochs, args.batch_size, args.crop_size) <= 0 or
            args.crop_size % 16 or args.num_workers < 0 or
            not np.isfinite(args.lr) or args.lr <= 0):
        parser.error('invalid epochs, batch size, crop size, workers or learning rate')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = sorted({str(path.resolve()) for directory in args.train_dirs
                    for path in Path(directory).glob('*.jpg')
                    if not path.name.startswith('._')})
    train_paths, val_paths = split_scenes(
        paths, args.holdout_fraction, args.seed, args.pair_prefix)
    if not args.pair_prefix:
        parser.error('Text calibration requires a nonempty night filename prefix')
    for files in (train_paths, val_paths):
        domains = {Path(path).name.startswith(args.pair_prefix) for path in files}
        if domains != {False, True}:
            parser.error('Text calibration needs day and night images in both grouped subsets')
    print('Calibration train images: {}; internal holdout: {}'.format(
        len(train_paths), len(val_paths)))
    device = torch.device(args.device)
    model = getattr(models, args.model_name)(pretrained=False)
    checkpoint_path = args.dual_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_trained_dual(model, checkpoint, args.model_name,
                      args.dora_rank, args.dora_alpha, args.dora1_rank, args.dora1_alpha)
    signature = backbone_signature(model)
    branch_metadata = dict(day_branch_signature=branch_signature(model, 'dora1'),
                           night_branch_signature=branch_signature(model, 'dora'))
    model.requires_grad_(False).eval().to(device)
    teacher = CLIPTeacher(args.clip_model, device, args.clip_cache_dir,
                          args.clip_negative_prompt, args.clip_positive_prompt)
    projection = CLIPProjection(model.backbone_channels, teacher.embedding_dim).to(device)
    optimizer = torch.optim.Adam(projection.parameters(), lr=args.lr)
    loaders = {}
    for index, (split, files) in enumerate((('train', train_paths), ('val', val_paths))):
        loaders[split] = DataLoader(
            CalibrationImages(files, args.crop_size, split == 'train', args.pair_prefix),
            batch_size=args.batch_size, shuffle=split == 'train',
            num_workers=args.num_workers, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed + index))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    for epoch in range(args.epochs):
        metrics = {}
        for split, loader in loaders.items():
            training = split == 'train'
            projection.train(training)
            total, count, correct, positive_cosine = 0.0, 0, 0, 0.0
            scene_embeddings = {}
            for batch in tqdm(loader, desc='{} {}/{}'.format(
                    split, epoch + 1, args.epochs)):
                inputs, labels, indices = batch
                labels = labels.to(device)
                with torch.no_grad():
                    features = extract_calibration_features(model, inputs.to(device), labels)
                with torch.set_grad_enabled(training):
                    embeddings = projection(features)
                    logits = text_alignment_logits(embeddings, teacher.text_centres,
                                                   args.clip_temperature)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                total += loss.item() * inputs.size(0)
                count += inputs.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                positive_cosine += (logits.detach() * args.clip_temperature).gather(
                    1, labels[:, None]).sum().item()
                if not training:
                    for index, label, embedding in zip(indices.tolist(), labels.tolist(),
                                                       embeddings.detach().cpu()):
                        name = Path(loader.dataset.paths[index]).name
                        scene = name[len(args.pair_prefix):] if label == 1 else name
                        scene_embeddings.setdefault(scene, {})[label] = embedding
            metrics[split] = total / count
            metrics[split + '_accuracy'] = correct / count
            metrics[split + '_positive_cosine'] = positive_cosine / count
            if not training:
                deltas = [pair[0] - pair[1] for pair in scene_embeddings.values()
                          if 0 in pair and 1 in pair]
                metrics['val_direction_cosine'] = (
                    torch.nn.functional.cosine_similarity(
                        torch.stack(deltas), teacher.text_direction.detach().cpu(),
                        dim=-1, eps=1e-3).mean().item() if deltas else None)
        print('Epoch {}: train={:.5f}, holdout={:.5f}'.format(
            epoch + 1, metrics['train'], metrics['val']))
        print('Holdout day/night accuracy={:.3f}, positive cosine={:.3f}'.format(
            metrics['val_accuracy'], metrics['val_positive_cosine']))
        print('Holdout paired night-to-day direction cosine={}'.format(
            metrics['val_direction_cosine']))
        if metrics['val'] < best:
            best = metrics['val']
            torch.save(dict(
                projection_state_dict={key: value.detach().cpu()
                                       for key, value in projection.state_dict().items()},
                model_name=args.model_name, clip_model=args.clip_model,
                input_dim=model.backbone_channels, embedding_dim=teacher.embedding_dim,
                image_normalization='imagenet', backbone_signature=signature,
                calibration_mode='dual_dora', alignment_target=TEXT_ALIGNMENT_TARGET,
                day_prompt=args.clip_positive_prompt, night_prompt=args.clip_negative_prompt,
                text_centres=teacher.text_centres.detach().cpu(),
                temperature=args.clip_temperature,
                source_checkpoint=os.path.abspath(checkpoint_path),
                **branch_metadata,
                calibration_args=vars(args), holdout_loss=best,
                holdout_accuracy=metrics['val_accuracy'],
                holdout_positive_cosine=metrics['val_positive_cosine'], epoch=epoch + 1,
                holdout_direction_cosine=metrics['val_direction_cosine'],
                train_paths=train_paths, holdout_paths=val_paths), args.output)
    print('Saved best projection: {}'.format(args.output))


if __name__ == '__main__':
    main()
