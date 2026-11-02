"""Visualize fused and regression features (the MAN attention is removed)."""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from utils.counting_visualization import (
    add_model_arguments, load_visualization_model, visualization_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CCTrans regression features')
    parser.add_argument('--data-dir', default='data/example.jpg', help='input image')
    parser.add_argument('--model-path', default='model/best_model.pth')
    parser.add_argument('--device', default='0', help='GPU index or cpu')
    parser.add_argument('--output-dir', default='')
    add_model_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device if args.device != 'cpu' else ''
    device = visualization_device(args.device)
    model = load_visualization_model(args, device)
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.model_path), 'features')
    os.makedirs(output_dir, exist_ok=True)
    with Image.open(args.data_dir) as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    inputs = transform(rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        stages = model.forward_backbone(inputs)
        fused, regression = model.regression.forward_features(*stages)
    # Channel-mean activation magnitude, not an attention probability map.
    for name, feature in [('fused', fused), ('regression', regression)]:
        heatmap = feature[0].abs().mean(dim=0).cpu().numpy()
        heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
        heatmap = (heatmap - heatmap.min()) / max(float(np.ptp(heatmap)), 1e-8)
        colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.4,
                                  colored, 0.6, 0)
        cv2.imwrite(os.path.join(output_dir, name + '.png'), overlay)
    print('Saved feature overlays to {}'.format(output_dir))


if __name__ == '__main__':
    main()
