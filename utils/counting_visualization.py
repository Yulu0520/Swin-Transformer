"""Shared CLI and checkpoint handling for density/feature visualization."""

import torch

from .counting_model import build_dual_dora_model, build_model, load_checkpoint


def add_model_arguments(parser):
    parser.add_argument('--model-name', choices=['swin_t_trans', 'swin_l_trans'],
                        default='swin_t_trans')
    parser.add_argument('--dual-route', choices=['baseline', 'dora', 'dora1'],
                        default='dora1')
    parser.add_argument('--dora-rank', type=int, default=4)
    parser.add_argument('--dora-alpha', type=float, default=4.0)
    parser.add_argument('--dora1-rank', type=int, default=4)
    parser.add_argument('--dora1-alpha', type=float, default=4.0)


def load_visualization_model(args, device):
    checkpoint = load_checkpoint(args.model_path)
    state = checkpoint.get('model_state_dict', checkpoint)
    dual = any('.parametrizations.weight.0.dora1.' in key for key in state)
    adapter = any('.parametrizations.weight.' in key for key in state)
    if dual and checkpoint.get('stage', 'dora1') == 'dora1':
        model = build_dual_dora_model(args, args.model_path)
    else:
        model = build_model(args, args.model_path, expect_dora=adapter)
        if dual:
            model.set_adapter_route(args.dual_route)
    return model.to(device).eval()


def visualization_device(device):
    return torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
