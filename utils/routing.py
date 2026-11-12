"""Hard image routing shared by mix fine-tuning and evaluation."""

import math
import os

import numpy as np
import torch


ROUTES = ('dora1', 'dora')  # router classes are day=0, night=1


def route_from_filename(name):
    if os.path.basename(name).lower().startswith('night_'):
        return 'night', 'dora'
    return 'day', 'dora1'


def select_routes(mode, names, router_inputs=None, router=None, device=None):
    if mode == 'filename':
        domains, routes = zip(*(route_from_filename(name) for name in names))
        return list(domains), list(routes), [1.0] * len(names)
    if mode != 'classifier' or router is None:
        raise ValueError('Classifier routing needs a trained router')
    router.eval()
    with torch.no_grad():
        logits = router(router_inputs.to(device))
        if logits.shape != (len(names), 2):
            raise ValueError('Router output must have shape [batch, 2]')
        confidence, labels = logits.softmax(1).max(1)
    labels = labels.cpu().tolist()
    return (['day' if index == 0 else 'night' for index in labels],
            [ROUTES[index] for index in labels], confidence.cpu().tolist())


def forward_routed_regression(model, inputs, routes):
    """Run one frozen backbone route/image, then one shared trainable head batch."""
    if len(routes) != len(inputs) or not routes or any(route not in ROUTES for route in routes):
        raise ValueError('Expected one day/night adapter route per input')
    groups, indices = [], []
    with torch.no_grad():
        for route in ROUTES:
            selected = [index for index, value in enumerate(routes) if value == route]
            if selected:
                model.set_adapter_route(route)
                groups.append(model.forward_backbone(inputs[selected]))
                indices.extend(selected)
        inverse = torch.tensor(indices, device=inputs.device).argsort()
        stages = [torch.cat([group[stage] for group in groups], dim=0)[inverse]
                  for stage in range(3)]
    return model.regression(*stages)


def split_large_image(inputs, maximum_size=3584):
    _, _, height, width = inputs.shape
    if height < maximum_size and width < maximum_size:
        return [inputs]
    h_splits, w_splits = math.ceil(height / maximum_size), math.ceil(width / maximum_size)
    h_step, w_step = height // h_splits, width // w_splits
    return [inputs[:, :, i * h_step:(i + 1) * h_step if i + 1 < h_splits else height,
                   j * w_step:(j + 1) * w_step if j + 1 < w_splits else width]
            for i in range(h_splits) for j in range(w_splits)]


def count_metrics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if not len(errors):
        return dict(images=0, mae=None, rmse=None)
    return dict(images=len(errors), mae=float(np.abs(errors).mean()),
                rmse=float(np.sqrt(np.square(errors).mean())))
