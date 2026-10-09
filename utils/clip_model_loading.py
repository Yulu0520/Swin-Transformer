"""Load both trained branches for calibration or night-only adaptation."""

import math


def load_trained_dual(model, checkpoint, model_name, dora_rank, dora_alpha,
                      dora1_rank, dora1_alpha, active_route='dora',
                      trainable_route=None):
    state = checkpoint.get('model_state_dict', checkpoint)
    settings = dict(model_name=model_name, dora_rank=dora_rank,
                    dora_alpha=dora_alpha, dora1_rank=dora1_rank,
                    dora1_alpha=dora1_alpha)
    if 'model_state_dict' in checkpoint:
        if checkpoint.get('adapter_type') != 'parallel_dora_dora1':
            raise ValueError('Expected a trained parallel DoRA/DoRA1 checkpoint')
        for name, expected in settings.items():
            actual = checkpoint.get(name)
            matches = (isinstance(actual, (int, float)) and
                       math.isclose(actual, expected)) if isinstance(expected, float) else actual == expected
            if not matches:
                raise ValueError('Dual checkpoint {} mismatch'.format(name))
    for route in ('dora', 'dora1'):
        for suffix in ('dora_A_q', 'dora_B_q', 'dora_A_v', 'dora_B_v',
                       'magnitude_q', 'magnitude_v'):
            if not any('.parametrizations.weight.0.{}.{}'.format(route, suffix)
                       in name for name in state):
                raise ValueError('Dual checkpoint is missing trained {} parameters'.format(route))
    count = model.enable_dual_dora(
        dora_rank, dora_alpha, dora1_rank, dora1_alpha,
        active_route=active_route, trainable_route=trainable_route)
    model.load_state_dict(state)
    return count
