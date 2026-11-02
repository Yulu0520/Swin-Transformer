"""Architecture checks shared by training, inference, and visualization."""

ARCHITECTURE = 'swin_cctrans_regression_v1'
OUTPUT_STRIDE = 8


def validate_counting_checkpoint(checkpoint):
    """Reject MAN heads and incompatible metadata before loading any weights."""
    state = checkpoint.get('model_state_dict', checkpoint)
    if any(key.startswith(('encoder.', 'feature_adapter.', 'reg_layer_0.'))
           for key in state):
        raise ValueError(
            'Legacy MAN checkpoint is incompatible with CCTrans Regression. '
            'Train a new baseline, then retrain DoRA and DoRA1.')
    if 'model_state_dict' in checkpoint:
        if checkpoint.get('architecture') != ARCHITECTURE:
            raise ValueError('Checkpoint architecture must be {}'.format(ARCHITECTURE))
        if checkpoint.get('output_stride') != OUTPUT_STRIDE:
            raise ValueError('Checkpoint output_stride must be 8')
    if 'regression.v1.0.weight' not in state or 'regression.res.3.weight' not in state:
        raise ValueError('Checkpoint does not contain the CCTrans Regression head')
    return checkpoint
