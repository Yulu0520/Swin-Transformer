"""Architecture checks shared by training, inference, and visualization."""

LEGACY_ARCHITECTURE = 'swin_cctrans_regression_v1'
ARCHITECTURE = 'swin_cctrans_regression_v2_mdc_preconv'
OUTPUT_STRIDE = 8


def architecture_for_mdc(mdc_mode):
    if mdc_mode not in ('legacy', 'preconv'):
        raise ValueError('Unknown MDC mode: {}'.format(mdc_mode))
    return ARCHITECTURE if mdc_mode == 'preconv' else LEGACY_ARCHITECTURE


def checkpoint_mdc_mode(checkpoint):
    """Identify raw .pth files as well as metadata checkpoints from weight keys."""
    state = checkpoint.get('model_state_dict', checkpoint)
    prefixes = tuple('regression.pre{}.'.format(index) for index in (1, 2, 3))
    has_preconv = any(key.startswith(prefixes) for key in state)
    if has_preconv and not all(prefix + '0.weight' in state for prefix in prefixes):
        raise ValueError('Checkpoint has incomplete MDC pre-convolution branches')
    return 'preconv' if has_preconv else 'legacy'


def validate_model_mdc(model, checkpoint):
    mode = checkpoint_mdc_mode(checkpoint)
    if model.mdc_mode != mode:
        raise ValueError("Checkpoint MDC mode '{}' does not match --mdc-mode {}. "
                         "Use --mdc-mode {} for this checkpoint, or train a new baseline "
                         "for the changed structure; --backbone-checkpoint can reuse a clean backbone."
                         .format(mode, model.mdc_mode, mode))


def validate_counting_checkpoint(checkpoint):
    """Reject MAN heads and incompatible metadata before loading any weights."""
    state = checkpoint.get('model_state_dict', checkpoint)
    if any(key.startswith(('encoder.', 'feature_adapter.', 'reg_layer_0.'))
           for key in state):
        raise ValueError(
            'Legacy MAN checkpoint is incompatible with CCTrans Regression. '
            'Train a new baseline, then retrain DoRA and DoRA1.')
    if 'model_state_dict' in checkpoint:
        if checkpoint.get('architecture') not in (ARCHITECTURE, LEGACY_ARCHITECTURE):
            raise ValueError('Unsupported checkpoint architecture: {}'.format(checkpoint.get('architecture')))
        if checkpoint.get('output_stride') != OUTPUT_STRIDE:
            raise ValueError('Checkpoint output_stride must be 8')
    if 'regression.v1.0.weight' not in state or 'regression.res.3.weight' not in state:
        raise ValueError('Checkpoint does not contain the CCTrans Regression head')
    mode = checkpoint_mdc_mode(checkpoint)
    if 'model_state_dict' in checkpoint:
        if checkpoint['architecture'] != architecture_for_mdc(mode):
            raise ValueError('Checkpoint architecture does not match its MDC weight keys')
        if checkpoint.get('mdc_mode', mode) != mode:
            raise ValueError('Checkpoint mdc_mode metadata does not match its weights')
    return checkpoint
