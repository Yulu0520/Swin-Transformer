"""Strict checkpoint construction for the standalone single-DoRA workflow."""

from models import swin_c_multibatch as models
from utils.stage3_bank import FEATURE_POSITIONS, baseline_fingerprint, validate_bank


def standalone_state(state):
    """Keep trained nighttime DoRA when importing a legacy parallel checkpoint."""
    converted = {}
    for name, value in state.items():
        if any(marker in name for marker in (
                '.parametrizations.weight.0.dora1.',
                '.parametrizations.weight.0.lora.')):
            continue
        converted[name.replace('.parametrizations.weight.0.dora.',
                               '.parametrizations.weight.0.')] = value
    return converted


def load_research_model(checkpoint, model_name=None, dora_rank=4, dora_alpha=4.0):
    """Load baseline, standalone/legacy DoRA, or self-contained Phase 3 weights."""
    config = checkpoint.get('research_config', {})
    saved_name = config.get('model_name', checkpoint.get('model_name'))
    if model_name is not None and saved_name and model_name != saved_name:
        raise ValueError('Checkpoint Swin variant does not match requested model')
    model_name = model_name or saved_name
    if model_name not in ('swin_t_trans', 'swin_l_trans'):
        raise ValueError('A supported Swin model name is required')
    state = standalone_state(checkpoint.get('model_state_dict', checkpoint))
    model = getattr(models, model_name)(pretrained=False)
    has_dora = any('.parametrizations.weight.0.dora_A_q' in key for key in state)
    rank = config.get('dora_rank', checkpoint.get('dora_rank', dora_rank))
    alpha = config.get('dora_alpha', checkpoint.get('dora_alpha', dora_alpha))
    if has_dora:
        model.enable_dora(rank, alpha)
    cross_stages = [number for number in (3, 4)
                    if 'stage{}_cross_attention.codebook'.format(number) in state]
    if len(cross_stages) > 1:
        raise ValueError('Checkpoint must contain only one active memory stage')
    has_cross = bool(cross_stages)
    stage = 'cross_attention' if has_cross else ('dora' if has_dora else 'baseline')
    if config and config.get('stage') != stage:
        raise ValueError('Checkpoint stage metadata does not match its weights')
    if has_cross:
        if not config or not has_dora:
            raise ValueError('Phase 3 checkpoint requires configuration and nighttime DoRA')
        memory_stage = cross_stages[0]
        if config.get('memory_stage', 3) != memory_stage:
            raise ValueError('Checkpoint memory stage does not match its weights')
        cross_key = 'stage{}_cross_attention.codebook'.format(memory_stage)
        bank = dict(config['memory_metadata'], codebook=state[cross_key])
        validate_bank(bank, model_name, config['memory_size'], memory_stage)
        if (config.get('feature_position', FEATURE_POSITIONS[3] if memory_stage == 3 else None)
                != FEATURE_POSITIONS[memory_stage]
                or config.get('feature_dim', bank['feature_dim'] if memory_stage == 3 else None)
                != bank['feature_dim']):
            raise ValueError('Checkpoint memory feature position or dimension is invalid')
        model.enable_cross_attention(
            bank['codebook'], config['cross_attention_heads'],
            config['alpha_init'], config['beta_init'], config['train_fusion'],
            memory_stage=memory_stage)
    model.load_state_dict(state, strict=True)
    fingerprint = baseline_fingerprint(state)
    if config.get('baseline_weights_sha256', fingerprint) != fingerprint:
        raise ValueError('Checkpoint baseline fingerprint does not match weights')
    if has_cross and config['memory_metadata']['baseline_weights_sha256'] != fingerprint:
        raise ValueError('Memory bank and nighttime DoRA have different baselines')
    model.research_config = dict(config, model_name=model_name, stage=stage,
                                 dora_rank=rank if has_dora else 0,
                                 dora_alpha=alpha if has_dora else 0,
                                 baseline_weights_sha256=fingerprint,
                                 count_loss_weight=config.get(
                                     'count_loss_weight', checkpoint.get('count_loss_weight', 0.0)))
    if has_cross:
        model.research_config.update(memory_stage=memory_stage,
                                     feature_dim=bank['feature_dim'],
                                     feature_position=FEATURE_POSITIONS[memory_stage])
    model.set_inference_domain('night')
    return model


def attach_memory(model, bank, heads=8, alpha_init=1.0, beta_init=0.1,
                  train_fusion=True, memory_stage=None):
    config = model.research_config
    if config['stage'] != 'dora':
        raise ValueError('Phase 3 must start from a nighttime DoRA checkpoint')
    validate_bank(bank, config['model_name'], memory_stage=memory_stage)
    memory_stage = bank['stage']
    if bank['baseline_weights_sha256'] != config['baseline_weights_sha256']:
        raise ValueError('Memory bank and nighttime DoRA have different baselines')
    model.enable_cross_attention(bank['codebook'], heads, alpha_init, beta_init,
                                 train_fusion, memory_stage=memory_stage)
    model.research_config = dict(
        config, stage='cross_attention', memory_size=bank['memory_size'],
        memory_stage=memory_stage, feature_dim=bank['feature_dim'],
        feature_position=FEATURE_POSITIONS[memory_stage],
        cross_attention_heads=heads, alpha_init=alpha_init, beta_init=beta_init,
        train_fusion=train_fusion,
        memory_metadata={key: value for key, value in bank.items()
                         if key != 'codebook'})


def research_payload(model):
    # The codebook is stored exactly once, as a persistent model buffer.
    config = dict(model.research_config)
    if config['stage'] == 'baseline':
        config['baseline_weights_sha256'] = baseline_fingerprint(model.state_dict())
    return {'format_version': 2 if 'memory_stage' in config else 1, 'workflow': 'memory',
            'stage': config['stage'], 'model_name': config['model_name'],
            'dora_rank': config['dora_rank'], 'dora_alpha': config['dora_alpha'],
            'count_loss_weight': config.get('count_loss_weight', 0.0),
            'adapter_type': 'dora' if config['dora_rank'] else None,
            'research_config': config, 'model_state_dict': model.state_dict()}
