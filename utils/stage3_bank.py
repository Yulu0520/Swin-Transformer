"""Stage 3/4 memory banks; module path retained for Stage 3 compatibility."""

import hashlib
from contextlib import nullcontext

import numpy as np
import torch


STAGE3_DIMS = {'swin_t_trans': 384, 'swin_l_trans': 768}
STAGE4_DIMS = {'swin_t_trans': 768, 'swin_l_trans': 1536}
STAGE_DIMS = {3: STAGE3_DIMS, 4: STAGE4_DIMS}
FEATURE_POSITIONS = {3: 'before_stage4_patch_merging', 4: 'before_backbone_norm'}


def load_file(path):
    # Legacy trainers stored MAE/RMSE as NumPy scalars. Allow only those known
    # scalar constructors, while keeping weights-only loading on newer torch.
    allowed = [np.float64(0).__reduce__()[0], np.dtype,
               type(np.dtype('float64')), type(np.dtype('float32'))]
    context = (torch.serialization.safe_globals(allowed)
               if hasattr(torch.serialization, 'safe_globals') else nullcontext())
    with context:
        try:
            return torch.load(path, map_location='cpu', weights_only=True)
        except TypeError as error:
            if 'weights_only' not in str(error):
                raise
            # Support the repository's torch 1.12 minimum.
            return torch.load(path, map_location='cpu')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_fingerprint(state):
    """Identify raw baseline tensors consistently with or without DoRA attached."""
    digest = hashlib.sha256()
    normalized = {}
    for name, value in state.items():
        if (name.startswith(('stage3_cross_attention.', 'stage4_cross_attention.'))
                or '.parametrizations.weight.0.' in name):
            continue
        name = name.replace('.parametrizations.weight.original', '.weight')
        normalized[name] = value
    for name, value in sorted(normalized.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_bank(bank, model_name=None, memory_size=None, memory_stage=None):
    if not isinstance(bank, dict) or 'codebook' not in bank:
        raise ValueError('Memory bank must include codebook and metadata')
    variant = bank.get('model_name')
    stage = bank.get('stage')
    dim = STAGE_DIMS.get(stage, {}).get(variant)
    codebook = bank['codebook']
    if (dim is None or not isinstance(codebook, torch.Tensor)
            or bank.get('feature_dim') != dim
            or codebook.ndim != 2
            or tuple(codebook.shape) != (bank.get('memory_size'), dim)
            or codebook.shape[0] <= 0
            or not codebook.is_floating_point()
            or not torch.isfinite(codebook).all()):
        raise ValueError('Invalid Stage 3/4 memory bank shape or metadata')
    # Old Stage 3 banks predate the position field. Stage 4 must be explicit.
    position = bank.get('feature_position', FEATURE_POSITIONS[3] if stage == 3 else None)
    if position != FEATURE_POSITIONS[stage]:
        raise ValueError('Memory bank feature position does not match the supported pre-norm path')
    if memory_stage is not None and memory_stage != stage:
        raise ValueError('Memory bank stage {} does not match --memory-stage {}'.format(stage, memory_stage))
    if model_name is not None and model_name != variant:
        raise ValueError('Memory bank Swin variant does not match model')
    if memory_size is not None and memory_size != codebook.shape[0]:
        raise ValueError('Memory bank size does not match --memory-size')
    if not bank.get('baseline_weights_sha256'):
        raise ValueError('Memory bank is missing baseline identity')
    return bank


class TokenReservoir:
    """Uniform priority sampling with a fixed RAM bound and deterministic RNG."""
    def __init__(self, capacity, seed):
        if capacity <= 0:
            raise ValueError('Token capacity must be positive')
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.tokens = None
        self.priorities = np.empty(0)

    def add(self, tokens):
        tokens = np.asarray(tokens, dtype=np.float32)
        if tokens.ndim != 2 or not np.isfinite(tokens).all():
            raise ValueError('Expected finite [N, C] sampled memory tokens')
        priorities = self.rng.random(len(tokens))
        if self.tokens is not None:
            tokens = np.concatenate((self.tokens, tokens))
            priorities = np.concatenate((self.priorities, priorities))
        if len(tokens) > self.capacity:
            selected = np.argpartition(priorities, -self.capacity)[-self.capacity:]
            tokens, priorities = tokens[selected], priorities[selected]
        self.tokens, self.priorities = tokens, priorities


def cluster_tokens(tokens, memory_size=256, seed=42, batch_size=4096):
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits
    if tokens is None or len(tokens) < memory_size:
        raise ValueError('Not enough sampled training tokens for the memory size')
    # One CPU thread avoids nondeterministic reduction order across runs.
    with threadpool_limits(limits=1):
        estimator = MiniBatchKMeans(
            n_clusters=memory_size, random_state=seed, batch_size=batch_size,
            n_init=3, reassignment_ratio=0.01)
        estimator.fit(tokens)
    return torch.from_numpy(estimator.cluster_centers_.copy()).float()
