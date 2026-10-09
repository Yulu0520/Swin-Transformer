"""Fixed CLIP text centres and differentiable whole-image alignment losses."""

import math

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_NIGHT_PROMPT = 'a photo of a crowd at night'
DEFAULT_DAY_PROMPT = 'a photo of a crowd during the day'
TEXT_ALIGNMENT_TARGET = 'clip_text_day_night_v1'


def rgb_to_imagenet(rgb):
    mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (rgb - mean) / std


def cosine_loss(predicted, target):
    return (1 - F.cosine_similarity(
        predicted.float(), target.detach().float(), dim=-1, eps=1e-6)).mean()


def text_alignment_logits(projected, text_centres, temperature=0.07):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Text alignment temperature must be finite and positive')
    if text_centres.ndim != 2 or text_centres.shape[0] != 2:
        raise ValueError('Expected two text centres ordered day, night')
    if projected.ndim != 2 or projected.shape[1] != text_centres.shape[1]:
        raise ValueError('Projected features and text centres must share an embedding dimension')
    image = F.normalize(projected.float(), dim=-1, eps=1e-6)
    text = F.normalize(text_centres.detach().float(), dim=-1, eps=1e-6)
    return image @ text.T / temperature


def text_alignment_loss(projected, text_centres, domain_labels, temperature=0.07):
    """Two-centre DICL: day=0, night=1; no object labels or image teacher."""
    return F.cross_entropy(
        text_alignment_logits(projected, text_centres, temperature), domain_labels.long())


def directional_loss(night, day, text_direction):
    # Limit derivatives when paired day/night displacement is very small.
    return (1 - F.cosine_similarity(
        day.detach().float() - night.float(),
        text_direction.detach().float(), dim=-1, eps=1e-3)).mean()


def warmup_factor(epoch_index, warmup_epochs):
    return min(1.0, epoch_index / warmup_epochs) if warmup_epochs else 1.0


class CLIPTeacher(nn.Module):
    """Encode and cache two frozen text centres; never call encode_image."""
    def __init__(self, model_name, device, download_root,
                 negative_prompt=DEFAULT_NIGHT_PROMPT,
                 positive_prompt=DEFAULT_DAY_PROMPT, cached_centres=None):
        super().__init__()
        if cached_centres is not None:
            centres = cached_centres.detach().float().to(device)
            if (centres.ndim != 2 or centres.shape[0] != 2 or centres.shape[1] == 0 or
                    not torch.isfinite(centres).all() or
                    torch.any(centres.norm(dim=-1) < 1e-6)):
                raise ValueError('Invalid cached day/night text centres')
            self._set_centres(centres)
            return
        try:
            import clip
        except ImportError as error:
            raise ImportError('Install OpenAI CLIP with: pip install '
                              'git+https://github.com/openai/CLIP.git') from error
        if not hasattr(clip, 'load'):
            raise ImportError('The installed clip package is not OpenAI CLIP')
        model, _ = clip.load(model_name, device=device, jit=False,
                             download_root=download_root)
        model.requires_grad_(False).eval()
        with torch.no_grad():
            # Same order as domain labels and branch IDs throughout training.
            tokens = clip.tokenize([positive_prompt, negative_prompt]).to(device)
            centres = F.normalize(model.encode_text(tokens).float(), dim=-1)
        self._set_centres(centres)

    def _set_centres(self, centres):
        centres = F.normalize(centres.detach().float(), dim=-1)
        direction = centres[0:1] - centres[1:2]
        if direction.norm().item() < 1e-6:
            raise ValueError('CLIP day and night prompts have no direction')
        self.embedding_dim = centres.shape[1]
        self.register_buffer('text_centres', centres)
        self.register_buffer('text_direction', F.normalize(direction, dim=-1))
        # Only cached embeddings are needed. Release the full CLIP model after
        # text encoding, including its unused visual encoder.
        self.eval()

    def train(self, mode=True):
        return super().train(False)
