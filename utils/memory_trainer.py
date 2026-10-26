"""Three standalone training phases sharing the original crowd-counting losses."""

import os

from models import swin_c_multibatch as models
from utils.memory_workflow import (attach_memory, load_research_model,
                                   research_payload)
from utils.regression_trainer_cosine_multibatch import RegTrainer
from utils.stage3_bank import baseline_fingerprint, load_file, validate_bank


class MemoryTrainer(RegTrainer):
    def build_model(self):
        args = self.args
        if args.resume:
            self.model = load_research_model(load_file(args.resume), args.model_name,
                                             args.dora_rank, args.dora_alpha)
            if self.model.research_config['stage'] != args.stage:
                raise ValueError('Resume checkpoint belongs to a different phase')
            config = self.model.research_config
            for name in ('crop_size', 'is_gray'):
                if name in config and config[name] != getattr(args, name):
                    raise ValueError('Resume preprocessing differs: ' + name)
            return
        if args.stage == 'baseline':
            self.model = getattr(models, args.model_name)(
                pretrained=args.pretrained_backbone,
                pretrained_path=args.pretrained_path)
            self.model.research_config = {
                'stage': 'baseline', 'model_name': args.model_name,
                'dora_rank': 0, 'dora_alpha': 0,
                'baseline_weights_sha256': baseline_fingerprint(self.model.state_dict()),
            }
        elif args.stage == 'dora':
            self.model = load_research_model(load_file(args.baseline_checkpoint),
                                             args.model_name)
            if self.model.research_config['stage'] != 'baseline':
                raise ValueError('Phase 2 needs the best daytime baseline')
            self.model.enable_dora(args.dora_rank, args.dora_alpha)
            self.model.research_config.update(
                stage='dora', dora_rank=args.dora_rank, dora_alpha=args.dora_alpha)
        elif args.stage == 'cross_attention':
            # Strictly load Phase 2 BEFORE adding any new parameters.
            self.model = load_research_model(load_file(args.dora_checkpoint),
                                             args.model_name,
                                             args.dora_rank, args.dora_alpha)
            bank = validate_bank(load_file(args.memory_bank_path), args.model_name,
                                 args.memory_size)
            attach_memory(self.model, bank, args.cross_attention_heads,
                          args.alpha_init, args.beta_init, not args.freeze_fusion)
        else:
            raise ValueError('Memory workflow supports baseline, dora, cross_attention')
        config = self.model.research_config
        for name in ('crop_size', 'is_gray'):
            if name in config and config[name] != getattr(args, name):
                raise ValueError('Phase preprocessing must match baseline: ' + name)
        config.update(crop_size=args.crop_size, is_gray=args.is_gray,
                      downsample_ratio=args.downsample_ratio)

    def checkpoint_metadata(self):
        return {key: value for key, value in research_payload(self.model).items()
                if key != 'model_state_dict'}

    def best_weights_payload(self):
        return research_payload(self.model)

    def best_updated(self):
        alias = {
            'baseline': 'baseline_best.pth',
            'dora': 'night_dora_best.pth',
            'cross_attention': 'night_cross_attention_best.pth',
        }[self.args.stage]
        # A relative symlink names rank 1 without storing an extra model copy.
        temporary = os.path.join(self.save_dir, '.' + alias)
        if os.path.lexists(temporary):
            os.unlink(temporary)
        os.symlink('best_model.pth', temporary)
        os.replace(temporary, os.path.join(self.save_dir, alias))

    def setup(self):
        super().setup()
        if self.top_models.records:
            self.best_updated()
