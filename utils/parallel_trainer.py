"""Night branch warmup followed by shared day/night MAN fine-tuning."""

from pathlib import Path

from test import build_dual_dora_model
from .finetune_trainer import FineTuneTrainer, file_digest, load_training_checkpoint


PARALLEL_ARCHITECTURE = 'swin_man_night_pfa_mdc_v1'


class ParallelTrainer(FineTuneTrainer):
    @property
    def validation_router_mode(self):
        return 'fixed' if self.args.stage == 'fusion' else self.args.router_mode

    def prepare_model(self, model_args, source, resumed):
        checkpoint = resumed if resumed else load_training_checkpoint(source)
        state = checkpoint.get('model_state_dict', checkpoint)
        contains_fusion = any(key.startswith('night_') for key in state)
        if 'model_state_dict' in checkpoint:
            expected = ((self.args.stage,) if resumed else
                        ('dora1', 'finetune') if self.args.stage == 'fusion' else ('fusion',))
            if checkpoint.get('stage') not in expected:
                raise ValueError('Parallel stage source must be one of: ' + ', '.join(expected))
            self.check_source_settings(checkpoint)
        if (self.args.stage == 'joint' or resumed) and not contains_fusion:
            raise ValueError('Joint/resume requires complete trained night-fusion weights')
        if self.args.stage == 'fusion' and not resumed and contains_fusion:
            raise ValueError('Use --resume for fusion weights; new fusion starts from MAN weights')
        model = build_dual_dora_model(model_args, source)
        if not contains_fusion:
            model.enable_night_fusion(self.args.fusion_alpha)
        model.enable_parallel_training(self.args.stage)
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        prefixes = (('night_branch.', 'night_fusion.') if self.args.stage == 'fusion' else
                    ('feature_adapter.', 'encoder.', 'reg_layer_0.'))
        if not trainable or any(not name.startswith(prefixes) for name in trainable):
            raise RuntimeError('Unexpected trainable parameters for ' + self.args.stage)
        return model

    def configure_train_mode(self):
        self.model.parallel_train_mode(self.args.stage)

    def checkpoint_metadata(self):
        return dict(architecture=PARALLEL_ARCHITECTURE, output_stride=16)

    def extra_config(self):
        root = Path(__file__).resolve().parents[1]
        return dict(stage=self.args.stage, architecture=PARALLEL_ARCHITECTURE,
                    trainable_policy=('night_modules_v1' if self.args.stage == 'fusion' else
                                      'crowd_aware_and_regression_only_v1'),
                    fusion_alpha_init=self.args.fusion_alpha,
                    parallel_code_digests={name: file_digest(root / name) for name in (
                        'models/night_fusion.py', 'models/swin_c_multibatch.py',
                        'models/module_consistency_multibatch.py',
                        'models/transformer_cosine_multibatch.py', 'datasets/crowd.py',
                        'utils/parallel_trainer.py', 'test.py')})
