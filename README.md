# Boosting Crowd Counting via Multifaceted Attention — Swin DoRA

This project uses a Swin backbone followed by a 512-channel feature projection,
the two-layer multibatch MAN encoder, and a density-map head. Training retains
Bayesian and cosine-consistency losses. The original paper is
[Boosting Crowd Counting via Multifaceted Attention](https://arxiv.org/abs/2203.02636).
`structure.png` illustrates the original VGG-based MAN architecture.

## Current experiment: Swin-T with one Q/V + MLP DoRA branch

The default experiment now uses **Swin-T, one DoRA branch, and no router**.
DoRA1 is not constructed. Every image uses the same adapter, regardless of its
filename or day/night appearance.

Each of Swin-T's 12 blocks adapts:

- attention Q and V; K remains unchanged;
- MLP Linear 1 (`C -> 4C`, torchvision `mlp.0`);
- MLP Linear 2 (`4C -> C`, torchvision `mlp.3`).

The MLP adapters use independent low-rank direction factors and one magnitude
per output channel. Their B factors start at zero; magnitudes are initialized
from the loaded baseline weights, preserving the initial model output within
floating-point tolerance. Biases, baseline weights, normalization layers,
patch embedding/merging, attention output projections, feature projection,
MAN encoder, and density head stay frozen. Only Q/V and MLP adapter parameters
are optimized. Adapter training keeps dropout/stochastic depth in eval mode
while allowing gradients through the frozen network.

With rank 4 for both Q/V and MLP, Swin-T has **278,208 trainable adapter
parameters**: 79,488 in Q/V and 198,720 in its 24 MLP Linear layers.

## Train

Install dependencies with `pip install -r requirements.txt`.
The current defaults are `--model-name swin_t_trans --stage dora`,
`--adapter-mode single --adapter-targets qv_mlp`, crop size 384, batch size 4,
rank/alpha 4 for Q/V and MLP, adapter learning rate 2e-4, and 10 epochs.
The trainer requires one CUDA GPU. Pass dataset and checkpoint paths explicitly
when moving the project to another machine.

Train Q/V + MLP adapters from an existing clean baseline:

```bash
python train.py --stage dora \
  --baseline-checkpoint /path/to/baseline/best_model.pth \
  --dora-train-dir /path/to/night/train \
  --dora-val-dir /path/to/mix/val \
  --dora-rank 4 --dora-alpha 4 \
  --mlp-dora-rank 4 --mlp-dora-alpha 4
```

Warm-start from a trained Q/V DoRA checkpoint, including an old parallel
DoRA/DoRA1 checkpoint:

```bash
python train.py --stage dora \
  --dora-checkpoint /path/to/old_dora/best_model.pth \
  --dora-train-dir /path/to/night/train \
  --dora-val-dir /path/to/mix/val
```

This retains the raw baseline and trained **DoRA** branch, drops DoRA1 (or the
legacy LoRA branch), then initializes the new MLP adapters from the loaded MLP
weights. It starts a new optimizer and epoch schedule. For weight-only legacy
checkpoints, supply the original Q/V rank and alpha. Metadata-bearing
checkpoints must match the requested training settings.

Add `--freeze-qv-dora` to train only MLP adapters while holding Q/V DoRA fixed.
Use `--adapter-targets qv` for the single-branch Q/V-only comparison.
Keep the same baseline, datasets, seed, crop, and training schedule when
comparing Q/V-only against Q/V + MLP; the latter has more trainable parameters.

To train a baseline first and then the single adapter:

```bash
python train.py --stage both --data-dir /path/to/day \
  --dora-train-dir /path/to/night/train --dora-val-dir /path/to/mix/val
```

The baseline root must contain `train/` and `val/`. Each crowd-counting split
contains `.jpg` images and matching `.npy` point annotations; training
annotations include the nearest-neighbor distance column. Validation uses
full images. The density head has stride 16, so `--downsample-ratio` must be 16.

Resume a matching single-branch epoch checkpoint with:

```bash
python train.py --stage dora --resume /path/to/run/5_ckpt.tar
```

Rank/alpha, targets, and `--freeze-qv-dora` must match. Use
`--dora-checkpoint` for old-architecture migration rather than `--resume`.
The epoch `.tar` retains optimizer state. A `.pth` resume restores weights
only and starts a new schedule.

Both single-branch best `.pth` and epoch `.tar` checkpoints contain model name,
targets, Q/V and MLP rank/alpha, and checkpoint format version. Best `.pth`
files use a `model_state_dict` wrapper rather than the old raw state dict;
use the project loaders to read them. Partial or mismatched weights are
rejected with strict loading.

Epochs are displayed as 1-based. Validation start indices are zero-based;
defaults validate after epochs 1 and 6 in a 10-epoch run. Logs and checkpoints
are saved under `model/<model-name>/<stage>/<run>/`.

```bash
tensorboard --logdir model
```

Swin-L remains available through `--model-name swin_l_trans`. Baseline
pretraining reads the official ImageNet-22K checkpoint from
`--pretrained-path` (default `pre_models/swin_large_patch4_window12_384_22k.pth`).
Swin-T uses torchvision's ImageNet weights. `--no-pretrained-backbone` disables
ImageNet initialization for baseline training.

## Test without a router

The default evaluates **only one DoRA checkpoint**. No separate baseline,
DoRA1, or router checkpoint is required. The saved target and rank/alpha
metadata reconstructs the correct architecture automatically.

```bash
python test.py --dora-checkpoint /path/to/new_run/best_model.pth \
  --data-root /path/to/SHA --domains day night mix --split val \
  --output-csv single_dora_results.csv
```

If `--dora-checkpoint` is omitted, the latest best checkpoint under
`model/swin_t_trans/dora/` is used. For raw legacy `.pth` weights, supply
`--dora-rank`/`--dora-alpha` and, if applicable,
`--mlp-dora-rank`/`--mlp-dora-alpha`. Old dual checkpoints are evaluated through
the retained DoRA branch only. The summary CSV contains per-domain MAE/RMSE;
no routing CSV is written in single mode.

`save_attention.py` and `save_density_and_var.py` accept the same adapter
rank/alpha arguments and load baseline or single-adapter checkpoints in eval
mode. They also extract the DoRA branch from old parallel checkpoints.

## Optional original parallel Q/V workflow

The old router and DoRA1 code remains available explicitly. It supports Q/V
adapters only; parallel MLP adapters are not part of this experiment.

```bash
python train.py --adapter-mode parallel --adapter-targets qv --stage dora1 \
  --dora-checkpoint /path/to/parallel_dora/best_model.pth \
  --dora1-train-dir /path/to/night/train --dora1-val-dir /path/to/day/val

python test.py --evaluation-mode compare --router-mode classifier \
  --router-checkpoint /path/to/router/best_model.pth \
  --dora1-checkpoint /path/to/parallel_dora1/best_model.pth \
  --data-root /path/to/SHA --domains mix
```

The old routing convention remains day -> DoRA1 and night -> DoRA. To compare
baseline, DoRA, and optional parallel checkpoints without routing, use
`--evaluation-mode compare --router-mode fixed` and the corresponding explicit
checkpoint arguments. The standalone router is trained with `train_router.py`.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover real Swin-T initial-output preservation, 12 Q/V + 24 MLP adapter
counts, gradients and frozen weights, single-branch migration, checkpoint
round trips with nondefault alpha, and router-free evaluation.

## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{lin2022boosting,
  title={Boosting Crowd Counting via Multifaceted Attention},
  author={Lin, Hui and Ma, Zhiheng and Ji, Rongrong and Wang, Yaowei and Hong, Xiaopeng},
  booktitle={CVPR},
  year={2022}
}
```
