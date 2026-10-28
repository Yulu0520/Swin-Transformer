# Boosting-Crowd-Counting-via-Multifaceted-Attention

The independent daytime baseline → fixed Stage 3/4 memory bank → nighttime
single-DoRA → memory Cross-Attention workflow is documented in
[docs/MEMORY_WORKFLOW.md](docs/MEMORY_WORKFLOW.md), including complete training,
evaluation, ablation, and CPU test commands. Select it with `--workflow memory`;
the legacy double-DoRA workflow remains available.

New Cross-Attention training defaults to `--memory-stage 4`: both the daytime
bank and nighttime queries use Stage 4 features **before backbone norm**. Fusion
is followed by the original norm, feature adapter, MAN, and density head.
Use `--memory-stage 3` to reproduce the earlier insertion before Stage 4 patch
merging. Existing baseline/DoRA weights can be reused; Stage 4 requires a newly
built bank and newly trained Cross-Attention projections.

```bash
python build_memory_bank.py --memory-stage 4 \
  --baseline-checkpoint /path/to/baseline_best.pth \
  --data-dir /path/to/SHA/day --output stage4_memory_bank.pt

python train.py --workflow memory --stage cross_attention --memory-stage 4 \
  --dora-checkpoint /path/to/night_dora_best.pth \
  --memory-bank-path stage4_memory_bank.pt \
  --dora-train-dir /path/to/SHA/night/train --dora-val-dir /path/to/SHA/night/val

python test.py --workflow memory --experiment C \
  --checkpoint model/swin_t_trans/cross_attention/stage4/RUN/best_model.pth \
  --data-root /path/to/SHA --domains day night --split val
```

Runs are separated under `cross_attention/stage3/` and `cross_attention/stage4/`.
Explicit checkpoints infer their stage automatically; `test.py --memory-stage`
can additionally check it. Without an explicit checkpoint, experiment C searches
Stage 4 by default; pass `--memory-stage 3` to find Stage 3 runs, including the
original flat layout. Resume restores the checkpoint's stage. The old
`build_stage3_bank.py` remains a Stage 3-only entrypoint.

Training includes an auxiliary crop count L1 loss, matching the `L1 model`
project: `Bayesian + consistency + count_loss_weight * count_L1`. The default
`--count-loss-weight 0.1` applies to every trained stage; use `0` for the original
objective. Count L1 compares each density-map sum with `targets.sum()`, preserving
fractional boundary targets, then averages absolute errors across the batch.
It adds no model parameters and does not change density output or inference.
Step/epoch logs include raw and weighted count loss, plus epoch
`train/crop_count_mae` (fractional crop targets, unlike the retained-point-count
`train/mae`). Epoch checkpoints and memory-workflow best checkpoints record the
coefficient; the current CLI value controls it when resuming, including old
checkpoints that have no coefficient. Full-image validation MAE/RMSE and top-five
checkpoint selection remain unchanged.
Official Implement of CVPR 2022 paper 'Boosting Crowd Counting via Multifaceted Attention'

This version replaces VGG19 with a Swin-Large backbone (2+2+18+2 = 24
attention blocks). The stage-4 feature is projected from 1536 to 512 channels
and passed to the original multibatch MAN Transformer encoder and density-map
head. The original Bayesian and cosine-consistency losses remain in use.
The CCST count head and CCST trainer are not used.

[arxiv](https://arxiv.org/pdf/2203.02636.pdf) | [知乎](https://zhuanlan.zhihu.com/p/478023612) | [B站](https://www.bilibili.com/video/BV13Y411u7r5?share_source=copy_web)

![image](structure.png)

## Train

Install dependencies with `pip install -r requirements.txt`. By default,
`python train.py` first trains the full 24-block baseline on `SHA/clean/train`
and selects its best checkpoint on `SHA/clean/val`. It then freezes that
baseline and trains Q/V DoRA in all 24 Swin attention blocks on
`SHA/hazy/train`, selecting the best result on `SHA/mix/val`. The MAN head
and custom Transformer stay frozen during DoRA training.

DoRA keeps the low-rank Q/V direction update and learns a separate magnitude
for every Q/V output vector. K remains unchanged. The magnitude is initialized
from the clean baseline, so attaching DoRA does not change the model output
before adaptation starts. The parallel DoRA1 branch is present but frozen
during this stage and is never called by the DoRA forward route.

The optional third stage trains a second DoRA branch named DoRA1 in parallel
with the trained DoRA branch. Both branches read the same frozen baseline Q/V
weight, and exactly one branch is evaluated per forward. During DoRA1 training,
the baseline and complete DoRA branch are frozen and DoRA is not called; only
DoRA1's direction and magnitude parameters are optimized. Both branches start
from the clean baseline output. K, the MAN encoder, and the density head are
shared and frozen.

```bash
python train.py
python train.py --stage baseline
python train.py --stage dora \
  --baseline-checkpoint model/swin_l_trans/baseline/RUN/best_model.pth

# Third stage: freeze baseline + DoRA and train only the DoRA1 branch.
python train.py --stage dora1 \
  --dora-checkpoint model/swin_t_trans/dora/RUN/best_model.pth \
  --dora1-train-dir /path/to/night/train \
  --dora1-val-dir /path/to/day/val \
  --dora-rank 4 --dora-alpha 4 \
  --dora1-rank 4 --dora1-alpha 4

# Run all three stages in sequence.
python train.py --stage all \
  --dora1-train-dir /path/to/night/train \
  --dora1-val-dir /path/to/day/val
```

The default training crop is **256×256**. The official ImageNet-22K
`swin_large_patch4_window12_384_22k.pth` is read from `pre_models/`, or from
`--pretrained-path`. Its filename describes pretraining; it does not set this
project's input size. The 12×12 Swin attention windows are padded as needed
for 256×256 crops. `--no-pretrained-backbone` starts from random weights.
The optional `--model-name swin_t_trans` retains the earlier 12-block Swin-T
model; the default `swin_l_trans` has 24 blocks.

Dataset paths can be changed with `--data-dir`, `--dora-train-dir`,
`--dora-val-dir`, `--dora1-train-dir`, and `--dora1-val-dir`. The adapter
directory names are explicit, so experiments such as night training and day
validation do not depend on the runtime route names. Each split contains
`.jpg` images and matching `.npy` point
annotations. `--resume <epoch_ckpt.tar>` continues the selected stage; with
the default `--stage both`, it resumes the baseline before DoRA starts.

Training and validation progress are displayed with tqdm. TensorBoard scalar
logs are stored under `model/<model-name>/<stage>/<run>/tensorboard/` and can be viewed with:

```
tensorboard --logdir model
```

Epoch numbers shown in logs, progress bars, TensorBoard, and checkpoint names
are 1-based.

Each baseline, DoRA, and DoRA1 run retains the best five validation weights by
`MAE + 2 * RMSE` (lower is better). Every validation result can enter this
ranking, even when it does not improve the best-ever score. Equal scores keep
the earlier result; non-finite metrics are ignored. Until five validations
have completed, fewer than five weights may exist.

`best_model.pth` is always rank 1; `best_model_rank2.pth` through
`best_model_rank5.pth` hold the remaining ranks. `top_models.json` records each
file's epoch, MAE, RMSE, and score. Use `--save-top-k N` to change the limit.
`--save-all` overrides this limit: it keeps `best_model.pth` plus a numbered
archive for every historical best, without a retention cap. Independently,
`--max-model-num` controls the number of recent epoch `.tar` resume files.

Resume files include the ranking. Resuming copies retained weights into the
new run directory without changing the source run. Keep the latest `.tar`,
`top_models.json`, and retained weights together; a stale ranking or missing
weight causes an error rather than silently restoring incorrect weights.
Legacy resume files recover only their known best model (its original epoch
is recorded as unknown). Automatic evaluation selects `best_model.pth`;
pass an explicit checkpoint path to evaluate another rank.

The stages have separate schedules:

| Stage | Epochs | Validate every | First validation index |
| --- | ---: | ---: | ---: |
| Clean baseline | `--pretrain-epochs 500` | `--pretrain-val-epoch 5` | `--pretrain-val-start 100` |
| Hazy/mix DoRA | `--dora-epochs 1200` | `--dora-val-epoch 5` | `--dora-val-start 500` |

The validation starts are **zero-based**. With these defaults, baseline first
validates after epoch 101 and DoRA first validates after epoch 501. `--max-epoch`,
`--val-epoch`, and `--val-start` are aliases for the three DoRA options.


## Test

```bash
python test.py --data-root SHA --split val \
  --baseline-checkpoint model/swin_l_trans/baseline/RUN/best_model.pth \
  --dora-checkpoint model/swin_l_trans/dora/RUN/best_model.pth \
  --dora1-checkpoint model/swin_l_trans/dora1/RUN/best_model.pth \
  --dora-rank 4 --dora-alpha 4 \
  --dora1-rank 4 --dora1-alpha 4 \
  --dual-route dora1
```

To route every mixed-set image by its filename, use a DoRA1-stage dual-DoRA
checkpoint. Files named `IMG_*` use the baseline+DoRA1 branch, while files named
`night_IMG_*` use the baseline+DoRA branch. The summary CSV contains overall
MAE/RMSE; the routing CSV records the selected branch and count error for every
image.

```bash
python test.py \
  --model-name swin_t_trans \
  --router-mode filename \
  --data-root /home/mmslab-1080/Desktop/img2img-turbo/SHA \
  --domains mix --split val \
  --dora1-checkpoint /path/to/dora1/best_model.pth \
  --output-csv filename_router_summary.csv \
  --routing-csv filename_router_details.csv
```

Filename routing evaluates only the dual-DoRA checkpoint; separate baseline and
DoRA checkpoint arguments are not needed in this mode.

### Image-content day/night router

Train the MobileNetV3-Small router with filename-derived labels. Filenames are
used only as training/validation targets; the trained router receives image
pixels only.

```bash
python train_router.py \
  --data-root /home/mmslab-1080/Desktop/img2img-turbo/SHA/mix \
  --epochs 20 --batch-size 32 --device cuda:0
```

Use its best checkpoint for content-based dynamic routing. The router resizes a
separate copy to 224x224; the crowd-counting input keeps its original size.

```bash
python test.py \
  --model-name swin_t_trans \
  --router-mode classifier \
  --router-checkpoint model/day_night_router/RUN/best_model.pth \
  --data-root /home/mmslab-1080/Desktop/img2img-turbo/SHA \
  --domains mix --split val \
  --dora1-checkpoint /path/to/dora1/best_model.pth \
  --output-csv classifier_router_summary.csv \
  --routing-csv classifier_router_details.csv
```

The routing CSV includes the filename-derived expected domain for evaluation,
the image-router prediction and confidence, the selected adapter, and the count
error. Runtime branch selection remains:

- predicted day -> baseline + DoRA1
- predicted night -> baseline + DoRA

Epoch `.tar` checkpoints contain stage, model name, optimizer, and both adapter
settings. Best `.pth` checkpoints contain model weights; supply the DoRA and
DoRA1 rank/alpha values when loading a parallel `.pth` checkpoint. The active
route is selected at runtime and is not stored as a learned parameter.


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
