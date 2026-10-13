# Boosting-Crowd-Counting-via-Multifaceted-Attention
Official Implement of CVPR 2022 paper 'Boosting Crowd Counting via Multifaceted Attention'

This version replaces VGG19 with Swin-T by default (2+2+6+2 = 12 attention
blocks), with optional Swin-L (2+2+18+2 = 24 blocks). The stage-4 feature is
projected to 512 channels and passed to the multibatch MAN Transformer encoder.
By default only **stage 3 and the MAN encoder output** enter fusion. Stage 3
receives a 1x1 projection to 512 channels. Both branches are aligned to the
density grid (input height/16 by input width/16, rounded down), normalized with
channel-wise LayerNorm without learned affine parameters, and concatenated to
1024 channels. A learned 1x1 convolution projects 1024 to 512 channels without
an additional activation, then passes it to the existing density-map head.
Selected stage branches use area resizing, while MAN uses bilinear resizing.

`--fusion-stages` selects one or more of stages 1, 2, 3 in both `train.py` and
`test.py`. MAN output is always included, after the selected stages in ascending
order. Only selected stage projections are created. Swin still executes all
four stages to obtain stage 4 for MAN.

```bash
python train.py --stage all --fusion-stages 3      # default: stage 3 + MAN
python train.py --stage all --fusion-stages 2      # stage 2 + MAN
python train.py --stage all --fusion-stages 2 3    # 1536 -> 512 fusion
python train.py --stage all --fusion-stages 1 2 3  # former 2048 -> 512 fusion
```
The original Bayesian and cosine-consistency losses remain in use; consistency
features still come only from MAN's stage-4 encoder path.
The CCST count head and CCST trainer are not used.

Train a new baseline with this fusion architecture before training DoRA or
DoRA1. The selected stage projections and fusion convolution train with the
baseline and are frozen along with MAN and the density head during adapter
training. Checkpoints from the former equal-sum architecture lack the fusion
convolution; stage-4-only checkpoints also lack the stage projections. Neither
can be loaded directly with the existing strict checkpoint loaders. Swin
backbone pretraining remains supported. Existing four-branch concat-convolution
checkpoints can still be loaded with `--fusion-stages 1 2 3`. They cannot be
loaded directly into the default two-branch architecture. Keep the fusion
selection identical across baseline, DoRA, DoRA1, resume, and evaluation.
Epoch checkpoints and best-weight JSON metadata record `fusion_stages`; plain
`.pth` files are checked using their stage projection keys and fusion dimensions.

[arxiv](https://arxiv.org/pdf/2203.02636.pdf) | [知乎](https://zhuanlan.zhihu.com/p/478023612) | [B站](https://www.bilibili.com/video/BV13Y411u7r5?share_source=copy_web)

![image](structure.png)

## Train

Install dependencies with `pip install -r requirements.txt`. By default,
`python train.py` defaults to the three-stage Swin-T pipeline. It first trains
the baseline on the configured day training split and validates on day. It
then freezes that baseline and trains Q/V DoRA in all 12 Swin attention blocks
on the configured night training split, validating on night, followed by the
DoRA1 stage using its independently configured data directories. The MAN head
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

Learning rate and weight decay are independent for each counting stage:

```bash
python train.py --stage all \
  --lr 5e-6 --weight-decay 1e-5 \
  --dora-lr 2e-4 --dora-weight-decay 1e-5 \
  --dora1-lr 1e-4 --dora1-weight-decay 5e-5
```

`--lr` and `--weight-decay` apply to the baseline. DoRA and DoRA1 use their
own options (each defaults to learning rate `2e-4` and weight decay `1e-5`).
When resuming an epoch `.tar`, optimizer moments are restored and weight decay
uses the selected stage's current setting. If the checkpoint contains a plateau
scheduler, its settings, progress, and decayed learning rate are also restored.
Use `--reset-scheduler` to restart the schedule with the current stage learning
rate and scheduler options. Older checkpoints without scheduler state start a
new schedule with the current options; `--lr-scheduler none` disables scheduling
and uses the stage learning rate.

Each stage defaults to its own `ReduceLROnPlateau` scheduler, monitoring
validation **RMSE** in minimization mode. The shared rule defaults are factor
`0.5`, patience `3` non-improving validation calls, relative improvement threshold
`1e-4`, cooldown `0`, and minimum learning rate `1e-7`. The fourth consecutive
non-improving validation triggers a reduction; patience counts validations,
not training epochs. Scheduling begins at the existing stage validation start.

```bash
python train.py --stage all \
  --scheduler-factor 0.5 --scheduler-patience 3 \
  --scheduler-min-lr 1e-7 \
  --pretrain-val-start 0 --dora-val-start 0 --dora1-val-start 0
```

Epoch `.tar` files include scheduler state for resuming. TensorBoard records
the learning rate at training epochs and after validation, and `train.log`
records reductions. Best-weight JSON files record the actual learning rate
used to train the selected epoch, before the post-validation scheduler step.

Each stage has an independent training crop size: baseline defaults to
**256×256**, while DoRA and DoRA1 each default to **384×384**.

```bash
python train.py --stage all \
  --baseline-crop-size 256 --dora-crop-size 384 --dora1-crop-size 256
```

`--baseline-crop-size`, `--dora-crop-size`, and `--dora1-crop-size` control
their respective stages independently. Changing `--dora-crop-size` does not
change DoRA1's crop size. The stage-specific defaults take precedence over
the legacy shared `--crop-size` option. Crop sizes must be positive and
divisible by `--downsample-ratio`.
The chosen stage crop size is also used by the Bayesian posterior and saved
in checkpoint metadata. Validation continues to use full images without
random cropping.

The official ImageNet-22K
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
are 1-based. The two stages have separate schedules:

Each counting stage also saves independent validation minima as
`best_mae.pth` and `best_rmse.pth` in its run directory. These files contain the
complete model state dict (including frozen weights and both adapters when
attached), in the same format as `best_model.pth`. Each is overwritten only
when its own metric strictly improves. The existing `best_model.pth` selection
by `MAE + 2 * RMSE` is retained for the sequential training pipeline.
Companion `best_mae.json` and `best_rmse.json` files record the selected epoch,
both validation metrics, true MSE (`RMSE ** 2`), validation directory and image
count, model/stage, crop size, adapter settings, active route, seed, and weights
path. These best `.pth` files do not contain optimizer state or the standalone
day/night router. Epoch `.tar` checkpoints retain the optimizer and independent
best records for resuming; older checkpoints without those records begin
tracking independent minima at their next validation. Running Python training
processes must be restarted/resumed to use this code change. No historical
best-weight files are reconstructed from logs.

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
