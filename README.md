# Boosting-Crowd-Counting-via-Multifaceted-Attention
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
stays frozen during DoRA training; the custom Transformer now has its own
DoRA adapters as described below.

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
from the clean baseline output. K weights, MAN base weights, and the density
head are shared and frozen. The selected MAN adapter branch is trainable.

### MAN encoder DoRA

Both MAN encoder layers now adapt five projections: `global_q`, `global_v`
(the shared value projection), `local_left_q`, `local_right_q`, and `local_q`.
The four K projections, input biases, `out_proj`, feed-forward layers,
LayerNorms, feature adapter, and density head remain frozen in adapter stages.
Each selected 512x512 projection has independent low-rank direction and
per-output magnitude parameters. Swin and MAN share the existing rank/alpha
arguments; at rank 4 the MAN adds 46,080 parameters per branch.

Swin and MAN switch routes together. During the DoRA stage, only their DoRA
branches are optimized; during DoRA1, only their DoRA1 branches are optimized.
Both branches use the same raw baseline weights and are never chained or
evaluated together. MAN's packed nine-projection weight is composed once per
attention forward and reused for its nine slices.

The training and evaluation entrypoints strictly load new checkpoints and
also accept old Swin-only checkpoints. Missing MAN adapters are initialized
from the checkpoint's MAN weights with zero direction updates, preserving
the old output before further training. Such adapters have not been trained:
to learn both MAN branches, start the DoRA stage from a baseline, then train
DoRA1 from the resulting new DoRA checkpoint. Starting DoRA1 from an old
Swin-only checkpoint trains only MAN DoRA1 and leaves MAN DoRA at baseline.
Old adapter-stage `.tar` optimizer states cannot be resumed with the expanded
parameter set; use their weights via `--baseline-checkpoint` (baseline stage
checkpoint) or `--dora-checkpoint` (DoRA stage checkpoint) to start a new stage.
New `.tar` checkpoints support normal `--resume` including the MAN optimizer
state. Incomplete MAN adapter checkpoints are rejected rather than filled in.

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
are 1-based. The two stages have separate schedules:

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
