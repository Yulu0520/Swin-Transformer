# Boosting-Crowd-Counting-via-Multifaceted-Attention
Official Implement of CVPR 2022 paper 'Boosting Crowd Counting via Multifaceted Attention'

This version replaces VGG19 with a Swin-Large backbone (2+2+18+2 = 24
attention blocks). The stage-4 feature is projected from 1536 to 512 channels
and passed to the original multibatch MAN Transformer encoder and density-map
head. The original Bayesian and cosine-consistency losses remain in use.
The CCST count head and CCST trainer are not used.
Training also includes an auxiliary count L1 loss with weight 0.005 by default.

[arxiv](https://arxiv.org/pdf/2203.02636.pdf) | [知乎](https://zhuanlan.zhihu.com/p/478023612) | [B站](https://www.bilibili.com/video/BV13Y411u7r5?share_source=copy_web)

![image](structure.png)

## Train

Install dependencies with `pip install -r requirements.txt`. The current CLI
defaults to `--model-name swin_t_trans --stage dora1` and local dataset/checkpoint
paths. Use explicit paths for your installation. `--stage both` trains the
baseline and then Q/V DoRA; `--stage all` also trains DoRA1 and fine-tunes the
shared MAN modules. Baseline uses day/train and day/val; DoRA uses night/train
and mix/val. The MAN head and custom Transformer stay frozen during DoRA
and DoRA1 training. Swin-T has 12 blocks; optional Swin-L has 24 blocks.

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

# Run all four stages in sequence (also supply a trained router checkpoint).
python train.py --stage all \
  --dora1-train-dir /path/to/night/train \
  --dora1-val-dir /path/to/day/val
```

### Auxiliary count L1 loss

The training objective is `Bayesian + consistency + count_loss_weight * count_L1`.
For each crop, `count_L1` compares the sum of the predicted density map with
`targets.sum()`, takes the absolute difference, then averages over the batch.
The target preserves the existing fractional weights for people at crop
boundaries; empty crops have a target count of zero. No extra count head is used.

Use `--count-loss-weight 0.005` (default) to enable the auxiliary objective, or
`--count-loss-weight 0` for the original objective. The setting applies to each
stage being trained; existing baseline and adapter checkpoints remain loadable.
To train an adapter with this loss without retraining the baseline:

```bash
python train.py --stage dora \
  --baseline-checkpoint /path/to/baseline/best_model.pth \
  --count-loss-weight 0.1
```

TensorBoard records raw and weighted count loss at step and epoch level.
`train/crop_count_mae` uses fractional crop targets, whereas the existing
`train/mae` uses the number of retained annotation points. Full-image validation
MAE/RMSE and checkpoint selection are unchanged. Epoch `.tar` checkpoints record
`count_loss_weight`; the CLI value controls the objective when resuming.

The default training crop is **256×256**. The official ImageNet-22K
`swin_large_patch4_window12_384_22k.pth` is read from `pre_models/`, or from
`--pretrained-path`. Its filename describes pretraining; it does not set this
project's input size. The 12×12 Swin attention windows are padded as needed
for 256×256 crops. `--no-pretrained-backbone` starts from random weights.
The default `--model-name swin_t_trans` uses the 12-block Swin-T model and
torchvision ImageNet weights. Optional `swin_l_trans` has 24 blocks and uses
the local Swin-L checkpoint described above.

Dataset paths can be changed with `--data-dir`, `--dora-train-dir`,
`--dora-val-dir`, `--dora1-train-dir`, and `--dora1-val-dir`. The adapter
directory names are explicit, so experiments such as night training and day
validation do not depend on the runtime route names. Each split contains
`.jpg` images and matching `.npy` point
annotations. `--resume <epoch_ckpt.tar>` continues the selected stage; with
`--stage both`, it resumes the baseline before DoRA starts.

Training and validation progress are displayed with tqdm. TensorBoard scalar
logs are stored under `model/<model-name>/<stage>/<run>/tensorboard/` and can be viewed with:

```
tensorboard --logdir model
```

Epoch numbers shown in logs, progress bars, TensorBoard, and checkpoint names
are 1-based. The two stages have separate schedules:

| Stage | Epochs | Validate every | First validation index |
| --- | ---: | ---: | ---: |
| Clean baseline | `--pretrain-epochs 600` | `--pretrain-val-epoch 5` | `--pretrain-val-start 50` |
| Hazy/mix DoRA | `--dora-epochs 1200` | `--dora-val-epoch 5` | `--dora-val-start 200` |

The validation starts are **zero-based**. With these defaults, baseline first
validates after epoch 51 and DoRA first validates after epoch 201. `--max-epoch`,
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


## Fine-tune the shared MAN modules

`--stage finetune` loads the best completed DoRA1 model, freezes the entire
Swin backbone (including its output norm), both DoRA branches, and the image
router. Only `feature_adapter`, the two-layer MAN `encoder` (including its
learnable region attention), and `reg_layer_0` are optimized. The frozen
backbone stays in evaluation mode; the trainable MAN modules use training mode.
`--stage all` now runs baseline -> DoRA -> DoRA1 -> fine-tune.

Fine-tuning defaults to `mix/train` and `mix/val`, Adam with LR `1e-7`, shared
`--weight-decay 1e-5`, and 50 epochs. It keeps the current Bayesian,
cosine-consistency, and count L1 losses (`--count-loss-weight 0.005` currently).
The model output stride is 16. The default fine-tune paths point to the local
`img2img-turbo/SHA/mix` dataset; override them for another installation.

```bash
python train.py --stage finetune \
  --dora1-checkpoint /path/to/dora1/best_model.pth \
  --finetune-train-dir /path/to/SHA/mix/train \
  --finetune-val-dir /path/to/SHA/mix/val \
  --router-checkpoint /path/to/router/best_model.pth \
  --model-name swin_t_trans \
  --dora-rank 12 --dora-alpha 12 \
  --dora1-rank 8 --dora1-alpha 8 \
  --finetune-lr 1e-7 --finetune-epochs 50
```

**Match rank/alpha to the source checkpoint.** The values above are the current
training defaults; older raw `.pth` weights may use 4/4 for both branches.
Metadata-bearing source checkpoints are checked against `--model-name` and
provide adapter settings when loading; the fine-tune settings must match.

The default `--router-mode classifier` runs the frozen router on a 224x224 copy
of each original full image, before random cropping or augmentation. Training
routes are cached once because the router is fixed. Day uses DoRA1; night uses
DoRA. Each batch is divided by route, with sample-weighted gradients accumulated
before one Adam step. Use `--router-mode filename` instead to select routes by
`night_` versus other filenames; this mode does not need a router checkpoint.

Validation runs once before any update as **epoch 0**, then after every training
epoch. TensorBoard contains mix, true-day, and true-night MAE/RMSE; true domains
come from filenames, independently of router predictions. Models are ranked by
mix `MAE + 2 * RMSE`; ties prefer earlier epochs. The run directory contains:

- Up to five `best_epoch_NNN.pth` files, with model and adapter metadata. Epoch 0
  can remain in the top five.
- `ranking.json`, sorted best first, with epoch, MAE, RMSE, score and filename.
- `last_checkpoint.tar`, containing the latest completed epoch, model, Adam
  state, ranking, configuration, and Python/NumPy/Torch/DataLoader RNG states.

Resume from `last_checkpoint.tar`, supplying the same data, router, loss, LR,
weight decay, model, batch, crop and seed settings used originally:

```bash
python train.py --stage finetune \
  --resume model/swin_t_trans/finetune/RUN/last_checkpoint.tar \
  --finetune-train-dir /path/to/SHA/mix/train \
  --finetune-val-dir /path/to/SHA/mix/val \
  --router-checkpoint /path/to/router/best_model.pth \
  --dora-rank 12 --dora-alpha 12 --dora1-rank 8 --dora1-alpha 8
```

Resume continues in the original run directory at the next epoch, without
repeating epoch 0. `--finetune-epochs` is the total target epoch and may be
increased. Image and annotation contents, router weights and loss implementation
are fingerprinted; incompatible settings or missing ranked weights cause an
error. Recovery is at epoch boundaries: an interrupted epoch is replayed.
Ranked weights and the latest checkpoint are saved with atomic replacement;
pruning happens after the latest checkpoint commits.

Compare the fine-tuned model with its original DoRA1 model on the same mix set:

```bash
python test.py \
  --finetune-checkpoint model/swin_t_trans/finetune/RUN/best_epoch_010.pth \
  --compare-before-after \
  --router-checkpoint /path/to/router/best_model.pth \
  --data-root /path/to/SHA --split val \
  --output-csv finetune_summary.csv --routing-csv finetune_routing.csv
```

Use `--router-mode filename` if that was the training route mode. The source
DoRA1 path is remembered; `--dora1-checkpoint` can supply its new location, but
its contents must match the original. Without `--compare-before-after`, only
the fine-tuned model is evaluated. This evaluation always uses `mix/<split>`;
it reports overall mix and filename-derived day/night metrics. The routing CSV
has one row per image per model (`before`/`after`), including true/predicted
domain, route, confidence, true/predicted count, and error. Fine-tune metadata
provides the model and adapter settings automatically. Router mode and classifier
weights must match training. `last_checkpoint.tar` is also loadable for testing.


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
