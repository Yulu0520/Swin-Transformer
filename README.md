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
and custom Transformer stay frozen during DoRA training.

DoRA keeps the low-rank Q/V direction update and learns a separate magnitude
for every Q/V output vector. K remains unchanged. The magnitude is initialized
from the clean baseline, so attaching DoRA does not change the model output
before adaptation starts. The parallel LoRA branch is present but frozen during
this stage and is never called by the DoRA forward route.

The optional third stage adds a LoRA branch in parallel with the trained DoRA
branch. Both branches read the same frozen baseline Q/V weight, and exactly one
branch is evaluated per forward: `DoRA(W_base)` or `W_base + LoRA_delta`. During
LoRA training, the baseline and complete DoRA branch are frozen and DoRA is not
called; only LoRA A/B matrices are optimized. LoRA B starts at zero, so its
initial route exactly preserves the clean baseline weight. K, the MAN encoder,
and the density head are shared and frozen. `--lora-train-dir` and
`--lora-val-dir` have no default paths and must be supplied for `--stage lora`
or `--stage all`.

```bash
python train.py
python train.py --stage baseline
python train.py --stage dora \
  --baseline-checkpoint model/swin_l_trans/baseline/RUN/best_model.pth

# Third stage: freeze baseline + DoRA and train only LoRA Q/V matrices.
python train.py --stage lora \
  --dora-checkpoint model/swin_t_trans/dora/RUN/best_model.pth \
  --lora-train-dir /path/to/lora/train \
  --lora-val-dir /path/to/lora/val \
  --dora-rank 4 --dora-alpha 4 \
  --lora-rank 4 --lora-alpha 4

# Run all three stages in sequence. LoRA paths are intentionally explicit.
python train.py --stage all \
  --lora-train-dir /path/to/lora/train \
  --lora-val-dir /path/to/lora/val
```

The default training crop is **256×256**. The official ImageNet-22K
`swin_large_patch4_window12_384_22k.pth` is read from `pre_models/`, or from
`--pretrained-path`. Its filename describes pretraining; it does not set this
project's input size. The 12×12 Swin attention windows are padded as needed
for 256×256 crops. `--no-pretrained-backbone` starts from random weights.
The optional `--model-name swin_t_trans` retains the earlier 12-block Swin-T
model; the default `swin_l_trans` has 24 blocks.

Clean/hazy/mix paths can be changed with `--data-dir`, `--dora-train-dir`, and
`--dora-val-dir`. Each split contains `.jpg` images and matching `.npy` point
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
  --lora-checkpoint model/swin_l_trans/lora/RUN/best_model.pth \
  --dora-rank 4 --dora-alpha 4 \
  --lora-rank 4 --lora-alpha 4 \
  --hybrid-route lora
```

To route every mixed-set image by its filename, use a parallel LoRA-stage
checkpoint. Files named `IMG_*` use the baseline+LoRA branch, while files named
`night_IMG_*` use the baseline+DoRA branch. The summary CSV contains overall
MAE/RMSE; the routing CSV records the selected branch and count error for every
image.

```bash
python test.py \
  --model-name swin_t_trans \
  --router-mode filename \
  --data-root /home/mmslab-1080/Desktop/img2img-turbo/SHA \
  --domains mix --split val \
  --lora-checkpoint /path/to/lora/best_model.pth \
  --output-csv filename_router_summary.csv \
  --routing-csv filename_router_details.csv
```

Filename routing evaluates only the hybrid checkpoint; separate baseline and
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
  --lora-checkpoint /path/to/lora/best_model.pth \
  --output-csv classifier_router_summary.csv \
  --routing-csv classifier_router_details.csv
```

The routing CSV includes the filename-derived expected domain for evaluation,
the image-router prediction and confidence, the selected adapter, and the count
error. Runtime branch selection remains:

- predicted day -> baseline + LoRA
- predicted night -> baseline + DoRA

Epoch `.tar` checkpoints contain stage, model name, optimizer, and both adapter
settings. Best `.pth` checkpoints contain model weights; supply the DoRA and
LoRA rank/alpha values when loading a parallel `.pth` checkpoint. The active
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
