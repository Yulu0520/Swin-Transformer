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
are 1-based. The two stages have separate schedules:

| Stage | Epochs | Validate every | First validation index |
| --- | ---: | ---: | ---: |
| Clean baseline | `--pretrain-epochs 500` | `--pretrain-val-epoch 5` | `--pretrain-val-start 100` |
| Hazy/mix DoRA | `--dora-epochs 1200` | `--dora-val-epoch 5` | `--dora-val-start 500` |

The validation starts are **zero-based**. With these defaults, baseline first
validates after epoch 101 and DoRA first validates after epoch 501. `--max-epoch`,
`--val-epoch`, and `--val-start` are aliases for the three DoRA options.


## Whole-image crowd text alignment and night-to-day guidance

The auxiliary training branch now follows MonoCMoE's **direct text-centre
supervision** idea at image level. There is no object detector, bounding-box
label, RoI-Align, or head/background classifier. Each full input feature map
is globally average-pooled, linearly projected and normalized by one shared
adapter `P`:

```text
Day crowd image   -> baseline Swin + DoRA1 -> global pooling + P -> z_day
Night crowd image -> baseline Swin + DoRA  -> global pooling + P -> z_night

"a photo of a crowd during the day" -> frozen CLIP text encoder -> T_day (label 0)
"a photo of a crowd at night"       -> frozen CLIP text encoder -> T_night (label 1)
```

The two labels describe illumination of crowd images, not object categories.
CLIP image embeddings are **not used** in adapter calibration or DoRA training.
The text encoder runs once to create normalized centres; they are cached and
the full CLIP model, including its unused visual encoder, is released. The
counting backbone, MAN, density head and day/night inference routes retain
their existing structure. This is an image-level, illumination-specific
adaptation of MonoCMoE's multi-class region alignment, not an exact reproduction
of its illumination-invariant class-centre objective. There is no BiLaLoRA.

### 1. Prepare the trained branches

Use a dual checkpoint containing a day-trained DoRA1 and night DoRA. If the
existing DoRA1 was trained on night images, first train it on day images:

```bash
python train.py --stage dora1 \
  --dora-checkpoint /path/to/night_dora/best_model.pth \
  --dora1-train-dir /home/mmslab-1080/Desktop/img2img-turbo/SHA/day/train \
  --dora1-val-dir /home/mmslab-1080/Desktop/img2img-turbo/SHA/day/val
```

The existing epoch schedules and dataset defaults are not changed. Review the
day reference's counting accuracy before using it to guide night adaptation.

### 2. Train the shared adapter with two-centre text contrastive loss

Install CLIP in the training environment:

```bash
pip install -r requirements-clip.txt
```

Freeze both Swin/DoRA branches. Train only `P` with the DICL-style objective:

```text
logits[k] = cosine(z, T_k) / temperature
L_align = -log( exp(logits[correct_domain]) / sum_k exp(logits[k]) )
```

A day image is compared against both text centres with label 0; a night image
uses label 1. Swin features do not start aligned with text merely because they
have the same dimension: this loss learns a correspondence to these two
specified centres. It does not establish alignment to every CLIP concept.

```bash
python train_clip_projection.py \
  --model-name swin_t_trans \
  --dual-checkpoint /path/to/day_dora1/best_model.pth \
  --train-dirs /home/mmslab-1080/Desktop/img2img-turbo/SHA/day/train \
               /home/mmslab-1080/Desktop/img2img-turbo/SHA/night/train \
  --dora-rank 4 --dora-alpha 4 --dora1-rank 4 --dora1-alpha 4 \
  --clip-temperature 0.07 \
  --crop-size 384 --epochs 20 --batch-size 8 --device cuda:0 \
  --output model/clip_projection/crowd_text_projection.pth
```

Supply the rank/alpha values of your source checkpoint. Day samples use DoRA1,
night samples use DoRA, identified by filename prefix `night_` (override with
`--pair-prefix`). Both versions of each scene remain together in the internal
10% holdout. Supply train directories only, never benchmark val/test splits.
The teacher and adapter see whole model-input patches, without object crops.

Calibration reports holdout contrastive loss, day/night accuracy, positive
text-centre cosine, and the paired `z_day - z_night` direction's cosine to the
text direction. The best holdout-loss checkpoint stores these metrics,
prompts, cached text centres, and fingerprints of the baseline and both adapter
branches, including scaling. A loss decrease alone is not proof of counting
improvement or complete semantic alignment.

**Existing CLIP image-teacher projection checkpoints are incompatible.**
Re-run this text-supervised calibration. `--baseline-checkpoint` calibration is
removed: use the actual dual checkpoint consumed by the next training stage.

### 3. Continue night DoRA with alignment and direction constraints

```bash
python train.py --stage dora \
  --model-name swin_t_trans \
  --dual-checkpoint /path/to/day_dora1/best_model.pth \
  --dora-train-dir /home/mmslab-1080/Desktop/img2img-turbo/SHA/night/train \
  --dora-val-dir /home/mmslab-1080/Desktop/img2img-turbo/SHA/night/val \
  --clip-stages dora \
  --clip-projection-checkpoint model/clip_projection/crowd_text_projection.pth \
  --clip-pair-root /home/mmslab-1080/Desktop/img2img-turbo/SHA/day/train \
  --clip-align-weight 0.1 --clip-direction-weight 0.001 \
  --clip-temperature 0.07 --clip-warmup-epochs 5 \
  --dora-epochs 20 --dora-val-start 0
```

Weights and temperature above are initial experiment settings, not tuned values.
The explicit validation start is needed when overriding your longer default
schedule with a short run. Use the **same dual checkpoint** for calibration and
fresh adaptation. Pair `night_IMG_x.jpg` with `IMG_x.jpg` in the day training
root; `--clip-pair-prefix` controls the removed prefix. Missing pairs or unequal
image sizes fail explicitly. Both images share resizing, crop and flipping.
Random grayscale augmentation is disabled in paired runs.

This stage fixes the baseline, day DoRA1, calibrated `P`, cached text centres,
MAN and density head. Only night DoRA updates. The night alignment loss uses
label 1 against both text centres; the optional directional constraint uses:

```text
delta_V = stopgrad(z_day) - z_night
delta_T = T_day - T_night
L_direction = 1 - cosine(delta_V, delta_T)
L_total = L_Bayesian + L_consistency
          + warmup * (align_weight * L_align_night + direction_weight * L_direction)
```

The fixed day reference is computed before the night forward. Gradients through
the fixed adapter update night DoRA, while the counting losses protect density
and count predictions. Direction similarity constrains the difference's angle;
it does not generate a day RGB image or force day/night embeddings to be equal.
`--clip-negative-prompt` and `--clip-positive-prompt` override the night/day
crowd prompts; use the same prompt pair for calibration and adaptation.

Both guidance weights default to zero. For controlled comparisons, keep the
same dual initialization, paired root, augmentation and schedule:

| Experiment | `--clip-align-weight` | `--clip-direction-weight` |
| --- | ---: | ---: |
| Counting control | 0 | 0 |
| Text alignment only | 0.1 | 0 |
| Text direction only | 0 | 0.001 |
| Alignment + direction | 0.1 | 0.001 |

The old `--clip-pair-weight` image-teacher objective is removed; its flag is
accepted only with zero to prevent old commands silently changing meaning.
Warmup starts at zero in epoch 1 and reaches full weight in epoch
`warmup_epochs + 1`; zero warmup enables guidance immediately. Logs record
`clip_alignment_loss`, `clip_direction_loss` and the warmup factor. Evaluate
MAE/RMSE on day, night and mix; text-centre accuracy is an auxiliary diagnostic.

Fresh runs validate baseline, day branch, initial night branch, model and prompt
identities. Epoch `.tar` resumes embed the fixed adapter, text centres and loss
configuration. Resume with the same text-guidance arguments; no external adapter
file or CLIP model download is needed on resume. Old image-teacher epoch runs
cannot silently resume the changed objective. The counting best `.pth` format
and inference routes remain available, and `test.py` accepts new alignment-only
or alignment+direction epoch checkpoints for dual routing. CLIP and the adapter
are not needed for counting inference.

CPU verification without downloading CLIP weights:

```bash
python -m unittest discover -s tests -v
```

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
