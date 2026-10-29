# Crowd counting with a text-free Stable Diffusion 1.5 backbone

The default model is `sd15_man`. It replaces Swin with the SD 1.5 VAE Encoder
and denoising U-Net while retaining the original MAN Crowd-Aware Module,
Refined Density Module, and training losses.

## Architecture

```text
RGB [-1, 1]
  -> SD 1.5 VAE encoder + quant_conv
  -> posterior mode (no random sample, no scaling, no added noise)
  -> U-Net, one forward at timestep 0, self-attention enabled
  -> final decoder image features (320 channels, stride 8)
  -> average pooling to stride 32 -> 1x1 projection to 512 channels
  -> original two-layer MAN encoder
  -> interpolate to stride 16 -> original density head -> ReLU
```

This follows the latent-mode and timestep settings in the official
[T2ICount implementation](https://github.com/cha15yq/T2ICount/blob/main/models/reg_model.py).
It uses the final decoder features so every retained U-Net stage participates
in baseline training. The noise-prediction convolution, unused VAE decoder,
and post-quantization decoder convolution are removed.

No tokenizer or Text Encoder is loaded. Every U-Net Transformer block has its
text attention (`attn2`) and associated normalization removed; self-attention
(`attn1`) and feed-forward layers remain. This explicitly skips cross-attention
rather than supplying an empty prompt. HSCM and RRC are not part of this model.

The model returns `(density_map, consistency_features)`, as before. Density
maps have shape `[B, 1, H//16, W//16]`, and their sum is the predicted count.
Training crops must be divisible by 16. Inputs are padded internally to multiples
of 64 and cropped in feature space; arbitrary validation images are split into
balanced non-overlapping tiles bounded by `--crop-size` to limit attention memory.
Training, validation and test inputs all use `[-1, 1]` normalization.

## Three training stages

| Stage | Updated parameters | Frozen parameters | Forward route |
| --- | --- | --- | --- |
| baseline | VAE Encoder, U-Net, feature projection, MAN encoder, density head | Adapters absent | baseline |
| dora | DoRA direction A/B and magnitude on U-Net self-attention Q/V | Entire baseline and DoRA1 | baseline + DoRA |
| dora1 | DoRA1 direction A/B and magnitude on U-Net self-attention Q/V | Entire baseline and trained DoRA | baseline + DoRA1 |

Adapters are attached to `attn1.to_q` and `attn1.to_v` in all down, middle,
and up blocks. They do not alter K, the attention output projection, the VAE,
or the MAN modules during adapter training. Each adapter reads the same raw
baseline weight. They are parallel branches; DoRA1 is not stacked on DoRA.
Both start from the trained baseline, with zero direction updates and baseline
weight magnitudes. Only the selected branch executes.

Baseline training includes gradients through VAE encoding. Adapter stages use
`no_grad()` only for the frozen VAE, preserving the graph from the unchanged
loss through the frozen counting modules back to the active U-Net adapter.
The frozen network stays in evaluation mode during adapter stages.

The unchanged objective is:

```text
Bayesian loss + cosine consistency loss + count_loss_weight * count L1
```

`--count-loss-weight` defaults to `0.1`. The count target preserves fractional
people at crop boundaries and zero-person crops. No extra count head is used.

## Setup and weights

Install a compatible PyTorch/torchvision pair for your CUDA environment, then:

```bash
pip install -r requirements.txt
```

The SD implementation uses Diffusers 0.32.2 and requires PyTorch >= 2.1.
Only the `vae/` and `unet/` components are loaded from
`stable-diffusion-v1-5/stable-diffusion-v1-5`. Initial pretrained loading needs
network access or cached weights. `--pretrained-path /path/to/sd15` can select
a local Diffusers-format directory containing those two subdirectories.
A single original `.ckpt` file is not accepted by this option.

Training checkpoints contain the full retained model. Resuming or evaluating
a checkpoint constructs the SD 1.5 architecture locally and loads its weights;
it does not download pretrained components. Swin checkpoints cannot initialize
this SD model. `--no-pretrained-backbone` explicitly starts from random weights.

## Training commands

Each training/validation directory contains `.jpg` images and corresponding
`.npy` point annotations. The default data paths are the existing day/night
paths in `train.py`; override them for your dataset.

Run all three stages, passing each stage's best checkpoint to the next:

```bash
python train.py --model-name sd15_man --stage all \
  --data-dir /path/to/baseline \
  --dora-train-dir /path/to/dora/train \
  --dora-val-dir /path/to/dora/val \
  --dora1-train-dir /path/to/dora1/train \
  --dora1-val-dir /path/to/dora1/val
```

Run stages separately:

```bash
python train.py --stage baseline --data-dir /path/to/baseline
python train.py --stage dora \
  --baseline-checkpoint /path/to/baseline/best_model.pth
python train.py --stage dora1 \
  --dora-checkpoint /path/to/dora/best_model.pth
```

The current defaults are 384x384 crops, batch size 4, and 10 epochs per stage.
Set independent schedules with `--pretrain-epochs`, `--dora-epochs`, and
`--dora1-epochs`, and learning rates with `--lr`, `--dora-lr`, `--dora1-lr`.
Validation occurs every 5 epochs starting at internal index 0 (after the first
epoch). The validation intervals/starts are independently configurable for
all three stages. Baseline full tuning includes VAE gradients and needs more
memory than a frozen VAE; reduce `--batch-size` to fit the GPU.

DoRA and DoRA1 each default to rank 4 and alpha 4. Use `--dora-rank`,
`--dora-alpha`, `--dora1-rank`, and `--dora1-alpha` to configure them.
Epoch `.tar` checkpoints record stage, optimizer, adapter settings, and the
fixed SD feature settings. `--resume` resumes the selected stage. Best `.pth`
checkpoints contain model weights; supply matching rank/alpha values when
loading them. Checkpoints and TensorBoard logs are saved under
`model/<model-name>/<stage>/<run>/`.

## Evaluation and routing

```bash
python test.py --model-name sd15_man --router-mode fixed \
  --baseline-checkpoint /path/to/baseline/best_model.pth \
  --dora-checkpoint /path/to/dora/best_model.pth \
  --dora1-checkpoint /path/to/dora1/best_model.pth \
  --data-root /path/to/SHA --domains day night mix --split val
```

Existing routing remains available with `--router-mode filename` or
`--router-mode classifier`. Both use a DoRA1-stage checkpoint containing both
branches. Classifier routing additionally needs `--router-checkpoint`.
The router receives its own image preprocessing; the counting model uses
SD normalization. Day selects DoRA1 and night selects DoRA.

Legacy Swin architectures remain selectable using `--model-name swin_t_trans`
or `swin_l_trans`. Historical documentation is in
[docs/SWIN_LEGACY.md](docs/SWIN_LEGACY.md); its old defaults are archived.

## Verification

```bash
python -m unittest discover -s tests -v
```

The SD tests use real Diffusers VAE/U-Net blocks with reduced channel widths
on CPU. They check baseline gradients and updates (including VAE), exclusive
adapter updates, inactive-branch exclusion, shared-baseline initialization,
checkpoint transitions, normalization, output dimensions and tiling.
They do not establish full-size pretrained SD accuracy, GPU memory use, or
training convergence.

Original MAN paper: [Boosting Crowd Counting via Multifaceted Attention](https://arxiv.org/abs/2203.02636).
