# Boosting-Crowd-Counting-via-Multifaceted-Attention
Official Implement of CVPR 2022 paper 'Boosting Crowd Counting via Multifaceted Attention'

This version replaces VGG19 with a Swin-Large backbone (2+2+18+2 = 24
attention blocks). The stage-4 feature is projected from 1536 to 512 channels
and passed to the original multibatch MAN Transformer encoder and density-map
head. The original Bayesian and cosine-consistency losses remain in use.
The CCST count head and CCST trainer are not used.
Training also includes an auxiliary count L1 loss with weight 0.005 by default.

## 日夜平行特徵模型（新增）

白天只執行原本的 MAN/Crowd-aware；黑夜才執行 Stage 2/3 的 PFA（亦稱 PRA）＋MDC。
Stage 4 只送往 Crowd-aware，不送入 PFA。每張圖片只執行一次選定路由的 Swin。

```text
day   -> Swin/DoRA1 -> stage4 -> feature_adapter + MAN -> resize stride16 -----> shared regression
night -> Swin/DoRA  -> stage4 -> feature_adapter + MAN -> resize stride16 -> C --+
                    -> stage2/3 -> PFA + MDC -> resize + 1x1 projection -> M ----+-> fusion -> shared regression
```

白天不呼叫 PFA、MDC、對齊投影或 fusion；Stage 2/3 仍作為 Swin 的正常前置階段執行。
黑夜 PFA 把 Stage 3 對齊 Stage 2 的尺寸，兩者以 CBR（Conv/BatchNorm/ReLU）投影至
256 通道後相加。MDC 並行使用 dilation 1/2/3，各輸出 128 通道，串接後加上 384 通道旁路。
這部分改寫自參考分支的 CCTrans Regression，但沒有獨立密度頭。

黑夜的 384 通道 MDC 特徵先縮放到 C 的實際尺寸，再以 1×1 Conv 投影至 512 通道。
融合為 `C + alpha * H(concat(C, M))`；H 是 `1x1 Conv(1024->512) + GELU + 3x3 Conv(512->512)`。
`alpha` 是可學習純量，`--fusion-alpha 0.1` 指定初始化值，不限制訓練後的正負號。
兩種路徑共用既有 `reg_layer_0`，輸出仍是 `[B, 1, floor(H/16), floor(W/16)]`。
這是起始實驗設計，並不保證優於原模型。

### 五階段訓練

| 階段 | 資料與路由 | 更新參數 |
| --- | --- | --- |
| `baseline` | 原本 day/train、day/val | 完整 Swin + MAN + regression |
| `dora` | 原本 night/train、mix/val，DoRA | 只有 DoRA |
| `dora1` | 原本 night/train、day/val，DoRA1 | 只有 DoRA1 |
| `fusion` | `--fusion-train-dir`、`--fusion-val-dir`，固定 DoRA | 新增 PFA/MDC、投影、fusion |
| `joint` | mix/train、mix/val，依 router 選日夜 | 只有 Crowd-aware（MAN 投影、MAN encoder）與共用 regression |

前 3 階段沒有新增模組，舊 baseline/DoRA/DoRA1 權重的結構保持相容。
`fusion` 可從原本 DoRA1 或原本 `finetune` 完整模型開始，並初始化新增模組。
`joint` 必須從完成的 `fusion` 模型開始。原本只調 MAN 的 `--stage finetune` 仍保留。
`--stage all` 現在依序執行這 5 階段；`--stage both` 仍只執行前 2 階段。

若已有雙 DoRA checkpoint，直接執行新增的兩階段即可：

```bash
python train.py --stage parallel \
  --dora1-checkpoint /path/to/dora1_or_finetune/best_model.pth \
  --fusion-train-dir /path/to/SHA/night/train \
  --fusion-val-dir /path/to/SHA/night/val \
  --joint-train-dir /path/to/SHA/mix/train \
  --joint-val-dir /path/to/SHA/mix/val \
  --router-mode classifier --router-checkpoint /path/to/router/best_model.pth \
  --dora-rank 12 --dora-alpha 12 --dora1-rank 8 --dora1-alpha 8
```

rank/alpha 必須符合來源 checkpoint；舊的 4/4 權重需改用對應值。
也可用 `--router-mode filename`，不需要 router checkpoint。
單獨第 4 階段用 `--stage fusion --dora1-checkpoint ...`；第 5 階段用
`--stage joint --fusion-checkpoint .../fusion/RUN/best_epoch_NNN.pth`。
`fusion` 的路由固定為 DoRA，與檔名或分類器無關。
`joint` 的分類器看完整原圖的 224×224 副本，再做計數 crop；路由快取一次。
不同路由分組 forward，按樣本比例累積 loss/梯度，再做一次 optimizer step。

兩個新階段都維持 `Bayesian + MAN cosine consistency + 0.005 * count L1`，
Bayesian 與 count 監督最終密度圖；黑夜的 consistency 仍取 MAN 內部特徵。
沒有 DM-Count、額外分支密度頭或分支對齊 loss；fractional crop targets 保持原規則。
日夜共用 regression，因此 joint 更新也可能改變白天表現，請同時查看 day/night/mix 指標。

`fusion` 預設 Adam、LR `1e-4`、50 epochs；既有 MAN/regression 與 Swin/adapters 都凍結。
新增 BatchNorm 統計在此階段學習；梯度穿過凍結的 regression 回傳到新增模組。
`joint` 預設 Adam、LR `1e-7`、50 epochs；只更新 Crowd-aware 與 regression。
Swin、adapters、router、PFA/MDC、對齊投影、fusion 卷積與 alpha 全部凍結且維持 eval mode；
新增 BatchNorm 的 running statistics 與 affine 參數也全部凍結。
黑夜仍會經過凍結的 fusion，梯度能穿過 fusion 回到 Crowd-aware；不使用 no_grad 包住 fusion。
可用 `--fusion-lr/--fusion-epochs/--joint-lr/--joint-epochs` 調整。

新階段沿用 epoch 0 驗證、每 epoch 驗證、`MAE + 2*RMSE` 前五名排名、原子存檔與 RNG 續訓。
輸出位於 `model/<model-name>/fusion|joint/<run>/`，包含帶模型/adapter metadata 的
`best_epoch_NNN.pth`、`ranking.json` 與 `last_checkpoint.tar`。
續訓必須指定單一階段（`fusion` 或 `joint`），搭配 `--resume .../last_checkpoint.tar`；
資料、router、loss、rank、LR、batch、crop、seed、初始 alpha 等設定必須一致。
可增加目標 epochs。`--device cpu` 可執行小資料驗證；不會自動開始正式長時間訓練。

### 測試新增模型

```bash
python test.py --parallel-checkpoint /path/to/joint/best_epoch_NNN.pth \
  --router-mode classifier --router-checkpoint /path/to/router/best_model.pth \
  --data-root /path/to/SHA --domains mix --split val \
  --output-csv parallel_summary.csv --routing-csv parallel_routing.csv
```

新入口自動讀取模型與 adapter metadata；joint 的動態路由模式及 classifier 權重必須符合訓練。
`--router-mode fixed --dual-route dora|dora1` 可作固定路由診斷。
`--compare-before-after` 會與該階段的來源權重比較（joint 的 before 是 fusion；fusion 的 before 是原 MAN），
來源搬移後可用 `--dora1-checkpoint` 指定新位置，內容指紋必須相同。
mix 摘要含 `subset=overall/day/night`，詳細 CSV 記錄每張圖片實際選定的路由與計數誤差。

```bash
python -m unittest discover -s tests -v
```

[arxiv](https://arxiv.org/pdf/2203.02636.pdf) | [知乎](https://zhuanlan.zhihu.com/p/478023612) | [B站](https://www.bilibili.com/video/BV13Y411u7r5?share_source=copy_web)

![image](structure.png)

## Train

Install dependencies with `pip install -r requirements.txt`. The current CLI
defaults to `--model-name swin_t_trans --stage dora1` and local dataset/checkpoint
paths. Use explicit paths for your installation. `--stage both` trains the
baseline and then Q/V DoRA; `--stage all` runs all five stages described above.
Baseline uses day/train and day/val; DoRA uses night/train
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

# Run all five stages using filename routing (or provide a trained classifier).
python train.py --stage all \
  --router-mode filename \
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
For the new architecture, `--stage all` runs baseline -> DoRA -> DoRA1 -> fusion -> joint.
This section documents the retained legacy `--stage finetune` path.

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
