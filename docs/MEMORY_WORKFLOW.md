# 三階段 Stage 3／Stage 4 Daytime Memory Bank 研究流程

新流程以 `--workflow memory` 與既有 double-DoRA 流程分開；`--stage cross_attention`
會自動選用 memory workflow。舊版 `dora1`、Router 與 checkpoint 載入功能保留。
所有階段可獨立執行，不必重新訓練前置階段。

## Stage 4：目前新實驗的預設

`train.py` 與新建庫入口 `build_memory_bank.py` 預設 `--memory-stage 4`。
保留 `--memory-stage 3` 作為對照，舊版 checkpoint 也能載入。下方原有完整流程範例
仍明確指定 Stage 3；Stage 4 建庫／訓練／測試指令如下。

```text
最佳白天 Baseline（eval、凍結）
  → 白天 train 圖片 → 完整 Swin Stage 1–4
  → Norm 前 Stage 4 特徵 → token 取樣 → MiniBatchKMeans → 固定 bank

夜間圖片 → Phase 2 Swin + DoRA（凍結，完整 Stage 1–4）
  → Norm 前 Stage 4 特徵 → W_Q → Q
固定白天 Stage 4 bank → W_K / W_V → K / V
  → Multi-Head Cross-Attention → W_O
  → α × Night Stage 4 + β × Cross
  → 原 Backbone Norm → 原 Feature Adapter → 原 MAN → 原 Head → Density
```

| 項目（384×384 裁切） | Stage 3 | Stage 4 |
| --- | --- | --- |
| 特徵位置 | features[:6] 後、Patch Merging 前 | features[7] 完成後、Backbone Norm 前 |
| 空間大小／Query 數量 | 24×24／576 | 12×12／144 |
| Swin-T／Swin-L channels | 384／768 | 768／1536 |
| K=256 時的 bank | [256,384]／[256,768] | [256,768]／[256,1536] |

Stage 4 的 bank 和 Query 必須都取 Norm 前，不新增 LayerNorm、sigmoid 或其他融合公式。
Stage 4 融合後不再執行 Patch Merging 或 Stage 4。Head 輸出仍為 stride 16 density map。
Baseline、DoRA、Norm、Adapter、MAN、Head 全部凍結，只更新 Q/K/V/O 與未固定的 α/β。
Density loss 到新模組的梯度保留；Bayesian、consistency、Count L1 與前五名保存規則不變。

```bash
python build_memory_bank.py --memory-stage 4 \
  --baseline-checkpoint model/swin_t_trans/baseline/RUN/baseline_best.pth \
  --data-dir /path/to/SHA/day --memory-size 256 \
  --tokens-per-image 256 --max-tokens 65536 --seed 42 \
  --device cuda:0 --output model/stage4_memory_bank.pt

python train.py --workflow memory --stage cross_attention --memory-stage 4 \
  --dora-checkpoint model/swin_t_trans/dora/RUN/night_dora_best.pth \
  --memory-bank-path model/stage4_memory_bank.pt --memory-size 256 \
  --dora-train-dir /path/to/SHA/night/train --dora-val-dir /path/to/SHA/night/val \
  --cross-attention-heads 8 --alpha-init 1.0 --beta-init 0.1 \
  --cross-attention-epochs 1200 --count-loss-weight 0.1 \
  --crop-size 384 --batch-size 4 --device 0

python test.py --workflow memory --experiment C \
  --checkpoint model/swin_t_trans/cross_attention/stage4/RUN/night_cross_attention_best.pth \
  --data-root /path/to/SHA --domains day night --split val --device 0 \
  --output-csv stage4_val_results.csv
```

可沿用同一份 Phase 1／Phase 2，**Stage 3 bank 與 Cross-Attention 投影不能直接當成
Stage 4 使用**，需要重建 bank 並從 Phase 2 初始化新的 Phase 3。預設 384 裁切時，
Stage 4 每張最多只有 144 tokens，`tokens-per-image=256` 會取出全部 144 個，不重複補足。
300 張最多 43,200 個，若 max_tokens=65,536 就不會再因總量上限淘汰；仍然是裁切後特徵。

checkpoint／bank 記錄 memory_stage、feature_dim、feature_position，Stage 4 的 position
為 `before_backbone_norm`。缺少新欄位的舊 Stage 3 checkpoint 依原架構還原；Stage 4
必須有明確位置資訊。bank 與模型的 stage、variant、channel、K、baseline 指紋不符會報錯。
Stage 4 codebook 仍只在 model_state_dict 保存一次，推論不重新建庫。

新輸出目錄分開為 `model/<model-name>/cross_attention/stage3/RUN/` 和
`model/<model-name>/cross_attention/stage4/RUN/`。舊版 `cross_attention/RUN/` 仍可作為
Stage 3 載入。`--resume` 以 checkpoint 的 memory_stage 為準；明確指定推論 checkpoint
時自動還原，亦可加 `test.py --memory-stage 4` 驗證。未指定 checkpoint 時，Experiment C
預設只搜尋 Stage 4；指定 `--memory-stage 3` 才會搜尋新／舊 Stage 3 目錄。

比較 Stage 3／4 時，固定相同的 Phase 2、資料、K、loss 係數與 α/β 訓練方式。
Stage 4 並不保證比較準或解決 β 縮小；仍保留兩路原始／加權 norm 記錄。

## 原 Stage 3 流程與共同設定

## 模型與梯度

```text
夜間影像 [B,3,H,W]
  → Swin features[:6]（Stage 1–3，包含 Night Q/V DoRA）
  → Stage 3 [B,H3,W3,C3]
  → Q projection；固定 Day codebook [K,C3] → 獨立 K/V projections
  → 8-head Cross-Attention（縮放 sqrt(C3/8)）→ O projection
  → α × Night + β × Cross（α=1.0、β=0.1，獨立且不受限）
  → features[6] Patch Merging → features[7] Stage 4
  → backbone norm → 原 Feature Adapter → 原 MAN → 原 Regression Head
  → Density Map [B,1,floor(H/16),floor(W/16)]
```

一般可整除的輸入，H3=H/16、W3=W/16；奇數大小依 torchvision 的 patch padding。
Swin-T 的 C3=384、Stage 4 channels=768；Swin-L 的 C3=768、Stage 4 channels=1536。
預設 K=256，bank 分別為 `[256,384]`／`[256,768]`。

| 模組 | Phase 1 | Phase 2 | Phase 3 |
| --- | --- | --- | --- |
| 原 Swin、Stage 4、Adapter、MAN、Head | 更新 | 凍結 | 凍結 |
| 單一 Q/V DoRA | 不掛載 | 更新 | 凍結 |
| Memory Bank | 不參與 | 不參與 | 固定 buffer |
| Cross-Attention Q/K/V/O | 不掛載 | 不掛載 | 更新 |
| α、β | 不掛載 | 不掛載 | 更新，可選固定 |

Phase 2/3 使用 eval 模式關閉凍結網路的 dropout/stochastic depth，同時保留 autograd。
Stage 3 模式的 Stage 4，以及兩種模式的 Norm、Adapter、MAN、Head 沒有被 `no_grad()` 包住，因此 density loss 能回傳到
Cross-Attention 與 α/β。Optimizer 只收集 `requires_grad=True` 的參數，並檢查名稱範圍。

訓練目標現在為 `Bay_Loss + consistency loss + count_loss_weight × count_L1`；
consistency 仍對 MAN 回傳的各層特徵計算。
沒有新增 GAN 或 feature alignment loss。測試另外只用 Bayesian counting loss backward，
確認 Q/K/V/O、α、β 的非零梯度，不依賴 consistency loss 才有梯度。
原 density ReLU 保留；若真實資料訓練出現整張負 pre-activation，ReLU 仍可能造成
計數梯度為零，需要由真實訓練觀察，這次沒有改變既有 Head。

Count L1 沿用 `/home/mmslab-1080/Desktop/L1 model` 的方式：逐張比較 density map
加總與裁切後 `targets.sum()`，取絕對值後對 batch 平均。目標保留裁切邊界的人數權重，
無人裁切目標為 0；不是 pixel-wise density loss，也不是模型權重 L1 正則化。
`--count-loss-weight 0.1` 為預設，`--count-loss-weight 0` 恢復原目標函數。
適用所有訓練階段；不改變模型、Memory Bank、α/β 或凍結規則，也不保證 β 不再縮小。

例如只在 Phase 3 啟用，保留現有 Phase 1/2 與 bank：

```bash
python train.py --workflow memory --stage cross_attention --memory-stage 3 \
  --dora-checkpoint model/swin_t_trans/dora/RUN/night_dora_best.pth \
  --memory-bank-path model/stage3_memory_bank.pt \
  --dora-train-dir /path/to/SHA/night/train \
  --dora-val-dir /path/to/SHA/night/val --count-loss-weight 0.1
```

若要從已訓練 Phase 3 微調，可改用 `--resume .../best_model.pth`；模型設定與 α/β
沿用 checkpoint，而 count loss 係數由本次 CLI 決定。用 `.tar` 續訓也遵循同一規則，
但 epoch 與 Optimizer 仍照原本方式還原。新 `.tar` 與 memory workflow 最佳 `.pth`
記錄 `count_loss_weight`，舊 checkpoint 不需要轉檔。若重新訓練 Phase 1 改變 baseline
權重，仍須針對該 baseline 重建 bank，並訓練對應 DoRA。

TensorBoard／train.log 加入原始及加權後 count loss（step、epoch），另記錄
`train/crop_count_mae`；原 `train/mae` 仍使用裁切保留點數，兩者定義不同。
驗證／測試的整圖 MAE、RMSE 與前五名評分方式維持原樣。

## 環境與資料

```bash
conda activate MAN
pip install -r requirements.txt
```

資料結構：`/path/to/SHA/{day,night}/{train,val,test}/`。
每個 split 包含 `.jpg` 與同名 `.npy` 標註；training annotations 沿用 `[x,y,nearest_distance]`。
以下替換 `/path/to/SHA`，以及各階段產生的 `RUN` 目錄。CPU 可把訓練／測試的
`--device 0` 改為 `--device cpu`，建庫的 `--device cuda:0` 改為 `--device cpu`。
Swin-L 使用 `--model-name swin_l_trans`，並提供官方 Swin-L 的 `--pretrained-path`；
不載入預訓練可用 `--no-pretrained-backbone`。

### Phase 1：白天 baseline

```bash
python train.py --workflow memory --stage baseline \
  --model-name swin_t_trans --data-dir /path/to/SHA/day \
  --pretrain-epochs 500 --pretrain-val-epoch 5 --pretrain-val-start 100 \
  --crop-size 384 --batch-size 4 --device 0 --save-top-k 5
```

完成後選用 `model/swin_t_trans/baseline/RUN/baseline_best.pth`。
此階段不掛載 DoRA、Memory Bank 或 Cross-Attention。

### 最佳 baseline 固定後建立 Memory Bank

```bash
python build_stage3_bank.py \
  --baseline-checkpoint model/swin_t_trans/baseline/RUN/baseline_best.pth \
  --data-dir /path/to/SHA/day --memory-size 256 \
  --tokens-per-image 256 --max-tokens 65536 --batch-size 4096 \
  --seed 42 --device cuda:0 --output model/stage3_memory_bank.pt
```

程式載入最佳 baseline，設為 eval 並凍結，**只讀取 `--data-dir/train`**，
不從訓練期間累積混合不同 epoch 的特徵。重用 `Crowd.train_transform` 的 normalization、
resize、crop、gray augmentation、flip；固定 seed、排序與單 worker，讓建庫可重現。
crop size／is_gray 從新 checkpoint 自動讀取，若明確指定不相符的值會報錯。
舊版純 state_dict 需補 `--model-name`，以及與原 baseline 相符的 `--crop-size`／`--is-gray`。

每張圖只取樣指定數量 tokens，再以 priority reservoir 將 RAM 使用限制在
`max_tokens × C3`；MiniBatchKMeans 對這個有界樣本池訓練，並固定 CPU thread=1。
不把整個資料集的全部 tokens 留在記憶體。儲存 codebook、Stage=3、維度、Swin variant、K、
建庫參數、training split 路徑、baseline checkpoint SHA256 與原始 baseline 權重指紋。
此版使用一般 MiniBatchKMeans，沒有 VQ、head-aware clustering 或動態更新。
建庫 API 參考 [MiniBatchKMeans 官方文件](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)。

### Phase 2：夜間單一 DoRA

```bash
python train.py --workflow memory --stage dora \
  --model-name swin_t_trans \
  --baseline-checkpoint model/swin_t_trans/baseline/RUN/baseline_best.pth \
  --dora-train-dir /path/to/SHA/night/train \
  --dora-val-dir /path/to/SHA/night/val \
  --dora-rank 12 --dora-alpha 12 --dora-lr 2e-4 \
  --dora-epochs 1200 --dora-val-epoch 5 --dora-val-start 0 \
  --crop-size 384 --batch-size 4 --device 0
```

完成後選用 `model/swin_t_trans/dora/RUN/night_dora_best.pth`。
此流程直接使用原本 `enable_dora()`／`QVDoRA`，不建立 dora1 或 Router。
K、原 Backbone、Feature Adapter、MAN、Head 全部凍結。

### Phase 3：Stage 3 Cross-Attention 與雙權重融合

```bash
python train.py --workflow memory --stage cross_attention --memory-stage 3 \
  --model-name swin_t_trans \
  --dora-checkpoint model/swin_t_trans/dora/RUN/night_dora_best.pth \
  --memory-bank-path model/stage3_memory_bank.pt --memory-size 256 \
  --dora-train-dir /path/to/SHA/night/train \
  --dora-val-dir /path/to/SHA/night/val \
  --cross-attention-heads 8 --alpha-init 1.0 --beta-init 0.1 \
  --cross-attention-lr 2e-4 --max-grad-norm 1.0 \
  --cross-attention-epochs 1200 --cross-attention-val-epoch 5 \
  --crop-size 384 --batch-size 4 --device 0
```

完成後選用 `model/swin_t_trans/cross_attention/stage3/RUN/night_cross_attention_best.pth`。
checkpoint 先嚴格載入 Phase 2，再掛載新模組，最後才建立 Optimizer。
bank 必須與 DoRA 的原始 baseline 權重指紋一致；Swin variant、Stage、C3、K 都會檢查。

每個 `--tensorboard-log-interval` 記錄 α、β、Night/Cross 的平均 token L2 norm、
加權後的兩組 norm，另保留原有 loss、validation MAE/RMSE。
Fusion 輸入／輸出、loss 與 gradient norm 有 NaN/Inf 檢查，梯度使用指定 max norm clipping。
α、β 不經 sigmoid，不限制加總或範圍。

### Checkpoint 與續訓

各階段沿用前五名：分數 `MAE + 2×RMSE`，越低越好，第一名仍叫 `best_model.pth`，
其餘為 `best_model_rank2.pth` 等，`top_models.json` 記錄排名。
`baseline_best.pth`／`night_dora_best.pth`／`night_cross_attention_best.pth` 是
第一名的相對 symlink，不另外複製大型權重。搬移單一 checkpoint 時應複製 symlink 的目標內容。
`--save-top-k`、`--save-all`、`--max-model-num` 的語意延續 README。

新流程的每個最佳 `.pth` 都包含完整 model_state_dict 與 research_config。
Phase 3 的 baseline、DoRA、Q/K/V/O、α/β 與 codebook buffer 都在 state_dict；
config 只存 bank metadata，不再存另一份 codebook。推論不需要外部 bank 檔，也不執行 K-Means。
舊版 raw baseline、standalone DoRA、dual DoRA、DoRA/LoRA 權重可載入；轉入研究流程時，
只提取其已訓練的 nighttime DoRA，舊版第二分支不帶入新模型。
舊版 raw DoRA 需要提供正確 rank/alpha，尤其 alpha 無法從 tensor shape 推得。

```bash
python train.py --workflow memory --stage cross_attention --memory-stage 3 \
  --resume model/swin_t_trans/cross_attention/stage3/RUN/100_ckpt.tar \
  --model-name swin_t_trans --dora-train-dir /path/to/SHA/night/train \
  --dora-val-dir /path/to/SHA/night/val \
  --cross-attention-epochs 1200 --crop-size 384 --device 0
```

`.tar` 會還原模型設定、Optimizer、epoch 與排行榜；resume 不需要再指定外部 bank。
需保留該最新 `.tar`、同目錄的排名權重與 `top_models.json`。
新 run 會複製保留權重，不刪除原 run；過期 `.tar` 的排行榜與現有檔案不一致時報錯。
crop size／is_gray 必須與原實驗一致；resume 時模型結構設定以 checkpoint 為準。
舊流程的 `.tar` Optimizer 結構與新的單 DoRA 不同，請以 checkpoint 作為新階段初始化，
不要用 `--resume` 跨 workflow 還原 Optimizer。

### A／B／C 評估

```bash
python test.py --workflow memory --experiment A \
  --checkpoint model/swin_t_trans/baseline/RUN/baseline_best.pth \
  --model-name swin_t_trans --data-root /path/to/SHA --domains day night \
  --split test --device 0 --output-csv experiment_A.csv

python test.py --workflow memory --experiment B \
  --checkpoint model/swin_t_trans/dora/RUN/night_dora_best.pth \
  --model-name swin_t_trans --data-root /path/to/SHA --domains day night \
  --split test --device 0 --output-csv experiment_B.csv

python test.py --workflow memory --experiment C \
  --checkpoint model/swin_t_trans/cross_attention/stage3/RUN/night_cross_attention_best.pth \
  --model-name swin_t_trans --data-root /path/to/SHA --domains day night mix \
  --split test --device 0 --output-csv experiment_C.csv
```

白天一定走 raw baseline，不啟用 DoRA 或 Cross-Attention；夜晚依 A/B/C 使用對應模型。
`mix` 沿用資料命名規則：`night_*` 為夜晚，其他為白天。新流程不需要分類 Router。
MAE、RMSE 與 `split_large_image` 切 patch 邏輯維持原本實作。可用 `--split val` 評估驗證集。
若沒提供 `--checkpoint`，會依 experiment 尋找對應階段最新的 `best_model.pth`。

新 Phase 1、2 checkpoint 也能由原本 `test.py` 的載入函式讀取；新 Phase 3 請使用 memory workflow。

### 消融

- 在 Experiment C 測試指令加入 `--disable-cross-attention`，夜晚只走 Baseline + DoRA。
- 在 Phase 3 訓練指令加入 `--freeze-fusion`，固定 α/β，Q/K/V/O 仍可訓練。
- 在建庫及 Phase 3 同時修改 `--memory-size K`，比較不同 K；同一模型的對照庫維持同樣 K。
- Phase 3 訓練可傳入同 shape、同 baseline 指紋且 metadata 完整的對照 `.pt` bank。
  Experiment C 評估也接受 `--memory-bank-path /path/to/control_bank.pt` 作為固定 bank 替換。
  這是外部對照庫介面；建庫程式本身仍只產生 K-Means bank，沒有加入其他 clustering。

## 測試與限制

加入 Stage 4 後在 MAN 環境（PyTorch 2.7.1）執行完整 suite：60 項，58 通過、0 失敗、
2 項 CUDA unavailable skip。建庫新／舊入口、訓練與測試 CLI 的 `--help`、語法檢查與
`git diff --check` 通過。

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p test_stage3_memory.py -v
python -m unittest discover -s tests -p test_stage4_memory.py -v
git diff --check
```

新增測試包含原 forward 等價、T/L Stage 3 維度、bank 序列化、attention 手算公式與縮放、
Bayesian loss 對 Q/K/V/O、α/β 的梯度、凍結參數完全不變、白天 bypass、夜間 bank 使用、
legacy checkpoint 相容、完整 Phase 3 checkpoint roundtrip、seed 重現與有界取樣。
另外以臨時合成 JPG/NPY 實際跑 baseline → 建庫 → single DoRA → Cross-Attention → resume
與 day/night 評估，檢查訓練入口、資料集、loss、排名與別名的整合。

Swin-T 在 CPU 執行實際 forward/backward；Swin-L 用真實模型的 meta tensors 驗證 Stage 3／4
拓樸與 shape，尚未執行 Swin-L 實際資料訓練。CUDA 測試在有可用裝置時才執行，否則明確 skip。
沒有宣稱完成真實資料訓練、改善 MAE/RMSE 或取得研究結論；仍需實際資料、訓練好的 baseline、
bank 品質、不同 seed 的完整實驗與 α/β 長期穩定性驗證。

## 本次新增或修改的檔案

| 檔案 | 原因 |
| --- | --- |
| `models/swin_c_multibatch.py` | Stage 3/4 提取與插入位置、day/night bypass，保留原 Head |
| `models/swin_dora.py` | 單 DoRA 可暫停，使白天精確取用 raw baseline 權重；參數鍵相容 |
| `models/memory_cross_attention.py`、`models/stage3_cross_attention.py` | 共用 Q/K/V/O 與融合，保留 Stage 3 舊匯入與參數名稱 |
| `utils/stage3_bank.py` | Stage 3/4 格式與位置檢查、baseline 指紋、有界 reservoir、K-Means |
| `build_memory_bank.py`、`build_stage3_bank.py` | 共用 Stage 3/4 建庫；舊入口固定 Stage 3，僅使用白天 train |
| `utils/memory_workflow.py` | 新／舊 checkpoint 嚴格載入、單 DoRA 提取、bank 身分檢查與完整輸出 |
| `utils/memory_trainer.py` | 三階段的建模、凍結、checkpoint metadata、最佳權重別名 |
| `utils/regression_trainer_cosine_multibatch.py` | 重用原 loss／驗證，新增建模 hooks、CPU、clipping、fusion logging |
| `utils/trainer.py` | 避免同一秒內續訓重用原輸出目錄 |
| `train.py` | workflow／cross_attention 階段與超參數，memory 流程預設夜間驗證集 |
| `test.py` | A/B/C、新模型白天／夜晚推論、bank 替換與停用消融 |
| `tests/test_stage3_memory.py`、`tests/test_stage4_memory.py` | 兩種位置的結構、梯度、凍結、checkpoint、合成資料全流程與 CUDA 條件測試 |
| `requirements.txt` | 加入 scikit-learn；threadpoolctl 由其依賴提供 |
| `README.md`、本文件 | 說明入口、可重現指令、相容性與驗證範圍 |

`datasets/crowd.py`、MAN、Regression Head、Bayesian／consistency loss 本身沒有因本次需求更動。
工作目錄原先已有的 `train.py` 超參數及 `losses/bay_loss.py` 修改予以保留。
