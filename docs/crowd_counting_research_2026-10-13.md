# 目前人群計數架構的誤差診斷與論文研究

整理日期：2026-10-13。範圍：目前工作區的 Swin-T/Swin-L、兩層 MAN、四路 concat-convolution 融合、雙 Q/V DoRA、日夜 router。這份整理沒有修改模型或啟動訓練。

## 1. 證據與限制

目前訓練紀錄與既有 CSV 屬於不同 run，不能混作同一次實驗：

| 證據 | 資料域 | 最佳 MAE/RMSE 或既有結果 | 說明 |
|---|---|---|---|
| baseline/1013-113128/train.log | day/val | 第 201 epoch：63.46 / 98.30 | 已記錄的最低 RMSE；第 426 epoch 的 MAE 為 63.43，但 RMSE 為 103.90 |
| dora/1013-133009/train.log | night/val | 第 331 epoch：75.54 / 126.55 | 本次讀取快照的最低 RMSE，run 尚在進行；兩階段驗證域不同，不能直接比較適應增益 |
| test_results.csv | day/val | 64.106 / 97.956 | 舊 classifier-router run |
| test_results.csv | night/val | 75.017 / 120.735 | 舊 classifier-router run |
| test_results.csv | mix/val | 69.561 / 109.937 | 舊 classifier-router run；checkpoint 路徑是 dora1/1013-042059 |

目前目錄沒有上述舊 dora1 checkpoint；無法確認 CSV 是否來自現行融合架構。需要用現行 checkpoint 重跑，才有可比較的逐張結果。

真正的 MSE = mean(error²)，RMSE = sqrt(MSE)。目前 trainer 的 mse 與 best_mse 變數、MSE 日誌實際都是 RMSE。固定樣本集合上，最小化兩者的模型排序相同。密度圖的逐格 MSE 不是總人數 MSE。

## 2. 舊逐張 CSV 的分析

只分析 domain=mix 的 364 列，避免將 day、night、mix 重複計入。

| 項目 | 結果 |
|---|---:|
| 日夜路由正確率 | 96.154%，350/364 |
| 平方誤差最大的 37 張之平方誤差占比 | 71.556% |
| 路由正確樣本之平方誤差占比 | 95.598% |
| day 平均偏差，prediction−GT | +13.112 人 |
| night 平均偏差，prediction−GT | −28.440 人 |
| 夜間 GT≥1000 的 18 張 MAE / RMSE | 241.010 / 304.016 |
| 夜間 GT≥1000 的平均偏差 | −178.053 人 |

大誤差案例：night_IMG_63，GT=1121，prediction=404.782，路由正確；night_IMG_90，GT=2256，prediction=1681.125，路由正確。它們支持先查高人數夜間漏計，但不能只靠 count 推斷一定是小人頭或低照度原因，仍須看圖片、點標註和密度圖。

## 3. 優先確認的程式與實驗設定

1. **同一 checkpoint、同一資料集比較各分支。** baseline route、固定 dora、固定 dora1、依檔名正確日夜標籤路由、classifier route 全部評估。依 GT 選誤差較低分支只能當離線診斷，不能當可部署成績。
2. **DoRA1 資料域。** 目前預設 night/train 訓練、day/val 驗證，推論卻分配給 day。若它定位為白天專家，先比較 baseline 白天結果，再用 day/train 或包含白天 replay 的 mix/train 訓練；這是實驗語意不一致，並非已證明的程式錯誤。
3. **提前驗證。** baseline 首次驗證為 epoch 201，恰好也是已記錄的最低 RMSE；無法排除更早有更好的模型。下一 run 從 epoch 1 開始，每 5 epoch 驗證，另存 best-MAE 與 best-RMSE。
4. **優化器。** 現行 baseline 的骨幹與新 MAN/融合/回歸頭共用 5e-6 學習率，且沒有 scheduler。測試參數分組、短 warmup 與衰減。建議起始搜尋 backbone 5e-6~1e-5、新模組 5e-5~1e-4；數值是本案實驗建議，不是論文保證。
5. **crop 設定。** baseline 預設 256，兩個 adapter 預設 384。先做相同 crop 的對照，不把更大 crop 自動視為更好。訓練 crop count 與驗證全圖 count 的 MAE/RMSE不能直接比較。
6. **資料配對。** 若 night 為日圖轉換，需查頭部幾何是否保留、標註是否仍對齊；day/night 同一原圖及其衍生圖應放在同一 split。182 張 val 與 SHA 標準 test 數量一致，但需核對檔案名後才能判定是否使用官方 test 做選模。
7. **無背景選項。** Bay_Loss 目前 res[:-1]、res[-1] 的處理預設最後一項是背景；若 use_background=False，最後一項會是人頭。切換前須先改成依背景設定分支處理。預設 use_background=True 不受這個特殊問題影響。

## 4. 論文及可接到目前架構的方法

### A. MAN，CVPR 2022：監督取捨與正規化

[原文](https://arxiv.org/html/2203.02636)

原文針對尺度變化使用可學習區域注意力，並用 Instance Attention 降低點標註噪聲。保留誤差較小的 90% 標註是作者設計，不是目前 topk 的程式錯誤。原文使用 4 層 encoder、正規項係數 100；目前是 2 層、直接將 cosine 項以係數 1 相加。骨幹、token 網格及 reduction 已改變，不能直接照搬係數。

建議：只改 supervision keep ratio，比較 0.9、0.95、1.0。若 1.0 能改善高人數夜間 RMSE，支持有效困難人頭被當作噪聲忽略的假說；若變差，保留穩健篩選。此處「訓練中捨棄 10% 點」與「評估最差 10% 圖」不是同一件事。

目前一筆 baseline epoch 600 的 Bayesian/consistency loss 約為 12.7604/0.0118；DoRA 快照末尾約為 26.8164/0.0370。數值上不能宣稱 cosine 已壓過主損失，仍需量測梯度。先做 λ=0、1、10，必要時再試更大權重，並記錄 loss 與梯度範數。

### B. DM-Count，NeurIPS 2020：總量與空間分布一起監督

[原文](https://arxiv.org/html/2009.13077) · [官方程式](https://github.com/cvlab-stonybrook/DM-Count)

作者用 count L1、normalized density 的 optimal transport、total variation distance 共同訓練，以避免過度依賴 Gaussian 點標註假設。TV 在這裡是兩個機率分布間的距離，不是常見的影像梯度平滑項。

UCF-QNRF 的 Table 5：只有 count loss 為 MAE/RMSE 103.1/175.9，加 OT 為 89.3/161.3，完整三項為 85.6/148.3。這是該論文內部消融，不能拿來預測本案必然改善多少。

建議：先在现有 Bayesian+cosine 上加一個小權重總量 L1，再另開完整 DM-Count 對照，不一開始把多種 OT/Bayesian 損失全部混合。現行 crop target 有邊界比例權重，新增 count target 要明確選擇與 Bayesian 一致的 target.sum()，或重新定義 hard inside-point count，不能未加檢查就用 len(points)。完整 OT 實作亦要定義邊界權重與空樣本。

若目標特别是 RMSE，可在 L1 穩定後比較小權重的 count 平方項或 Huber；這是本案推論，不是 DM-Count 的原始 count loss。平方項會更強烈關注大誤差，也會放大錯標，需先檢查資料及梯度尺度。

### C. Generalized Loss，CVPR 2021：透視與點標註監督

[原文](https://openaccess.thecvf.com/content/CVPR2021/html/Wan_A_Generalized_Loss_Function_for_Crowd_Counting_and_Localization_CVPR_2021_paper.html)

作者提出基於 unbalanced optimal transport 的 generalized loss，並設計 perspective-guided transport cost，改善計數與定位。可作為 Bayesian 或 DM-Count 的替代損失實驗；若本案遠近人頭尺寸差異明顯，值得測試。nearest-neighbor distance 只能當弱尺度 proxy，不等同真實透視標註；極密集及孤立人頭都可能偏差。

### D. STEERER，ICCV 2023：尺度選擇與逐層繼承

[原文](https://arxiv.org/html/2308.10468) · [官方程式](https://github.com/taohan10200/STEERER)

作者指出不同尺度的特徵不一定都適合每個區域；FSIA 選擇適合的特徵，MSIL 用區域選擇監督逐步繼承，最後使用最高解析度輸出。

目前把所有特徵降到 stride 16，MAN 又在 stride 32 處理，對小人頭可能有資訊限制，但需圖片與消融證實。建議先把 density stride 16 改為 8，保留 stage 2 的較細网格；同步修改 posterior stride、標註對應及輸出大小。384 crop 的網格由 24×24 變 48×48；融合與回歸部分空間用量约為 4 倍，非整個模型都 4 倍。

再測試四路特徵的空間 gate；普通 concat+1×1 已能學習通道混合，不是「完全沒有選擇」。空間 gate 是額外的逐位置尺度選擇。簡單 gate 只是受 STEERER 啟發的消融，不是完整重現 FSIA/MSIL。

### E. MPCount，CVPR 2024：跨域特徵與前景區域輔助監督

[原文](https://openaccess.thecvf.com/content/CVPR2024/html/Peng_Single_Domain_Generalization_for_Crowd_Counting_CVPR_2024_paper.html) · [官方程式](https://github.com/Shimmer93/MPCount)

作者使用 attention memory bank、content error mask、attention consistency loss，以及 patch-wise 人頭有無分類，處理未見域和密度標籤模糊。

建議先抽出輕量 patch occupancy 分支，接在融合特徵後，用點標註產生區域人頭有無標籤，和密度損失共同訓練。先做輔助監督、再測 soft gating，避免硬遮罩把夜間真實人頭全部刪掉。它主要針對背景假陽性，不能預期單靠它修好夜間高人數低估。

### F. UGSDA，ACM MM 2024：風格多樣性與密度一致性

[作者研究頁](https://chenlab.comp.polyu.edu.hk/Papers/paper1_guanchen/paper1_guanchen.html) · [出版資訊](https://research.polyu.edu.hk/en/publications/domain-agnostic-crowd-counting-via-uncertainty-guided-style-diver/)

作者使用 global style、local uncertainty perturbations、density distribution consistency 提升未知域泛化。對本案可先做不改人頭幾何的亮度、gamma、噪聲、色溫與對比增強；再比較同一圖片兩種風格的 count/density consistency。這只是簡化概念實驗，不是完整 UGSDA。

若有幾何對齊且標註可靠的 day/night 成對圖，可用清晰圖 teacher 指導夜圖 student。亮圖教師須先可靠；密度對齊需保留總量及幾何，不能只要求總人數一樣而忽略空間位置。新論文 [Lightweight and effective crowd counting and localization in diverse low-visibility conditions，2026](https://www.sciencedirect.com/science/article/pii/S0031320326003043) 的公開摘要使用 restoration-guided teacher 與 multidimensional KD；本次僅核對公開摘要，不引用未核對的細節或數字。

### G. 背景錯誤分析，WACV 2021

[原文](https://arxiv.org/abs/2003.13759)

作者發現背景假陽性會抵消前景漏計，使總 count 看似正確，但密度圖不可靠；加入前景分割分支抑制背景。建議畫出最差案例密度圖，分辨漏人與背景多算，再決定是否投入 occupancy/segmentation。

### H. DoRA，ICML 2024：適配容量對照

[原文](https://arxiv.org/abs/2402.09353) · [官方程式](https://github.com/NVlabs/DoRA)

原文的 magnitude/direction 分解用來縮小低秩適配與 full fine-tuning 的差距；它不是 crowd counting 成績保證。本文據本案提出的實驗：rank 4/8/16 固定 alpha/r；比較 Q/V-only、Q/K/V、attention output 或部分 MLP 適配；再以部分／全量 fine-tuning 當診斷對照。高 rank 不一定更泛化。

先比較只 DoRA 與 DoRA+fusion/head。若增加 head 可訓練性，須使用分支專屬 head，或以 day/night replay 保護共用 head；否則會改到白天模型，不能仍宣稱只更新夜間分支。

## 5. 實驗執行順序

| 優先度 | 實驗 | 主要回答的問題 |
|---|---|---|
| P0 | 現行 checkpoint 分支與路由完整評估、資料配對與 split 稽核 | 誤差來自計數、路由或資料？ |
| P0 | epoch 1 起驗證，另存 best-MAE/RMSE | 是否漏掉早期最佳 checkpoint？ |
| P1 | 分組學習率、warmup/decay | 新模組是否學得太慢，或後期訓練失穩？ |
| P1 | keep ratio 0.9/0.95/1.0，其他設定固定 | 高誤差真實人頭是否被排除？ |
| P1 | 加總量 L1，再比較小權重 Huber/平方項 | 訓練是否欠缺直接計數約束？ |
| P1 | DoRA1 改 day 訓練或白天直接用 baseline | 路由與專家訓練語意是否一致？ |
| P2 | 只 DoRA vs DoRA+分支 head/fusion | 凍結解碼器是否限制跨域校準？ |
| P2 | 同 crop 對照，stride 16 vs 8 | 尺度／解析度是否限制計數？ |
| P2 | Bayesian vs 完整 DM-Count | 分布監督是否有幫助？ |
| P3 | 空間尺度 gate、occupancy、成對蒸餾 | 依密度圖診斷選擇新的模組 |

每個實驗先固定 seed、資料 split、輸入策略、checkpoint 來源與其他超參數；前段只改一項。最後只組合各自有效的修改，至少跑 3 個 seed。記錄 day/night/mix 的 MAE、真正 MSE、RMSE、prediction−GT 偏差、GT count 分箱、最差 10% 平方誤差占比、GPU記憶體與推論時間。若 day/night 是同場景成對資料，bootstrap 以原場景成組抽樣，避免把相關樣本當獨立樣本。

最終選模使用獨立 validation；若現有 182 張實際是 SHA 官方 test，應從 train 另切 validation，避免持續用官方 test 選超參數。

結論：優先處理證據已支持的早期選模、高人數夜間漏計、資料域與更新範圍問題，再用 DM-Count、STEERER、MPCount/UGSDA 的概念做有對照的消融。沒有任何已查論文直接驗證目前 Swin+MAN+dual DoRA 組合會改善固定百分比。
