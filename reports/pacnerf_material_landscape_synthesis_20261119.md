# PAC-NeRF 材料參數與 WonderPlay 目標：跨實驗證據綜合

日期：2026-11-19。本文整合既有九場景 50-update 訓練、1D sensitivity、四場景真正 2D grid、Droplet E／GT PLY Chamfer、Trophy 正確摩擦角 sweep，以及四場景 joint Sobol。**本文只重分析已保存資料，沒有啟動新的 forward 或訓練。**

## 摘要

目前沒有一個能解釋全部場景的單一故障原因。證據支持四種不同情況：

1. **局部方向與遠處低谷不同。** Droplet 的 E 從初值 `3.0e-7` 小幅增加時 WonderPlay loss 上升；單獨 E 和聯合 E、ν 的 5-update 控制都把 E 降低。較遠處的換算 GT E `1.597e-4` 卻有低得多的 loss 與 3D Chamfer。這是已取樣的 loss barrier，能解釋「往 GT 的整體趨勢好、局部 Adam 卻往反方向走」；不能簡化為 Adam 符號錯誤。
2. **代理目標與影片品質的最佳方向依區域而變。** Letter 的 GT 注入使 FG EPE 明顯改善，WonderPlay loss 改善很少；Bird 的 E 在 1D 影像 EPE 上比 WonderPlay 更敏感。Playdoh／Cat 的 joint 樣本在 global 範圍內 WP 與 FG EPE 對齊，但在 local 範圍內「低 WP／高 PSNR」的排名關係反轉。Cream 的 WP 與 FG EPE 在 joint local 樣本中只有弱相關。
3. **觀測可能對部分參數不敏感，但結論僅限測過的條件。** 在七點 1D slice 中，Playdoh ν 與 Toothpaste viscosity 的四項指標變化都小；Toothpaste joint 樣本的中央 50% WonderPlay loss 很窄。這些不構成全域不可辨識或「任意參數都能重建相同影片」的證明。
4. **數值、梯度、GT 換算限制不能略過。** Torus 在目前 Warp forward 下，換算 GT 點的 WP／FG 均比初始差；Playdoh／Toothpaste 的完整 renderer AD／FD 尚未通過；Cream／Toothpaste viscosity 沒有認證的 GT 單位換算。Toothpaste joint 採樣另有 136 個 nonfinite，會讓有限值圖產生選擇偏差。

**最重要的研究判斷：**目前是「objective／observation sensitivity gap、局部 loss barrier、模型／GT 對齊問題、數值或梯度限制」的組合。不能只靠延長訓練、只看最低 loss，或把所有情況先稱作 ill-posed。

## 1. 資料與可比較的問題

| 實驗 | 範圍 | 改變／固定 | 可以支持的結論 |
|---|---|---|---|
| [九場景 WonderPlay 訓練](pacnerf_wonderplay_9scene_final_20261111.md) | 八場景 50 update；Letter 原紀錄 74 update，另有共同 0–50 曲線 | optimizer 同時改 active parameters | 當前 optimizer 實際移動與最後影片結果；**不能**單獨隔離某個參數的因果作用 |
| [標準化 1D sensitivity](pacnerf_standardized_sensitivity_results_20261117.md) | 23 條七點曲線，146/147 有限 | 每次只改一個 active parameter | 指定 reference 附近的條件靈敏度；不能推出整個參數空間 |
| [真正 2D grid](../outputs/pacnerf_2d_night_20261116/summary.json) | Droplet、Letter、Torus、Bird 各 20×20；有效 grid 314/307/385/400 | 同時改兩個 active parameters，其餘場景條件固定 | 該二維範圍的有限格點形狀；邊界最佳點不能稱為全域最小值 |
| [Droplet E＋Chamfer](../outputs/droplet_E_dual_loss_20261115/dense/results.csv) | 83 個有限 E 點 | 固定 ν 與場景，對同一 forward 計算 WP 與逐幀 3D Chamfer | Droplet 這條 E 路徑的局部 barrier 和兩種 supervision 的偏好 |
| [Trophy 正確角度掃描](../outputs/trophy_friction_0_60_20261115/results.csv) | 0°–60° 每度一點，61/61 有限 | E=0.01 固定，正確五參數傳遞 | 角度條件曲線；**舊** `pacnerf_material_landscape_20261115/trophy` 的 WP 值傳錯參數，禁用 |
| [四場景 joint Sobol](../outputs/pacnerf_joint_sampling_20261118/analysis/FORMAL_ANALYSIS.md) | Playdoh、Cat、Cream、Toothpaste；4,629 有限、203 nonfinite、168 pending | 三／四個 active parameters 同時改；3/4 global、1/4 local | 已抽樣 joint 點的指標對齊與低 loss 參數分散；3D 圖不是固定第四參數的曲面 |

所有 forward 採用現有 Warp MPM、固定 observed RAFT target 與 WonderPlay rasterized physical pixel flow；joint 任務明確使用 `--max-points 20000`。FG／Whole EPE 與 PSNR 來自[獨立 evaluator](pacnerf_standardized_sensitivity_protocol_20261117.md)。FG mask 是 GT RGBA alpha `==255`，nearest resize 到 256×480；FG／Whole 用同一對 RAFT flow，只差平均範圍。Whole 包括未歸零的背景 RAFT 預測，因此和 FG 排名可以不同。各場景 raw WP 數值沒有共同尺度，跨場景只比較**相對變化、排名與條件方向**。

## 2. 九場景：訓練結果與 landscape 的一致性

以下訓練數值取同一 WonderPlay 方法的 update 0→50 [診斷表](wonderplay_diagnosis_20261113/README.md)；Letter 的 74-update 最終影片不拿來充當 50-update。RE 是目前 scorer 的材料相對誤差，Cream／Toothpaste 為不含 viscosity 的部分 RE。`↑`／`↓` 是指標數值變化，不自動等於好壞。

| Scene | 50-update 現象 | 相應的 forward 證據 | 最保守的機制判讀 |
|---|---|---|---|
| **Letter** | WP −9.31%；FG `0.897→0.801`；RE 仍 `2428%→1295%` | GT 注入 FG `0.665`、WP `0.799`；2D sampled WP 最佳在 `(μ,κ)≈(318,50k)`，FG 最佳在 `(60,567k)`；1D ν：WP 範圍 0.06%、FG 2.12% | **proxy gap／材料尺度問題**：FG 改善不保證 RE 接近 GT；有限 grid 的 WP／FG 偏好也不完全重合 |
| **Droplet** | WP −3.32%；FG `6.200→6.669` 變差；E `3.0e-7→2.11e-7` 遠離 GT | 初值附近 E↑ 使 WP↑；dense E 掃描過 barrier 後 GT 附近 WP、Chamfer 均顯著低；2D GT 注入 WP／FG 都較初始好 | **局部 barrier＋遠距離更新不足** 是直接解釋；FG 變差顯示只下降 WP 還不足以保證影像動態改善 |
| **Torus** | WP −3.33%；FG `7.618→7.719`；RE `80.63%→98.13%` | 真正 2D grid：GT 注入 WP `26.365`、FG `8.140`，都比 initial `24.534`／`7.652` 差；WP sampled best 在 E 下邊界 | **目前 Warp forward／GT 對齊或目標偏好問題**；不能只責怪 optimizer 沒往 GT |
| **Bird** | WP 約 −0.006%；FG `8.487→8.412`；RE `83.20%→99.89%` | 1D E 的 WP 範圍 1.67%、FG 11.29%；2D WP 最佳 E≈`0.0243`，FG 最佳在掃描上界 E≈`0.0777` | **proxy sensitivity gap**：對 E 的影片 EPE 訊號比 WP 強；ν 接近下界的訓練結果另需看參數化與邊界 |
| **Playdoh** | WP −3.63%；FG `7.658→7.660` 幾乎不變；RE `76.16%→71.64%` | 1D yield 對 WP／PSNR 變化 11.42%／11.66%；1D ν 四指標都近乎平坦；joint global WP–FG ρ `0.982`，local WP–PSNR favorable ρ `−0.992` | **yield 敏感但局部影像指標偏好不同**；完整 renderer AD／FD 未解，無法把訓練限制全歸因 loss |
| **Cat** | WP −0.32%；FG `6.199→5.421`；RE `77.26%→78.48%` | 1D yield 對 WP 範圍 32.74%；joint global WP–FG ρ `0.909`，local WP–PSNR favorable ρ `−0.941`；GT 注入 FG 明顯較訓練後好 | **有效 flow 改善，但材料辨識／局部 PSNR 偏好仍不同**；50 次微小更新不足以證明可達 GT |
| **Cream** | WP −15.35%；FG `4.999→4.431`，PSNR `23.74→23.42` 反降；部分 RE 變差 | 初始 reference 的 1D E／ν WP 範圍 24.32%／27.31%；joint local WP–FG ρ `0.248`，選 WP top 5% 的 FG 中位數比直接選 FG top 5% 高 `0.223 px` | **objective/evaluation gap 候選**，但四參數同時變動與 viscosity GT 未認證，尚不能指出哪一軸造成 |
| **Toothpaste** | WP −0.29%；FG `8.214→8.231`；部分 RE 幾乎不動 | 1D viscosity 四指標皆低於 1% 範圍；joint 604/1050 有效點的 WP 在 `20–21`，但只有 6 點距 sampled best `15.52` 不超過 5%；joint global／local 失敗率 `9.1%／18.6%` | **局部低靈敏度／數值篩選／梯度與初始影片錯位混合**；紫色大片是 plateau，不是大片最佳解 |
| **Trophy** | WP −0.126%；角度 `10°→10.15°`，GT 40°；FG 略差 | 正確 61 點 sweep：10°→40° WP `11.636→9.876`、FG `5.153→4.864`，PSNR `21.995→21.641`；WP 最佳 sampled angle 60°、FG 59°、PSNR 15° | **角度步長過小＋PSNR/flow 偏好分歧**；舊傳參錯誤 sweep 必須排除 |

這張表是**證據分類**，不是九場景故障率統計。單一場景可能同時有 optimizer、代理目標、模型與梯度問題。50-update 訓練同時改多參數；1D 在指定 reference 固定其餘參數，所以兩者的方向不必完全一致。

訓練使用約 `lr=0.003` 的 Adam，正值材料在 log10 座標更新。若每次 E 的 log10 步長約為 `0.003` 且方向不變，50 次只相當於 `10^(50×0.003)≈1.41` 倍；Droplet、Bird、Cat 等場景的換算 GT E 與初值相差數百至數千倍。這說明**更新距離不足**是合理限制，但不能解釋 Droplet 的局部反向坡度、Torus 的 GT 注入變差，或 Playdoh／Cat 的局部 WP–PSNR 反向。因此延長 epoch 並非這九場景共同的充分解法。

## 3. 最有力的受控案例：Droplet E

Droplet 的[初值密集掃描](../outputs/droplet_E_root_cause_20261115/local_sweep/results.csv)在固定 ν=`0.4995` 時給出：

| E solver | WP loss |
|---:|---:|
| `2.85e-7` | 14.1173 |
| **`3.00e-7` initial** | **14.1485** |
| `3.15e-7` | 14.1786 |

因此 initial 附近 **E 增加會讓 WP 上升**。以 `z_E=log10(E)` 更新的[單獨 E 控制](../outputs/droplet_E_root_cause_20261115/single_E_control_v2_ds1.0_ep5/training.jsonl)和[聯合 E＋ν 控制](../outputs/droplet_E_root_cause_20261115/joint_E_nu_control_v2_ds1.0_ep5/training.jsonl)各跑 5 次，兩者的 E 都 `3.00e-7→約2.90e-7`，WP 均下降；第一步 AD `dL/dz_E≈+1.28`、Adam step 約 `−0.003`。兩種控制的 E 方向**沒有翻轉**。這支持局部 optimizer 按當前 loss 下降，不支持「聯合 ν 更新才把 E 推反」的說法。完整 AD／FD gate 的可用範圍仍應以原紀錄為準，這些控制不證明所有 renderer 梯度處處正確。

同一 E、同一 forward 的[83 點 WP＋3D Chamfer 並排圖](../outputs/droplet_E_dual_loss_20261115/dense/wp_vs_gt_ply_chamfer.png)顯示 WP 先在 `E≈8.10e-7` 達到 sampled 高點 `14.498`，再向大 E 下降；換算 GT E≈`1.597e-4` 的 WP=`5.707`，sampled WP 最低在 `2.694e-4`（`5.230`）。獨立 PLY Chamfer 在 initial=`0.007546`、換算 GT=`0.003370`；sampled 最低在 `1.274e-4`（`0.003349`）。**兩個 loss 都支持較大的 E 能改善結果，但 GT 附近不完全是 WP 最低點；局部 WP barrier 足以讓小步長優化先朝反方向。**

Chamfer 是 framewise、world-coordinate 幾何距離，不是 GT optical flow，也不需要 PLY 粒子 ID。PLY 的[來源、座標及模型限制](droplet_ply_gt_feasibility_20261115.md)已調查：檔案只有 xyz，與官方可選 GT archive 尚無 checksum 認證；原始 Taichi viscous fluid 與 Warp Newtonian 也未證明動力學等價。因此 Chamfer 的低谷是獨立幾何證據，**不是** Warp 轉換 GT E 已被證明為物理真值。

## 4. 1D、2D、joint 三種圖合起來說什麼

### 4.1 條件 sensitivity 不等於全域可辨識性

[九場景七點 1D 報告](pacnerf_standardized_sensitivity_results_20261117.md)指出 Newtonian Letter／Droplet 的 E 對 WP 變化約 28.78%／30.01%，Plasticine 的主要訊號則是 yield（Cat 32.74%，Playdoh 11.42%），Cream 初始 reference 的 E／ν 都明顯。另一方面，Playdoh ν 與 Toothpaste viscosity 四指標都小於探索性的 1% flatness cutoff。這些結論只在七點範圍和其餘參數固定的 reference 成立。Letter、Droplet、Torus、Bird、Playdoh、Cat、Trophy 固定其他材料於**換算 GT active set**；Cream／Toothpaste 因 viscosity GT 未認證，固定於**全 initial set**。不應把兩種 reference 混為一張材料排名。

### 4.2 真正 2D grid 顯示不同低谷與邊界效應

[四場景 2D summary](../outputs/pacnerf_2d_night_20261116/summary.json)把 initial 與換算 GT 直接放到同一 forward／evaluator 下：

| Scene | Initial→GT WP | Initial→GT FG EPE | Initial→GT PSNR | 對圖的關鍵讀法 |
|---|---:|---:|---:|---|
| Droplet | `14.149→5.712` | `6.195→4.256` | `19.43→21.58` | WP／FG 都往高 μ 改善；兩者 sampled best 同在掃描右上邊界 `(400,200k)`，尚未包住低谷 |
| Letter | `0.809→0.799` | `0.897→0.665` | `30.33→33.44` | WP 改善很小、FG 明顯改善；WP 與 FG sampled minima 在不同 μ、κ 區域 |
| Torus | `24.534→26.365` | `7.652→8.140` | `18.98→18.58` | 在目前 forward 下換算 GT 對三項均較差；WP sampled best 在 E 下邊界 |
| Bird | `17.620→17.458` | `8.487→7.330` | `19.58→19.66` | WP 動很小而 FG 大幅改善；FG sampled best 在 E 上邊界 |

圖中的白色格點是參數限制或 nonfinite／缺值，不是高 loss。各場景的 WP 與 FG 圖應成對閱讀：[Droplet WP](../outputs/pacnerf_2d_night_20261116/droplet_wonderplay_loss.png)／[FG](../outputs/pacnerf_2d_night_20261116/droplet_fg_epe.png)、[Torus WP](../outputs/pacnerf_2d_night_20261116/torus_wonderplay_loss.png)／[FG](../outputs/pacnerf_2d_night_20261116/torus_fg_epe.png)、[Bird WP](../outputs/pacnerf_2d_night_20261116/bird_wonderplay_loss.png)／[FG](../outputs/pacnerf_2d_night_20261116/bird_fg_epe.png)。當 sampled best 落在掃描邊界，只能說該有限網格的最佳點在邊界。

### 4.3 三／四參數 joint 點揭露「條件依賴」，不是平滑 3D surface

[joint 分層統計](../outputs/pacnerf_joint_sampling_20261118/analysis/FORMAL_ANALYSIS.md)將 global／local Sobol 分開：Playdoh／Cat 的 global `ρ(WP,FG)` 分別 `0.982／0.909`，但 local 的「低 WP／高 PSNR」有利相關係數是 `−0.992／−0.941`。Cream local `ρ(WP,FG)=0.248`，WP 最佳 5% 的 FG 中位數比直接選 FG 最佳 5% 高 `0.223 px`。Toothpaste 的 loss 20–21 plateau 佔有效點 `604/1050`，然而只有 `6/1050` 在 sampled best 的 5% 內；[聚焦圖](../outputs/pacnerf_joint_sampling_20261118/analysis/toothpaste_loss_focus.png)比紫色 3D 點雲更適合判讀。

[三維 PNG 索引](../outputs/pacnerf_joint_sampling_20261118/analysis/three_dimensional/README.md)中的每點是真實 forward；四參數場景將第四軸按 median 分成兩排。這不等於固定第四參數。Playdoh／Cat 的低 WP 遠距參數點對確實存在，卻只證明 sampled scalar 指標相近；**沒有**逐像素影片相同或結構性 ill-posedness 的證據。Toothpaste top-WP `E×yield` 正相關是補償候選，需固定 ν／viscosity 的受控 2D 才能驗證。Cream／Toothpaste 的 nonfinite 使好點分布受到選擇偏差。

### 4.4 為何 FG EPE 與 Whole EPE 可以反向

正式 evaluator 對兩者使用**相同的 observed/rendered RAFT flow**；FG 只在 GT alpha 前景平均，Whole 在全部 256×480 pixels 平均，背景 flow 沒有設為零。[Bird 像素檢查](pacnerf_standardized_sensitivity_protocol_20261117.md)顯示前景只佔約 3.46%。既有 Bird 2D 兩點出現實際排名反轉：`(E,ν)=(0.0007494,0.1974)` 的 FG／Whole 是 `8.571／7.305`，另一點 `(0.0112328,0.0684)` 是 `8.772／7.078`。第一點 FG 較好、Whole 較差；由相同 mask 權重反推的背景 EPE 約 `7.260` 對 `7.017`，足以解釋反轉。這支持該例是**前景／背景加權效應**，並未在 Bird 的可視化檢出 mask/frame 錯位；不能據此保證每一場景的 RAFT 背景估計都有物理意義。

## 5. 證據等級與尚未解決的問題

| 可說到哪一步 | 目前證據 | 還不能說什麼 |
|---|---|---|
| **已觀察** | 固定其他參數的 Droplet E 在 initial 附近 WP 局部斜率；同 forward 的 Chamfer；四場景真 2D 的有限格點；Trophy 正確五參數角度曲線 | 連續空間全域最小值、GT Warp 動力學完全正確 |
| **強候選** | Cream joint WP/FG gap；Letter/Bird 1D 或 2D 的 WP/EPE 靈敏度差；Playdoh/Cat local WP/PSNR 反向排名 | 哪個單一材料參數造成、可微 loss 必定朝哪方向更新 |
| **尚未證實** | Toothpaste E×yield 補償、多組參數產生近同影片、影片本身辨識力不足 | 結構性 ill-posedness、唯一性或可識別性定理 |

此外，[Original vs WonderPlay 50-update 比較](../outputs/pacnerf_original50_vs_wonderplay_20261111/comparison.md)是結果對照，兩套訓練 schedule／optimizer 不同，**不是**只換 loss 的受控 ablation。Letter 的 WonderPlay 最終影片使用既有 74-update checkpoint，與其他八場景的 50-update 結果不能視為同樣訓練預算。Playdoh／Toothpaste 的完整 renderer AD／FD discrepancy 尚在，[九場景報告](pacnerf_wonderplay_9scene_final_20261111.md)列出限制。這些限制會影響「loss landscape 是否足以預測 training」的解釋。

## 6. 後續最小驗證順序

本輪分析不新增 GPU 任務。若要把候選機制升級為受控結論，按資訊價值排序：

1. **Cream objective gap：**固定 ν、viscosity 等兩軸，在已觀察的低 WP／低 FG 區域做 paired 2D forward；四項指標同點計算。若 WP 改善而 FG 穩定變差，才判定同一條件下 proxy 不對齊。
2. **Toothpaste compensation：**固定 ν 與 viscosity，在 `E×yield` 做 coarse 2D；同時記 nonfinite 邊界與參數範圍。若低 WP 出現穩定斜向低谷，才支持參數補償。完整 renderer AD gate 需另修／驗證，不能拿 50-update 訓練作唯一證據。
3. **Playdoh／Cat local PSNR 反向：**在 local reference 附近選固定其餘材料的短路徑逐點 forward，驗證 WP／FG／PSNR 排名反轉是否維持；若維持，再查 motion proxy 與 photometric 渲染偏好的差別。
4. **Torus forward 對齊：**先審核換算 GT 的物理參數、初始幾何／時序／外力與官方模擬差異，再決定是否有意義調 optimizer；目前 GT 注入在 WP、FG 均較初始差。

## 再現與閱讀路徑

先看本報告與三張精選圖：[Droplet E 雙 loss](../outputs/droplet_E_dual_loss_20261115/dense/wp_vs_gt_ply_chamfer.png)、[四場景 joint 分層統計](../outputs/pacnerf_joint_sampling_20261118/analysis/formal_stratified_alignment.png)、[Toothpaste plateau](../outputs/pacnerf_joint_sampling_20261118/analysis/toothpaste_loss_focus.png)。想追特定場景，再進入 [九場景訓練 EPE／PSNR／RE 曲線](../outputs/pacnerf_original50_vs_wonderplay_20261111/original_vs_project_accuracy_curves_9_scenes.png)、[九場景 1D sensitivity](../outputs/pacnerf_standardized_sensitivity_20261117/analysis.md)、[四場景真 2D](../outputs/pacnerf_2d_night_20261116/summary.json)、[joint 逐點 CSV](../outputs/pacnerf_joint_sampling_20261118/analysis/samples.csv)與[訓練逐次紀錄](../outputs/pacnerf_wonderplay_50_20261110/consolidated_training)。

分析使用已保存的有限 forward 點，沒有做 IID p-value 或把 Sobol 點當隨機獨立樣本。所有 `sampled best` 都只代表給定範圍與成功點中的最小值。以這些限制呈現，報告可以支持後續研究決策，也避免把漂亮的色塊誤寫成可辨識性的數學證明。

Droplet 的並排圖可由已保存 CSV 重畫：

```bash
/disk2/ccchen/miniconda3/envs/physflow/bin/python scripts/plot_droplet_wp_chamfer_comparison.py
```

這支程式只讀取既有 `results.csv`，不會執行 GPU forward。
