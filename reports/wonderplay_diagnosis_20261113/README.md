# PAC-NeRF WonderPlay：為何 loss 下降，影片與材料 RE 卻常沒有改善？

日期：2026-11-13。本文只診斷目前的 **WonderPlay rasterized pixel-flow + Warp reverse AD** 訓練。九場景的訓練與評估結果已存在；本文沒有重跑或覆蓋那些 checkpoint。影片請看[三欄對照總覽](../../outputs/pacnerf_original50_vs_wonderplay_20261111/final_three_way_videos/overview_last_frames.png)與[逐場景影片](../../outputs/pacnerf_original50_vs_wonderplay_20261111/final_three_way_videos/manifest.json)，順序為 WonderPlay、Original、觀測 GT。Original 只用來提供視覺參考，以下診斷和改進對象是 WonderPlay。

## 先說結論

1. **目前訓練的 loss 不是影片重建誤差，也不是材料 RE。** 它把 MPM 產生的 Gaussian 2D 位移 rasterize 成 pixel flow，再與觀測影格的固定 RAFT flow 比較。PSNR、重新渲染影像的 RAFT EPE、材料 RE 都是另外計算。因此 loss 下降不保證影片或參數靠近 GT。
2. **學習率與參數距離不匹配，確定是原因之一。** 九個 run 的 Adam learning rate 都是 `0.003`；E 用 `log10(E)` 座標。除 Letter 外，多數場景的 E 初值距換算後 GT 有 18–2532 倍，而 50 個同方向、大小約 `0.003` 的 log 座標步長只相當於約 `10^0.15 = 1.41` 倍。Trophy 的摩擦角直接用角度座標，50 次只從 10° 到 10.15°，GT 為 40°。
3. **小學習率不是全部。** Torus、Bird、Cream 的 E 反而往遠離 GT 的方向走；它們的初始完整 loss AD/FD 已通過。Trophy 把 learning rate 提到 `0.5` 做 5 次 reverse AD 更新後，材料 RE 改善，但 FG EPE 仍略變差。這些證據表示現有 flow 代理目標與材料辨識、影像準確度不一致，不能只把訓練延長或放大步長。
4. **不能宣稱九場景反傳都正確。** Playdoh、Toothpaste 的完整 renderer AD/FD 尚未通過；兩者只通過分段 MPM 路徑與完整梯度同方向檢查。其他場景的初始局部 gate 通過，也不能保證訓練全程、所有可見性切換都正確。
5. **影片可以提供材料資訊，但目前系統尚不能判定它足夠。** 單視角、幾何和時序誤差、參數耦合可能令多組材料產生相似光流。GT 注入與一條參數路徑的 loss 掃描已完成；尚需逐參數敏感度和 coverage 測試，才能判斷哪些材料能由單段影片唯一識別。不能把目前的失敗直接歸咎於「影片本身沒有資訊」。

**2026-11-13 實驗更新：** 上述 GT forward 注入與同一 WonderPlay loss 的參數路徑掃描已完成。九個 GT 注入 forward 均為有限值；七個場景的 GT 注入 loss 低於 WonderPlay update 50，但從初值往 GT 的路徑在 Droplet、Bird、Cat、Cream、Toothpaste 會先遇到較高 loss。這將問題進一步分成「步長不足」、「局部地形障礙」及「代理 loss/forward 不對齊」，詳見下節。

## 目前 loss 究竟怎麼算？

[實作](../../utils/wonderplay_flow_objective.py)在每個相鄰影格對 `t` 執行以下步驟：

1. MPM 模擬**全部粒子**，取得 Gaussian 的位置與 covariance。flow objective 以固定 stride 最多選取 `--max-points 20000` 顆 Gaussian；它不是 MPM 粒子上限。
2. 將每顆 Gaussian 在 `t` 與 `t+1` 的位置投影到影像平面，得到像素位移 `Δᵢ`。
3. Rasterizer 按深度排序，利用 covariance、opacity 與前方透射率 `Tᵢ` 合成 `F̂ₜ(p) = Σᵢ Tᵢ(p) αᵢ(p) Δᵢ`。目前 flow 背景為零，沒有對總 alpha 再正規化；見[rasterizer CUDA forward](../../WonderPlay/submodules/depth_diff_gaussian_rasterization_min/cuda_rasterizer/forward.cu)。
4. 固定 target `Fₜ^RAFT` 由觀測的白底影格預先透過 RAFT 算出，尺寸為 `256×480`。固定 mask `Mₜ` 是觀測影像縮小後 `min(R,G,B) < 0.95` 的像素。
5. 每個影格對的 loss 是 `0.5 × mean((F̂ₜ[p] − Fₜ^RAFT[p])²)`，mean 涵蓋 `Mₜ` 內的像素與兩個 flow 通道；最後對影格對再取 mean。

**這不是 `RAFT(rendered RGB)` 的 EPE，也沒有 RGB photometric、silhouette、深度、材料 GT 的直接懲罰。** 評估用的 FG EPE 則是先渲染完整影像、對渲染影格重新跑 RAFT，再與觀測 RAFT 比較；見[獨立 evaluator](../../scripts/evaluate_universal_reference.py)。這是相關但不同的函數。視覺上缺失、形狀不符或物體與 GT 錯位，不會像 RGB/silhouette loss 那樣被直接量到。當 rasterizer 的 Gaussian 覆蓋不到觀測 mask 時，該處預測 flow 接近零；這可能減弱校正錯位的梯度，需以逐影格 coverage map 實測。

材料 RE 由[scorer](../../scripts/score_universal_pacnerf_re.py)把 solver 參數轉為具名場景 GT 的物理單位，再平均 `|估計−GT|/|GT|`。Cream、Toothpaste 的黏度換算尚未認證，因此下表是 **μ、κ、yield stress 的部分 RE**。GT 使用[data/pacnerf/groundtruth.json](../../data/pacnerf/groundtruth.json)；`gt_table2` 是無具名場景映射的編號實例，不能直接混用。

## 九場景證據

下表全部比較 **update 0 與 update 50**，包括 Letter；Letter 的三欄最終影片則保留已有的 WonderPlay update 74。loss 取[每次更新紀錄](../../outputs/pacnerf_wonderplay_50_20261110/consolidated_training)，EPE/PSNR 取[獨立曲線重評](../../outputs/pacnerf_original50_vs_wonderplay_20261111/accuracy_curve)，RE 取[逐次參數換算](../../outputs/pacnerf_original50_vs_wonderplay_20261111/material_re_curves_9_scenes.json)。這裡的百分比是**同一方法、同一 loss 的變化**，沒有跨方法比較 raw loss。

| Scene | WonderPlay loss 變化 | FG EPE，px | PSNR，dB | 材料 RE | 初始 gradient gate | 判讀 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Letter | −9.31% | 0.897 → 0.801 | 30.33 → 30.82 | 2427.77% → 1294.65% | 完整通過 | 影片改善，但 RE 仍非常大。 |
| Droplet | −3.32% | 6.200 → 6.669 | 19.43 → 19.58 | 99.72% → 99.84% | 初始 MPM 分段；最終完整通過 | EPE、RE 不隨 loss 改善。 |
| Torus | −3.33% | 7.618 → 7.719 | 18.98 → 18.94 | 80.63% → 98.13% | 初始完整通過；最終 ν FD 不確定 | E、ν 往遠離 GT 方向，ν 到下界。 |
| Bird | −0.006% | 8.487 → 8.412 | 19.58 → 19.58 | 83.20% → 99.89% | 初始完整通過；最終 ν FD 不確定 | loss 幾乎沒變，ν 到下界。 |
| Playdoh | −3.63% | 7.658 → 7.660 | 20.87 → 20.96 | 76.16% → 71.64% | MPM 分段通過；完整 renderer 未解決 | RE 改善，影片動態仍不對。 |
| Cat | −0.32% | 6.199 → 5.421 | 28.99 → 28.96 | 77.26% → 78.48% | 初始完整通過 | EPE 改善，RE 和 PSNR 略變差。 |
| Cream | −15.35% | 4.999 → 4.431 | 23.74 → 23.42 | 99.50% → 99.69%* | 初始完整通過 | flow 改善，影像與材料未同步改善。 |
| Toothpaste | −0.29% | 8.214 → 8.231 | 22.30 → 22.28 | 99.89% → 99.91%* | MPM 分段通過；完整 renderer 未解決 | 幾乎沒有有效改善。 |
| Trophy | −0.126% | 5.151 → 5.214 | 21.99 → 22.00 | 75.00% → 74.63% | 初始完整通過 | 更新幅度不足，FG EPE 略變差。 |

`*` 部分 RE，不含未認證的 viscosity 單位換算。數值變化很小的場景仍可能受浮點或 RAFT 重評波動影響，不應把千分位差異過度解讀。完整九場景[影片](../../outputs/pacnerf_original50_vs_wonderplay_20261111/final_three_way_videos/manifest.json)與[ E、ν、μ、κ 每 update 曲線](../../outputs/pacnerf_original50_vs_wonderplay_20261111/original_vs_project_E_nu_mu_kappa_9_scenes.png)可直接檢查參數和形狀。

## 新實驗：GT 注入與初值到 GT 的 loss 路徑

使用同一組相機、時間戳、MPM 和 `--max-points 20000` 的 WonderPlay objective，對**初始、WonderPlay update 50、GT 注入**三組參數各做 forward；沒有 optimizer update。GT 的 E、ν、yield stress、摩擦角依現有 scorer 轉成 solver 單位。Cream、Toothpaste 的 viscosity 單位未認證，因此固定在初始值，屬於**部分 GT 注入**。[GT 注入參數與獨立影片評估](../../outputs/pacnerf_wonderplay_diagnostics_20261113/gt_injection/summary.json)、[三點 WonderPlay loss](../../outputs/pacnerf_wonderplay_diagnostics_20261113/wonderplay_loss_triplets/summary.json)均已保存。三點測試的初始 loss 與原訓練紀錄相符，差異約為浮點重跑量級。

| Scene | FG EPE：WP u50 → GT 注入，px | WonderPlay loss：WP u50 → GT 注入 | 直接判讀 |
| --- | ---: | ---: | --- |
| Letter | 0.801 → **0.664** | **0.734 → 0.799** | GT 影片較好，代理 loss 卻較差；明確的目標不一致。 |
| Droplet | 6.669 → **4.256** | 13.679 → **5.712** | GT 在 forward 中可大幅改善，訓練未達到。 |
| Torus | **7.719 → 8.141** | **23.717 → 26.365** | 目前 GT 換算/forward 下，GT 反而較差；需查模型、初始條件與量測定義。 |
| Bird | 8.412 → **7.330** | 17.619 → **17.458** | GT 較好，但 loss 差距不大。 |
| Playdoh | 7.660 → **4.568** | 16.353 → **8.886** | GT 在 forward 中大幅較好；完整 renderer gradient 仍待解。 |
| Cat | 5.421 → **3.237** | 8.851 → **4.354** | GT 在 forward 中大幅較好。 |
| Cream | 4.431 → **4.008** | 6.913 → **6.687** | 部分 GT 注入；PSNR 23.42 → 23.28，視覺指標仍不一致。 |
| Toothpaste | 8.231 → **8.122** | 20.242 → **19.859** | 部分 GT 注入，改善非常小；forward/幾何/時序仍可能主導影片差距。 |
| Trophy | 5.214 → **4.860** | 11.622 → **9.876** | GT 角度方向有較低 loss；正式訓練的角度步長太小。 |

接著沿初值到 GT 的**optimizer 座標**掃描：正值參數使用 log10、ν 使用 `log10(1−2ν)`、摩擦角線性，取路徑比例 `0, 0.1, 0.25, 0.5, 0.75, 0.9, 1`。結果見[八場景 loss 路徑圖](../../outputs/pacnerf_wonderplay_diagnostics_20261113/gt_path_loss_curves.png)及[九場景彙整 JSON（含八場景原始逐點 loss）](../../outputs/pacnerf_wonderplay_diagnostics_20261113/diagnostic_summary.json)。這是**沿一條路徑的離散掃描**，不是全參數空間的全域最小值證明。

- **Trophy：** loss 從 11.636 沿 GT 方向持續降到 9.876。主要可先修步長與訓練長度，但 FG EPE 不一定單調改善。
- **Playdoh：** 沿 GT 方向從 16.969 降到 8.886，沒有看到該取樣路徑上的障礙；步長與未解的 renderer gradient 都需要處理。
- **Droplet、Bird、Cat、Cream、Toothpaste：** 從初值往 GT 的前段，取樣 loss 先升後降。例如 Cat 8.879 → 9.853（路徑 25%）→ 4.354（GT）；Cream 8.166 → 9.088 → 6.687。這種有限路徑上的高 loss 區域可讓局部 Adam 往別的方向下降。只把 learning rate 加大，可能越過不穩定區，也可能使 loss 上升，必須受控驗證。
- **Torus：** 初始 24.534、訓練後 23.717、GT 26.365；掃描中 GT 方向大多更差。這是目前模型/觀測/GT 單位或初始條件不對齊的優先診斷場景，不能直接把「走向 GT」當作 optimizer 目標。
- **Letter：** 未做路徑掃描，但 GT 注入的影片 EPE/PSNR 較好、WonderPlay proxy loss 較高，已足夠證明目前代理目標不能單獨代表影片品質。

### Learning rate：已證實是限制，但不是唯一原因

[elastic/newtonian trainer](../../scripts/train_projected_flow_gradient.py)與[material trainer](../../scripts/check_pacnerf_material_flow_ad.py)的既有 run 都使用 `Adam(lr=0.003)`；material trainer 的 CLI 預設也是 `0.003`，未直接沿用場景 physics config 的 `lr` 排程。除了摩擦角以外，活躍參數通常在 `log10(E)`、`log10(Y)`、`log10(η)`、`log10(1−2ν)` 等座標上更新。這令不同場景的實際物理參數步長差異很大。

按[具名場景 GT](../../data/pacnerf/groundtruth.json)與目前[換算公式](../../scripts/score_universal_pacnerf_re.py)推回 solver 的內部 E，GT/初始 E 約為：Droplet **532×**、Torus **18.5×**、Bird **388×**、Playdoh **47×**、Cat **1732×**、Cream **157×**、Toothpaste **2532×**。這是**在目前 scorer 的 E、μ、κ 定義下推得**，不等於已證明 GT 參數會在 Warp forward 中穩定地重現影片。Letter 的 GT E 則比初值小約 8.4 倍。50 個 `0.003` 的步長不足以橫跨許多場景的尺度差距。

Trophy 是一個直接對照：正式 run 使用 `lr=0.003`，50 次後摩擦角 10° → 10.15°。另保存在[短程 `lr=0.5` reverse AD 對照](../../outputs/pacnerf_wonderplay_optical_loss_diagnostics_20261112/trophy_lr05_ds1.0_ep5_ds1.0_ep5/training.jsonl)中，5 次後變成 **12.495°**，loss **11.636 → 11.404**，RE **75.0% → 68.8%**；其[獨立評估](../../outputs/pacnerf_wonderplay_optical_loss_diagnostics_20261112/trophy_lr05_eval/evaluation.json)的 FG EPE 約 **5.185**，比初始 **5.151** 略差，Whole EPE 則 **5.522 → 5.368**。這顯示放大步長能提高參數移動量，但仍不保證所有視覺指標改善；該短程實驗不是新的正式 50-update 結果。

### 梯度與 objective：不能用一次 loss 下降判定正確

Torus、Bird、Cat、Cream、Trophy 的**初始完整 loss** reverse AD/central FD gate 通過；Letter 也通過。Droplet 初始只通過 MPM 分段，同方向，最終完整 gate 通過。這些結果支持「所有場景完全沒有反傳」這個說法**不成立**。但局部 gate 不能驗證整段訓練，也不能驗證影像 metric 或材料 RE 的方向。

Playdoh、Toothpaste 的完整 rasterizer gradient 仍有顯著 AD/FD discrepancy，尤其是 source-position/visibility 路徑；[既有九場景報告](../pacnerf_wonderplay_9scene_final_20261111.md)列出分段數據。Toothpaste 的 yield stress 完整 FD discrepancy 在目前 50-update gate 中仍明顯超過 10%，MPM-only 段則接近。這兩個場景不能用 50 次 loss 下降替代完整 renderer 驗證，也不能僅放寬 threshold。

### 前向影片與可辨識性：目前沒有排除模型/時序問題

Toothpaste 的觀測 RAFT 在前六個 frame pair 的平均前景位移約 8–17 px，後段降到約 0–4 px；最終模擬影像重新跑 RAFT 則從約 0.6 px 逐步增到約 9 px，前段慢、後段仍在動。其第 5 幀的物體中心在渲染影像約 `y=192`，GT 約 `y=326`；訓練前後這個差距幾乎不變。它的第 0 幀渲染前景面積約是 GT 的 1.78 倍，以白底簡單閾值估計的 silhouette IoU 約 0.56。這些是**診斷性影像量測**，不是正式 benchmark 指標，但能直接解釋為何影片看起來差很多。要區分材質、初速、重力、時間對齊、幾何或 renderer 的責任，還需要受控 forward 實驗。

多數其他場景的第 0 幀渲染與 GT 粗略 silhouette IoU 約 0.88–0.95，顯示它們不全是起始形狀問題；晚期可降得很低，例如 Bird 最後一幀的簡單閾值 IoU 近零。單一影片的光流也可能對不同材料組合不敏感：若參數敏感度矩陣近似退化，影片本身就不足以唯一識別所有參數。但目前沒有做該測試，**不能據此宣稱影片資訊不足**。

## 接下來應做的事：按診斷順序，不先盲目重跑 50 次

1. **根據已完成的 GT forward 追查剩餘 forward 差距。** 九場景 GT 注入均有限，且 Droplet、Bird、Playdoh、Cat 明顯比訓練結果好；先對 Torus、Toothpaste 的 GT 注入影片做逐影格輪廓、相機、時間戳、初速與 stress 單位核對。Cream、Toothpaste 需先認證 viscosity 換算，再做完整 GT 注入。若 GT 參數仍產生錯誤影片，先修 forward 模型或校準，不應只調 optimizer。
2. **把 flow 目標拆開看。** 每對影格保存觀測 RAFT、rasterized physical flow、渲染影像的 RAFT、Gaussian 累積 alpha/coverage、前景 mask、逐像素 residual 與逐對 loss；量化 observed mask 中有多少像素沒有模擬 Gaussian 覆蓋，及每對影格的梯度大小。這能檢驗「錯位後 loss 對參數沒梯度」是否真實發生。
3. **修完整 renderer AD 路徑。** 對 Playdoh/Toothpaste 分別固定 MPM state，逐項檢查 `means3D`、covariance、`delta`、opacity 的 backward 和多個合理 FD step；另外紀錄 Gaussian 排序、contributor count、alpha clamp、visibility 切換。只有在局部平滑區間同方向且相對誤差合理時，再信任其訓練 gradient。保留 `--max-points 20000`，不能靠降低取樣數通過正式 gate。
4. **接續已完成的 loss 路徑掃描，測可辨識性。** 目前只有初值→GT 的七點 WonderPlay loss；下一步在代表性場景逐一擾動 E、ν、Y、η或φ，估計觀測 flow 對參數的 sensitivity matrix，並對路徑點計算 FG/Whole EPE、PSNR、輪廓。若 rank 不足或多個參數高度耦合，改成分階段優化或加入額外視角/幀數，而不是只加 epoch。
5. **重設 optimizer 並受控驗證。** 在 forward 與 AD gate 可信後，為各參數選擇能跨越合理尺度的 learning rate 或 trust-region/line-search；監控參數是否撞邊界，並用同一 forward loss 做接受測試。Trophy 可先用較大角度步長做完整 50-update 對照；Torus、Bird 的 ν 下界與 E 反向移動也要單獨檢查。
6. **若最終目標是影片相似度，讓 loss 真正觀察到影片差異。** 保留 WonderPlay rasterized pixel-flow 主項，評估在其上加入可微 silhouette/photometric、遮擋與物體位置項，或校正 rasterized physical flow 與 RAFT(rendered RGB) 之間的代理偏差；先以小型受控場景和 GT-injection ablation 證明有幫助，再擴到九場景。這仍使用 reverse AD，不改成 FD optimizer。
7. **用多個指標作為完成條件。** 每次正式訓練記錄 loss、參數、RE、逐影格 FG/Whole EPE、PSNR、silhouette/coverage、渲染影片；同時保留未通過的 gradient gate 與數值不穩定標記。僅有 loss 下降不算材料辨識或影片重建成功。

## 目前可下的判斷

已完成的 GT forward 和 loss 路徑實驗進一步確認：**optimizer 尺度不匹配** 是 Trophy 等場景的直接限制；**有限路徑上的高 loss 區域**讓 Droplet、Bird、Cat、Cream、Toothpaste 的局部優化不容易走向 GT；**代理目標與影像品質不一致**在 Letter、Cream 等場景可直接觀察；Torus 在目前模型下連 GT 注入的 loss/EPE 都更差。局部 reverse AD 在多數場景有證據支持，但 Playdoh/Toothpaste 的完整 renderer 路徑仍待修正或釐清可見性非平滑性。影片是否足以唯一識別全部材料參數，仍需 coverage、單參數 sensitivity、完整黏度 GT 和 forward 校準實驗後回答。
