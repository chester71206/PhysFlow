# Letter 材質反推：點投影 flow loss 與 WonderPlay 式像素 flow loss

日期：2026-11-09。這份報告說明目前程式實際做的事，以及已完成的 Letter
50 次更新實驗；舊 WonderPlay 200 次更新工作已在第 140 次停止，checkpoint 保留。新的 Letter 點版重跑已依使用者要求停止；其餘 PAC-NeRF 場景改排 WonderPlay 式像素 flow 訓練。

## 研究問題與共同部分

兩種方法都從同一個 Letter 3D Gaussian 場景、相機和物理初始設定出發，讓
MPM 依材質參數 `E, ν` 產生粒子軌跡。監督訊號是從固定觀測影像取得的 RAFT
2D flow；RAFT 權重不受訓練更新，且兩種方法都不對「render RGB → RAFT」
整條鏈做反向傳播。參數梯度沿「材質 → 完整 MPM 時域 rollout → 粒子位置 →
2D flow loss」回傳。兩種方法的差異在於**如何把 3D 粒子運動轉成預測的
2D flow，以及在哪些位置比較**。

記 `θ=(E,ν)`，第 `t` 幀第 `i` 顆 Gaussian 的世界座標為 `X_{t,i}(θ)`，
相機投影為 `π_t`，觀測 RAFT flow 為 `F_t^{obs}(u)`。所有距離以 RAFT flow
解析度的像素為單位；Letter 有 16 個 source frames、15 個 flow pairs。

## 舊方法：在可見 Gaussian 中心比較

對每顆被選中的 Gaussian，直接計算投影中心的位移：

`d_{t,i}(θ) = π_{t+1}(X_{t+1,i}(θ)) − π_t(X_{t,i}(θ))`。

在當前 source 投影點取樣固定的觀測 flow，再對固定選中點做均方誤差：

`L_point(θ) = mean_t [ 0.5 mean_{i∈V_t} ||d_{t,i}(θ) − F_t^{obs}(π_t(X_{t,i}(θ)))||² ]`。

`V_t` 在初始軌跡上選定，後續訓練不重新選：先按最大點數等間距抽樣，
排除 opacity ≤ 0.02、視錐外及相機後方的點；以每個像素的最近中心深度
作近似 z-buffer，再要求投影位置落在觀測影像的非白色區域。保留下來的點
在 loss 中**等權**。Opacity 只參與選點門檻，Gaussian 的屏幕大小、彼此
重疊的 alpha 混合和像素覆蓋率不進入 loss。[程式](../utils/projected_flow_objective.py)

好處是簡單、計算量較低，而且梯度直接對應物理點的投影位移。限制是把
Gaussian 當作一個中心點；某顆點即使對影像只有很小的實際貢獻，入選後仍
可與主導畫面的點有相同權重。初始可見性決策固定，也無法反映後續遮擋變化。

## 新方法：將 Gaussian 位移 rasterize 成像素 flow

對每顆選中 Gaussian 仍計算上述 `d_{t,i}`，但另外將其 source 位置、
covariance 與固定 opacity 送入 WonderPlay/HUGS 的 flow rasterizer。對像素
`u`，依深度排序的 Gaussian 用 `α_{t,i}(u)` 表示該像素的 opacity 乘上
Gaussian footprint；其前方透射率為
`T_{t,i}(u)=∏_{j<i}(1−α_{t,j}(u))`。程式的 flow 合成為

`F_t^{pred}(u;θ) = Σ_i T_{t,i}(u) α_{t,i}(u) d_{t,i}(θ)`。

這裡**沒有除以** `Σ_i Tα`；部分透明或低覆蓋的像素，其預測 flow 會向
零位移衰減。然後只在固定的觀測前景 mask `M_t` 上比較：

`L_pixel(θ) = mean_t [ 0.5 mean_{u∈M_t,c∈{x,y}} (F_{t,c}^{pred}(u;θ) − F_{t,c}^{obs}(u))² ]`。

在我們的實作中，`M_t` 是將觀測 RGB 縮到 RAFT 解析度後，以任一通道
`<0.95` 判定的前景；它不是每次重新渲染的 alpha mask。Gaussian opacity
本身也固定，並未作為要反推的參數；但其像素貢獻及遮擋順序會隨運動改變。
前景以外的背景 flow 沒有直接受罰。這次實作從同一組約 102,473 顆
MPM/Gaussian 粒子中，每幀等間距選取 995 顆作為像素 flow rasterization
輸入，並使用模擬輸出的動態 covariance；這不是把物理模擬縮到 995 顆。
[新 objective](../utils/wonderplay_flow_objective.py)；
[rasterizer 合成式](../WonderPlay/submodules/depth_diff_gaussian_rasterization_min/cuda_rasterizer/forward.cu)。

WonderPlay 論文本身用物理模擬生成的像素 flow **控制影片生成**，並非用同一
個 flow loss 反推 `E,ν`。我們借用其「投影運動 → Gaussian rasterization →
像素 flow」的表示，接上自己的材質反推 objective；不可聲稱 WonderPlay
論文已驗證本研究的材質辨識方法。
[原論文（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_WonderPlay_Dynamic_3D_Scene_Generation_from_a_Single_Image_and_ICCV_2025_paper.pdf)

## 對照表

| 項目 | 點投影 loss | WonderPlay 式像素 loss |
|---|---|---|
| 比較單位 | 入選 Gaussian 的投影中心 | 觀測前景中的像素 |
| 預測 flow | 同一粒子中心的前後 2D 位移 | 位移經 Gaussian footprint、opacity、深度順序合成 |
| Opacity | `>0.02` 選點門檻 | 每個像素的 alpha 混合權重；數值固定 |
| 可見性 | 初始軌跡的近似 z-buffer，集合固定 | 像素混合隨幾何變化；比較用的觀測前景 mask 固定 |
| Gaussian covariance | 不使用 | 決定像素 footprint，沿用模擬輸出 |
| 損失數值 | 點平均 MSE | 像素平均 MSE；**不可直接和左欄比大小** |
| 物理梯度 | MPM AD → 點位移 | MPM AD → 位移與 covariance → rasterizer backward |

## 已完成的 Letter 實驗

兩個 projected 方法從相同 Letter 初始化與訓練設定開始，各取第 50 次
更新的參數，用同一套完整時域 renderer、RAFT 與 16 幀做獨立評估。
Original 列為歷史參考，但它當年訓練的 MPM scale 不同（見下方限制）。

| 方法（各取第 50 次更新） | 最終 E | 最終 ν | FG EPE ↓ | Whole EPE ↓ | PSNR ↑ |
|---|---:|---:|---:|---:|---:|
| Original | 0.00015000 | 0.489779 | 0.8091 | **0.5875** | 30.8076 |
| 點投影 loss | 0.00010813 | 0.499861 | 0.8014 | 0.6611 | 30.8210 |
| 像素 flow loss | 0.00010684 | 0.499860 | **0.7992** | 0.6518 | **30.8388** |

像素版相對點版，FG EPE 低約 0.0022 px、Whole EPE 低約 0.0093 px、
PSNR 高約 0.0178 dB。兩者差異很小；這不是「明顯優越」的證據，也不是
材質 E/ν 更接近 GT 的證據。[同 50 次指標圖](../outputs/letter_three_methods_equal50/independent_metrics_equal50.png)、
[影格並排圖](../outputs/letter_three_methods_equal50/frame_08_gt_three_methods_equal50.png)。

像素版自己的訓練 loss 在 50 次更新中由約 3.1132 降至 3.0567。
圖上接近直線，是因為每步下降量緩慢地從約 0.00123 降至 0.000995；
用線性擬合的 `R²≈0.9992`，但非每步完全等量。固定學習率的 Adam 加上
這段局部平滑參數路徑，足以形成這種外觀。新舊 loss 的數值刻度不同，
不能用 3.06 與 0.69 判定方法好壞；應比較同套獨立評估指標。
[訓練 loss 與逐步變化圖](../outputs/letter_three_methods_equal50/loss_and_update_changes.png)。

像素版在 Letter 初始點和第 50 次 checkpoint 的 E/ν 梯度都做過 AD 對
多尺度中心有限差分的檢查並通過。這支持兩個已測點附近的實作梯度，
**不保證整個參數空間、其他材質或所有遮擋切換都平滑**。
[初始梯度檢查](../outputs/letter_wonderplay_flow_check_ds1.0_ep1/gradient_check.json)、
[第 50 次梯度檢查](../outputs/letter_wonderplay_flow_epoch2_to_50_ds1.0_ep49/post_gradient_check.json)。

## 比較的限制與後續實驗

1. 這次兩邊都模擬約 102,473 顆粒子，但 loss 取樣不同：像素版設定上限
   1,000，實際抽到 **995 顆**；點版設定上限 20,000，實際抽到 **17,079 顆**。
   這是為了先在 2080 Ti 上驗證新 rasterizer 與 MPM AD 是否能串接，卻讓
   50 次比較同時改了 loss **及**取樣密度。要檢驗「只改演算法」，應固定
   同一份粒子 ID（例如 995 顆）重跑兩個 loss、同樣從 epoch 0 訓練
   50 次，再以同一套獨立 renderer 評估。僅在報表中挑 995 個舊點或
   重新畫舊曲線，無法補救已經不同的訓練梯度。
2. Original 歷史訓練的 `scale_origin=1.5407542`，兩種 projected 訓練則為
   `0.7703771`；雖然最終獨立評估採相同 config，Original 不可視為嚴格
   同一物理初始化下的 loss 消融。
3. 點 loss 的選點與像素 loss 的前景 mask 都是固定規則；像素版的 flow
   仍以固定觀測 RAFT 為目標，不等於對最終 RGB 影片直接最小化 PSNR/EPE。
4. 單一 scene、單一起點、單一相機及 50 次更新不能證明材質唯一可辨識。
   若要研究 ill-posedness，需跨起點／參數掃描並逐幀比較不同參數的影片。
5. 舊 WonderPlay 200 次更新工作已在第 140 次停止；新的 Letter 像素版
   採 20,000 上限從 epoch 0 重跑。Letter 點版不再重跑；先前 995 對
   17,079 顆的 50 次比較仍不能視為純 loss 消融。尚未完成的 200 次
   PSNR/EPE 不應當成結果。

## 建議給教授的結論句

「我們先前的 loss 直接比較物理 Gaussian 中心的投影位移與觀測 optical
flow，opacity 只用於可見點篩選。新方法參考 WonderPlay，把 Gaussian 的
投影位移按其 footprint、opacity 和遮擋次序 rasterize 成像素 flow，再與
觀測 flow 比較。Letter 的 50 次更新結果顯示新方法可反向傳播，獨立 FG
EPE 與 PSNR 略有改善，但差距很小；由於採樣點數不同，還不能把改善全
歸因於像素合成，也不能宣稱材質參數已被唯一辨識。」
