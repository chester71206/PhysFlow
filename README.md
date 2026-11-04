# PhysFlow: projected-flow material optimization

此分支保存最終 optical-flow **reverse AD** 流程，保留
`simulation_memory_safe.py` 作為原方法比較。中間 simulation 實驗、搜尋式
optimizer 與訓練輸出不包含在本次發布中；Letter 場景模型資料已納入。

```
E, nu -> checkpointed MPM -> projected particle motion -> flow loss -> Adam
                                                         ^
                               fixed observed RAFT flow --|
```

梯度穿過完整 MPM rollout 及投影；RAFT 只從觀測影像抽取固定 target。
這是替代 observation objective，沒有修復原本 renderer + RAFT backward。
FD 僅檢查 AD，**不做參數搜尋或更新**。production trainer 支援
**elastic、strain-rate Newtonian**；其餘材料明確回報 unsupported。
Letter 曾完成 50 epochs 並通過最終梯度檢查，但不代表其他場景都已驗證，
也不代表材質辨識誤差已足夠低。每次新實驗都重新檢查梯度。

## 1. Clone 與安裝

需要 Linux x86_64、NVIDIA GPU、CUDA **11.8 toolkit**（含 `nvcc`）及 C++ compiler。
建議 24 GB GPU；實際記憶體隨場景而變。使用 Python 3.9、Torch 2.2.0+cu118、
torchvision 0.17.0、Warp 0.10.1、Taichi 1.5.0。

```bash
git clone --branch projected-flow-final https://github.com/chester71206/PhysFlow.git
cd PhysFlow
conda env create -f environment.yml
conda activate physflow
export CUDA_HOME=/usr/local/cuda-11.8  # 改成新機器的 CUDA 11.8 toolkit 路徑
bash scripts/install_flow.sh
python scripts/smoke_flow_cpu.py
```

安裝腳本固定 PyTorch3D commit 與 Python dependencies，從 repo 原始碼重新編譯
simple-knn/rasterizer，不使用本機 `.so`。rasterizer 原始碼對齊原機器實際載入的
PGSR 版本，另以 `release/rasterizer_source.json` 保存 SHA256；安裝檢查會比對 wrapper。
首次抽取 flow 時 torchvision 會下載
RAFT weights；離線機器可搬移原機器 `~/.cache/torch/hub/checkpoints/`。

如果以 Git bundle 搬移程式，clone 指令也可以改成：

```bash
git clone --branch projected-flow-final /path/to/physflow_projected_flow_final_20261103.bundle PhysFlow
```

## 2. 場景資料

`projected-flow-final` 已包含 `model/pacnerf/letter/` 的相機、影像、RGBA
masks、Gaussian checkpoint 和幾何資料。跑 Letter 材質 sweep 前先檢查：

```bash
python scripts/check_flow_data.py --scene letter
```

其他 PAC-NeRF 場景仍需另行搬移。請在有原資料的機器匯出：

```bash
python scripts/export_flow_assets.py --source /path/to/existing/PhysFlow \
  --output /path/to/pacnerf_flow_assets.tar.gz
# 例如只搬 Droplet 可加 --scenes droplet
```

複製 archive、`.sha256`、`.manifest.json` 至新機器，在 archive 目錄檢查後還原：

```bash
sha256sum -c pacnerf_flow_assets.tar.gz.sha256
cd /path/to/new/PhysFlow
tar -xzf /path/to/pacnerf_flow_assets.tar.gz
python scripts/check_flow_data.py --scene letter
```

```
model/pacnerf/letter/
  cameras.json
  images_generated/r_42_0.png ...
  masks/a_42_0.png ...                 # supplied RGBA observations
  point_cloud/iteration_*/point_cloud.ply
  pcd/0.ply ...                       # 原方法 geometry loss 需要
data/pacnerf/timing/letter.json       # Git 內的真實時間戳
data/pacnerf/groundtruth.json         # Git 內的材質 GT，僅用於報表
```

亦可取得 [PhysFlow 預處理 PAC-NeRF 場景](https://drive.google.com/drive/folders/1h79csXUK4clx5Udmsje-9vTwuQ3rBlel?usp=sharing)，
但非 Letter 實驗還需要相對應的 supplied RGBA `masks/`；直接匯出原機器資料能保持觀測一致。
Letter sweep 無須搬移整個 `PAC_NeRF_data/`、`PAC_NERF_original_data/` 或 `outputs/`，
必要時間戳、材質 GT 和起點參數已整理進此分支。

## 3. 最終訓練入口

先對完整場景檢查梯度，不更新參數：

```bash
GPU=0 bash scripts/train_flow.sh \
  --model-path model/pacnerf/letter \
  --physics-config config/gradient_repair/letter_flow_ad.json \
  --output outputs/letter_check --check-only
```

開始訓練，每次使用新的 output prefix：

```bash
GPU=0 bash scripts/train_flow.sh \
  --model-path model/pacnerf/letter \
  --physics-config config/gradient_repair/letter_flow_ad.json \
  --output outputs/letter_flow --epochs 50 --lr 0.003
```

輸出在 `outputs/letter_flow_ds1.0_ep50/`：

- `gradient_check.json`：E、nu 各三種 h 的 central FD vs AD。
- `training.jsonl`：每次完整 rollout 的 loss、AD gradient 與更新前後參數。
- `ckpt/sim.json`、`ckpt/optimizer.pt`：材質參數、Adam state、固定 visibility。
- `post_gradient_check.json`：最終 checkpoint 再檢查梯度。
- `timing_manifest.json`、`kinematics_manifest.json`：來源時間與場景轉換。

門檻：六組 stencil 全部同號、max relative error ≤10%、median ≤5%。
初始未通過不訓練，最終未通過保留 checkpoint 並回報失敗。
`--resume path/to/optimizer.pt --epochs N` 表示**再更新 N 次**。
模型/config 路徑與資料簽章需一致，既有 checkpoint 不能直接在不同絕對路徑 resume；
新機器可由相同資料/config 重新開始。

## 4. 全 PAC-NeRF 原方法／新方法比較

```bash
GPU=0 nohup bash scripts/run_pacnerf.sh outputs/pacnerf_comparison \
  > pacnerf_comparison.log 2>&1 < /dev/null &
```

九個場景每場 original → projected-flow AD，各 50 epochs。新方法只跑
Letter、Droplet、Torus、Bird，其他五個場景標示 unsupported，原方法仍執行。
訓練與獨立評估在同張 GPU 串行執行。

`results.md`、`results.csv`、`status.json` 約每 15 秒更新 epoch 及可用材質 RE，
每項完成後補上獨立評估的 foreground/whole **EPE**、half MSE、**PSNR**。
兩方法的訓練 loss 與 optimizer 不同；場景初始化、GT 與評估流程相同。
材質 RE 是 `abs(estimated-GT)/abs(GT)`；Markdown 是百分比、CSV 是比例。
Cream/Toothpaste viscosity 轉換未確立，完整 mean RE 留空。

單獨評估 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python scripts/evaluate_universal_reference.py \
  --model-path model/pacnerf/letter \
  --physics-config config/gradient_repair/letter_flow_ad.json \
  --checkpoint-json outputs/letter_flow_ds1.0_ep50/ckpt/sim.json \
  --output outputs/letter_flow_evaluation --dataset pacnerf
```

### 只跑 Letter：原方法與 projected-flow loss，然後畫圖

這個入口在同一張 GPU **串行**執行：原方法 `simulation_memory_safe.py`
50 個更新、projected physical flow vs fixed observed RAFT flow 50 個更新；
各自訓練後用相同的 native-renderer/RAFT 評估最終 checkpoint，最後輸出
loss、材質 RE、逐影格 PSNR/EPE 三張 PNG 與 CSV/報告。

```bash
GPU=8 bash scripts/run_letter_two_methods.sh outputs/letter_two_methods/letter
```

輸出在 `outputs/letter_two_methods/letter/detailed_comparison/`：
`loss_curves.png`、`material_accuracy.png`、`evaluation_accuracy.png`、
`training_curves.csv`、`frame_metrics.csv`、`REPORT.md`。
訓練與獨立評估 log 在每個方法的 `train.log`、`evaluation.log`。
入口拒絕已存在的輸出路徑，避免覆蓋既有實驗。

如兩個方法已訓練及評估完成，可只重畫圖，不佔 GPU：

```bash
python scripts/report_letter_two_methods.py --base outputs/letter_two_methods/letter
```

兩種訓練 loss 定義不同，原方法還每十個更新移動一次監督視窗；
只能各自看收斂趨勢。PSNR/EPE 目前是**最終 checkpoint** 的逐影格／flow-pair
評估，並非每個更新都有影像指標。若原方法和 projected 訓練時的
`scale_origin` 不同，材質 RE 曲線依各自訓練 scale 計算，報告也列共同評估
scale 下的結果。

## Letter 目前參數的 E/nu 局部曲線（建議先跑）

以 `data/pacnerf/letter_material_starts/projected_flow_ad.json` 為唯一中心。
E-only 曲線**固定 nu**、將 E 從 0.75 E0 掃到 1.25 E0，共 41 點，
先測中心和最近的正負擾動。nu-only 曲線固定 E；其餘為局部 7×7
E/nu 網格。所有點用同一套 renderer/RAFT 評估，沒有分 original/projected
評估方法。預設共 101 個不重複點，每次最多新增 10 次，可續跑。

```bash
GPU=8 bash scripts/run_letter_local_E_nu.sh --plan-only
GPU=8 MAX_NEW=10 bash scripts/run_letter_local_E_nu.sh
```

`outputs/letter_local_E_nu/E_only_curves.png` 直接回答「加一點 E 是否改善
PSNR/EPE」；`analysis.md` 列出每個 E 點相對目前 checkpoint 的變化。
要換另一個起點，使用 `--checkpoint path/to/sim.json`，並指定新的
`OUTPUT=...`。`--e-span`、`--e-points` 可改掃描範圍和密度。
已有的七點實驗若在預設位置，會沿用中心點評估。

## Letter 材質參數接近 GT 的另一種診斷實驗

完成上面的安裝與 Letter 場景資料還原後，可從本分支保存的 original 與
projected-flow AD 起點，沿物理單位的 μ、κ 各取七個點直到論文 GT。
所有點使用相同的 config、時間對齊、renderer、RAFT 和影像目標；
這是固定參數路徑的評估，**不執行材質訓練**。

```bash
GPU=0 bash scripts/run_letter_material_to_gt.sh --plan-only
GPU=0 bash scripts/run_letter_material_to_gt.sh
```

`GPU` 選擇可用 GPU；`POINTS`（預設 7）、`SPACING`（`log` 或 `linear`）、
`OUTPUT` 可用環境變數指定。同一指令可在中斷後接續；每完成一點即更新
`outputs/letter_material_to_gt/results.md`、`results.csv`、`status.json`，
完成的點也有 `frame_metrics.csv`。結束後生成 `metric_curves.png` 和 PDF。
首次執行可能下載 RAFT weights。要只讀取已完成結果而不使用 GPU，可加
`--reuse-only`。本分支包含兩個起點的材質 JSON、共同評估的 kinematics
manifest、材質 GT、Letter 場景與 Gaussian checkpoint。

若要另外檢查物理單位 μ/κ 朝 GT 的方向，可用先前的
material landscape 掃描。它同時改變內部 E 和 nu，**不能單獨回答
「增加 E 是否改善 PSNR」**。預設兩條各 25 點的 GT 路徑、original/projected/GT
三個中心的四種正反方向，以及 GT 周圍 5×5 網格；重複參數只評估一次，
合計 153 次獨立 forward 評估。會讀取上一段七點 sweep 的相同結果，
每次最多新增 10 次 GPU 評估，可用同一指令續跑：

```bash
GPU=8 MAX_NEW=10 bash scripts/run_letter_material_landscape.sh --plan-only
GPU=8 MAX_NEW=10 bash scripts/run_letter_material_landscape.sh
# 想自訂其他起點（物理單位 μ、κ）：
GPU=8 MAX_NEW=10 OUTPUT=outputs/letter_landscape_extra \
  bash scripts/run_letter_material_landscape.sh --start low_mu=50,100000
```

輸出 `results.md`/`results.csv`、`path_curves.png`、`direction_curves.png`、
`gt_neighborhood.png`、`analysis.md` 與 `counterexamples.json`。
`analysis.md` 計算已量測相鄰點的 EPE/PSNR 單調性；
`counterexamples.json` 列出材質 RE 增大但 PSNR 增加超過 0.05 dB
或前景 EPE 降低超過 0.01 px 的點。這些都是有限採樣的觀察，
不能證明未採樣區間連續平滑或 optimizer 梯度正確。

## 程式範圍

| 檔案 | 用途 |
| --- | --- |
| `scripts/train_projected_flow_gradient.py` | 最終 production AD 訓練 |
| `flow_scene.py` | 場景初始化；無搜尋或舊訓練 loop |
| `utils/projected_flow_objective.py` | 固定 observation/visibility 的 projected-flow loss |
| `mpm_solver_warp/repaired_temporal_rollout.py` | immutable state、checkpoint recomputation、segmented reverse |
| `mpm_solver_warp/analytic_*_adjoint.py`, `polar_adjoint.py` | 已驗證 VJP |
| `release/frozen_mpm.json` | 十個凍結 physics 檔案的 SHA256 |
| `simulation_memory_safe.py` | 原方法基準，仍有原 temporal gradient 限制 |

`deterministic_p2g_diagnostic.py`、`deterministic_g2p_adjoint.py` 雖沿用早期名稱，
仍是 production adjoint 的必要依賴。其他中間 simulation 沒有加入。
原作者 `simulation.py` 與資料準備工具仍保留，不是此分支建議訓練入口。

本分支基於 [PhysFlow (CVPR 2025)](https://github.com/zhuomanliu/PhysFlow)，
[原始 README](docs/UPSTREAM_README.md)、[論文](https://arxiv.org/abs/2411.14423)。
保留原 repo 及 CUDA extensions 的 LICENSE。
