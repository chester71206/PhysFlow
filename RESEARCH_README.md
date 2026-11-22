# PhysFlow / PAC-NeRF 研究交接與輸出索引

更新：2026-11-22（台北）。**新對話請先讀本檔，再讀相關實驗的 metadata／report；不要只憑資料夾名稱推測設定。**

這個 workspace 的研究主題是：用 Warp MPM 模擬 PAC-NeRF 場景，透過 WonderPlay rasterized physical pixel flow 對固定的 observed RAFT flow 計算訓練 loss，並用獨立 evaluator 測量 FG EPE、Whole EPE、PSNR，分析材料參數能否被辨識。`outputs/` 是本機產物，已被 `.gitignore` 排除；切換對話仍可讀，但單純 `git pull` 到另一台機器**不會帶過去**。完整 104 個頂層輸出目錄及大小見 [outputs/CATALOG.md](outputs/CATALOG.md)。

## 先看這幾個結果

| 想回答的問題 | 首選檔案 | 注意 |
| --- | --- | --- |
| 九場景 Original 與 WonderPlay 訓練及評估 | [正式九場景報告](reports/pacnerf_wonderplay_9scene_final_20261111.md)、[50-update 比較](outputs/pacnerf_original50_vs_wonderplay_20261111/comparison.md) | Letter 的既有最終 WonderPlay 是 74 updates；其餘八場景是 50。比較不等於只換 loss 的受控 ablation。 |
| 合成 Press／Bending／Stretching 的 WonderPlay 可辨識性 | [三運動總報告](PAC-NeRF/data/synthetic_motion_study/REPORT.md)、[共同 ν sensitivity 圖](PAC-NeRF/data/synthetic_motion_study/nu_sensitivity_by_view.png)、[Stretching multi-start](PAC-NeRF/data/stretching_bar/multistart/multistart_paths.png) | Exact GT Flow 在 matched FEM forward 約零；RAFT 可讓連續最低點偏離 GT。Bending 有兩個高 E／高 ν 發散點；PAC-NeRF MPM 施力條件仍未對齊。 |
| MPM Press Block 的 GT minimum 與兩起點訓練 | [MPM 專項報告](PAC-NeRF/data/press_block_mpm/REPORT.md)、[FD Adam 訓練報告](PAC-NeRF/data/press_block_mpm/training/REPORT.md)、[參數路徑](PAC-NeRF/data/press_block_mpm/training/training_paths.png)、[三視角影片](PAC-NeRF/data/press_block_mpm/preview_3view.mp4) | matched Warp MPM 的 exact flow 在 GT 得零且為抽樣最低；RGB RAFT 的 ν 最低點偏至 .305（GT .32）。較近起點的 Exact FD Adam 80 步恢復至 19.815 kPa／.31524，遠起點失敗；RAFT 近起點走向偏移 ν。此為 FD Adam，非移動平板邊界的 reverse AD 驗證。 |
| **Press Block 物理稽核與修正版** | [舊場景物理失敗原因](PAC-NeRF/data/press_block_mpm_native/physics_audit_20261120/REPORT.md)、[新場景 A/B/C 報告](PAC-NeRF/data/press_block_mpm_physical_v3_penalty_10000_ground_10000_cap_2mm_plate_90mm_dt_1920/REPORT.md)、[新三視角影片](PAC-NeRF/data/press_block_mpm_physical_v3_penalty_10000_ground_10000_cap_2mm_plate_90mm_dt_1920/preview_3view.mp4) | 原 native 場景雖然 forward/AD 重播通過，物理上仍有初始沉降、17.94 mm 平板穿透及 1000 粒子長期漂移；其影片與訓練結論不能當物理 GT 驗收。修正版 8000 粒子、單側接觸已有 A/B/C 實測，但新接觸核尚未接入 PhysFlow reverse AD；暫停其 RAFT/loss/訓練。 |
| **2026-11-22 遠距起點與新場景** | [遠距 Gaussian WonderPlay 報告](outputs/synthetic_wonderplay_farstart_20261122/REPORT.md)、[獨立 Blender flow 偏差](outputs/synthetic_wonderplay_blender_bias_20261122/REPORT.md)、[合成影片批次](outputs/synthetic_physics_batch_20261121/README.md) | 棒球兩組、足球高 E 組在 120 步恢復 GT；足球低 E／高 ν 組曾撞 ν 上界，120 步仍偏離，續跑 270 步才達預設門檻。這只驗證 Gaussian 自洽目標。Blender 可見表面 flow 在 GT 有非零 loss／梯度，從 GT 訓練也會降低 loss 並偏離 GT。新橡皮鴨低落差場景通過自動 full QA，但自然度仍待人工複核。 |
| **E 相差 10／100 倍（待 GPU）** | [實驗設定與即時產物](outputs/synthetic_wonderplay_order_magnitudes_20261122/README.md) | 前述 0.5／2 倍不足以回答此問題。新實驗測 E/GT=10、0.1、100、0.01，ν 從 GT 起跑；改用 log E 最佳化與寬邊界。啟動時所有 GPU 均被其他工作佔用，背景執行器只會在裝置空閒後開始。未產生 probe／final 前沒有收斂結論。 |
| GT／Original／WonderPlay 訓練後影片 | [三欄影片清單](outputs/pacnerf_original50_vs_wonderplay_20261111/final_three_way_videos/manifest.json)、[最後影格總覽](outputs/pacnerf_original50_vs_wonderplay_20261111/final_three_way_videos/overview_last_frames.png) | 左到右為 WonderPlay、Original、觀測 GT；Letter 用 WonderPlay u74。 |
| 研究結論與限制 | [跨實驗綜合](reports/pacnerf_material_landscape_synthesis_20261119.md) | 區分 proxy sensitivity、觀測敏感度、局部 barrier、forward mismatch；不要預設 ill-posed。 |
| Bird 的訓練軌跡與 2D loss | [Bird 四格圖](outputs/bird_training_landscape_diagnostic_20261119/bird_loss_landscape_vs_training.png)、[同 Warp 合成目標診斷](outputs/bird_synthetic_flow_diagnostic_20261119/REPORT.md) | 合成目標在 GT 取得零 loss 只驗證同一 forward 的自洽性，不證明外部資料對齊。 |
| Letter／Droplet／Torus 的訓練軌跡與 2D loss | [診斷圖目錄](outputs/pacnerf_training_landscape_diagnostic_20261119/)、[原 20×20 grid](outputs/pacnerf_2d_night_20261116/summary.json) | Letter／Droplet 後續補掃另存於下述進行中目錄；原圖可能尚未包含補點。 |
| Trophy 單參數摩擦角 | [訓練＋掃描圖](outputs/pacnerf_training_landscape_diagnostic_20261119/trophy_friction_loss_landscape_vs_training.png)、[61 點 CSV](outputs/trophy_friction_0_60_20261115/results.csv) | 40° 是 PAC-NeRF 表中的 GT，並非目前 Warp＋WonderPlay loss 的保證最小點。 |

## 原始資料與方法地圖

- **新受控 press 場景（2026-11-20）：**[場景與重跑說明](PAC-NeRF/data/press_block/README.md)、[三視角影片](PAC-NeRF/data/press_block/preview_3view.mp4)、[WonderPlay 診斷](PAC-NeRF/data/press_block/diagnostic/REPORT.md)。3D 四面體 FEM 的局部下壓／回彈資料；標準 mask 的三視角 5×5 網格中，同 forward 精確 flow 與 RGB RAFT target 都以 GT 格點最低，ν 對 flow 較不敏感；單視角物體 mask 有微小 GT 排名翻轉。PAC-NeRF 現有 BC 不能表示移動平板，且訓練環境缺 `torch_scatter`；不能把資料集 FEM 與 PAC-NeRF forward 當作已對齊。`PAC-NeRF/data/` 被 git 忽略，搬機須另複製。

- **九場景訓練原始紀錄與 checkpoint：**[outputs/pacnerf_wonderplay_50_20261110/](outputs/pacnerf_wonderplay_50_20261110/)；原方法／WonderPlay 的獨立評估與比較圖在 [outputs/pacnerf_original50_vs_wonderplay_20261111/](outputs/pacnerf_original50_vs_wonderplay_20261111/)。
- **一維 sensitivity：**[結果與定義](reports/pacnerf_standardized_sensitivity_results_20261117.md)、[逐點資料](outputs/pacnerf_standardized_sensitivity_20261117/)。
- **二維 20×20 forward grid：**[四場景結果](outputs/pacnerf_2d_night_20261116/results.csv)（Bird／Torus 掃 E×ν；Letter／Droplet 掃物理 μ×κ）。`sampled minimum` 只是在有效採樣格點中最低，不是連續空間全域最小值。
- **Trophy 角度：**[修正後 0°–60° 掃描](outputs/trophy_friction_0_60_20261115/four_metrics.png)，固定其他材料參數，每 1° 一點，61/61 有效。**不要使用**舊的 `outputs/pacnerf_material_landscape_20261115/trophy/` 作為角度結論；那版曾有五參數傳遞位置錯誤。[原因報告](reports/trophy_friction_sweep_root_cause_20261115.md)。
- **Droplet E 的獨立幾何對照：**[WonderPlay flow 與 GT PLY 3D Chamfer](outputs/droplet_E_dual_loss_20261115/dense/wp_vs_gt_ply_chamfer.png)、[PLY 可行性報告](reports/droplet_ply_gt_feasibility_20261115.md)。Chamfer 不是 GT optical flow。
- **四場景 joint Sobol：**[正式分析](outputs/pacnerf_joint_sampling_20261118/analysis/FORMAL_ANALYSIS.md)、[樣本 CSV](outputs/pacnerf_joint_sampling_20261118/analysis/samples.csv)。Playdoh／Cat／Cream／Toothpaste 同時改三或四個 active parameters；完成 12 小時排程時有 4,629 有效、203 失敗、168 尚未嘗試，不能把產生的 5,000 個點都當作有效實驗。
- **Letter 專項：**[舊全域材料掃描分析](outputs/letter_material_landscape/analysis.md)、[E×ν forward](outputs/letter_en_landscape_full_ds1.0_ep1/)。`letter_material_landscape` 約 4.2 GiB，是目前最大目錄之一，含大量逐點 evaluator 輸出；**未清理**。
- **共用評估定義：**[scripts/evaluate_universal_reference.py](scripts/evaluate_universal_reference.py)。FG EPE 使用 GT RGBA alpha `==255` 並 nearest resize 到 RAFT 尺寸；Whole EPE 對同一批 RAFT flow 的整張影像平均；PSNR 比較 rendered RGB 與目標 RGB。

## 目前仍在跑：Letter／Droplet 無 ν scan cutoff 補點

2026-11-20 約 02:00（台北）的快照：[輸出](outputs/letter_droplet_grid_completion_20261119/)、[log](outputs/letter_droplet_grid_completion_20261119/full.log)，GPU 9 的 launcher PID `18670`、worker PID `25506`。計畫 219 個新增點；當時 195 點已有結果，其中 189 有效、6 失敗。包含初始→完整 GT 路徑和原 2D grid 曾被診斷 cutoff 排除的格點；每點獨立 forward＋正式 evaluator，失敗會記錄。**PID／數量只是一個時間快照，新對話必須重新查。** 不要移動、刪除、覆蓋這個目錄，也不要干擾 GPU 9 工作。

即時查看：

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'run_letter_droplet_grid_completion|complete_letter_droplet_grid|pacnerf_material_landscape'
tail -n 3 outputs/letter_droplet_grid_completion_20261119/full.log
python - <<'PY'
import json
from pathlib import Path
files = list(Path('outputs/letter_droplet_grid_completion_20261119/points').glob('*/result.json'))
rows = [json.loads(p.read_text()) for p in files]
print('attempted', len(rows), 'finite', sum(bool(r.get('finite')) for r in rows), 'failed', sum(not r.get('finite') for r in rows))
PY
```

Launcher 在所有點嘗試完後會自動重畫圖與 summary；`COMPLETE` 標記表示整個計畫已嘗試，**不表示每點 finite**。可手動從保存結果重畫：

```bash
python scripts/report_letter_droplet_completion.py --root outputs/letter_droplet_grid_completion_20261119
python scripts/plot_pacnerf_training_landscapes.py --scene letter --scene droplet \
  --supplement-root outputs/letter_droplet_grid_completion_20261119 \
  --output outputs/pacnerf_training_landscape_completed_20261119
```

## 解讀與待解問題

1. PAC-NeRF 論文的材料 GT 是原資料流程的真值；目前 Warp MPM 的固定幾何、其他材料參數、初速度、邊界、相機、渲染與 observed RAFT target 不保證完全等價。GT 點不在現有 loss／PSNR 最佳區域時，先核對 forward 條件，勿直接宣稱 optimizer 壞掉或資料無法辨識。
2. WonderPlay training loss、FG EPE、Whole EPE、PSNR 是不同量；訓練 loss 下降不保證 EPE／PSNR 或材料 RE 同步改善。不同 scene 的 raw loss 值不可直接比較。
3. Bird 同 Warp 合成目標實驗在 GT 取得零 loss；仍需檢驗外部觀測資料的 geometry／motion／render 對齊。Trophy 的 40° 在目前 61 點 sweep 上不是 WP 或 PSNR 的最佳；逐幀 PSNR 顯示差異主要在後段影格，[圖](outputs/pacnerf_training_landscape_diagnostic_20261119/trophy_psnr_15_vs_GT40_by_frame.png)。
4. Playdoh／Toothpaste 完整 renderer 的 AD／FD 檢查仍有限制；Cream／Toothpaste viscosity 的物理 GT 單位換算尚未認證。細節見[九場景報告](reports/pacnerf_wonderplay_9scene_final_20261111.md)。

## 輸出整理規則

`outputs/` 中有主要結果、smoke、失敗重試、舊設定與原始逐點資料。這次只建立索引，**沒有搬移或刪除**，因為報告、續跑腳本和 checkpoint 路徑可能引用原位置。需要整理磁碟時，先按 [完整目錄清單](outputs/CATALOG.md) 找出候選目錄，再核對使用中的程序、原始結果和引用關係；尤其不要把 `smoke` 名稱直接當成安全刪除依據。

更新完整目錄與大小：

```bash
python scripts/refresh_output_catalog.py
```

## 給下一個對話的最短交接指令

> 請先讀 `RESEARCH_README.md`、`outputs/CATALOG.md`，再讀與我問題相關的 `reports/` 和各實驗 metadata。先用 `ps`／log 核對目前是否仍有 GPU 工作；README 的 PID 和完成數只是快照。保留現有 checkpoints 與逐點輸出，不要把舊失敗實驗混進正式結果。回答時區分已觀察、推論、尚未驗證，並附上具體輸出檔案。
