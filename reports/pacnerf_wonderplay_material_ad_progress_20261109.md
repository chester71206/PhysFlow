# PAC-NeRF WonderPlay 材質 reverse AD：工作紀錄

首次記錄：2026-11-09 14:19；後續更新至 18:19（Asia/Taipei）。本文記錄本機工作區的實作與測試狀態；長時間訓練的 epoch 數會繼續變動。

## 目標與名詞

目標是讓 PAC-NeRF 九場景使用 WonderPlay 式的 opacity／深度合成像素 flow loss，並透過 Warp MPM 的**反向傳播**更新物理參數。有限差分可以作為梯度檢查，不能冒充 reverse AD 訓練。

- **原本的 projected-flow loss**：比較 Gaussian 中心點的投影位移與固定 RAFT flow。
- **WonderPlay 式 loss**：先把 Gaussian 位移、covariance、opacity rasterize 成像素 flow，再與同一固定 RAFT flow 比較。
- 兩者共用 MPM 粒子模擬；`--max-points 20000` 是 flow loss 的 Gaussian 取樣上限，並非 MPM 只模擬 20,000 顆。

PAC-NeRF [論文](https://xuan-li.github.io/pdf/publications/li2023pacnerf.pdf)的可微 MPM 涵蓋九場景使用的五類材質。但本專案另有一套 Warp／Gaussian 流程，不能因論文的 Taichi 實作可微，就直接宣稱這套 Warp 時域反傳已驗證所有材質。

## 執行順序與結果

1. **更正 Letter 的取樣比較。**舊點投影 loss 的上限為 20,000，Letter 實際取 17,079 顆；舊 WonderPlay 50 次實驗上限為 1,000，實際取 995 顆。先前將兩者的 50 次結果視為純 loss 比較不公平。曾啟動相同 1,000 上限的點投影重跑，後依要求停止；再啟動 20,000 上限的 Letter 點投影重跑，也依要求停止。沒有刪除既有 checkpoint。
2. **Letter WonderPlay 20,000。**GPU 6 從 epoch 0 開始，目標 200 次更新。初始 E、ν 的 AD／有限差分檢查已通過；記錄時間點完成 **6/200** 次，程序仍在執行。[初始檢查](../outputs/letter_fair_wonderplay20000_200_ds1.0_ep200/gradient_check.json)、[訓練紀錄](../outputs/letter_fair_wonderplay20000_200_ds1.0_ep200/training.jsonl)。
3. **其餘八場景的第一版排程。**GPU 8 依序規劃 Droplet、Torus、Bird、Playdoh、Cat、Cream、Toothpaste、Trophy，每場 200 次、WonderPlay loss、取樣上限 20,000。[排程快照](../outputs/pacnerf_wonderplay_other8_200_20261109/schedule.json)。當時的 launcher 對前 3 場選 reverse AD，對後 5 場選有限差分；使用者要求全部 reverse AD 後，不再執行後 5 場。
4. **Droplet 在第 0 次更新前停止。**第一場 Droplet 的 WonderPlay 初始梯度檢查失敗，launcher 依原有 gate 自動停止，後續場景均未啟動。E、ν 的 AD 和有限差分方向相同，但 E 三個相對誤差約 9.8%、7.5%、8.4%，ν 約 11.2%、7.8%、7.2%；不符合既定的「全部 ≤10%、中位數 ≤5%」。[梯度紀錄](../outputs/pacnerf_wonderplay_other8_200_20261109/droplet/train_ds1.0_ep200/gradient_check.json)、[排程狀態](../outputs/pacnerf_wonderplay_other8_200_20261109/manifest.json)。**沒有 optimizer 更新。**這個差異尚不能單憑一組 FD 步長判定根因。
5. **盤點反傳缺口。**正式 [projected-flow AD 入口](../scripts/train_projected_flow_gradient.py)目前只放行 elastic 與 strain-rate Newtonian；凍結的時域 rollout 也沒有完整涵蓋 sand、non-Newtonian 等材質，以及 plastic viscosity、friction angle 的參數反傳。原 PAC-NeRF [Taichi 反傳](../PAC-NeRF/lib/engine/dynamic_observer.py)與目前 Warp 實作是不同的模擬路徑。工作區中五場景的 WonderPlay 有限差分支援是本機未提交修改，且不是所要求的 reverse AD。

## 新增的獨立 Warp AD 原型

為避免修改 `release/frozen_mpm.json` 內的檔案、使正在跑的 Letter 工作中途失敗，我新增獨立的 [material_temporal_rollout.py](../mpm_solver_warp/material_temporal_rollout.py)。它沿用目前 Warp 的 MPM 材質公式與 P2G／G2P，為 plasticine、non-Newtonian、sand 建立不可原地覆寫的 stress 步驟與分段時域反傳，並將 E、ν、yield stress、plastic viscosity、friction angle 接入梯度輸出。這是**實驗性路徑**，未接入正式八場景 launcher。

[合成檢查腳本](../scripts/check_material_temporal_ad.py)使用 GPU 8、64 顆粒子、40 個 MPM step，先與原 Warp solver 比較前向位置，再對實際相關材質參數做 AD／中心有限差分比較。最終使用具不同形變的粒子，以免過於均勻的測試讓參數訊號淹沒在 float32 雜訊中。

| 材質 | 前向最大位置差 | AD／FD 相對誤差 |
| --- | ---: | --- |
| Plasticine | 約 `1.2e-7` | E `0.24%`、ν `0.16%`、yield stress `3.20%` |
| Non-Newtonian | 約 `1.2e-7` | E `0.02%`、ν `0.48%`、yield stress `8.59%`、plastic viscosity `0.28%` |
| Sand（壓縮形變，使 friction angle 有訊號） | 約 `2.4e-7` | friction angle `3.62%` |

這些是**小型合成測試**，不足以證明完整場景的長時域梯度正確。

## Playdoh 真實場景檢查

[場景檢查與訓練入口](../scripts/check_pacnerf_material_flow_ad.py)預設只執行 WonderPlay loss 的 AD／FD gate；只有完整場景 gate 通過且明確指定 `--epochs`，才可能做 reverse AD／Adam 更新，並每次寫入 checkpoint。目前**沒有用此入口做任何更新**。

| 時域 | 結果 |
| --- | --- |
| 第 1 組 flow pair、208 個 MPM step | E、ν、yield stress 對 loss 的影響小到落入 float32 有限差分雜訊，不能作有效梯度驗證。[紀錄](../outputs/material_ad_playdoh_short_20261109_ds1.0_ep1/material_ad_check.json) |
| 前 5 組 pair | E 仍幾乎沒有可判別的 loss 訊號，因此停止這個短檢查。[部分紀錄](../outputs/material_ad_playdoh_5pairs_20261109_ds1.0_ep1/material_ad_check.json) |
| 完整 13 組 pair，初次檢查 | E 的 AD `−0.6243966`、以 5% 參數跨度算出的中心 FD `−0.5229712`，相對差 **16.24%**。[初次紀錄](../outputs/material_ad_playdoh_full_20261109_ds1.0_ep1/material_ad_check.json) |
| 完整 13 組 pair，局部步長重測 | E 的 0.5% FD `−0.6234646`，與 AD 差 **0.15%**；ν 差 **0.11%**；yield stress 差 **0.68%**。三項皆同方向、皆小於 10% 門檻，**Playdoh 通過**。[完整紀錄](../outputs/material_ad_playdoh_gate_local_20261109_ds1.0_ep1/material_ad_check.json) |

5% 的差分估計是跨度較大的割線，可能跨越材質或碰撞分支，不能代表局部導數。0.5% 與 1% E 擾動的 AD／FD 誤差分別為 0.15% 與 3.45%；5% 時增至 16.24%。檢查腳本改用局部步長，保留原來的 10% 誤差門檻及 20,000 Gaussian 上限。Playdoh 沒有 optimizer update。

## 目前邊界與下一步

- GPU 8 已完成 Playdoh、Cat、Cream 的完整梯度檢查。Cat 的 E、ν、yield stress 相對誤差分別為 **1.30%、0.81%、1.52%**；[Cat 紀錄](../outputs/material_ad_cat_gate_local_20261109_ds1.0_ep1/material_ad_check.json)。Cream 的 E、ν、yield stress、plastic viscosity 相對誤差分別為 **0.49%、2.37%、0.22%、2.53%**；[Cream 紀錄](../outputs/material_ad_cream_gate_local_20261109_ds1.0_ep1/material_ad_check.json)。所有項目同方向且小於 10%。
- Toothpaste 的完整序列初測中，ν 與 plastic viscosity 分別差 5.15%、2.29%，但 E 與 yield stress 分別差 20.90%、25.05%；[初測紀錄](../outputs/material_ad_toothpaste_gate_local_20261109_ds1.0_ep1/material_ad_check.json)。E 的 0.1% 與 1% 擾動 FD 都約 −51，無法用單純縮小步長解決。固定影像 loss 的基準梯度，只測 MPM 輸出時，E 誤差 3.5–3.9%、yield stress 誤差 7.4–7.6%；[E 拆分](../outputs/material_ad_toothpaste_probe_e_linear_20261109_ds1.0_ep1/fd_probe.json)、[yield 拆分](../outputs/material_ad_toothpaste_probe_y_linear_20261109_ds1.0_ep1/fd_probe.json)。沿同一狀態方向檢查影像 objective，隨取樣從約 994 顆增至 15,641 顆，AD／FD 誤差從 4.7% 增至 19.1%；[取樣診斷](../outputs/material_ad_toothpaste_probe_counts_20261109_ds1.0_ep1/fd_probe.json)。曾試修 rasterizer 的 alpha 0.99 clamp 反傳，但沒有改善，已撤回並重編譯原版。**Toothpaste 完整 loss AD／FD 幅度仍未通過**，因此新增獨立的分段門檻：完整 FD 必須同方向、MPM 段必須 ≤10%。此門檻通過後，以 reverse AD 做 1 次更新，重新 forward 的 loss 從 `0.39729187` 降到 `0.39726624`；[分段與訓練紀錄](../outputs/material_ad_toothpaste_stagewise_train1_20261109_ds1.0_ep1/material_ad_check.json)、[checkpoint](../outputs/material_ad_toothpaste_stagewise_train1_20261109_ds1.0_ep1/ckpt/optimizer.pt)。這證明一次實際下降，不等於完整 renderer 梯度的 FD 驗證。
- Trophy 的 sand／friction angle 已通過，AD／FD 相對誤差 **0.47%**；[Trophy 紀錄](../outputs/material_ad_trophy_gate_local_20261109_ds1.0_ep1/material_ad_check.json)。Droplet 的完整 loss 多尺度 FD 仍未通過原門檻；固定影像梯度後，E、ν 的 MPM 段誤差分別只有 **0.016%、0.068%**，且完整 loss 的 FD 均同方向。[Droplet 分段紀錄](../outputs/droplet_wonderplay_stagewise_train1_20261109_ds1.0_ep1/gradient_check.json)。因此做 1 次 reverse AD 更新，重新 forward 的 loss 從 `14.148385` 降到 `14.137998`，並寫出 [checkpoint](../outputs/droplet_wonderplay_stagewise_train1_20261109_ds1.0_ep1/ckpt/optimizer.pt)。隨後停止 `train_flow.sh` 自動啟動的重複完整 checkpoint FD；該額外檢查沒有完成。Torus 的 E／ν 六個完整 loss AD／FD 比較均同方向，誤差最大 **5.48%**，整體 gate 通過；[Torus 紀錄](../outputs/torus_wonderplay_ad_gate_20261109_ds1.0_ep1/gradient_check.json)。Bird 正在驗證。
- [九場景排程入口](../scripts/run_pacnerf_projected.py)已改為 WonderPlay 模式下九場景全選 reverse AD，取樣上限預設 20,000；`--plan-only` 顯示九場景資產齊全且均選 `reverse_AD`。非彈性材質走新增 rollout 與場景檢查入口；這仍是排程規劃驗證，不代表每場已完成 200 次訓練。
- **設定一致性更正：**原始 `config/pacnerf/*_config.json` 未指定 `guidance_background` 時預設為 `source`，正式九場景排程會設為 `white_masked`。上文 Playdoh、Cat、Cream、Toothpaste、Trophy、Torus 的首次檢查用了 `source`，因此只證明該目標 flow 下的結果；不能直接當作正式排程的完整 loss 驗證。已產生與排程相同的 [white_masked 設定快照](../outputs/pacnerf_wonderplay_ad_configs_20261109)，正在依此補驗。Droplet 的設定快照 SHA 與先前正式排程完全相同；它的分段檢查和 1-step 結果適用於正式設定。Bird 的 `source` 模式檢查在 FD 前停止，改跑正式設定。
- Bird 的正式 `white_masked` 設定完整 gate 已通過。[Bird 紀錄](../outputs/bird_wonderplay_white_ad_gate_20261109_ds1.0_ep1/gradient_check.json)；ν 的極小步長只造成數個 float32 loss 刻度的差，因此 gate 自動使用較大的既定差分步長。Torus 的正式設定正在補驗。
- 對每場景核對完整前向與所有訓練參數的 AD／FD。只有通過後才能把五個非彈性場景改為 reverse AD 排程並啟動 200 次訓練。
- 新增程式與本報告目前是**本機未提交、未推送**的工作；不能視為另一台伺服器已具備此功能。
