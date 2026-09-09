# supplemental_pipeline 程式碼說明

這份文件是在 2026-08-30 依照目前工作區的程式碼整理的說明。重點不是重講論文，而是幫你看懂這個 `supplemental_pipeline/` 資料夾實際上在做什麼，以及哪些地方只是工程補接、哪些地方可能真的有問題。

## 先講結論

這個資料夾本質上是「把論文前半段缺的流程，用外部 repo 和幾個 wrapper 串起來」的整合層，不是論文作者官方的一鍵 pipeline。

它補的流程大致是：

1. 用外部方法建立 3D Gaussian 或 4D Gaussian。
2. 把外部方法輸出的資料格式轉成 PhysFlow 能吃的場景格式。
3. 用 SAM2Point 做 2D mask，再把 mask 粗略投回 3D Gaussian。
4. 用 CogVideo 產生 `images_generated/` 當目標運動 supervision。
5. 用 OpenAI 視覺模型猜材質，再把結果塞進 PhysFlow config。
6. 最後呼叫 `simulation.py` 跑物理反演。

所以你可以把它理解成：

- `external/` 是「借來的演算法本體」
- 這些 `.py` 是「膠水程式」

這也是為什麼它看起來很多地方都像 wrapper，而不是完整演算法。

## 目錄怎麼看

### 重要資料夾

- `external/`
  - 第三方 repo，不是這份補充程式碼的核心邏輯。
  - 目前主要包含 `PGSR`、`Grid4D`、`splatt3r`、`SAM2Point`。
- `workspaces/`
  - 中間產物工作目錄。
  - 例如抽出來的影格、COLMAP 結果、重建輸出、材質 JSON。
- `__pycache__/`
  - Python 自動產生，不用讀。

### 真正需要看的頂層程式

- `common.py`
- `bootstrap_external.py`
- `extract_video_frames.py`
- `prepare_colmap_scene.py`
- `prepare_dynamic_video_workspace.py`
- `prepare_pacnerf_multiview.py`
- `run_splatt3r_infer.py`
- `convert_to_physflow_scene.py`
- `run_sam2_masks.py`
- `project_masks_to_gaussians.py`
- `query_openai_material.py`
- `create_material_config.py`
- `run_cogvideo_cli.py`
- `run_full_pipeline.py`

## 整體資料流

### A. 3D/4D 重建

- `run_full_pipeline.py` 依照 `--reconstruction-method` 決定走哪條路。
- `pgsr`
  - 複製多視角影像到 workspace。
  - 如果輸入其實是 `PAC_NeRF_data/<scene>` 這種 `all_data.json + data/` 格式，會先自動抽出一個 time slice，轉成純多視角圖片資料夾。
  - 呼叫 `external/PGSR` 做 preprocess 和 train。
- `grid4d`
  - 先把影片抽成影格並整理成 COLMAP scene。
  - 呼叫 `external/Grid4D/train.py`。
- `splatt3r`
  - 吃一張或兩張圖，輸出一個 Gaussian PLY。

### B. 格式轉接到 PhysFlow

- `convert_to_physflow_scene.py` 會把外部方法輸出的內容整理成：
  - `model/custom/<scene>/point_cloud/iteration_x/point_cloud.ply`
  - `model/custom/<scene>/input.ply`
  - `model/custom/<scene>/cameras.json`
  - `model/custom/<scene>/images/`
  - `model/custom/<scene>/cfg_args`

### C. 運動區域與 guidance

- `run_sam2_masks.py`
  - 對 `images/` 做影片式 segmentation。
- `project_masks_to_gaussians.py`
  - 把 2D mask 投影回 Gaussian center。
  - 產出 `moving_part_points.ply`。
- `run_cogvideo_cli.py`
  - 用單張圖加文字 prompt 生成目標運動影格，存到 `images_generated/`。

### D. 材質初始化

- `query_openai_material.py`
  - 問 OpenAI 視覺模型，猜材質和參數。
- `create_material_config.py`
  - 把猜出的材料資訊套進現有 PhysFlow config template。

### E. 物理反演

- `run_full_pipeline.py` 最後呼叫根目錄的 `simulation.py`。

## 逐檔案說明

## `README.md`

### 它在幹嘛

- 說明這個資料夾的定位。
- 列出各條路徑的 CLI 用法。
- 強調這不是官方完整重現流程。

### 你該怎麼理解

- 這份 README 比較像「另一個 AI 對自己補的東西做的使用說明」。
- 它不是 PhysFlow 原 repo 的正式文件。

### 問題

- 有些敘述比實際程式更樂觀。
- 例如 README 說 dynamic video 會把「原始動態 frame」複製到 guidance，但實作上 `prepare_dynamic_video_workspace.py` 是把 `images/` 複製成 `guidance_frames/`，而 `images/` 通常是經過 COLMAP undistort 之後的影格，不一定是原始 frame。

## `__init__.py`

### 它在幹嘛

- 幾乎沒邏輯，只是讓 `supplemental_pipeline` 可以當 Python package。

### 問題

- 沒什麼問題，可以忽略。

## `common.py`

### 它在幹嘛

- 放共用路徑常數。
- 放共用工具函式。

### 重要內容

- `SUPPLEMENTAL_ROOT`、`REPO_ROOT`、`EXTERNAL_ROOT`、`WORKSPACE_ROOT`
  - 把資料夾位置集中管理。
- `REPOSITORIES`
  - 定義第三方 repo 名稱、GitHub 路徑與本地目的地。
- `MATERIAL_CONFIG_TEMPLATES`
  - 不同材料對應到哪個既有 PhysFlow config。
- `ensure_dir`
  - 建資料夾。
- `read_json` / `write_json`
  - 讀寫 JSON。
- `run_command`
  - 印出指令後用 `subprocess.run(..., check=True)` 執行。
- `copy_or_link`
  - 複製或建立 symlink。
- `list_sorted_files`
  - 依檔名排序列出固定副檔名檔案。
- `normalize_material_type`
  - 把像 `jelly`、`plastic` 這些 alias 映到內部材料名稱。
- `download_github_tarball`
  - 直接下載 GitHub tarball 解壓，不走 `git clone`。

### 你該怎麼理解

- 這一檔是整個補充 pipeline 的基底。
- 後面幾乎每支 script 都靠它拿 repo 路徑和跑外部指令。

### 問題

- `download_github_tarball` 只下載原始碼，不會處理各 repo 的 Python env、weights、submodule、CUDA build。
- 所以 `bootstrap_external.py` 完成後，不代表外部方法真的可跑。
- `download_github_tarball` 直接 `extractall`，雖然來源是 GitHub tarball，風險不算高，但這不是很嚴謹的解壓方式。

## `bootstrap_external.py`

### 它在幹嘛

- 下載 `REPOSITORIES` 裡列出的外部 repo。
- 可選擇額外幫 SAM2Point 跑 checkpoint 下載腳本。

### 實際流程

1. 解析 `--repos`。
2. 逐個呼叫 `download_github_tarball(...)`。
3. 如果指定 `--download-sam2-checkpoints`，執行 `SAM2Point/checkpoints/download_ckpts.sh`。

### 你該怎麼理解

- 它只是 source code bootstrap，不是完整安裝器。

### 問題

- 不處理 conda env、pip 套件、CUDA extension build。
- 所以這支成功，不代表 `PGSR`、`Grid4D`、`splatt3r`、`SAM2Point` 能直接在同一個 Python 環境裡 import。

## `extract_video_frames.py`

### 它在幹嘛

- 把影片拆成連號影格，提供 Grid4D 或其他流程使用。

### 實際流程

1. 用 `imageio` 讀影片。
2. 根據 `--frame-stride` 或 `--target-fps` 算保留間隔。
3. 可選擇 resize。
4. 依序輸出成 `00000.png` 這種格式。

### 你該怎麼理解

- 這支是很標準的前處理腳本。
- 給 Grid4D 用時，目的是把原本影片整理成容易進 COLMAP 的 frame sequence。

### 問題

- `target_fps` 是用整數 stride 近似，不是精準重採樣。
- 如果原影片 fps 不整除目標 fps，抽樣頻率會有誤差。
- 這不算 bug，比較像簡化實作。

## `prepare_colmap_scene.py`

### 它在幹嘛

- 把 `scene_dir/input/` 裡的影格轉成 COLMAP-style scene。

### 實際流程

1. 檢查 `input/` 是否存在。
2. 清掉舊的 `database.db`、`images/`、`sparse/0`。
3. 跑 `colmap feature_extractor`。
4. 跑 `colmap exhaustive_matcher`。
5. 跑 `colmap mapper`。
6. 如果沒指定 `--skip-undistort`，再跑 `colmap image_undistorter`。
7. 否則直接把 `input/` 複製到 `images/`。

### 你該怎麼理解

- 這是標準「影片 frame -> COLMAP scene」的整理器。
- `Grid4D` 或 `PGSR` 這類 3DGS 方法通常需要這種資料格式。

### 問題

- `clear_path()` 完全沒被用到，屬於死碼。
- 沒處理 mapper 產出多個 model 的情況，只是假設 `sparse/0` 一定是對的。
- 如果 `--skip-undistort`，後續投影類流程會更依賴 `input` 本身是否已經接近無畸變。

## `prepare_dynamic_video_workspace.py`

### 它在幹嘛

- 把原始動態影片或一串 frames 整理成 Grid4D 可訓練的 workspace。

### 實際流程

1. 建 `workspaces/grid4d_video/<scene-name>/input`。
2. 如果輸入是影片：
   - 呼叫 `extract_video_frames.py`。
3. 如果輸入是現成 frames：
   - 直接複製並重新命名成連號檔名。
4. 呼叫 `prepare_colmap_scene.py` 跑 COLMAP。
5. 把 `images/` 再複製一份到 `guidance_frames/`。

### 你該怎麼理解

- 這支是 `grid4d` 路徑前半段的總控。
- 它把「抽幀 + COLMAP + guidance frame 準備」包在一起。

### 問題

- 如果你走 `--frames-dir` 路徑，`--frame-stride`、`--max-frames`、`--target-fps`、`resize` 這些參數根本不會生效，因為 `copy_frames()` 只是全複製。
- 它把 `images/` 複製成 `guidance_frames/`。若 `images/` 是 COLMAP undistort 後的影像，那 guidance 就不是原始影片 frame。
- README 對這件事的描述和實作不完全一致。

## `prepare_pacnerf_multiview.py`

### 它在幹嘛

- 專門把 PAC-NeRF 原始 scene 格式轉成 `PGSR` 可直接吃的 multiview image folder。

### 為什麼需要它

- `PAC_NeRF_data/torus` 這類資料夾不是單純圖片集合。
- 它裡面有：
  - `all_data.json`
  - `data/*.png`
- `all_data.json` 每筆同時帶有：
  - `file_path`
  - `time`
  - `c2w`
  - `intrinsic`
- 所以它本質上是「多個 time step，每個 time step 又有多個 view」。
- 但 `PGSR` 要吃的是「單一 time step 的多視角圖片」。

### 實際流程

1. 檢查某個資料夾是不是 PAC-NeRF scene。
2. 讀 `all_data.json`。
3. 收集所有可用的 `time`。
4. 選一個 `time`：
   - 可用 `--time`
   - 或 `--time-index`
   - 若都沒給，預設選最接近 `0` 的 time
5. 只保留該 time 對應的所有 view。
6. 複製成一個純圖片資料夾。
7. 另外寫一份 `pacnerf_import_metadata.json`，把來源影像、`c2w`、`intrinsic`、選到哪個 time 記下來。

### 你該怎麼理解

- 它不是重建器。
- 它只是 PAC-NeRF 原始資料格式和 `PGSR` 之間的 adapter。

### 問題 / 限制

- 它目前只做「單一 time slice -> multiview folder」。
- 還沒有把 `all_data.json` 的相機資訊直接拿去跳過 COLMAP 或直接建立 PhysFlow scene。
- 換句話說，它仍然是走 `PGSR` 那條重建路，而不是直接利用 PAC-NeRF 已知相機。

## `run_splatt3r_infer.py`

### 它在幹嘛

- 呼叫 `external/splatt3r` 做 inference，輸出 Gaussian PLY。

### 實際流程

1. 接受一張或兩張圖片。
2. 若只給一張，直接複製成兩張相同的輸入。
3. 把 `splatt3r` repo 相關路徑插進 `sys.path`。
4. 從 Hugging Face 下載 checkpoint。
5. 載入 model。
6. 跑 `pred1, pred2 = model(img1, img2)`。
7. 用 `utils.export.save_as_ply` 存成 PLY。

### 你該怎麼理解

- 這是一個很薄的 wrapper。
- 核心演算法不在這裡，在 `external/splatt3r`。

### 問題

- 單張圖時把同一張圖複製成兩張，只是權宜作法，不保證符合 `splatt3r` 原本假設。
- 這比較像「能跑」而不是「方法上很乾淨」。
- 沒檢查 `external/splatt3r` 是否真的已存在、依賴是否齊全。

## `convert_to_physflow_scene.py`

### 它在幹嘛

- 把外部重建結果轉成 PhysFlow `model/<scene>/...` 需要的格式。

### 兩種主要情況

#### 1. `pgsr` / `grid4d`

- 從外部輸出複製：
  - `point_cloud/iteration_*`
  - `input.ply`
  - `cameras.json`
  - `cfg_args` 或自動補一份
- 可另外把 `images_dir` 複製到 `scene_dir/images`

#### 2. `splatt3r` / `generic`

- 把某個 PLY 直接當成 `point_cloud/iteration_1/point_cloud.ply`
- 也複製一份到 `input.ply`
- 如果沒有 `camera_json`，就用第一張 reference image 自動造一個只有單相機的 `cameras.json`

### `make_default_camera_json()` 在做什麼

- 讀整個 PLY 的 bounding box。
- 算一個中心和半徑。
- 把相機放在物體前方一段距離。
- 焦距用固定 60 度 FOV 反推。

### 你該怎麼理解

- 這一檔是「資料格式轉接器」。
- 對 `PGSR` / `Grid4D` 路徑來說，比較合理，因為那些方法本來就會產生 `cameras.json`。
- 對 `splatt3r` / `generic` 路徑來說，這邊在硬湊 PhysFlow 需要的相機資訊。

### 問題

- `splatt3r` / `generic` 的自動相機是很粗糙的假相機，只保證 PhysFlow 有東西可讀，不保證和輸入圖片真的對齊。
- 這代表後面如果依賴 `cameras.json` 做投影、view selection、bbox 設定，結果可能很飄。
- 如果 `gaussian_ply` 是資料夾，它只拿找到的第一個 `.ply`，不保證那個就是你真正要的輸出。
- `--link` 只在 `copy_or_link` 路徑上有效；`splatt3r/generic` 那段還是直接 `shutil.copy2`。

## `run_sam2_masks.py`

### 它在幹嘛

- 呼叫 `external/SAM2Point/segment.py`，對一串 frames 輸出每張對應的二值 mask。

### 三種 prompt

- `point`
  - 用一個點當 prompt。
- `box`
  - 用一個框當 prompt。
- `mask`
  - 先用 image predictor 在第一張圖生成 mask，再用 video predictor 往後傳播。

### 你該怎麼理解

- 這支也是純 wrapper。
- 真正的 segmentation 邏輯在 `external/SAM2Point/segment.py`。

### 問題

- `mask` prompt 名稱有點誤導，因為這裡沒有讓你直接傳入一張現成 mask；它其實還是要你給一個點，讓 SAM2 先在第一張圖長出 mask。
- `output_dir` 不會先清空；如果之前跑過、這次影格數比較少，可能留下舊檔。

## `project_masks_to_gaussians.py`

### 它在幹嘛

- 把每個 2D mask 反投影到 Gaussian center 上。
- 只要某個 Gaussian 在夠多視角都落在 mask 內，就把它當成 moving part。
- 最後輸出 `moving_part_points.ply`。

### 實際流程

1. 找最新的 `point_cloud/iteration_*`。
2. 讀所有 Gaussian center 的 `(x, y, z)`。
3. 讀 `cameras.json`。
4. 讀 masks。
5. 用每個 camera 把所有 3D 點投影到 2D。
6. 看投影點是否落在白色 mask 區域。
7. 對每個 Gaussian 累積票數。
8. 達到 `min_votes` 與 `vote_frac` 門檻者保留。

### 你該怎麼理解

- 這是整份補充 pipeline 裡最像「自己另外發明一個近似方法」的地方。
- 它不是 SAM2Point 官方直接提供給 PhysFlow 的標準輸出格式。

### 問題

- 這支有一個實際 bug。
  - 它先用 `camera["img_name"]` 對應 mask 檔名。
  - 但只要中間有某些 camera 找不到對應 mask，它會把 `None` 過濾掉，再直接 `zip(cameras, ordered_masks)`。
  - 這會讓後面的 camera 和 mask 發生錯位配對。
- 這支沒有做遮擋測試。
  - 只要某個 3D 點投影後落在 mask 裡，就算它在物體後面、其實被前景擋住，也會被投票。
  - 對稠密 Gaussian 來說，這會很容易把整塊厚度都選進去。
- 它也沒有用深度圖或 rasterization 去決定「畫面上真正可見的是哪個 Gaussian」。
- 所以它比較像粗略的 moving-part selector，不是幾何上很嚴謹的 2D-to-3D 分割。

## `query_openai_material.py`

### 它在幹嘛

- 把一張圖片和一段 prompt 丟給 OpenAI 視覺模型。
- 希望模型回一個 JSON，內含材質類別與物理參數。

### 實際流程

1. 讀圖並轉成 base64 data URI。
2. 呼叫 OpenAI API。
3. 取回 `choices[0].message.content`。
4. `json.loads(...)` 後寫到輸出檔。

### 你該怎麼理解

- 這支想做的是「用多模態模型幫你初始化 material config」。
- 它只管生成一個材料 JSON，沒有直接碰 PhysFlow config。

### 問題

- 它的 prompt 沒有明確規定 JSON schema。
- 但下游 `create_material_config.py` 直接假設輸出一定有 `material_type` 這個 key。
- 如果模型回的是 `{ "material": "foam", ... }`，這支雖然會成功寫檔，但下一步會直接炸掉。
- 沒做 API 錯誤處理、rate limit 處理、非 JSON 回覆處理。
- 這支用的是原始 HTTP 請求加 chat completions 格式，能不能穩定對應你目前安裝與帳號的 OpenAI 用法，要看環境。

### 建議的理想輸出格式

如果你要手動修正 `material_json`，至少要長這樣：

```json
{
  "material_type": "foam",
  "density": 120,
  "E": 0.03,
  "nu": 0.25,
  "yield_stress": 0.01,
  "plastic_viscosity": 0.05
}
```

## `create_material_config.py`

### 它在幹嘛

- 把 `material_json` 套到某個既有的 PhysFlow config template 上。

### 實際流程

1. 讀 `material_json`。
2. 取 `material_init["material_type"]`。
3. 用材料類型對應到某個 template。
4. 複製 template。
5. 保留一大批 PhysFlow 需要的全域欄位。
6. 把材料相關欄位改成 LLM 回傳的值。
7. 可選擇覆蓋 `default_camera_index` 與 `bbox_2d`。

### 你該怎麼理解

- 這不是「從零建立 config」。
- 它其實是「拿某個現成場景 config 當母版，再改材料參數」。

### 問題

- 這是另一個很大的風險點。
- Template 裡保留的欄位很多其實是 scene-specific，不是 material-specific：
  - `boundary_conditions`
  - `mpm_space_vertical_upward_axis`
  - `mpm_space_viewpoint_center`
  - `default_camera_index`
  - `move_camera`
  - `delta_a` / `delta_e` / `delta_r`
  - `line`
  - `bbox_2d`
- 這些值如果沿用 `torus`、`plane_960`、`hat` 之類範例場景，通常不會自動適合你的新場景。
- 換句話說，這支只是在「材質」層面做替換，沒有真正解決「新場景物理 config 該怎麼設」。
- 它依賴 `material_type` 這個 key，但上游 prompt 沒保證一定會回這個名字。
- 它也沒有驗證某些材料必填欄位是否真的存在。

## `run_cogvideo_cli.py`

### 它在幹嘛

- 用 Diffusers 的 `CogVideoXImageToVideoPipeline` 做 image-to-video 生成。

### 實際流程

1. 讀單張圖。
2. 載入 `THUDM/CogVideoX-5b-I2V`。
3. 依照 prompt 生成影片幀。
4. 把每張 frame 存成 `frame_00000.png` 這種格式。
5. 同時輸出 mp4。

### 你該怎麼理解

- 這支基本上就是把 repo 根目錄原本偏 demo 式的 `video_generator.py` 改成 CLI 化。

### 問題

- `output_dir` 不會先清空，若上次生成 49 張、這次只生成 25 張，資料夾可能殘留舊 frame。
- `seed` 預設只在 `0..255` 取值，範圍太小，重複機率高。
- `torch.Generator(device="cuda")` 是硬寫死 CUDA，如果環境不是 CUDA 或目前 pipeline 想走別的 device，會出錯。
- `torch_dtype=torch.bfloat16` 也假設硬體支援。

## `run_full_pipeline.py`

### 它在幹嘛

- 這是總控腳本。
- 負責依照參數把前面所有步驟串起來。

### 實際流程

1. 先決定是 `pgsr`、`grid4d` 還是 `splatt3r`。
2. 如果是 `pgsr` 且 `input-dir` 看起來像 PAC-NeRF scene，先自動呼叫 `prepare_pacnerf_multiview.py` 的邏輯，抽出單一 time 的 multiview images。
3. 做重建。
4. 呼叫 `convert_to_physflow_scene.py`。
5. 如果指定 segmentation 參數：
   - 跑 `run_sam2_masks.py`
   - 再跑 `project_masks_to_gaussians.py`
6. 如果指定 `video_prompt`：
   - 跑 `run_cogvideo_cli.py`
7. 如果指定 `openai_material`：
   - 跑 `query_openai_material.py`
8. 若 material JSON 存在：
   - 跑 `create_material_config.py`
9. 若 config 和 `images_generated/` 都存在：
   - 跑 `simulation.py`

### 你該怎麼理解

- 它不是很複雜的演算法，而是 orchestrator。
- 真正的價值是把多個外部工具接起來。

### 問題

- 它對 `mode` 和 `reconstruction-method` 的搭配沒有很強的限制，例如你仍可傳一些概念上不太合理的組合。
- `--sam2-prompt-type mask` 在這裡仍然要求 `--sam2-point`，因為底層實作其實不是直接吃使用者提供的 mask。這個命名會讓人誤解。
- 如果先走 Grid4D 複製了 `images_generated/`，後面又指定 `video_prompt`，CogVideo 會直接往同一個資料夾寫新 frame，但不會先清空舊內容。
- 它假設各外部 repo、checkpoint、依賴都已經在對的環境內可用，自己不做檢查。

## 這份補充 pipeline 最值得注意的問題

這裡把真正要小心的點集中列出來。

### 1. `project_masks_to_gaussians.py` 有實際錯位 bug

- 缺 mask 時，camera 和 mask 會錯位配對。
- 這不是單純近似，而是真的可能導致輸出錯。

### 2. 2D mask 投回 3D 的方法很粗

- 沒有遮擋判定。
- 沒有 depth-aware visible point selection。
- 會把被遮住的後方 Gaussian 也選進 moving part。

### 3. `create_material_config.py` 只是在改材質，不是在建立正確場景 config

- 它沿用別的場景 template 的 camera、bbox、boundary condition。
- 這對新場景通常不會自動正確。

### 4. `splatt3r` 路徑的相機是硬湊出來的

- `convert_to_physflow_scene.py` 自動生成的單相機只是占位用。
- 如果後面流程很依賴相機幾何，這條路的結果會很不穩。

### 5. `query_openai_material.py` 和 `create_material_config.py` 的 JSON 契約不穩

- 上游 prompt 沒明確要求 `material_type` 這個 key。
- 下游卻直接硬取 `material_init["material_type"]`。

### 6. 幾個輸出資料夾不會先清空

- `run_cogvideo_cli.py`
- `run_sam2_masks.py`

這會留下前一次執行的殘檔，尤其在影格數變少時更容易踩雷。

## 哪些地方是「可接受的研究原型」，哪些是「應該修」

### 可接受的研究原型

- `extract_video_frames.py` 用 stride 近似 target fps。
- `run_splatt3r_infer.py` 薄 wrapper。
- `bootstrap_external.py` 只負責抓 source code。

這些比較像「功能不完整，但合理」。

### 應該修

- `project_masks_to_gaussians.py` 的 camera/mask 錯位。
- `query_openai_material.py` 與 `create_material_config.py` 的 JSON schema 不一致。
- `create_material_config.py` 直接沿用別的場景模板，沒有提醒使用者 scene-specific 欄位仍需重設。
- `run_cogvideo_cli.py`、`run_sam2_masks.py` 沒清 output directory。
- `prepare_dynamic_video_workspace.py` 在 `frames-dir` 路徑忽略 frame sampling 相關參數。

## 我對這份程式碼的整體判斷

如果你的問題是「這份前半段程式碼是不是完全亂寫」，
答案是不是。它的主體邏輯其實是清楚的：

- 用外部 repo 做重建或分割
- 用 wrapper 轉成 PhysFlow 的資料格式
- 用 `run_full_pipeline.py` 串起來

但如果你的問題是「它是不是已經等於論文官方 pipeline，而且很穩」，
答案也不是。它更像：

- 能幫你把缺的步驟補齊
- 能讓流程大致跑通
- 但其中幾個關鍵步驟仍然是 heuristic 或 placeholder

尤其是：

- 3D moving-part 建立
- 單圖 Splatt3R 相機設定
- 新場景 material config 建立

這三塊都不是很扎實。

## 這次檢查我有做什麼

- 我逐支閱讀了 `supplemental_pipeline/` 頂層 Python 腳本。
- 我也對照了本 repo 的 `simulation.py`、`gs/scene/__init__.py`、`utils/decode_param.py` 等檔案，確認 PhysFlow 真正期待的場景格式。
- 我另外做了語法檢查：

```bash
python3 -m py_compile supplemental_pipeline/*.py
```

語法層面目前都能 compile。

### 但我沒有做的事

- 沒有真的把整條 pipeline 跑完。
- 沒有替每個外部 repo 安裝相依套件與權重後做 runtime 驗證。
- 沒有逐行審查 `external/` 裡的第三方專案。

所以這份文件能回答的是：

- 每支程式在做什麼
- 跟 PhysFlow 是怎麼接的
- 哪些地方看起來危險

但不能保證每條路徑在你目前環境裡一定可直接成功執行。
