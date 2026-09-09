# PhysFlow Supplemental Pipeline

這個資料夾是補在 PhysFlow repo 內的整合層，不是原作者原生提供的一鍵 pipeline。

它補了四件事：

1. `external/`
放論文 README 提到的外部方法原始碼：
- `PGSR`
- `Grid4D`
- `splatt3r`
- `SAM2Point`

2. 3DGS 匯入與轉接
- `run_splatt3r_infer.py`
- `convert_to_physflow_scene.py`

3. segmentation / video / material init
- `run_sam2_masks.py`
- `project_masks_to_gaussians.py`
- `run_cogvideo_cli.py`
- `query_openai_material.py`
- `create_material_config.py`

4. 總控腳本
- `run_full_pipeline.py`

## 這些檔案是我補的

全部都在這個資料夾：
- `supplemental_pipeline/common.py`
- `supplemental_pipeline/bootstrap_external.py`
- `supplemental_pipeline/run_splatt3r_infer.py`
- `supplemental_pipeline/convert_to_physflow_scene.py`
- `supplemental_pipeline/run_sam2_masks.py`
- `supplemental_pipeline/project_masks_to_gaussians.py`
- `supplemental_pipeline/query_openai_material.py`
- `supplemental_pipeline/create_material_config.py`
- `supplemental_pipeline/run_cogvideo_cli.py`
- `supplemental_pipeline/run_full_pipeline.py`

`external/` 裡的 repo 是我另外下載進來的外部原始碼，不是我手寫的。

## 外部 repo 對應論文步驟

### 1. 建立 3D Gaussian

- 多視角圖片：`external/PGSR`
  - preprocess:
    - `python3 scripts/preprocess/convert.py --data_path <dataset_root>`
  - train:
    - `python3 train.py -s <scene_path> -m <output_path> --max_abs_split_points 0 --opacity_cull_threshold 0.05`

- 動態影片 / 4DGS：`external/Grid4D`
  - train:
    - `python3 train.py -s <scene_path> -m <output_path> --conf <conf.py>`

- 單張圖 / 雙張圖：`external/splatt3r`
  - wrapper:
    - `python3 supplemental_pipeline/run_splatt3r_infer.py --image a.png --image b.png --output gaussians.ply`

### 2. 分離可動區域

- `external/SAM2Point`
- wrapper:
  - `python3 supplemental_pipeline/run_sam2_masks.py --frames-dir <dir> --output-dir <dir> --prompt-type point --point 320 240`
- 然後把 2D mask 投回 Gaussian：
  - `python3 supplemental_pipeline/project_masks_to_gaussians.py --scene-dir model/custom/<scene> --masks-dir model/custom/<scene>/masks_generated`

### 3. 生成目標動作影片

- `python3 supplemental_pipeline/run_cogvideo_cli.py --image <img> --prompt "..." --output-dir model/custom/<scene>/images_generated`

### 4. LLM 材質初始化

- 用 OpenAI 視覺模型產生材料 JSON：
  - `python3 supplemental_pipeline/query_openai_material.py --image <img> --output material.json`
- 再轉成 PhysFlow config：
  - `python3 supplemental_pipeline/create_material_config.py --material-json material.json --output-config config/custom/<scene>_config.json`

### 5. PhysFlow 物理反演

- `python3 simulation.py --model_path model/custom/<scene> --output_path outputs/custom/<scene> --physics_config config/custom/<scene>_config.json --n_epoch 50 --n_key_frame 13`

## 自動總控

### 多視角 + PGSR

```bash
python3 supplemental_pipeline/run_full_pipeline.py \
  --scene-name my_scene \
  --mode multiview \
  --reconstruction-method pgsr \
  --input-dir /path/to/multiview_images \
  --sam2-prompt-type point \
  --sam2-point 320 240 \
  --video-prompt "The object is gently squeezed."
```

如果 `--input-dir` 指到的是 PAC-NeRF 原始 scene 資料夾，或是一個包含同名 scene 子資料夾的根目錄，pipeline 也會自動先切出單一 time slice 再跑 PGSR。例如：

```bash
python3 supplemental_pipeline/run_full_pipeline.py \
  --scene-name torus \
  --mode multiview \
  --reconstruction-method pgsr \
  --input-dir /home/ccchen/PhysFlow/PAC_NeRF_data \
  --pacnerf-time-index 1 \
  --sam2-prompt-type point \
  --sam2-point 320 240 \
  --video-prompt "The torus is gently squeezed." \
  --openai-material
```

這時會先把：

- `PAC_NeRF_data/torus/all_data.json`
- `PAC_NeRF_data/torus/data/`

整理成：

- `supplemental_pipeline/workspaces/pacnerf_multiview/torus/images/`

只保留某一個 `time` 對應的多視角圖片，然後再接 PGSR。

### 單張/雙張圖 + Splatt3R

```bash
python3 supplemental_pipeline/run_full_pipeline.py \
  --scene-name my_single_image_scene \
  --mode single_image \
  --reconstruction-method splatt3r \
  --input-dir /path/to/images \
  --sam2-prompt-type box \
  --sam2-box 200 150 500 650 \
  --video-prompt "The object is softly pushed from the left."
```

### 動態影片 + Grid4D

```bash
python3 supplemental_pipeline/run_full_pipeline.py \
  --scene-name my_video_scene \
  --mode video \
  --reconstruction-method grid4d \
  --video-path /path/to/video.mp4 \
  --prepare-video-workspace \
  --frame-stride 2 \
  --physflow-python /home/ccchen/miniconda3/envs/physflow/bin/python \
  --grid4d-python /path/to/grid4d_env/bin/python
```

這條路會做：

1. 用 `extract_video_frames.py` 把影片抽成連號 frame  
2. 用 `prepare_colmap_scene.py` 跑 COLMAP，產生 `images/`、`sparse/0/`  
3. 用 `Grid4D train.py` 吃這個 Colmap-style scene  
4. 用 `convert_to_physflow_scene.py` 把 Grid4D 輸出搬成 `model/custom/<scene>`  
5. 把原始動態 frame 複製成 `model/custom/<scene>/images_generated/` 當 PhysFlow guidance  
6. 若你另外提供 material config，再直接接 `simulation.py`

如果你已經自己準備好 `Grid4D` 的 scene folder，就不用 `--prepare-video-workspace`，直接：

```bash
python3 supplemental_pipeline/run_full_pipeline.py \
  --scene-name my_video_scene \
  --mode video \
  --reconstruction-method grid4d \
  --input-dir /path/to/prepared_grid4d_scene \
  --grid4d-python /path/to/grid4d_env/bin/python \
  --physflow-python /home/ccchen/miniconda3/envs/physflow/bin/python
```

如果你只想先做到影片前處理，不想一次跑完，也可以拆開：

```bash
python3 supplemental_pipeline/prepare_dynamic_video_workspace.py \
  --scene-name my_video_scene \
  --video /path/to/video.mp4 \
  --frame-stride 2 \
  --colmap-bin colmap
```

它會在：

- `supplemental_pipeline/workspaces/grid4d_video/my_video_scene/input`
- `supplemental_pipeline/workspaces/grid4d_video/my_video_scene/images`
- `supplemental_pipeline/workspaces/grid4d_video/my_video_scene/sparse/0`
- `supplemental_pipeline/workspaces/grid4d_video/my_video_scene/guidance_frames`

建立好後，你可以手動跑 Grid4D：

```bash
/path/to/grid4d_env/bin/python supplemental_pipeline/external/Grid4D/train.py \
  -s supplemental_pipeline/workspaces/grid4d_video/my_video_scene \
  -m supplemental_pipeline/workspaces/reconstruction/my_video_scene
```

## 重要限制

1. 這不是保證零手動的一鍵研究重現
- `Grid4D` 仍需要你提供可用的 dataset/conf
- `SAM2Point` 官方 README 也寫了 custom 3D input 還沒完全釋出
- 單張圖用 `splatt3r` 時，我補的是可跑的 Gaussian 匯入與預設相機，不是論文作者原生的 PhysFlow 單張圖官方 pipeline

2. 我補的是工程整合層
- 外部方法本體還是依賴各自 repo
- 版本衝突、CUDA、權重下載，仍要按外部 repo 的需求處理

3. `project_masks_to_gaussians.py` 是補的近似轉接
- 它用相機投影把 2D mask 映回 Gaussian centers，產出 `moving_part_points.ply`
- 這不是 SAM2Point 官方直接輸出給 PhysFlow 的格式

## 建議使用順序

1. `python3 supplemental_pipeline/bootstrap_external.py`
2. 先單獨跑 3DGS 方法
3. 再 `convert_to_physflow_scene.py`
4. 再跑 segmentation / CogVideo / material init
5. 最後跑 `simulation.py`

## Python 環境建議

這套補充 pipeline 預設不假設所有外部 repo 跟 PhysFlow 共用同一個環境。

- `run_full_pipeline.py` 支援：
  - `--pgsr-python`
  - `--grid4d-python`
  - `--splatt3r-python`
  - `--sam2point-python`
  - `--physflow-python`

如果你已經有：
- `PGSR` 一個 conda env
- `Grid4D` 一個 conda env
- `SAM2Point` 一個 conda env
- `PhysFlow` 自己的 `physflow` env

那就可以明確指定：

```bash
python3 supplemental_pipeline/run_full_pipeline.py \
  --scene-name my_scene \
  --mode multiview \
  --reconstruction-method pgsr \
  --input-dir /path/to/images \
  --pgsr-python /path/to/pgsr_env/bin/python \
  --sam2point-python /path/to/sam2point_env/bin/python \
  --physflow-python /home/ccchen/miniconda3/envs/physflow/bin/python
```
