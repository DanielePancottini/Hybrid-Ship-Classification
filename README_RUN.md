# How to run

Below are concise steps to set up a Python environment and run training, evaluation and inference for this repository on Windows (PowerShell). Adjust paths and CUDA choices as needed.

## 1) Prerequisites

- Install Python 3.8+ (3.10 or 3.11 recommended).
- A suitable `pip` or virtual environment. GPU support requires a matching PyTorch / CUDA wheel — see the PyTorch selector at https://pytorch.org/get-started/locally/.

## 2) Create & activate a virtual environment (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3) Install dependencies

- Recommended: install a PyTorch build that matches your CUDA version first (use the selector on the PyTorch website). Example CPU-only install:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- Example CUDA 11.8 (replace with the correct command from pytorch.org for your setup):

```powershell
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

- Then install the remaining Python packages used by the project:

```powershell
pip install -r requirements.txt
```

## 4) Prepare the dataset

- Place the WaterScenes data under `data/WaterScenes` and ensure the split files `train.txt`, `val.txt`, `test.txt` exist (the repo expects these paths by default).

## 5) Run (examples)

- Visualize a batch and start training (default config and paths are defined in `train.py`):

```powershell
# from repo root
python train.py
```

- Run evaluation using the checkpoint files listed in `evaluate.py` (it checks the `./checkpoints` folder):

# How to run

Below are concise steps to set up a Python environment and run training, evaluation and inference for this repository on Windows (PowerShell). Adjust paths and CUDA choices as needed.

## 1) Prerequisites

- Install Python 3.8+ (3.10 or 3.11 recommended).
- A suitable `pip` or virtual environment. GPU support requires a matching PyTorch / CUDA wheel — see the PyTorch selector at https://pytorch.org/get-started/locally/.

## 2) Create & activate a virtual environment (PowerShell)

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
```

## 3) Install dependencies

- Recommended: install a PyTorch build that matches your CUDA version first (use the selector on the PyTorch website). Example CPU-only install:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- Example CUDA 11.8 (replace with the correct command from pytorch.org for your setup):

```powershell
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

- Then install the remaining Python packages used by the project:

```powershell
pip install numpy pandas matplotlib opencv-python Pillow tqdm
```

Note: This repository does not include pinned versions. For reproducible runs, create a `requirements.txt` and pin versions after you confirm a working environment.

## 4) Prepare the dataset

- Place the WaterScenes data under `data/WaterScenes` and ensure the split files `train.txt`, `val.txt`, `test.txt` exist (the repo expects these paths by default).

## 5) Convert raw radar CSVs → REVP `.npy` maps (preprocessing)

The repository contains preprocessing scripts that convert raw radar CSV point files into 4-channel image-like tensors (Range, Elevation, Doppler, Power) and save them as NumPy `.npy` files.

1. Ensure the dataset is unpacked at `data/WaterScenes/` with the following expected structure:

```
data/WaterScenes/
	image/                 # RGB images like 00001.jpg
	radar/                 # raw CSV radar files like 00001.csv
	detection/yolo/        # YOLO-format labels (00001.txt)
	train.txt
	val.txt
	test.txt
```

2. Run the preprocessing script from the repository root:

```powershell
python data/waterscenes_preprocess.py
```

What this does:
- Reads `data/WaterScenes/radar/<file_id>.csv` for each `file_id` found in the split files.
- Uses `preprocess/revp.REVP_Transform` to bin points into a `TARGET_SIZE` grid (default 320×320).
- Saves the transformed radar maps to `data/WaterScenes/radar_revp_npy/<file_id>.npy`.

Important settings:
- `ORIGINAL_IMAGE_SIZE` in `data/waterscenes_preprocess.py` is set to `(1080, 1920)` by default — change it if your images use a different resolution because the transform rescales `u,v` coordinates.
- `TARGET_SIZE` (default `(320,320)`) controls the final spatial resolution. If you change it, make sure dataset/model configs align (the code assumes 320×320 in several places).

## 6) Compute radar mean/std for normalization

After preprocessing, compute dataset statistics and copy them into `train.py`/`evaluate.py` to normalize radar channels.

Run the stats script (option A: from `preprocess` folder):

```powershell
cd preprocess
python get_radar_stats.py
cd ..
```

The script prints `Radar Mean` and `Radar STD` as lists of 4 floats. Copy these values into `RADAR_MEAN` and `RADAR_STD` in `train.py` and `evaluate.py` (or adjust the scripts to read them from a config file).

If you prefer to run from repo root, edit the `DATASET_ROOT` path at the top of `preprocess/get_radar_stats.py` to `os.path.abspath("./data/WaterScenes")` and run:

```powershell
python preprocess/get_radar_stats.py
```

## 7) Run examples (training / evaluation / inference)

From repo root:

```powershell
# Train with default parameters
python train.py

# Evaluate (uses checkpoints in ./checkpoints)
python evaluate.py

# Quick inference (edit CHECKPOINT_PATH in test_inference.py)
python test_inference.py
```

