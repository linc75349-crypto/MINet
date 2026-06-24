<div align="center">

# MINet: Multiscale Interactive Network for Real-Time Salient Object Detection of Strip Steel Surface Defects

[![Paper](https://img.shields.io/badge/Paper-TII%202024-blue)](https://ieeexplore.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch 1.4+](https://img.shields.io/badge/PyTorch-1.4+-red.svg)](https://pytorch.org/)

</div>

> **Note:** This is the English version of the documentation. For Chinese documentation, see [README.md](README.md).

---

## Table of Contents

- [News](#news)
- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

---

## News

- [2024] The paper has been accepted by **IEEE Transactions on Industrial Informatics (TII)**.
- [2024] Official code, trained model, and predicted saliency maps are released.

---

## Overview

MINet is a **real-time salient object detection network** designed specifically for detecting surface defects on strip steel in industrial settings. The network is built entirely from depthwise separable convolutions and does not require pretrained backbone weights.

### Key Features

- **Real-time performance**: Lightweight architecture achieves ~300 FPS on 368×368 inputs
- **Multiscale Interactive Module (MI Module)**: The core building block fuses multi-scale contextual information via parallel dilated convolutions (d={1,2,4,8}) and channel-wise interaction
- **Deep supervision**: Five side outputs at different resolutions provide auxiliary supervision during training
- **Fully scratch-trained**: No ImageNet pretraining required — the network trains from scratch

---

## Architecture

### MI Module

The MI Module consists of four parallel depthwise separable convolutions with dilation rates of 1, 2, 4, and 8. After extracting multi-scale features, a channel shuffle operation reorganizes channels across scales so that each output channel receives information from all four scales. Point-wise convolutions then fuse the shuffled features, and a residual connection adds the original input.

![MI Module](figures/MI.png)

### Network Overview

MINet follows a U-Net-like encoder-decoder structure with 5 stages:

| Stage | Encoder | Output Size | MI Modules | Decoder |
|:-----:|:-------:|:-----------:|:----------:|:-------:|
| 1 | Conv 3×3, stride 2 | 16×184×184 | 0 | DSConv → DSConv |
| 2 | DSConv stride 2 + 3×MI | 32×92×92 | 3 | DSConv → DSConv |
| 3 | DSConv stride 2 + 4×MI | 64×46×46 | 4 | DSConv → DSConv |
| 4 | DSConv stride 2 + 6×MI | 96×23×23 | 6 | DSConv → DSConv |
| 5 | DSConv stride 2 + 3×MI | 128×12×12 | 3 | DSConv → DSConv |

The decoder upsamples features via bilinear interpolation and fuses them with encoder skip connections through element-wise addition. All 5 decoder outputs participate in loss computation (deep supervision).

---

## Results

### Quantitative Comparison

![Quantitative Comparison](figures/quan.png)

### Qualitative Comparison

![Qualitative Comparison](figures/qual.png)

### Inference Speed

| Method | Input Size | FPS |
|:------:|:----------:|:---:|
| MINet | 368×368 | ~300 |

Run the speed benchmark:
```bash
python FPS.py
```

---

## Installation

### Prerequisites

- Python 3.7 or later
- PyTorch 1.4.0 or later
- CUDA (recommended for training)

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/Kunye-Shen/MINet.git
cd MINet
```

**2. Create a virtual environment (recommended)**
```bash
conda create -n minet python=3.9 -y
conda activate minet
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

If you need a specific PyTorch version for your CUDA, follow the [official instructions](https://pytorch.org/get-started/locally/) first, then install the remaining dependencies.

**4. Verify installation**
```bash
python -c "from model import MINet; import torch; print('OK')"
```

---

## Quick Start

### Download Resources

We provide the trained model weights and predicted saliency maps:

| Resource | Link |
|:--------:|:----:|
| Predicted saliency maps | [Google Drive](https://drive.google.com/drive/folders/1cj_Gd8EDIPvP4SpCdNWhS4G-fRMxm7VC?usp=drive_link) |
| Predicted saliency maps | [Baidu Netdisk](https://pan.baidu.com/s/1eBZX_1Nf_sWYVz2opp4f0A) (code: **rokb**) |

Place downloaded weights in the `model_save/` directory.

### Dataset Preparation

Organize your dataset as follows:

```
Dataset/
└── SD-saliency-900/
    ├── Img_train/    # Input images (.bmp)
    └── GT_train/     # Ground-truth masks (.png, white=defect, black=background)
```

Image and label files must be paired by name (e.g., `In_1.bmp` ↔ `In_1.png`).

All paths and hyperparameters are configured centrally in `config.py`.

### Run Inference

```bash
python test.py
```

Loads `model_save/MINet_best.pth` and generates side-by-side comparison images (input | prediction) in the `results/` directory.

### Train from Scratch

```bash
python train.py
```

Training hyperparameters (configurable in `config.py`):

| Parameter | Default | Description |
|:---------:|:-------:|:-----------:|
| Epochs | 80 | Total training epochs |
| Batch size | 32 | Samples per batch |
| Learning rate | 0.004 | Adam optimizer |
| LR schedule | StepLR (step=30, γ=0.5) | Decay every 30 epochs |
| Loss | BCE + SSIM | Hybrid loss summed over 5 side outputs |

Checkpoints are saved to `model_save/`:
- `MINet_best.pth` — best model by training loss
- `MINet.pth` — final model after 80 epochs
- `epoch_*.pth` — per-epoch snapshots
- `epoch_*_iter_*.pth` — checkpoints every 1000 iterations

### Configuration

All experiment settings are managed centrally in `config.py`:
- **Dataset paths**: modify `dataset` dict
- **Training hyperparameters**: modify `train` dict
- **Data augmentation**: modify `transform` dict
- **Model save paths**: modify `model` dict

### Measure Inference Speed

```bash
python FPS.py
```

Feeds 300 random 368×368 tensors through the model (100 runs, skip first 20 warm-up) and reports average FPS.

---

## Project Structure

```
MINet/
├── LICENSE                   # MIT License
├── CITATION.cff              # Citation metadata
├── README.md                 # Chinese documentation
├── README-EN.md              # English documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation config
├── config.py                 # Centralized configuration
├── train.py                  # Training script
├── test.py                   # Inference / visualization
├── FPS.py                    # Speed benchmark
├── data_loader.py            # Dataset class and transforms
├── model/
│   ├── __init__.py           # Package exports (MINet)
│   ├── basic.py              # Core building blocks
│   └── MINet.py              # Full network definition
├── pytorch_ssim/
│   └── __init__.py           # SSIM loss implementation
├── figures/                  # Paper figures
├── model_save/               # Model checkpoints
├── results/                  # Prediction outputs
└── Dataset/                  # Training data (user-provided)
    └── SD-saliency-900/
        ├── Img_train/
        └── GT_train/
```

---

## FAQ

**Q1: ModuleNotFoundError: No module named 'xxx'**

The package is not installed. Run `pip install xxx`.

**Q2: FileNotFoundError: cannot find dataset**

Check the dataset paths in `config.py` and ensure the directory structure is correct.

**Q3: CUDA out of memory**

Reduce `train.batch_size` in `config.py` (e.g., from 32 to 8 or 16).

**Q4: Python version compatibility**

- Python 3.12 may have compatibility issues with some packages. Python 3.9–3.11 is recommended.
- If you encounter `setuptools` errors, try `pip install 'setuptools<81'`.

**Q5: State dict key mismatch when loading model**

If you trained with a different config or code version, loading may fail. Try:
```python
net.load_state_dict(torch.load('model.pth'), strict=False)
```

**Q6: How to check if GPU is being used?**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
`True` = GPU is available; `False` = CPU only.

---

## Citation

If you use MINet in your research, please cite our paper:

```bibtex
@article{shen2024minet,
  title={MINet: Multiscale Interactive Network for Real-Time Salient Object Detection of Strip Steel Surface Defects},
  author={Shen, Kunye and Zhou, Xiaofei and Liu, Zhi},
  journal={IEEE Transactions on Industrial Informatics},
  year={2024}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

- [Kunye Shen](https://scholar.google.com.hk/citations?user=q6_PkywAAAAJ&hl=zh-CN)
- [Xiaofei Zhou](https://scholar.google.com.hk/citations?user=2PUAFW8AAAAJ&hl=zh-CN)
- [Zhi Liu](https://scholar.google.com.hk/citations?user=Sd5VB2cAAAAJ&hl=zh-CN)

For questions, please open an issue on GitHub.
