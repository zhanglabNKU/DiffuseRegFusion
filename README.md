# DiffuseRegFusion

This repository contains the official implementation preview of **DiffuseRegFusion**, a diffusion-based framework for deformable medical image registration and image fusion.

The current release is a lightweight research preview intended to show the training and testing workflow. Some model details, pretrained weights, dataset-specific processing utilities, and full experiment scripts are intentionally omitted at this stage. We will fully open-source the complete codebase, trained models, and detailed reproducibility instructions after the paper is accepted.

## Overview

DiffuseRegFusion takes paired medical images from two modalities, such as CT and MRI, and learns to estimate a deformation field while producing a fused image. The released preview includes:

- 2D paired CT-MRI dataset loading
- diffusion-based registration and fusion training code
- checkpoint saving and loading
- testing code for warped images, deformation fields, fused images, and basic metrics
- JSON configuration files for the released 2D workflow

Several internal model components and additional experimental branches have been simplified or removed from this preview. These parts will be released after acceptance.

## Environment

The code is implemented in Python with PyTorch. A typical environment should include:

```bash
python >= 3.8
pytorch
numpy
Pillow
SimpleITK
tqdm
tensorboard
matplotlib
```

Please install a PyTorch version that matches your CUDA runtime.

## Data Preparation

Prepare paired 2D images with the same file names under `CT` and `MRI` folders:

```text
datasets/CT-MRI/
  train/
    CT/
      case001.png
      case002.png
    MRI/
      case001.png
      case002.png
  test/
    CT/
      case101.png
    MRI/
      case101.png
```

Supported image formats are `.png`, `.jpg`, and `.jpeg`. The default paths are defined in:

- `config/train_2D.json`
- `config/test_2D.json`

Update the `dataroot` fields if your dataset is stored elsewhere.

## Training

Run training with:

```bash
python train_2D.py -c config/train_2D.json -gpu 0
```

Checkpoints and logs are written under `experiments/<experiment_name>_train_<timestamp>/`. The current training script saves model checkpoints every 50 epochs.

To resume training, set `path.resume_state` in the training config to the checkpoint prefix. For example, if the checkpoint files are:

```text
I1000_E50_gen_G.pth
I1000_E50_opt.pth
```

set the resume path to the prefix:

```text
/path/to/I1000_E50
```

## Testing

Run testing with a checkpoint prefix:

```bash
python test_2D.py -c config/test_2D.json -w /path/to/checkpoint/I1000_E50 -gpu 0
```

The `-w` argument should not include `_gen_G.pth` or `_opt.pth`. The script appends these suffixes automatically when loading the model.

Testing writes NIfTI outputs to the configured results directory under `experiments/<experiment_name>_test_<timestamp>/results/`, including moving images, fixed images, registered images, deformation fields, and fused images. It also reports SSIM, negative Jacobian determinant, Jacobian determinant statistics, and runtime.

## Current Release Status

This repository is not the final camera-ready release. Some implementation details and experiment-specific modules have been removed to avoid exposing incomplete or unstable code before peer review is complete.

After the paper is accepted, we will release:

- the complete model implementation
- pretrained checkpoints
- full training and evaluation scripts
- detailed configuration files
- dataset preprocessing and reproduction instructions

## Citation

Citation information will be added after the paper is accepted.
