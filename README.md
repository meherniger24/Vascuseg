# nnU-Net v2 from .npy → Training → Inference (Step-by-Step)

# What nnU-Net Expects

nnU-Net requires data in **NIfTI (.nii.gz)** format organized in a specific directory structure.

Expected structure:

nnUNet_raw/
└── DatasetXXX_NAME/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    └── dataset.json

File naming rules:

imagesTr/
- case_0001_0000.nii.gz
- case_0002_0000.nii.gz

labelsTr/
- case_0001.nii.gz
- case_0002.nii.gz

Important:
- `_0000` indicates image channel 0
- Labels must contain integer class values
- Background must be 0

---

# 1. Install nnU-Net v2

```bash
pip install -U nnunetv2
```
# 2. Set Environment Variables

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```
# 3. Prepare Input .npy Files

- Image shape: `(Z, Y, X)` or `(Y, X)`
- Label must match image shape
- Labels must be integer values
- Background class = `0`

# 4. Convert `.npy` → `.nii.gz`

Example conversion script:

```python
import os
import numpy as np
import SimpleITK as sitk

def npy_to_nifti(npy_path, out_path, spacing=(1.0,1.0,1.0), is_label=False):

    arr = np.load(npy_path)

    if arr.ndim == 2:
        arr = arr[None, ...]

    if is_label:
        arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(np.float32)

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)

    sitk.WriteImage(img, out_path, True)
