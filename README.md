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
