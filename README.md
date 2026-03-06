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

- Label must match image shape
- Labels must be integer values
- Background class = `0`

# 5. Create Dataset Folder

Example:

```bash
mkdir -p $nnUNet_raw/Dataset101_Vessels/imagesTr
mkdir -p $nnUNet_raw/Dataset101_Vessels/labelsTr
mkdir -p $nnUNet_raw/Dataset101_Vessels/imagesTs
```

Move the converted `.nii.gz` files into the appropriate folders.

---

# 6. Create `dataset.json`

Example:

```json
{
  "name": "Vessels",
  "tensorImageSize": "3D",
  "modality": { "0": "CT" },
  "labels": {
    "background": 0,
    "vessel": 1
  },
  "numTraining": 100,
  "numTest": 20
}
```

---

# 7. Verify Dataset Integrity

```bash
nnUNetv2_verify_dataset_integrity -d 101
```

---

# 8. Plan and Preprocess

```bash
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity
```

This step will:

- Determine optimal patch size  
- Analyze voxel spacing  
- Normalize intensities  
- Generate preprocessed training data
