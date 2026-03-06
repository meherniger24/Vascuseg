# nnU-Net v2 from .npy → Training → Inference (Step-by-Step)

This guide shows how to train nnU-Net v2 starting from many NumPy volumes (.npy) and their segmentation masks (.npy), by converting them into the format nnU-Net expects.

# What nnU-Net expects

nnU-Net v2 expects a dataset in nnU-Net Raw format:

Images: NIfTI (.nii.gz) in imagesTr/ and imagesTs/

Labels (masks): NIfTI (.nii.gz) in labelsTr/

A dataset.json describing modalities + labels

Naming rules (important)

Case ID format: case_0001, case_0002, ...

Image filename: case_0001_0000.nii.gz
(_0000 = channel 0; multi-channel uses _0001, _0002, etc.)

Label filename: case_0001.nii.gz

1) Install nnU-Net v2
