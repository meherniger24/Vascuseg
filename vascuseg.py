# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:57:28 2025

@author: meher
"""

import os
import numpy as np
import imageio
import nibabel as nib
import subprocess
from tempfile import TemporaryDirectory
import os

# Set required nnU-Net v2 environment variables
os.environ["nnUNet_raw"] = r"C:\Users\meher\spyder\nnUNet_raw"
os.environ["nnUNet_preprocessed"] = r"C:\Users\meher\spyder\nnUNet_preprocessed"
os.environ["nnUNet_results"] = r"C:\Users\meher\spyder\nnUNet_trained\nnUNet_results"

def save_prediction_as_binary_pngs(pred_path, output_dir, base_name):
    """
    Converts a predicted NIfTI volume into a series of binary PNG slices (0 or 255).
    """
    data = nib.load(pred_path).get_fdata()
    binary = (data > 0).astype(np.uint8)

    os.makedirs(output_dir, exist_ok=True)
    for i in range(binary.shape[2]):
        slice_img = (binary[:, :, i] * 1).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, f"{base_name}_{i:03}.png"), slice_img)


def chunked_png_prediction(input_dir, output_png_dir, chunk_size=100,
                           dataset_id="Dataset001_VesselSegmentation",
                           fold="2"):
    
    """
    Processes PNG stack in chunks:
      1. Converts each chunk to a temporary NIfTI file.
      2. Runs nnUNetv2 on the temporary NIfTI.
      3. Converts prediction back to binary PNG slices.
    """
    png_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    total_files = len(png_files)

    for chunk_id, start in enumerate(range(0, total_files, chunk_size)):
        end = min(start + chunk_size, total_files)
        chunk = png_files[start:end]

        with TemporaryDirectory() as tempdir:
            temp_nii_dir = os.path.join(tempdir, "imagesTs")
            temp_out_dir = os.path.join(tempdir, "predictions")
            os.makedirs(temp_nii_dir)
            os.makedirs(temp_out_dir)

            # Stack PNGs into 3D volume
            stack = []
            for f in chunk:
                img = imageio.imread(os.path.join(input_dir, f))
                if img.ndim == 3:
                    img = img[:, :, 0]
                stack.append(img)

            vol = np.stack(stack, axis=-1)  # (H, W, Z)
            vol = np.transpose(vol, (1, 0, 2))  # (W, H, Z) for NIfTI
            
            # Save temporary NIfTI file
            case_id = f"CASE_{chunk_id:03}_0000"
            nii_path = os.path.join(temp_nii_dir, f"{case_id}.nii.gz")
            nib.save(nib.Nifti1Image(vol.astype(np.uint8), affine=np.eye(4)), nii_path)

            # Run nnUNetv2 prediction
            subprocess.run([
                "nnUNetv2_predict",
                "-i", temp_nii_dir,
                "-o", temp_out_dir,
                "-d", dataset_id,
                "-c", "3d_fullres",
                "-f", fold,
                "--disable_tta"
            ], check=True)

            # Save binary prediction PNGs
            pred_nii = os.path.join(temp_out_dir, f"CASE_{chunk_id:03}.nii.gz")
            save_prediction_as_binary_pngs(pred_nii, output_png_dir, f"CASE_{chunk_id:03}")



if __name__ == "__main__":
    chunked_png_prediction(
        input_dir=r"C:\vascuseg\input(1-60)",
        output_png_dir=r"C:\vascuseg\binary(1-60)",
        dataset_id="Dataset001_VesselSegmentation",
        fold="2"
    )
