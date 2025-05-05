import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class BraTSDataset(Dataset):
    def __init__(self, data_dir, slice_step=5, max_patients=None, random_selection=False, random_seed=42):
        self.samples = []

        all_patients = [p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))]

        if random_selection:
            random.seed(random_seed)
            patients = random.sample(all_patients, k=min(max_patients or len(all_patients), len(all_patients)))
        else:
            patients = sorted(all_patients)[:max_patients]

        for patient in patients:
            patient_dir = os.path.join(data_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            flair_path = os.path.join(patient_dir, f"{patient}_flair.nii.gz")
            seg_path = os.path.join(patient_dir, f"{patient}_seg.nii.gz")

            try:
                flair = nib.load(flair_path).get_fdata()
                seg = nib.load(seg_path).get_fdata()
            except Exception as e:
                print(f"Loading Error {patient}: {e}")
                continue

            seg[seg == 4] = 3

            for z in range(0, flair.shape[2], slice_step):
                flair_slice = flair[:, :, z]
                seg_slice = seg[:, :, z]

                flair_slice = (flair_slice - np.mean(flair_slice)) / (np.std(flair_slice) + 1e-5)

                self.samples.append((
                    torch.tensor(flair_slice, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(seg_slice, dtype=torch.long)
                ))

        print(f"Loading Done. Total slices : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
