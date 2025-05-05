# Brain Tumor Segmentation with U-Net (BraTS 2021)

This project provides a PyTorch implementation of a 2D U-Net for brain tumor segmentation using the BraTS 2021 dataset (FLAIR modality only).

## Dataset

You must manually download the BraTS 2021 training dataset from the official website:

https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1?select=BraTS2021_Training_Data.tar

After downloading, move the file `BraTS2021_Training_Data.tar` into the `TrainingData/` directory.

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/Kubixcube/PROJ-H402.git
cd PROJ-H402/
```

### 2. Create and activate a Conda environment (recommended)

```bash
conda create -n brainseg python=3.10 -y
conda activate brainseg
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Extract the BraTS data

Make sure `BraTS2021_Training_Data.tar` is placed in `TrainingData/`, then run:

```bash
tar -xvf TrainingData/BraTS2021_Training_Data.tar -C TrainingData
```

You should now see folders like `BraTS2021_00000/`, `BraTS2021_00001/`, etc.

## Running the Project

Once everything is ready, start training:

```bash
python main.py
```

Predictions and evaluation results will be saved in the `results/` folder.
