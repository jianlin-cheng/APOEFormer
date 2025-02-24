# ADFormer
A multi-modal transformer model for studying Alzheimer's disease

---

## 1. Overview

This repository demonstrates a two-stage approach for combining **MRI** data with numeric (tabular) features (e.g., microbiome, blood biomarkers, clinical data) to predict a binary outcome (APOE4 status). The pipeline includes:

1. **Data Preprocessing:** Handling missing values (either via group averaging or dropping) and applying z-score normalization.
2. **Contrastive Pretraining:** Learning a shared embedding space for MRI + numeric features using a custom contrastive loss.
3. **Attention-based Classification:** Training a multi-head attention model (with an optional ensemble) to predict APOE4.

---

## 2. Data Loading and Preprocessing

- We have multiple CSVs: *Microbiome.csv*, *Blood_Metabolites.csv*, *Sirolimus_inflammatory_markers.csv*, *Sirolimus_Blood_Data.csv*, and *Other.csv*.
- Each file has rows indexed by **(Patient_ID, Timepoint)**.
- We merge them on **(Patient_ID, Timepoint)** to form a single DataFrame called **data**.

### Handling Missing Values:

- If only a few data points are missing in a column, we fill them via a mean or group-average imputation.
- If too many values are missing, we drop that column or those rows entirely to avoid introducing too many artificial values.

### Z-score Normalization:

- After handling missing data, we compute the mean and standard deviation for each numeric feature and apply:

  ```
  X_norm = (X - μ) / σ
  ```

- This ensures features have mean 0 and standard deviation 1, which often stabilizes MLP training.

---

## 3. MRI Preprocessing

- We store MRI volumes in **.nii** files, each representing **(patient_id, timepoint)**.
- If the data is 4D, we average the last dimension to get a 3D volume.
- In the code, each volume is stored as a PyTorch tensor in a dictionary:

  ```
  mri_dict[(patient_id, timepoint)] = (1, D, H, W) tensor
  ```

### Slicing for CLIP:

1. We pick a few 2D slices (center ±1) from the 3D volume, each slice is 128x128.
2. We normalize each slice to the range [0..255] and convert it to a PIL image (RGB).
3. We feed the 2D image into CLIP, which is a Vision Transformer that internally splits the image into patches.
4. CLIP outputs a 512-dimensional feature which we project to our desired embedding dimension (e.g., 64).
5. If multiple slices are taken, we average their embeddings to get one final vector for the entire MRI volume.

---

## 4. Data Splitting

- **Pretraining Split:** We need at least 19 patients, randomly selecting 16 for training and 3 for validation. We take up to 3 timepoints per patient.
- **Classification Split:** We need at least 23 patients, dividing them into 16 for training, 3 for validation, and 4 for testing. We use embeddings from the encoders for classification.

---

## 5. Contrastive Pretraining

- **CombinedContrastiveDataset:**
  - Forms positive samples (the same row repeated for MRI, Microbiome, Biomarker, Other) and negative samples (random mismatches).
  - Each sample yields four embeddings:

    ```
    e_MRI, e_Micro, e_Biom, e_Other
    ```

- **NT-Xent Loss:**
  - Embeddings that truly match (same patient/timepoint) are treated as positives, and all others are treated as negatives.
  - Using a temperature parameter, we push positives closer together and negatives farther apart in the embedding space.

- **Progressive Unfreezing:**
  - We initially unfreeze only the last two layers of CLIP.
  - If validation loss plateaus, we unfreeze more layers in increments of two, up to the entire model if needed.

---

## 6. Attention Classification

- **Embedding Extraction:**
  - We load the pretrained MRI and numeric encoders.
  - For each patient, we generate embeddings for each timepoint and either:
    1. **Average** across timepoints to form a single [4, embed_dim] sample per patient.
    2. **Treat each timepoint as separate tokens** (resulting in [T x 4, embed_dim] if T is the number of timepoints).

- **Stacked Multi-Head Attention Classifier:**
  - Takes input of shape [B, seq_len, embed_dim].
  - Applies multiple layers of multi-head self-attention, then merges the original and final embeddings with a skip connection.
  - Outputs a single logit for binary classification (APOE4 or not).

- **Ensemble:**
  - We train multiple attention models (each with a different seed).
  - At test time, we average their logits before thresholding at 0 to produce the final prediction.

---

## 7. Step-by-Step Training Procedure

1. **Preprocess the CSVs:** Fix missing data (imputation or dropping) and apply z-score normalization.
2. **Load MRI volumes:** Possibly 128x128x128 volumes (or 4D averaged to 3D).
3. **Split for pretraining:** 16 patients for training, 3 for validation.
4. **Contrastive training:** Up to 100 epochs or until no improvement; possibly unfreeze more CLIP layers.
5. **Save pretrained checkpoint.**
6. **Classification split:** Pick 23 patients (16 train, 3 val, 4 test).
7. **Train attention classifier:** Either by averaging timepoints or treating them as separate tokens.
8. **Ensemble and test evaluation:** Evaluate model performance using ensemble predictions.

---

## 8. Running the Code

- **Dependencies:**
  - `pytorch`, `torchvision`, `transformers` (for CLIP), `nibabel` (for NIfTI), `pandas`, `numpy`, etc.

- **Adjust Paths:**
  - In `load_data()` and `load_mri_data()`, ensure your CSV and MRI directories are correct.

- **Command:**

  ```bash
  python3 your_script.py
  ```

This script handles:
- Merging CSV data.
- Loading MRI volumes.
- Pretraining (contrastive).
- Saving a checkpoint.
- Attention-based classification.
- Printing final test accuracy.

---

## 9. Conclusion

This pipeline covers:
- Missing values (imputation or dropping).
- Z-score normalization of numeric features.
- Slicing 3D MRI volumes for CLIP-based feature extraction.
- Contrastive pretraining to align MRI + numeric features.
- Attention classification (optionally across multiple timepoints).

---
## 10. Embedding analysis for pretraining (Optional)

- **Dimensionality Reduction:**
  - Uses **t-SNE** for visualization.
  - Helps in understanding how well MRI and numeric features align in the learned embedding space.

- **Embedding Evaluation:**
  - Evaluates embeddings using **KNN**.
  - Measures classification accuracy to assess the quality of the learned representations.

- **Command:**

  ```bash
  python3 pre_train_and_embedding.py
  ```
