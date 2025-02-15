import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# For scaling and PCA in embedding analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import umap.umap_ as umap

#############################################
# 1) CSV Data Loading & Merging
#############################################
def load_data():
    def load_file(file_path):
        try:
            df = pd.read_csv(file_path)
            df.rename(columns=lambda x: x.strip(), inplace=True)
            return df
        except FileNotFoundError:
            print(f"⚠️ File not found: {file_path}")
            return pd.DataFrame()

    # Update paths as needed
    microbiome = load_file('/home/tmnthc/New/Microbiome.csv')
    blood_metabolites = load_file('/home/tmnthc/New/Blood_Metabolites.csv')
    inflammatory_markers = load_file('/home/tmnthc/New/Sirolimus_inflammatory_markers.csv')
    blood_data = load_file('/home/tmnthc/New/Sirolimus_Blood_Data.csv')
    other_data = load_file('/home/tmnthc/New/Other.csv')
    brain_cbf_imaging = load_file('/home/tmnthc/New/Brain_CBF_Imaging.csv')

    for name, df in zip(
        ["Microbiome", "Blood Metabolites", "Inflammatory Markers", "Blood Data", "Other", "Brain CBF Imaging"],
        [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data, brain_cbf_imaging]
    ):
        if df.empty:
            print(f"⚠️ Warning: {name} data is empty or missing.")
        else:
            print(f"{name} data loaded with shape: {df.shape}")

    # Drop APOE4 if it exists
    if 'APOE4' in other_data.columns:
        other_data.drop(columns=['APOE4'], inplace=True)

    # Make sure 'Patient_ID' and 'Timepoint' are string type
    for df in [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data, brain_cbf_imaging]:
        if 'Patient_ID' in df.columns:
            df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
        if 'Timepoint' in df.columns:
            df['Timepoint'] = df['Timepoint'].astype(str).str.strip()

    # Convert numeric columns, fill NAs with zero
    def force_numeric(df):
        numeric_cols = [col for col in df.columns if col not in ['Patient_ID', 'Timepoint']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.fillna(0)
    
    microbiome = force_numeric(microbiome)
    blood_metabolites = force_numeric(blood_metabolites)
    inflammatory_markers = force_numeric(inflammatory_markers)
    blood_data = force_numeric(blood_data)
    other_data = force_numeric(other_data)
    brain_cbf_imaging = force_numeric(brain_cbf_imaging)

    # Prefix columns so that they don't overlap
    def add_prefix(df, prefix):
        df = df.copy()
        cols = [col for col in df.columns if col not in ['Patient_ID', 'Timepoint']]
        df.rename(columns={col: f"{prefix}_{col}" for col in cols}, inplace=True)
        return df

    microbiome_numeric = add_prefix(microbiome, "Microbiome")
    blood_metabolites_numeric = add_prefix(blood_metabolites, "Biomarker")
    inflammatory_markers_numeric = add_prefix(inflammatory_markers, "Inflammatory")
    blood_data_numeric = add_prefix(blood_data, "Biomarker")
    other_numeric = add_prefix(other_data, "Other")
    brain_cbf_numeric = add_prefix(brain_cbf_imaging, "CBF")

    # Ensure we keep [Patient_ID, Timepoint] columns
    for df in [microbiome_numeric, blood_metabolites_numeric, inflammatory_markers_numeric,
               blood_data_numeric, other_numeric, brain_cbf_numeric]:
        df[['Patient_ID', 'Timepoint']] = df[['Patient_ID', 'Timepoint']]

    # Merge everything into a single DataFrame
    data = other_numeric.copy()
    for df in [microbiome_numeric, blood_metabolites_numeric, inflammatory_markers_numeric,
               blood_data_numeric, brain_cbf_numeric]:
        data = data.merge(df, on=['Patient_ID', 'Timepoint'], how='outer')
    data.fillna(0, inplace=True)
    print(f"Final merged data shape: {data.shape}")
    return data

#############################################
# 2) MRI Data Loading
#############################################
def load_mri_data(root_dir):
    """
    Scans a directory structure:
      root_dir/
         patient_id/
            timepoint_name.nii
    and loads each .nii file as a Torch tensor.
    Returns a dict with keys (patient_id, timepoint_name).
    """
    mri_dict = {}
    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        if os.path.isdir(patient_path):
            for timepoint_file in os.listdir(patient_path):
                if timepoint_file.endswith('.nii'):
                    timepoint_name = timepoint_file.split('.')[0]
                    timepoint_path = os.path.join(patient_path, timepoint_file)
                    mri_image = nib.load(timepoint_path).get_fdata()
                    if len(mri_image.shape) == 4:
                        # If 4D, average across the last dimension
                        mri_image = np.mean(mri_image, axis=-1)
                    mri_tensor = torch.tensor(mri_image, dtype=torch.float32).unsqueeze(0)
                    # Store in dict
                    mri_dict[(patient_id, timepoint_name)] = mri_tensor
    return mri_dict

#############################################
# 3) MRIClipEncoder
#############################################
class MRIClipEncoder(nn.Module):
    """
    MRI Encoder: Extracts a middle 2D slice from a 3D MRI, converts it to a PIL image,
    processes it through CLIP, and projects the resulting 512-d embedding to embed_dim.
    """
    def __init__(self, embed_dim=32):
        super().__init__()
        # Load CLIP model & processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Projection MLP from 512 -> embed_dim
        self.project = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, mri_batch):
        embeddings = []
        B = mri_batch.size(0)

        for i in range(B):
            volume_3d = mri_batch[i, 0]  # shape: (D, H, W)
            D = volume_3d.shape[0]
            mid_slice_idx = D // 2
            slice_2d = volume_3d[mid_slice_idx]

            # Convert to numpy
            np_slice = slice_2d.cpu().numpy()

            # Instead of np_slice.ptp():
            # either use np.ptp(np_slice) or max - min
            rng = np_slice.max() - np_slice.min()
            # rng = np.ptp(np_slice)  # alternative

            np_slice = (np_slice - np_slice.min()) / (rng + 1e-8)
            np_slice = (np_slice * 255).astype(np.uint8)

            # Convert to RGB PIL
            pil_img = Image.fromarray(np_slice, mode='L').convert("RGB")

            # The device for the input
            device = mri_batch.device
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Extract CLIP embeddings
            with torch.no_grad():
                img_feats = self.clip_model.get_image_features(**inputs)
                img_feats = F.normalize(img_feats, p=2, dim=-1)

            # Project
            out = self.project(img_feats)
            out = F.normalize(out, p=2, dim=-1)
            embeddings.append(out)

        # Stack into (B, embed_dim)
        return torch.cat(embeddings, dim=0)

#############################################
# 4) MLPEncoder
#############################################
class MLPEncoder(nn.Module):
    """
    Simple MLP Encoder for tabular data.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        # x: (B, input_dim)
        out = self.encoder(x)
        return F.normalize(out, p=2, dim=-1)

#############################################
# 5) CombinedContrastiveDataset
#############################################
class CombinedContrastiveDataset(Dataset):
    """
    Explicitly creates combined samples for contrastive learning.
    For each row, a positive sample is (i, i, i, i, i). Negative samples
    are random (i1, i2, i3, i4, i5) with at least 2 distinct indices.
    """
    def __init__(self, data, mri_dict, negative_sample_fraction=1.0, positive_repeat=100):
        self.data = data.reset_index(drop=True)
        self.mri_dict = mri_dict
        self.N = len(self.data)
        self.cbf_data = data.filter(like='CBF').values

        # Positive samples
        self.positive_samples = [(i, i, i, i, i) for i in range(self.N) for _ in range(positive_repeat)]

        # Negative samples
        num_negatives = int(self.N * negative_sample_fraction)
        negative_samples = []
        while len(negative_samples) < num_negatives:
            sample = [random.choice(range(self.N)) for _ in range(5)]
            if len(set(sample)) > 1:
                negative_samples.append(tuple(sample))

        self.negative_samples = negative_samples

        # Label = 1 for positive, 0 for negative
        self.samples = [(s, 1) for s in self.positive_samples] + [(s, 0) for s in self.negative_samples]
        print(f"Total combined samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        indices, label = self.samples[index]
        i, j, k, l, m = indices
        patient_id = str(self.data.loc[i, 'Patient_ID'])
        timepoint = str(self.data.loc[i, 'Timepoint'])

        # Get MRI data
        mri_tensor = self.mri_dict.get((patient_id, timepoint),
                                       torch.zeros((1,128,128,128), dtype=torch.float32))

        micro_tensor = torch.tensor(self.data.filter(like='Microbiome_').iloc[j].values.astype(np.float32))
        biom_tensor  = torch.tensor(self.data.filter(like='Biomarker_').iloc[k].values.astype(np.float32))
        other_tensor = torch.tensor(self.data.filter(like='Other_').iloc[l].values.astype(np.float32))
        cbf_tensor   = torch.tensor(self.cbf_data[m].astype(np.float32))

        return {
            "mri": mri_tensor,
            "micro": micro_tensor,
            "biom":  biom_tensor,
            "other": other_tensor,
            "cbf":   cbf_tensor,
            "patient_ids": torch.tensor(int(patient_id), dtype=torch.long),
        }

#############################################
# 6) Patient Contrastive Loss
#############################################
def patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, e_cbf, patient_ids, tau=0.1):
    """
    Stacks embeddings from all 5 modalities. We treat embeddings from the same patient
    as positives. Using an InfoNCE-like approach but grouping 5 embeddings at a time.
    """
    B, D = e_mri.shape
    # (5B, D)
    all_embeddings = torch.cat([e_mri, e_micro, e_biom, e_other, e_cbf], dim=0)

    # Similarities
    sim_matrix = torch.matmul(all_embeddings, all_embeddings.t()) / tau

    # Mark positives: same patient ID
    repeated_ids = patient_ids.repeat(5)  # shape: (5B,)
    diag_mask = torch.eye(5*B, dtype=torch.bool, device=sim_matrix.device)
    positive_mask = (repeated_ids.unsqueeze(0) == repeated_ids.unsqueeze(1)) & (~diag_mask)

    sim_exp = torch.exp(sim_matrix)
    sum_all = sim_exp.sum(dim=1)
    sum_pos = (sim_exp * positive_mask).sum(dim=1)
    eps = 1e-8

    loss = -torch.log((sum_pos + eps) / (sum_all + eps))
    return loss.mean()

#############################################
# 7) MultiModalEmbeddingModel
#############################################
class MultiModalEmbeddingModel(nn.Module):
    """
    Encodes each modality into an embedding of size embed_dim
    and returns (e_mri, e_micro, e_biom, e_other, e_cbf).
    """
    def __init__(self, micro_dim, biom_dim, other_dim, cbf_dim, embed_dim=32):
        super(MultiModalEmbeddingModel, self).__init__()
        # No 'device=' argument here
        self.mri_encoder   = MRIClipEncoder(embed_dim=embed_dim)
        self.micro_encoder = MLPEncoder(micro_dim, embed_dim)
        self.biom_encoder  = MLPEncoder(biom_dim, embed_dim)
        self.other_encoder = MLPEncoder(other_dim, embed_dim)
        self.cbf_encoder   = MLPEncoder(cbf_dim, embed_dim)

    def forward(self, mri, micro, biom, other, cbf):
        e_mri   = self.mri_encoder(mri)
        e_micro = self.micro_encoder(micro)
        e_biom  = self.biom_encoder(biom)
        e_other = self.other_encoder(other)
        e_cbf   = self.cbf_encoder(cbf)
        return e_mri, e_micro, e_biom, e_other, e_cbf

#############################################
# 8) Training & Validation Loops
#############################################
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        mri   = batch["mri"].to(device)
        micro = batch["micro"].to(device)
        biom  = batch["biom"].to(device)
        other = batch["other"].to(device)
        cbf   = batch["cbf"].to(device)
        pids  = batch["patient_ids"].to(device)

        optimizer.zero_grad()
        e_mri, e_micro, e_biom, e_other, e_cbf = model(mri, micro, biom, other, cbf)
        loss = patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, e_cbf, pids, tau=0.1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            mri   = batch["mri"].to(device)
            micro = batch["micro"].to(device)
            biom  = batch["biom"].to(device)
            other = batch["other"].to(device)
            cbf   = batch["cbf"].to(device)
            pids  = batch["patient_ids"].to(device)

            e_mri, e_micro, e_biom, e_other, e_cbf = model(mri, micro, biom, other, cbf)
            loss = patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, e_cbf, pids, tau=0.1)
            running_loss += loss.item()

    return running_loss / len(dataloader)

#############################################
# 9) Plotting Helpers
#############################################
def plot_metrics(train_losses, val_losses, epochs):
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved loss plot as loss.png")

#############################################
# 10) Embedding Analysis Helpers
#############################################
def knn_accuracy(embeddings, labels, k=1):
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(embeddings, labels)
    preds = knn.predict(embeddings)
    return (preds == labels).mean()

def tsne_visualization(embeddings, group_labels, full_labels, title="t-SNE Visualization"):
    tsne = TSNE(n_components=2, perplexity=20, learning_rate=500, n_iter=1500, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    unique_groups = np.unique(group_labels)
    cmap = plt.get_cmap('tab10', len(unique_groups))
    group_color = {group: cmap(i) for i, group in enumerate(unique_groups)}

    # Potential markers by modality
    marker_dict = {"mri": "o", "micro": "s", "biom": "D", "other": "^", "cbf": "v"}
    plt.figure(figsize=(10,8))
    plotted = {}
    for i, (x, y) in enumerate(tsne_embeddings):
        group = group_labels[i]
        # last part of label is the modality
        modality = full_labels[i].split('_')[-1].lower()
        marker = marker_dict.get(modality, 'o')
        legend_label = f"{group}_{modality}"
        if legend_label not in plotted:
            plt.scatter(x, y, color=group_color[group], marker=marker, s=80, label=legend_label)
            plotted[legend_label] = True
        else:
            plt.scatter(x, y, color=group_color[group], marker=marker, s=80)

    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig("tsne_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved t-SNE plot as tsne_plot.png")

def umap_visualization(embeddings, group_labels, full_labels, title="UMAP Visualization"):
    reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.05, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)

    unique_groups = np.unique(group_labels)
    cmap = plt.get_cmap('tab10', len(unique_groups))
    group_color = {group: cmap(i) for i, group in enumerate(unique_groups)}

    marker_dict = {"mri": "o", "micro": "s", "biom": "D", "other": "^", "cbf": "v"}
    plt.figure(figsize=(10,8))
    plotted = {}
    for i, (x, y) in enumerate(umap_embeddings):
        group = group_labels[i]
        modality = full_labels[i].split('_')[-1].lower()
        marker = marker_dict.get(modality, 'o')
        legend_label = f"{group}_{modality}"
        if legend_label not in plotted:
            plt.scatter(x, y, color=group_color[group], marker=marker, s=80, label=legend_label)
            plotted[legend_label] = True
        else:
            plt.scatter(x, y, color=group_color[group], marker=marker, s=80)

    plt.title(title)
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig("umap_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved UMAP plot as umap_plot.png")

def extract_individual_embeddings(model, dataloader, device):
    """
    Iterates through a DataLoader of single-modality items
    and returns arrays of (embeddings, group_labels, full_labels).
    """
    model.eval()
    embeddings = []
    group_labels = []
    full_labels = []
    with torch.no_grad():
        for batch in dataloader:
            data_list = batch["data"]
            group_list = batch["group_label"]
            full_list = batch["modality_label"]
            for i, x in enumerate(data_list):
                # Move x to device
                x = x.to(device)
                modality = full_list[i].split('_')[-1].lower()  # e.g., "mri"

                # Get appropriate encoder
                if modality == "mri":
                    emb = model.mri_encoder(x.unsqueeze(0))
                elif modality == "micro":
                    emb = model.micro_encoder(x.unsqueeze(0))
                elif modality == "biom":
                    emb = model.biom_encoder(x.unsqueeze(0))
                elif modality == "other":
                    emb = model.other_encoder(x.unsqueeze(0))
                elif modality == "cbf":
                    emb = model.cbf_encoder(x.unsqueeze(0))
                else:
                    raise ValueError(f"Unknown modality: {modality}")

                embeddings.append(emb.cpu().numpy())
                group_labels.append(group_list[i])
                full_labels.append(full_list[i])

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, group_labels, full_labels

#############################################
# 11) Main Training & Evaluation Script
#############################################
def main():
    data = load_data()
    mri_dict = load_mri_data("/home/tmnthc/CBF_imaging")

    # Example: train on first 16 patients, validate on next 3
    all_patients = np.unique(data['Patient_ID'])
    if len(all_patients) < 19:
        raise ValueError("Not enough patients. Need at least 19 patients.")
    train_patient_ids = all_patients[:16]
    val_patient_ids   = all_patients[16:19]
    print("Training Patient IDs:", train_patient_ids)
    print("Validation Patient IDs:", val_patient_ids)

    train_data = data[data['Patient_ID'].isin(train_patient_ids)]
    val_data   = data[data['Patient_ID'].isin(val_patient_ids)]

    # For each patient, keep only the first 3 timepoints
    train_data = train_data.sort_values(['Patient_ID', 'Timepoint']).groupby('Patient_ID').head(3).reset_index(drop=True)
    val_data   = val_data.sort_values(['Patient_ID', 'Timepoint']).groupby('Patient_ID').head(3).reset_index(drop=True)
    print(f"Training data rows: {len(train_data)} (expected 16*3 = 48)")
    print(f"Validation data rows: {len(val_data)} (expected 3*3 = 9)")

    # Create Datasets
    train_dataset = CombinedContrastiveDataset(train_data, mri_dict, negative_sample_fraction=1.0, positive_repeat=100)
    val_dataset   = CombinedContrastiveDataset(val_data,   mri_dict, negative_sample_fraction=1.0, positive_repeat=100)
    print(f"Number of training combined samples: {len(train_dataset)}")
    print(f"Number of validation combined samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=3, shuffle=False)

    # Identify feature dims
    micro_dim = train_data.filter(like='Microbiome_').shape[1]
    biom_dim  = train_data.filter(like='Biomarker_').shape[1]
    other_dim = train_data.filter(like='Other_').shape[1]
    cbf_dim   = train_data.filter(like='CBF_').shape[1]
    print(f"Microbiome feature dim: {micro_dim}")
    print(f"Biomarker feature dim:  {biom_dim}")
    print(f"Other feature dim:      {other_dim}")
    print(f"CBF Imaging feature dim:{cbf_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalEmbeddingModel(micro_dim, biom_dim, other_dim, cbf_dim, embed_dim=32)
    model = model.to(device)  # Move entire model (including submodules) to GPU if available

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 80
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, device)
        v_loss = validate_epoch(model, val_loader, device)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

    plot_metrics(train_losses, val_losses, epochs)

    # Save model
    torch.save(model.state_dict(), "model_checkpoint.pth")
    print("Model checkpoint saved as model_checkpoint.pth")

    ##################################################
    #  Embedding Analysis on Validation Set (Positive Only)
    ##################################################
    class PositiveIndividualModalityDataset(Dataset):
        """
        Returns each modality from each row. This helps us visualize embeddings individually.
        """
        def __init__(self, data, mri_dict):
            self.data = data.reset_index(drop=True)
            self.mri_dict = mri_dict
            # We'll handle 5 modalities: "mri", "micro", "biom", "other", "cbf"
            self.modalities = ["mri", "micro", "biom", "other", "cbf"]

        def __len__(self):
            return len(self.data) * len(self.modalities)

        def __getitem__(self, idx):
            row_idx = idx // len(self.modalities)
            modality_idx = idx % len(self.modalities)
            modality = self.modalities[modality_idx]

            patient_id = str(self.data.loc[row_idx, 'Patient_ID'])
            timepoint  = str(self.data.loc[row_idx, 'Timepoint'])
            group_label = f"{patient_id}_{timepoint}"

            if modality == "mri":
                data_point = self.mri_dict.get((patient_id, timepoint),
                                               torch.zeros((1,128,128,128), dtype=torch.float32))
            elif modality == "micro":
                data_point = torch.tensor(self.data.filter(like='Microbiome_').iloc[row_idx].values.astype(np.float32))
            elif modality == "biom":
                data_point = torch.tensor(self.data.filter(like='Biomarker_').iloc[row_idx].values.astype(np.float32))
            elif modality == "other":
                data_point = torch.tensor(self.data.filter(like='Other_').iloc[row_idx].values.astype(np.float32))
            elif modality == "cbf":
                data_point = torch.tensor(self.data.filter(like='CBF_').iloc[row_idx].values.astype(np.float32))
            else:
                raise ValueError(f"Unknown modality: {modality}")

            full_label = f"{group_label}_{modality}"
            return data_point, group_label, full_label

    def eval_collate_fn(batch):
        return {
            "data": [b[0] for b in batch],
            "group_label": [b[1] for b in batch],
            "modality_label": [b[2] for b in batch],
        }

    # Create a dataset of positive individual samples
    eval_dataset = PositiveIndividualModalityDataset(val_data, mri_dict)
    eval_loader  = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=eval_collate_fn)

    # Extract embeddings
    embeddings, group_labels, full_labels = extract_individual_embeddings(model, eval_loader, device)

    # Scale + (Optionally) PCA
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    if embeddings_scaled.shape[0] >= 50:
        pca = PCA(n_components=50, whiten=True)
        reduced_embeddings = pca.fit_transform(embeddings_scaled)
    else:
        reduced_embeddings = embeddings_scaled

    # Normalize row-wise
    normalized_embeddings = reduced_embeddings / np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)

    # KNN accuracy with k=1
    knn_acc = knn_accuracy(normalized_embeddings, group_labels, k=1)
    print(f"KNN Accuracy (k=1) on validation embedding analysis: {knn_acc * 100:.2f}%")

    # Visualizations
    tsne_visualization(normalized_embeddings, group_labels, full_labels,
                       title="t-SNE Visualization of Validation Embeddings")
    umap_visualization(normalized_embeddings, group_labels, full_labels,
                       title="UMAP Visualization of Validation Embeddings")

if __name__ == "__main__":
    main()
