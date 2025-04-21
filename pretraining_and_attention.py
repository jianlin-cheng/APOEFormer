#!/usr/bin/env python3

import random
import numpy as np
import torch

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T
import cv2  # for resizing heatmaps
import shap  # SHAP

###########################################
# Helper: Plot Loss Metrics
###########################################
def plot_metrics(train_losses, val_losses, total_epochs, filename="loss_plot.png"):
    epochs = range(1, total_epochs+1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved loss plot as {filename}")

###########################################
# 1) CSV Data Loading & Merging
###########################################
def load_file(file_path, drop_apoe4=True):
    try:
        df = pd.read_csv(file_path)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        if drop_apoe4 and 'APOE4' in df.columns:
            df.drop(columns=['APOE4'], inplace=True)
        return df
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_path}")
        return pd.DataFrame()

def load_data():
    # Load existing CSV files.
    microbiome = load_file('/home/tmnthc/tom/anaconda3/New1/Microbiome.csv')
    blood_metabolites = load_file('/home/tmnthc/tom/anaconda3/New1/Blood_Metabolites.csv')
    inflammatory_markers = load_file('/home/tmnthc/tom/anaconda3/New1/Sirolimus_inflammatory_markers.csv')
    blood_data = load_file('/home/tmnthc/tom/anaconda3/New1/Sirolimus_Blood_Data.csv')
    other_data = load_file('/home/tmnthc/tom/anaconda3/New1/Other.csv', drop_apoe4=True)
    # Load the new numeric modality.
    brain_cbf = load_file('/home/tmnthc/tom/anaconda3//New1/Brain_CBF_Imaging.csv', drop_apoe4=False)

    for name, df in zip(
        ["Microbiome", "Blood Metabolites", "Inflammatory Markers", "Blood Data", "Other", "Brain_CBF"],
        [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data, brain_cbf]
    ):
        if df.empty:
            print(f"⚠️ Warning: {name} data is empty or missing.")
        else:
            print(f"{name} data loaded with shape: {df.shape}")

    # Standardize Patient_ID and Timepoint columns.
    for df in [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data, brain_cbf]:
        if 'Patient_ID' in df.columns:
            df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
        if 'Timepoint' in df.columns:
            df['Timepoint'] = df['Timepoint'].astype(str).str.strip()

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
    brain_cbf = force_numeric(brain_cbf)

    def add_prefix(df, prefix):
        df = df.copy()
        cols = [c for c in df.columns if c not in ['Patient_ID','Timepoint']]
        df.rename(columns={c: f"{prefix}_{c}" for c in cols}, inplace=True)
        return df

    micro_num  = add_prefix(microbiome, "Microbiome")
    blood_met_num  = add_prefix(blood_metabolites, "Biomarker")
    blood_data_num = add_prefix(blood_data, "Biomarker")
    inflam_num = add_prefix(inflammatory_markers, "Biomarker")
    other_num  = add_prefix(other_data, "Other")
    cbf_num = add_prefix(brain_cbf, "mri_numeric")  # New modality

    biomarker_data = blood_met_num.merge(blood_data_num, on=['Patient_ID', 'Timepoint'], how='outer')
    biomarker_data = biomarker_data.merge(inflam_num, on=['Patient_ID', 'Timepoint'], how='outer')

    # Merge all data together.
    data = other_num.copy()
    for df2 in [micro_num, biomarker_data, cbf_num]:
        data = data.merge(df2, on=['Patient_ID','Timepoint'], how='outer')
    data.fillna(0, inplace=True)
    print(f"Final merged data shape: {data.shape}")
    return data

###########################################
# 2) MRI Data Loading (for brain imaging)
###########################################
def load_mri_data(root_dir):
    mri_dict = {}
    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        if os.path.isdir(patient_path):
            for tfile in os.listdir(patient_path):
                if tfile.endswith('.nii'):
                    timepoint_name = tfile.split('.')[0]
                    full_path = os.path.join(patient_path, tfile)
                    mri_image = nib.load(full_path).get_fdata()
                    # Do not average slices—if image is 4D with extra singleton dimension, squeeze it.
                    if len(mri_image.shape) == 4 and mri_image.shape[-1] == 1:
                        mri_image = np.squeeze(mri_image, axis=-1)
                    # Convert to torch tensor and add batch dimension.

                      # Normalization: subtract the mean and divide by the standard deviation.
                    mean_val = np.mean(mri_image)
                    std_val = np.std(mri_image)
                    if std_val != 0:
                        mri_image = (mri_image - mean_val) / std_val
                    else:
                        mri_image = mri_image - mean_val
                        
                    mri_tensor = torch.tensor(mri_image, dtype=torch.float32).unsqueeze(0)
                    mri_dict[(patient_id, timepoint_name)] = mri_tensor
    return mri_dict

###########################################
# Safe normalization and Augmentation Functions
###########################################
def safe_normalize(x, p=2, dim=-1, eps=1e-12):
    norm = x.norm(p, dim=dim, keepdim=True)
    return x / (norm + eps)

def advanced_image_augmentation():
    return T.Compose([
        T.RandomResizedCrop(128, scale=(0.8, 1.0)),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15)
    ])

def augment_numeric(x, noise_std=0.05):
    noise = torch.randn_like(x) * noise_std
    return x + noise

###########################################
# 3) MRIClipEncoder
###########################################
class MRIClipEncoder(nn.Module):
    def __init__(self, embed_dim=64, augment=False, dropout_p=0.1):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.unfreeze_layers = 2
        self._unfreeze_last_n_layers(self.unfreeze_layers)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.project = nn.Sequential(
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.augment = augment
        if self.augment:
            self.augmentation = advanced_image_augmentation()

    def _unfreeze_last_n_layers(self, n):
        for layer in self.clip_model.vision_model.encoder.layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"Unfroze last {n} layers of CLIP vision encoder.")

    def gradually_unfreeze(self, current_patience, max_patience, increment=2, max_layers=12):
        if current_patience >= max_patience and self.unfreeze_layers < max_layers:
            self.unfreeze_layers = min(self.unfreeze_layers + increment, max_layers)
            self._unfreeze_last_n_layers(self.unfreeze_layers)

    def forward(self, mri_batch):
        """
        Expects mri_batch of shape (B, 1, H, W, D).
        Now we permute so that D (the original last axis) becomes axis 0,
        and then slice over that.
        """
        device = mri_batch.device
        B = mri_batch.size(0)
        all_proj = []
        for i in range(B):
            # get the volume: originally (H, W, D)
            vol = mri_batch[i, 0]

            # if it isn’t 3‑D, fallback
            if vol.ndim != 3:
                vol = torch.zeros((128, 128, 128), dtype=torch.float32, device=device)

            # permute so that the original last axis is now first:
            # (H, W, D) → (D, H, W)
            vol = vol.permute(2, 0, 1)

            D, H, W = vol.shape
            slice_features = []

            for idx in range(D):
                # now vol[idx] is vol[:, :, idx] of the original tensor
                slice_2d = vol[idx].cpu().numpy()
                # … your existing normalization, PIL conversion, CLIP call, projection, etc. …
                rng = slice_2d.max() - slice_2d.min()
                slice_2d = (slice_2d - slice_2d.min()) / (rng + 1e-8)
                slice_2d = (slice_2d * 255.0).astype(np.uint8)
                pil_img = Image.fromarray(slice_2d, mode='L').convert("RGB")
                if self.training and self.augment:
                    pil_img = self.augmentation(pil_img)
                inputs = self.processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    feat = self.clip_model.get_image_features(**inputs)
                feat = safe_normalize(feat, p=2, dim=-1)
                proj = self.project(feat)
                proj = safe_normalize(proj, p=2, dim=-1)
                slice_features.append(proj)

            volume_embeddings = torch.cat(slice_features, dim=0)  # (D, embed_dim)
            all_proj.append(volume_embeddings.unsqueeze(0))       # (1, D, embed_dim)

        return torch.cat(all_proj, dim=0)  # (B, D, embed_dim)


###########################################
# 4) MLPEncoder for Numeric Data
###########################################
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=64, augment=False, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.layernorm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.layernorm2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, output_dim)
        self.layernorm3 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.augment = augment

    def forward(self, x):
        out1 = self.layernorm1(self.fc1(x))
        act1 = F.relu(out1)
        act1 = self.dropout(act1)
        out2 = self.layernorm2(self.fc2(act1))
        act2 = F.relu(out2)
        act2 = self.dropout(act2)
        res = act1 + act2
        out3 = self.layernorm3(self.fc3(res))
        return safe_normalize(out3, p=2, dim=-1)

###########################################
# 5) CombinedContrastiveDataset
###########################################
class CombinedContrastiveDataset(Dataset):
    def __init__(self, data, mri_dict, negative_sample_fraction=1, positive_repeat=10, augment=False):
        self.data = data.reset_index(drop=True)
        self.mri_dict = mri_dict
        self.N = len(self.data)
        self.augment = augment
        # Create positive pairs (same sample repeated)
        self.positive_samples = [
            (i, i, i, i, i) for i in range(self.N) for _ in range(positive_repeat)
        ]
        # Create negative pairs (random mismatch)
        num_negatives = int(self.N * negative_sample_fraction)
        negative_samples = []
        while len(negative_samples) < num_negatives:
            sample = [random.choice(range(self.N)) for _ in range(5)]
            if len(set(sample)) > 1:
                negative_samples.append(tuple(sample))
        self.negative_samples = negative_samples

        # Label=1 for positives, label=0 for negatives
        self.samples = [(s, 1) for s in self.positive_samples] + [(s, 0) for s in self.negative_samples]
        print(f"Total combined samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        indices, label = self.samples[idx]
        i, j, k, l, m = indices

        pid_mri, tpt_mri = str(self.data.loc[i, "Patient_ID"]), str(self.data.loc[i, "Timepoint"])
        pid_micro, tpt_micro = str(self.data.loc[j, "Patient_ID"]), str(self.data.loc[j, "Timepoint"])
        pid_biom, tpt_biom = str(self.data.loc[k, "Patient_ID"]), str(self.data.loc[k, "Timepoint"])
        pid_other, tpt_other = str(self.data.loc[l, "Patient_ID"]), str(self.data.loc[l, "Timepoint"])
        pid_numeric, tpt_numeric = str(self.data.loc[m, "Patient_ID"]), str(self.data.loc[m, "Timepoint"])
    
        mri_tensor = self.mri_dict.get((pid_mri, tpt_mri), torch.zeros((1,128,128,128), dtype=torch.float32))
        micro_tensor = torch.tensor(self.data.filter(like="Microbiome_").iloc[j].values.astype(np.float32))
        biom_tensor = torch.tensor(self.data.filter(like="Biomarker_").iloc[k].values.astype(np.float32))
        other_tensor = torch.tensor(self.data.filter(like="Other_").iloc[l].values.astype(np.float32))
        mri_numeric_tensor = torch.tensor(self.data.filter(like="mri_numeric_").iloc[m].values.astype(np.float32))
    
        if self.augment and label == 1:
            micro_tensor = augment_numeric(micro_tensor)
            biom_tensor = augment_numeric(biom_tensor)
            other_tensor = augment_numeric(other_tensor)
            mri_numeric_tensor = augment_numeric(mri_numeric_tensor)
    
        sample_label = f"{pid_mri}_{tpt_mri}"
        return {
            "mri": mri_tensor,
            "micro": micro_tensor,
            "biom": biom_tensor,
            "other": other_tensor,
            "mri_numeric": mri_numeric_tensor,
            "sample_label": sample_label,
            "label": torch.tensor(label, dtype=torch.float32)
        }

def custom_collate(batch):
    collated = {}
    for key in batch[0]:
        if key == "sample_label":
            collated[key] = [d[key] for d in batch]
        elif key == "label":
            collated[key] = torch.stack([d[key] for d in batch])
        else:
            collated[key] = torch.stack([d[key] for d in batch])
    return collated

###########################################
# 6) Patient Contrastive Loss
###########################################
def patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, e_numeric, sample_labels, tau=0.5):
    """
    e_mri, e_micro, e_biom, e_other, e_numeric: (B, embed_dim)
    We combine them into a big matrix of shape (5*B, embed_dim).
    Then do typical contrastive cross-entropy.
    """
    B, D = e_mri.shape
    all_labels = sample_labels * 5
    unique_labels, inverse = np.unique(all_labels, return_inverse=True)
    labels = torch.tensor(inverse, device=e_mri.device)
    emb_all = torch.cat([e_mri, e_micro, e_biom, e_other, e_numeric], dim=0)
    sim_matrix = torch.matmul(emb_all, emb_all.t()) / tau
    diag_mask = torch.eye(5 * B, dtype=torch.bool, device=sim_matrix.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~diag_mask)
    neg_mask = ~pos_mask & (~diag_mask)
    sim_pos = sim_matrix * pos_mask.float()
    sim_neg = sim_matrix * neg_mask.float()
    eps = 1e-8
    sum_pos = torch.exp(sim_pos).sum(dim=1)
    sum_neg = torch.exp(sim_neg).sum(dim=1)
    loss = -torch.log((sum_pos + eps) / (sum_pos + sum_neg + eps))
    return loss.mean()

###########################################
# 7) MultiModalEmbeddingModel
###########################################
class MultiModalEmbeddingModel(nn.Module):
    def __init__(self, micro_dim, biom_dim, other_dim, numeric_dim, embed_dim=64, augment=False):
        super().__init__()
        self.mri_encoder = MRIClipEncoder(embed_dim=embed_dim, augment=augment)
        self.micro_encoder = MLPEncoder(input_dim=micro_dim, output_dim=embed_dim, augment=augment)
        self.biom_encoder = MLPEncoder(input_dim=biom_dim, output_dim=embed_dim, augment=augment)
        self.other_encoder = MLPEncoder(input_dim=other_dim, output_dim=embed_dim, augment=augment)
        self.numeric_encoder = MLPEncoder(input_dim=numeric_dim, output_dim=embed_dim, augment=augment)

    def forward(self, mri, micro, biom, other):
        e_mri = self.mri_encoder(mri)   # (B, D, embed_dim)
        # For non-MRI, we just do MLP on entire vector → (B, embed_dim)
        e_micro = self.micro_encoder(micro)
        e_biom  = self.biom_encoder(biom)
        e_other = self.other_encoder(other)
        return e_mri, e_micro, e_biom, e_other

###########################################
# 8) Training & Validation for Pretraining
###########################################
def train_epoch(model, loader, optimizer, device, epoch, val_loss_history, patience_threshold=5):
    model.train()
    total_loss = 0
    for batch in loader:
        mri = batch["mri"].to(device)
        micro = batch["micro"].to(device)
        biom = batch["biom"].to(device)
        other = batch["other"].to(device)
        mri_numeric = batch["mri_numeric"].to(device)
        sample_labels = batch["sample_label"]
        optimizer.zero_grad()
        # Forward
        e_mri, e_micro, e_biom, e_other = model(mri, micro, biom, other)
        e_numeric = model.numeric_encoder(mri_numeric)
        # We average over the slices for e_mri to get shape (B, embed_dim)
        # or we can also keep it separate, but let's do a simple mean here for contrastive:
        e_mri_mean = e_mri.mean(dim=1)
        loss = patient_contrastive_loss(e_mri_mean, e_micro, e_biom, e_other, e_numeric,
                                        sample_labels=sample_labels, tau=0.5)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)

    # This is where we can gradually unfreeze more CLIP layers if val loss is not improving
    if len(val_loss_history) > 0 and avg_loss >= max(val_loss_history[-patience_threshold:]):
        model.mri_encoder.gradually_unfreeze(
            current_patience=patience_threshold,
            max_patience=patience_threshold
        )
    return avg_loss

def validate_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            mri = batch["mri"].to(device)
            micro = batch["micro"].to(device)
            biom = batch["biom"].to(device)
            other = batch["other"].to(device)
            mri_numeric = batch["mri_numeric"].to(device)
            sample_labels = batch["sample_label"]
            e_mri, e_micro, e_biom, e_other = model(mri, micro, biom, other)
            e_numeric = model.numeric_encoder(mri_numeric)
            # Mean over slices for MRI
            e_mri_mean = e_mri.mean(dim=1)
            loss = patient_contrastive_loss(e_mri_mean, e_micro, e_biom, e_other, e_numeric,
                                            sample_labels=sample_labels, tau=0.5)
            total_loss += loss.item()
    return total_loss / len(loader)

###########################################
# 9) Stacked Multi-Head Attention Classifier
###########################################
class StackedAttentionClassifier(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True))
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.fuse = nn.Linear(embed_dim*2, embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, embeddings):
        """
        embeddings: (B, T, embed_dim)
        returns: (B, T) with a logit per token
        """
        residual = embeddings
        out = embeddings
        for layer in self.layers:
            attn_output, _ = layer(out, out, out)
            out = self.layernorm(out + self.dropout(attn_output))
        fused = torch.cat([residual, out], dim=-1)  # shape (B, T, 2*embed_dim)
        fused = F.relu(self.fuse(fused))            # shape (B, T, embed_dim)
        logits = self.classifier(fused)             # shape (B, T, 1)
        return logits.squeeze(-1)                   # shape (B, T)

###########################################
# 10) Dataset for Attention Classification
###########################################
class AttentionDatasetWithLabels(Dataset):
    """
    Creates a (T, embed_dim) embedding for each patient, 
    where T = 5 * (# MRI slices).
    Order of stacked embeddings: [MRI_slices, Micro_repeated, Biom_repeated, Other_repeated, Numeric_repeated]
    Then flatten to shape (T, embed_dim).
    """
    def __init__(self, patient_ids, data, mri_dict, model, device, patient_labels):
        self.samples = []
        self.device = device
        self.model = model
        for pid in patient_ids:
            patient_data = data[data["Patient_ID"] == pid]
            if len(patient_data) == 0:
                continue
            row = patient_data.iloc[0]
            mri_tensor = mri_dict.get((str(row["Patient_ID"]), str(row["Timepoint"])),
                                      torch.zeros((1,128,128,32), dtype=torch.float32))
            micro_tensor = torch.tensor(row.filter(like="Microbiome_").values.astype(np.float32))
            biom_tensor  = torch.tensor(row.filter(like="Biomarker_").values.astype(np.float32))
            other_tensor = torch.tensor(row.filter(like="Other_").values.astype(np.float32))
            mri_numeric_tensor = torch.tensor(row.filter(like="mri_numeric_").values.astype(np.float32))
    
            # Move to device and encode
            mri_tensor = mri_tensor.to(device).unsqueeze(0)  # (1, 1, H, W, D)
            micro_tensor = micro_tensor.to(device).unsqueeze(0)
            biom_tensor = biom_tensor.to(device).unsqueeze(0)
            other_tensor = other_tensor.to(device).unsqueeze(0)
            mri_numeric_tensor = mri_numeric_tensor.to(device).unsqueeze(0)
    
            with torch.no_grad():
                e_mri, e_micro, e_biom, e_other = model(mri_tensor, micro_tensor, biom_tensor, other_tensor)
                # e_mri: (1, D, 64)
                e_numeric = model.numeric_encoder(mri_numeric_tensor)  # (1, 64)
    
            e_mri = e_mri.squeeze(0)  # shape (D, 64)
            rep = e_mri.size(0)       # number of slices

            # Repeat each of the other embeddings D times so shapes match
            e_micro = e_micro.squeeze(0).unsqueeze(0).repeat(rep, 1)    # (D, 64)
            e_biom  = e_biom.squeeze(0).unsqueeze(0).repeat(rep, 1)     # (D, 64)
            e_other = e_other.squeeze(0).unsqueeze(0).repeat(rep, 1)    # (D, 64)
            e_numeric = e_numeric.squeeze(0).unsqueeze(0).repeat(rep, 1)# (D, 64)
    
            # Stack them: shape (5, D, 64)
            modalities = torch.stack([e_mri, e_micro, e_biom, e_other, e_numeric], dim=0)
            # Flatten first 2 dims -> (5*D, 64)
            sample_embedding = modalities.view(-1, modalities.size(-1))
    
            label = patient_labels.get(str(pid), 0)
            self.samples.append((sample_embedding, torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def attn_collate_with_labels(batch):
    embeddings_list, label_list = zip(*batch)
    embeddings = torch.stack(embeddings_list, dim=0)  # (B, 5*D, 64)
    labels = torch.stack(label_list, dim=0)           # (B,)
    return embeddings, labels

def main():
    import numpy as np
    import os
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    import shap

    num_runs = 20
    overall_accuracies = []
    test_accuracies = []
    shap_records = {mod: [] for mod in ["MRI", "Micro", "Biom", "Other", "Numeric"]}

    for run in range(1, num_runs + 1):
        print(f"\n\n===== RUN {run}/{num_runs} =====")

        # ---------------------
        # Pretraining Phase
        # ---------------------
        data = load_data()
        mri_dict = load_mri_data("/home/tmnthc/tom/anaconda3/T1")
        all_pats = np.unique(data["Patient_ID"])
        if len(all_pats) < 19:
            raise ValueError("Need at least 19 patients for pretraining.")
        selected_pats = np.random.choice(all_pats, size=19, replace=False)
        train_pats_pre = selected_pats[:16]
        val_pats_pre   = selected_pats[16:]
        print("Pretraining - Training Patient IDs:", train_pats_pre)
        print("Pretraining - Validation Patient IDs:", val_pats_pre)

        train_data = data[data["Patient_ID"].isin(train_pats_pre)]
        val_data   = data[data["Patient_ID"].isin(val_pats_pre)]
        train_data = train_data.sort_values(["Patient_ID", "Timepoint"]).groupby("Patient_ID").head(3).reset_index(drop=True)
        val_data   = val_data.sort_values(["Patient_ID", "Timepoint"]).groupby("Patient_ID").head(3).reset_index(drop=True)
        print(f"Training data rows: {len(train_data)}")
        print(f"Validation data rows: {len(val_data)}")

        train_set = CombinedContrastiveDataset(train_data, mri_dict, negative_sample_fraction=3, positive_repeat=2, augment=True)
        val_set   = CombinedContrastiveDataset(val_data,   mri_dict, negative_sample_fraction=3, positive_repeat=2, augment=False)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=custom_collate)
        val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, collate_fn=custom_collate)

        micro_dim_csv   = train_data.filter(like='Microbiome_').shape[1]
        biom_dim_csv    = train_data.filter(like='Biomarker_').shape[1]
        other_dim_csv   = train_data.filter(like='Other_').shape[1]
        numeric_dim_csv = train_data.filter(like='mri_numeric_').shape[1]
        print(f"Micro={micro_dim_csv}, Biom={biom_dim_csv}, Other={other_dim_csv}, Numeric={numeric_dim_csv}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiModalEmbeddingModel(
            micro_dim_csv, biom_dim_csv, other_dim_csv, numeric_dim_csv,
            embed_dim=64, augment=True
        ).to(device)

        checkpoint_path = "32slice.pth"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            print("Loaded pretrained model checkpoint (strict=False).")
        else:
            optimizer = torch.optim.AdamW([
                {'params': model.mri_encoder.clip_model.vision_model.encoder.parameters(), 'lr': 1e-4},
                {'params': model.mri_encoder.project.parameters(),                 'lr': 5e-3},
                {'params': model.micro_encoder.parameters(),                       'lr': 1e-3},
                {'params': model.biom_encoder.parameters(),                        'lr': 1e-3},
                {'params': model.other_encoder.parameters(),                       'lr': 1e-3},
                {'params': model.numeric_encoder.parameters(),                     'lr': 1e-3},
            ], weight_decay=1e-6)

            total_epochs = 200
            val_loss_history = []
            patience_threshold = 5
            for ep in range(1, total_epochs + 1):
                tr_loss = train_epoch(model, train_loader, optimizer, device,
                                      epoch=ep, val_loss_history=val_loss_history,
                                      patience_threshold=patience_threshold)
                val_loss = validate_epoch(model, val_loader, device)
                val_loss_history.append(val_loss)
                print(f"Pretraining Epoch {ep}/{total_epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
            plot_metrics(val_loss_history, val_loss_history, total_epochs, filename="pretraining_loss.png")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Pretraining checkpoint saved as {checkpoint_path}")

        # ---------------------
        # Attention Classification Phase
        # ---------------------
        other_df = pd.read_csv('/home/tmnthc/tom/anaconda3/New1/Other.csv')
        other_df.rename(columns=lambda x: x.strip(), inplace=True)
        other_df['Patient_ID'] = other_df['Patient_ID'].astype(str).str.strip()
        other_df['Timepoint']  = other_df['Timepoint'].astype(str).str.strip()
        patient_labels_df = other_df.groupby("Patient_ID")["APOE4"].max().reset_index()
        patient_apoe4 = {row["Patient_ID"]: int(row["APOE4"]) for _, row in patient_labels_df.iterrows()}
        all_patients = np.unique(other_df["Patient_ID"])
        if len(all_patients) < 23:
            raise ValueError("Need at least 23 unique patients for attention classification.")

        # Full-dataset loader for overall accuracy
        full_attn_dataset = AttentionDatasetWithLabels(all_patients, data, mri_dict, model, device, patient_apoe4)
        full_loader = DataLoader(full_attn_dataset, batch_size=4, shuffle=False, collate_fn=attn_collate_with_labels)

        # Single trial per run
        selected = np.random.choice(all_patients, size=23, replace=False)
        train_ids = selected[:16]
        val_ids   = selected[16:19]
        test_ids  = selected[19:23]
        print("Attention Trial patient split:")
        print("  Train IDs:", train_ids)
        print("  Validation IDs:", val_ids)
        print("  Test IDs:", test_ids)

        train_attn_dataset = AttentionDatasetWithLabels(train_ids, data, mri_dict, model, device, patient_apoe4)
        val_attn_dataset   = AttentionDatasetWithLabels(val_ids,   data, mri_dict, model, device, patient_apoe4)
        test_attn_dataset  = AttentionDatasetWithLabels(test_ids,  data, mri_dict, model, device, patient_apoe4)

        train_attn_loader = DataLoader(train_attn_dataset, batch_size=4, shuffle=True,  collate_fn=attn_collate_with_labels)
        val_attn_loader   = DataLoader(val_attn_dataset,   batch_size=4, shuffle=False, collate_fn=attn_collate_with_labels)
        test_attn_loader  = DataLoader(test_attn_dataset,  batch_size=4, shuffle=False, collate_fn=attn_collate_with_labels)

        ensemble_models = []
        for ens in range(1):
            attn_model = StackedAttentionClassifier(embed_dim=64, num_heads=4, num_layers=2, dropout=0.1).to(device)
            optimizer_attn = torch.optim.Adam(attn_model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion_attn = torch.nn.BCEWithLogitsLoss()

            best_val_loss = float('inf')
            epochs_no_improve = 0
            best_model_state = None

            for epoch in range(20):
                # training
                attn_model.train()
                total_loss = 0.0
                for embeddings, labels in train_attn_loader:
                    embeddings = embeddings.to(device)
                    labels = labels.to(device).float()
                    optimizer_attn.zero_grad()
                    logits = attn_model(embeddings)
                    logits_agg = logits.mean(dim=1)
                    loss = criterion_attn(logits_agg, labels)
                    loss.backward()
                    optimizer_attn.step()
                    total_loss += loss.item() * embeddings.size(0)
                avg_train_loss = total_loss / len(train_attn_dataset)

                # validation
                attn_model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for embeddings, labels in val_attn_loader:
                        embeddings = embeddings.to(device)
                        labels = labels.to(device).float()
                        logits = attn_model(embeddings)
                        logits_agg = logits.mean(dim=1)
                        loss = criterion_attn(logits_agg, labels)
                        total_val_loss += loss.item() * embeddings.size(0)
                avg_val_loss = total_val_loss / len(val_attn_dataset)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_model_state = attn_model.state_dict()
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= 5:
                    print(f"  Early stopping at epoch {epoch+1}, best val loss {best_val_loss:.4f}")
                    break
                if (epoch+1) % 5 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1}/20 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if best_model_state is not None:
                attn_model.load_state_dict(best_model_state)
            ensemble_models.append(attn_model)

        # overall accuracy on full dataset
        total_correct_full = 0
        total_samples_full = 0
        with torch.no_grad():
            for embeddings, labels in full_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device).float()
                ensemble_logits = sum(m(embeddings) for m in ensemble_models) / len(ensemble_models)
                logits_agg = ensemble_logits.mean(dim=1)
                preds = (logits_agg >= 0).float()
                total_correct_full += (preds == labels).sum().item()
                total_samples_full += labels.size(0)
        overall_acc = total_correct_full / total_samples_full if total_samples_full else 0.0
        print(f"Run {run} Overall Accuracy: {overall_acc*100:.2f}%")

        # test accuracy
        total_correct_test = 0
        total_samples_test = 0
        with torch.no_grad():
            for embeddings, labels in test_attn_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device).float()
                ensemble_logits = sum(m(embeddings) for m in ensemble_models) / len(ensemble_models)
                logits_agg = ensemble_logits.mean(dim=1)
                preds = (logits_agg >= 0).float()
                total_correct_test += (preds == labels).sum().item()
                total_samples_test += labels.size(0)
        trial_accuracy = total_correct_test / total_samples_test if total_samples_test else 0.0
        print(f"Run {run} Test Accuracy:   {trial_accuracy*100:.2f}%")

        # SHAP analysis
        if len(train_attn_dataset) > 0:
            sample_embedding, _ = train_attn_dataset[0]
            T = sample_embedding.shape[0]
            sample_input_flat = sample_embedding.view(-1).cpu().numpy()
            attn_model_for_shap = ensemble_models[0].eval()

            def attention_wrapper(X):
                X_t = torch.from_numpy(X).float().to(device)
                B = X_t.shape[0]
                X_t = X_t.view(B, T, 64)
                with torch.no_grad():
                    logits = attn_model_for_shap(X_t)
                    logits_agg = logits.mean(dim=1)
                return logits_agg.cpu().numpy()

            baseline = np.zeros((1, T * 64), dtype=np.float32)
            explainer = shap.KernelExplainer(attention_wrapper, baseline)
            shap_values = explainer.shap_values(np.expand_dims(sample_input_flat, axis=0), nsamples=100)[0]
            D = T // 5
            modality_indices = {
                "MRI":     slice(0 * D * 64, 1 * D * 64),
                "Micro":   slice(1 * D * 64, 2 * D * 64),
                "Biom":    slice(2 * D * 64, 3 * D * 64),
                "Other":   slice(3 * D * 64, 4 * D * 64),
                "Numeric": slice(4 * D * 64, 5 * D * 64),
            }
            print("Modality-level SHAP importances:")
            for mod, sl in modality_indices.items():
                val = np.sum(np.abs(shap_values[sl]))
                print(f"  {mod}: {val:.4f}")
                shap_records[mod].append(val)

        overall_accuracies.append(overall_acc)
        test_accuracies.append(trial_accuracy)

    # summary
    print("\n\n===== SUMMARY OVER 20 RUNS =====")
    for i in range(num_runs):
        print(f"Run {i+1:2d}: Overall={overall_accuracies[i]*100:5.2f}%   Test={test_accuracies[i]*100:5.2f}%")
    print(f"\nAverage Overall Accuracy: {np.mean(overall_accuracies)*100:.2f}%")
    print(f"Average Test    Accuracy: {np.mean(test_accuracies)*100:.2f}%")
    print("\nAverage SHAP importances per modality:")
    for mod, vals in shap_records.items():
        print(f"  {mod}: {np.mean(vals) if vals else 0.0:.4f}")


if __name__ == "__main__":
    main()
