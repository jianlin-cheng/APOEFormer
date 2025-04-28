#!/usr/bin/env python3
from torch.cuda.amp import GradScaler, autocast
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

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


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
# import cv2  # for resizing heatmaps
import shap  # SHAP

import torch
print(torch.__version__)
torch.cuda.empty_cache()


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
    microbiome = load_file('/home/tmnthc/New1/Microbiome.csv')
    blood_metabolites = load_file('/home/tmnthc/New1/Blood_Metabolites.csv')
    inflammatory_markers = load_file('/home/tmnthc/New1/Sirolimus_inflammatory_markers.csv')
    blood_data = load_file('/home/tmnthc/New1/Sirolimus_Blood_Data.csv')
    other_data = load_file('/home/tmnthc/New1/Other.csv', drop_apoe4=True)
    # Load the new numeric modality.
    brain_cbf = load_file('/home/tmnthc/New1/Brain_CBF_Imaging.csv', drop_apoe4=False)

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
    """
    Walks root_dir/<patient_id>/<timepoint>/*.nii(.gz)
    Returns dict[(patient_id, timepoint)] -> FloatTensor [1,H,W,D]
    with debug prints for each load.
    """
    mri_dict = {}
    total_loaded = 0

    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue
        print(f"[INFO] Patient '{patient_id}' found, scanning timepoints...")

        # each subfolder is a timepoint
        for timepoint in os.listdir(patient_path):
            tp_path = os.path.join(patient_path, timepoint)
            if not os.path.isdir(tp_path):
                continue
            print(f"  [INFO] Timepoint '{timepoint}' folder: {tp_path}")

            # find the first NIfTI in that folder
            nii_file = None
            for fn in os.listdir(tp_path):
                if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                    nii_file = fn
                    break

            if nii_file is None:
                print(f"  [WARN]   No .nii/.nii.gz file in {tp_path}, skipping timepoint.")
                continue

            full_path = os.path.join(tp_path, nii_file)
            print(f"  [INFO]   Found file '{nii_file}', loading...")

            # Load and normalize
            img = nib.load(full_path).get_fdata()
            if img.ndim == 4 and img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
            mean, std = img.mean(), img.std()
            img = (img - mean) / (std + 1e-8)

            # Convert to tensor
            mri_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
            mri_dict[(patient_id, timepoint)] = mri_tensor

            print(f"  [INFO]   Loaded tensor shape {mri_tensor.shape}")
            total_loaded += 1

    print(f"[INFO] Finished loading MRI data: {total_loaded} volumes added.")
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
    def __init__(self, embed_dim=32, augment=False, dropout_p=0.1):
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
        Batches all B*D slices into a single CLIP forward pass for speed.
        Returns: (B, D, embed_dim)
        """
        device = mri_batch.device
        B, _, H, W, D = mri_batch.shape

        # 1) Collect and preprocess all slices
        pil_images = []
        for i in range(B):
            vol = mri_batch[i, 0]               # (H, W, D)
            if vol.ndim != 3:
                vol = torch.zeros((H, W, D), dtype=torch.float32, device=device)
            # bring D-axis to front
            vol = vol.permute(2, 0, 1)          # (D, H, W)
            for slice2d in vol:
                arr = slice2d.cpu().numpy()
                with np.errstate(divide='ignore', invalid='ignore'):
                    minv = np.nanmin(arr)
                    maxv = np.nanmax(arr)
                    rng  = maxv - minv
                    if rng < 1e-6:
                        img8 = np.zeros_like(arr, dtype=np.uint8)
                    else:
                        norm = (arr - minv) / rng
                        norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
                        img8 = (norm * 255.0).astype(np.uint8)
                pil = Image.fromarray(img8, mode='L').convert("RGB")
                if self.training and self.augment:
                    pil = self.augmentation(pil)
                pil_images.append(pil)

        # 2) Batch through CLIP
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        feats = self.clip_model.get_image_features(**inputs)  # (B*D, 512)
        feats = safe_normalize(feats, p=2, dim=-1)

        # 3) Project and normalize
        projs = self.project(feats)                           # (B*D, embed_dim)
        projs = safe_normalize(projs, p=2, dim=-1)

        # 4) Reshape back to (B, D, embed_dim)
        projs = projs.view(B, D, -1)
        return projs



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
    def __init__(self, data, mri_dict, negative_sample_fraction=1, positive_repeat=1, augment=False):
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
    def __init__(self, micro_dim, biom_dim, other_dim, numeric_dim, embed_dim=32, augment=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.mri_encoder = MRIClipEncoder(embed_dim=embed_dim, augment=augment)
        self.micro_encoder = MLPEncoder(input_dim=micro_dim, output_dim=embed_dim, augment=augment)
        self.biom_encoder = MLPEncoder(input_dim=biom_dim, output_dim=embed_dim, augment=augment)
        self.other_encoder = MLPEncoder(input_dim=other_dim, output_dim=embed_dim, augment=augment)
        self.numeric_encoder = MLPEncoder(input_dim=numeric_dim, output_dim=embed_dim, augment=augment)

    def forward(self, mri, micro, biom, other, numeric):
        e_mri = self.mri_encoder(mri)   # (B, D, embed_dim)
        # For non-MRI, we just do MLP on entire vector → (B, embed_dim)
        e_micro = self.micro_encoder(micro)
        e_biom  = self.biom_encoder(biom)
        e_other = self.other_encoder(other)
        e_numeric = self.numeric_encoder(numeric)
        return e_mri, e_micro, e_biom, e_other, e_numeric

###########################################
# 8) Training & Validation for Pretraining
###########################################
def train_epoch(model, loader, optimizer, device, epoch,
                val_loss_history, patience_threshold=5,
                accum_steps=4, scaler=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        mri          = batch["mri"].to(device)
        micro        = batch["micro"].to(device)
        biom         = batch["biom"].to(device)
        other        = batch["other"].to(device)
        mri_numeric  = batch["mri_numeric"].to(device)
        sample_labels = batch["sample_label"]

        # mixed‐precision forward + loss
        with torch.cuda.amp.autocast():

            e_mri, e_micro, e_biom, e_other, e_numeric = model(mri, micro, biom, other, mri_numeric)
        
            e_mri_mean = e_mri.mean(dim=1)
            loss = patient_contrastive_loss(
                e_mri_mean, e_micro, e_biom, e_other, e_numeric,
                sample_labels=sample_labels, tau=0.5
            ) / accum_steps

        # backward + gradient accumulation
        scaler.scale(loss).backward()
        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += (loss * accum_steps).item()

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
            e_mri, e_micro, e_biom, e_other, e_numeric = model(mri, micro, biom, other, mri_numeric)

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
    def __init__(self, embed_dim=32, num_heads=4, num_layers=2, dropout=0.1):
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

import torch
from torch.utils.data import Dataset

def attn_collate_with_labels(batch):
    """
    Collate that stacks sample embeddings and labels.
    Assumes each sample embedding has identical first-dimension length.
    """
    embeddings_list, label_list = zip(*batch)
    embeddings = torch.stack(embeddings_list, dim=0)  # (B, T, E)
    labels = torch.stack(label_list, dim=0)           # (B,)
    return embeddings, labels

class AttentionDatasetWithLabels(Dataset):
    """
    For each patient, embeds *all* specified timepoints and concatenates
    their token sequences before classification.

    Args:
      patient_ids: list of patient ID strings to include
      data: merged DataFrame with columns Patient_ID, Timepoint, features...
      mri_dict: dict[(pid, tp)] -> CPU tensor [H, W, D]
      emb_model: pretrained MultiModalEmbeddingModel
      device: torch.device for inference
      patient_labels: dict[pid] -> int label
      timepoints: list of timepoint names to include per patient (e.g. ["Baseline","Washout","Post"])
    """
    def __init__(
        self,
        patient_ids,
        data,
        mri_dict,
        emb_model,
        attn_model,
        device,
        patient_labels,
        timepoints
    ):
        if attn_model is None:
            raise ValueError(
                "attn_model is None!  Make sure your attention‐head training loop ran "
                "and appended a valid model to `ensemble_models` before you call "
                "`AttentionDatasetWithLabels(...)`."
            )
        self.samples = []
        self.device = device
        self.emb_model = emb_model.to(device).eval()
        self.attn_model = attn_model.to(device).eval()

        E = emb_model.embed_dim  

        for pid in patient_ids:
            # collect per-timepoint token sequences
            seqs = []
            for tp in timepoints:
                # fetch row for pid,tp
                df_row = data[(data.Patient_ID == pid) & (data.Timepoint == tp)]
                if df_row.empty:
                    raise ValueError(f"Missing data for patient {pid}, timepoint {tp}")
                row = df_row.iloc[0]

                # load MRI
                mri_arr = mri_dict.get((pid, tp))
                if mri_arr is None:
                    raise ValueError(f"Missing MRI for {pid}-{tp}")
                mri_tensor = mri_arr.unsqueeze(0).to(device)

                # load static features
                micro = torch.tensor(
                    row.filter(like="Microbiome_").values.astype("float32")
                ).unsqueeze(0).to(device)
                biom  = torch.tensor(
                    row.filter(like="Biomarker_").values.astype("float32")
                ).unsqueeze(0).to(device)
                other = torch.tensor(
                    row.filter(like="Other_").values.astype("float32")
                ).unsqueeze(0).to(device)
                num   = torch.tensor(
                    row.filter(like="mri_numeric_").values.astype("float32")
                ).unsqueeze(0).to(device)

                # embed via combined model
                with torch.no_grad():
                    logits = self.attn_model  # ensure classifier has correct dtype
                    e_mri, e_micro, e_biom, e_other, e_num = self.emb_model(
                        mri_tensor, micro, biom, other, num
                    )  # e_mri: (1, D, E)

                # tile static embeddings across slices
                B, D, _ = e_mri.shape
                def tile(x): return x.unsqueeze(1).repeat(1, D, 1)
                seq_tp = torch.cat([
                    e_mri,
                    tile(e_micro),
                    tile(e_biom),
                    tile(e_other),
                    tile(e_num)
                ], dim=1)  # (1, 5*D, E)
                seqs.append(seq_tp.squeeze(0))  # (5*D, E)

            # concatenate across timepoints
            sample_embedding = torch.cat(seqs, dim=0)  # (T_total, E)
            label_val = patient_labels.get(pid, 0)
            label = torch.tensor(label_val, dtype=torch.float32)
            self.samples.append((sample_embedding, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def list_masked_paths(mask_root, timepoints):
    """
    Returns list of tuples (pid, tp, region, full_path)
    without loading any data.
    """
    import os
    paths = []
    for pid in os.listdir(mask_root):
        pid_dir = os.path.join(mask_root, pid)

        if not os.path.isdir(pid_dir):
            continue
        for tp in timepoints:
            tp_dir = os.path.join(pid_dir, tp)
            if not os.path.isdir(tp_dir):
                continue
            for fn in os.listdir(tp_dir):
                if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                    region = fn.rsplit(".nii",1)[0]
                    full = os.path.join(tp_dir, fn)
                    paths.append((pid, tp, region, full))
    return paths


import torch
import torch.nn.functional as F

class FullModel(torch.nn.Module):
    def __init__(self, emb_model, attn_model):
        super().__init__()
        self.emb  = emb_model
        self.attn = attn_model

    def forward(self, mri, micro, biom, other, numeric):
        # 1) embed everything
        e_mri, e_micro, e_biom, e_other, e_num = self.emb(
            mri, micro, biom, other, numeric
        )  # e_mri: (B, D, E), others: (B, E)

        B, D, E = e_mri.shape

        # helper to repeat non‐MRI embeddings across the D “slices”
        def tile(x):
            # x is (B, E) → become (B, D, E)
            return x.unsqueeze(1).repeat(1, D, 1)

        # 2) stack them as 5*D tokens of size E
        seq = torch.cat([
            e_mri,
            tile(e_micro),
            tile(e_biom),
            tile(e_other),
            tile(e_num),
        ], dim=1)  # <-- NOTE dim=1, so seq: (B, 5*D, E)

        # 3) run attention & mean‐pool over the 5*D “time” dimension
        logits = self.attn(seq).mean(dim=1)  # (B,)
        return logits

    

def evaluate_region_impacts_multi_timepoint(full_model, data_df, mri_orig, masked_paths, timepoints):
    """
    For each patient and each region, concatenates the 3 visits (Baseline, Washout, Post)
    into one long token sequence, once with the original MRIs and once with the same region masked
    at all three timepoints. Returns a DataFrame of Δ‐logits per (Patient, Region).
    
    full_model: embedding+attention wrapped model, eval + half on GPU
    data_df: merged pandas DataFrame with Patient_ID, Timepoint, and static features
    mri_orig: dict[(pid, tp)] -> CPU tensor [1,H,W,D] of original MRIs
    masked_paths: dict[(pid, tp, region)] -> filepath of masked NIfTI
    timepoints: list of 3 timepoint names, e.g. ["Baseline","Washout","Post"]
    """
    import torch, pandas as pd, nibabel as nib, numpy as np, time

    full_model.eval()
    device = next(full_model.parameters()).device
    results = []
    start = time.time()

    # gather unique patients & regions
    patients = sorted({pid for pid,_,_ in masked_paths})
    regions  = sorted({reg for *_, reg in masked_paths})

    for pid in patients:
        for region in regions:
            # check all three masks exist
            print(f"[DEBUG] Patient={pid} | Region={region}")
            keys_mask = [(pid, tp, region) for tp in timepoints]
            if not all(k in masked_paths for k in keys_mask):
                print(f"  [SKIP] Missing mask for one of {keys_mask}")
                continue

            #--- build original concatenated embedding ---#
            seqs_orig = []
            seqs_mask = []
            for tp in timepoints:
                # load and normalize original MRI
                img_o = mri_orig.get((pid, tp))
                if img_o is None:
                    seqs_orig = seqs_mask = None
                    break
                # load and normalize masked MRI
                path = masked_paths[(pid, tp, region)]
                arr = nib.load(path).get_fdata()
                arr = (arr - arr.mean())/(arr.std()+1e-8)
                img_m = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)

                # extract static features at this tp
                row = data_df[(data_df.Patient_ID==pid)&(data_df.Timepoint==tp)]
                if row.empty:
                    seqs_orig = seqs_mask = None
                    break
                row = row.iloc[0]
                def make(x):
                    v = pd.to_numeric(row.filter(like=x), errors='coerce').fillna(0).values
                    return torch.from_numpy(v.astype(np.float32)).unsqueeze(0)
                micro = make("Microbiome_").to(device).half()
                biom  = make("Biomarker_").to(device).half()
                other = make("Other_").to(device).half()
                num   = make("mri_numeric_").to(device).half()

                # run embeddings
                with torch.no_grad():
                    # original
                    e_o, m_mic, m_bio, m_oth, m_num = full_model.emb(
                        img_o.to(device).unsqueeze(0).half(),
                        micro, biom, other, num
                    )
                    # masked
                    e_m, _, _, _, _ = full_model.emb(
                        img_m.to(device).unsqueeze(0).half(),
                        micro, biom, other, num
                    )

                # tile static across slices and stack
                B, D, E = e_o.shape
                def tile(x): return x.unsqueeze(1).repeat(1, D, 1)
                seq_o = torch.cat([e_o, tile(m_mic), tile(m_bio), tile(m_oth), tile(m_num)], dim=1)
                seq_m = torch.cat([e_m, tile(m_mic), tile(m_bio), tile(m_oth), tile(m_num)], dim=1)
                seqs_orig.append(seq_o)
                seqs_mask.append(seq_m)

            if seqs_orig is None:
                continue

            # concatenate timepoints: each seq is (1,5*D,E), now (1,3*5*D,E)
            seq_orig = torch.cat(seqs_orig, dim=1)
            seq_mask = torch.cat(seqs_mask, dim=1)

            # final forward pass
            with torch.no_grad():
                logit_o = full_model.attn(seq_orig).mean(dim=1).item()
                logit_m = full_model.attn(seq_mask).mean(dim=1).item()

            results.append({
                "Patient": pid,
                "Region": region,
                "Logit_orig": logit_o,
                "Logit_mask": logit_m,
                "Delta": logit_o - logit_m
            })

    df = pd.DataFrame(results, columns=["Patient","Region","Logit_orig","Logit_mask","Delta"])
    print(f"[INFO] Multi‐TP region‐impact done in {time.time()-start:.1f}s; {len(df)} entries.")
    return df


class MaskedMultiTimeDataset(Dataset):
    def __init__(self, patient_ids, region, data_df, masked_paths,
                 model, attn_model, device, patient_labels, timepoints):
        super().__init__()
        self.device = device
        self.samples = []
        self.model = model.to(device).eval()
        self.attn  = attn_model.to(device).eval()

        for pid in patient_ids:
            # fetch label
            label_val = patient_labels[pid]
            label = torch.tensor(label_val, dtype=torch.float32)

            # build per-timepoint embeddings
            seqs = []
            for tp in timepoints:
                print(f"    → Timepoint={tp}")
                # find the one mask for this pid,tp,region
                path = masked_paths.get((pid,tp,region))
                if path is None:
                    raise ValueError(f"No mask for {pid},{tp},{region}")

                # load & normalize
                arr = nib.load(path).get_fdata()
                arr = (arr - arr.mean())/(arr.std()+1e-8)
                mri_mask = torch.from_numpy(arr.astype("float32")).unsqueeze(0).to(device)

                # static features
                row = data_df[(data_df.Patient_ID==pid)&(data_df.Timepoint==tp)].iloc[0]
                def make(x): 
                    vals = row.filter(like=x).astype(float).fillna(0).values
                    return torch.from_numpy(vals.astype("float32")).unsqueeze(0).to(device)
                micro = make("Microbiome_")
                biom  = make("Biomarker_")
                other = make("Other_")
                num   = make("mri_numeric_")

                # embed once, same as in FullModel‘s forward
                with torch.no_grad():
                    e_mri, e_mic, e_bio, e_oth, e_num = self.model(
                        mri_mask.unsqueeze(0), micro, biom, other, num
                    )  # e_mri: (1,D,E); others: (1,E)

                # tile & stack exactly as in FullModel
                D, E = e_mri.shape[1], e_mri.shape[2]
                def tile(x): return x.unsqueeze(1).repeat(1, D, 1)
                seq_tp = torch.cat([e_mri, tile(e_mic), tile(e_bio),
                                    tile(e_oth), tile(e_num)], dim=1)
                seqs.append(seq_tp.squeeze(0))  # (5*D, E)

            # concatenate all three timepoints
            sample_emb = torch.cat(seqs, dim=0)  # (3*5*D, E)
            self.samples.append((sample_emb, label))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
    
def main():
    import numpy as np
    import pandas as pd

    mask_root = "/home/tmnthc/mask"

    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wrap in DataParallel—this will split each batch across GPUs [0..N-1]
    
    print("Running on", device)
    scaler_pre = torch.cuda.amp.GradScaler()
    TIMEPOINTS = ["Baseline", "Washout", "Post"]


    num_runs = 1
    overall_accuracies = []
    test_accuracies = []
    shap_records = {mod: [] for mod in ["MRI", "Micro", "Biom", "Other", "Numeric"]}

    for run in range(1, num_runs + 1):
        print(f"\n\n===== RUN {run}/{num_runs} =====")

        # ---------------------
        # Pretraining Phase w/ Early Stopping (no checkpointing)
        # ---------------------
        data = load_data()
        mri_dict = load_mri_data("/home/tmnthc/T1")
        all_patients = np.unique(data["Patient_ID"])

        if len(all_patients) < 19:
            raise ValueError("Need at least 19 patients for pretraining.")

        # Pretraining split (this stays)
        selected = np.random.choice(all_patients, size=23, replace=False)
        train_ids = selected[:16]
        val_ids   = selected[16:19]
        test_ids  = selected[19:]
        print("Using patient split for BOTH phases — Train:", train_ids,
            "Val:", val_ids, "Test:", test_ids)
        # last 4 for attention test only

        print(f"\n===== RUN {run}/{num_runs} =====")
        print("Shared split this run:")
        print("  Train IDs:", train_ids)
        print("  Val   IDs:", val_ids)
        print("  Test  IDs:", test_ids)
        train_pats_pre = train_ids
        val_pats_pre   = val_ids
        print("Pretraining - Train IDs:", train_pats_pre)
        print("Pretraining - Val   IDs:", val_pats_pre)

        train_data = (
            data[data["Patient_ID"].isin(train_pats_pre)]
            .sort_values(["Patient_ID","Timepoint"])
            .groupby("Patient_ID").head(3)
            .reset_index(drop=True)
        )
        val_data = (
            data[data["Patient_ID"].isin(val_pats_pre)]
            .sort_values(["Patient_ID","Timepoint"])
            .groupby("Patient_ID").head(3)
            .reset_index(drop=True)
        )
        print(f"Train rows: {len(train_data)}, Val rows: {len(val_data)}")

        train_loader = DataLoader(
            CombinedContrastiveDataset(train_data, mri_dict,
                                       negative_sample_fraction=1,
                                       positive_repeat=1,
                                       augment=True),
            batch_size=16, shuffle=True, collate_fn=custom_collate, num_workers=1, pin_memory=True,prefetch_factor=4)
        
        val_loader = DataLoader(
            CombinedContrastiveDataset(val_data, mri_dict,
                                       negative_sample_fraction=1,
                                       positive_repeat=1,
                                       augment=False),
            batch_size=16, shuffle=False, collate_fn=custom_collate,  num_workers=1, pin_memory=True,prefetch_factor=4)
        

        micro_dim = train_data.filter(like='Microbiome_').shape[1]
        biom_dim  = train_data.filter(like='Biomarker_').shape[1]
        other_dim = train_data.filter(like='Other_').shape[1]
        num_dim   = train_data.filter(like='mri_numeric_').shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiModalEmbeddingModel(
            micro_dim, biom_dim, other_dim, num_dim,
            embed_dim=32, augment=True
        ).to(device)

        optimizer = torch.optim.AdamW([
            {'params': model.mri_encoder.clip_model.vision_model.encoder.parameters(), 'lr': 1e-4},
            {'params': model.mri_encoder.project.parameters(),                 'lr': 5e-3},
            {'params': model.micro_encoder.parameters(),                       'lr': 1e-3},
            {'params': model.biom_encoder.parameters(),                        'lr': 1e-3},
            {'params': model.other_encoder.parameters(),                       'lr': 1e-3},
            {'params': model.numeric_encoder.parameters(),                     'lr': 1e-3},
        ], weight_decay=1e-6)

        total_epochs = 1
        patience_pre = 15
        best_pre_val_loss = float('inf')
        epochs_no_improve_pre = 0
        best_pre_state = None
        val_loss_history = []

        for ep in range(1, total_epochs + 1):
            tr_loss = train_epoch(model, train_loader, optimizer, device,
                                  epoch=ep,
                                  val_loss_history=val_loss_history,
                                  patience_threshold=patience_pre,  scaler=scaler_pre)
            val_loss = validate_epoch(model, val_loader, device)
            torch.cuda.empty_cache()
            val_loss_history.append(val_loss)

            if val_loss < best_pre_val_loss:
                best_pre_val_loss = val_loss
                epochs_no_improve_pre = 0
                best_pre_state = model.state_dict()
            else:
                epochs_no_improve_pre += 1

            print(f"Pretrain Epoch {ep}/{total_epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

            if epochs_no_improve_pre >= patience_pre:
                print(f"→ Early stop pretraining at epoch {ep}, best val loss {best_pre_val_loss:.4f}")
                break

        # reload best pretraining weights
        if best_pre_state is not None:
            model.load_state_dict(best_pre_state)

       

        
        # Attention Classification Phase
        # ---------------------
        # Attention Classification Phase
        # ---------------------
        import pandas as pd
        from torch.utils.data import DataLoader

        # 1) Prepare patient‐level labels (APOE4)
        other_df = pd.read_csv('/home/tmnthc/New1/Other.csv')
        other_df.rename(columns=lambda x: x.strip(), inplace=True)
        other_df['Patient_ID'] = other_df['Patient_ID'].astype(str).str.strip()
        other_df['Timepoint']  = other_df['Timepoint'].astype(str).str.strip()
        patient_labels_df = other_df.groupby("Patient_ID")["APOE4"].max().reset_index()
        patient_apoe4 = {row["Patient_ID"]: int(row["APOE4"]) 
                        for _, row in patient_labels_df.iterrows()}

        all_patients = list(patient_apoe4.keys())
        if len(all_patients) < 23:
            raise ValueError("Need at least 23 unique patients for attention classification.")

        

        # 3) Train one or more attention heads
        ensemble_models = []
        ensemble_size = 1
        max_epochs     = 2
        early_stop_pat = 5

        for ens in range(ensemble_size):
            # a) instantiate your classifier
            attn_model = StackedAttentionClassifier(
                embed_dim=32, num_heads=4, num_layers=2, dropout=0.1
            ).to(device)

            print("Attention Phase using same split — Train:", train_ids,
            "Val:", val_ids, "Test:", test_ids)

            # b) build train/val datasets & loaders right here
            train_attn_ds = AttentionDatasetWithLabels(
                train_ids, data, mri_dict,
                model, attn_model, device, patient_apoe4, TIMEPOINTS
            )
            val_attn_ds = AttentionDatasetWithLabels(
                val_ids, data, mri_dict,
                model, attn_model, device, patient_apoe4, TIMEPOINTS
            )
            train_attn_loader = DataLoader(
                train_attn_ds, batch_size=16, shuffle=True,
                collate_fn=attn_collate_with_labels
            )

            # after train_attn_loader & val_attn_loader
            test_attn_dataset = AttentionDatasetWithLabels(
                test_ids, data, mri_dict,
                model,           # your frozen embedding backbone
                attn_model,  # your trained head
                device,
                patient_apoe4,
                TIMEPOINTS
            )
            test_attn_loader = DataLoader(
                test_attn_dataset,
                batch_size=16,      # or whatever batch size you used for train/val
                shuffle=False,
                collate_fn=attn_collate_with_labels
            )

            val_attn_loader = DataLoader(
                val_attn_ds, batch_size=16, shuffle=False,
                collate_fn=attn_collate_with_labels
            )

            # c) optimizer + loss
            optimizer_attn = torch.optim.Adam(
                attn_model.parameters(), lr=1e-3, weight_decay=1e-4
            )
            criterion_attn = torch.nn.BCEWithLogitsLoss()

            best_val_loss   = float('inf')
            epochs_no_improve = 0
            best_state = None

            # d) train / validate loop
            for epoch in range(1, max_epochs+1):
                # — train —
                attn_model.train()
                total_tr = 0.0
                for emb, lbl in train_attn_loader:
                    emb, lbl = emb.to(device), lbl.to(device)
                    optimizer_attn.zero_grad()
                    with torch.cuda.amp.autocast():
                        logits = attn_model(emb).mean(dim=1)
                        loss   = criterion_attn(logits, lbl)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(attn_model.parameters(), 1.0)
                    optimizer_attn.step()
                    total_tr += loss.item() * emb.size(0)
                train_loss = total_tr / len(train_attn_ds)

                # — validate —
                attn_model.eval()
                total_val = 0.0
                with torch.no_grad():
                    for emb, lbl in val_attn_loader:
                        emb, lbl = emb.to(device), lbl.to(device)
                        logits = attn_model(emb).mean(dim=1)
                        total_val += criterion_attn(logits, lbl).item() * emb.size(0)
                val_loss = total_val / len(val_attn_ds)

                # — check for improvement & early stop —
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = attn_model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epoch == 1 or epoch % 5 == 0:
                    print(f"Head {ens+1}, Epoch {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f}")
                if epochs_no_improve >= early_stop_pat:
                    print(f"→ Early stopping head {ens+1} at epoch {epoch}")
                    break

            # reload best weights, add to ensemble
            if best_state is not None:
                attn_model.load_state_dict(best_state)
            ensemble_models.append(attn_model)
            print(f"  → Appended head {ens+1}, ensemble size={len(ensemble_models)}")


            print(f"Trained {len(ensemble_models)} attention head(s).")

        # 5) Overall accuracy on full dataset
        full_attn_dataset = AttentionDatasetWithLabels(
            all_patients, data, mri_dict,
            model, ensemble_models[0], device, patient_apoe4, TIMEPOINTS
        )
        full_loader = DataLoader(full_attn_dataset, batch_size=16, shuffle=False, collate_fn=attn_collate_with_labels)

        correct, total = 0, 0
        with torch.no_grad():
            for embeddings, labels in full_loader:
                embeddings = embeddings.to(device)
                labels     = labels.to(device)
                logits = ensemble_models[0](embeddings).mean(dim=1)
                preds  = (logits >= 0).float()
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        print(f"Overall Accuracy: {100*correct/total:.2f}%")

        # 6) Test accuracy
        correct, total = 0, 0
        with torch.no_grad():
            for embeddings, labels in test_attn_loader:
                embeddings = embeddings.to(device)
                labels     = labels.to(device)
                logits = ensemble_models[0](embeddings).mean(dim=1)
                preds  = (logits >= 0).float()
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        print(f"Test Accuracy:    {100*correct/total:.2f}%")

            # ─── Multi‐timepoint, per‐region masked evaluation ─────────────────────────
        # build a dict for quick lookup
        masked_paths = {
            (pid, tp, region): path
            for pid, tp, region, path in list_masked_paths(mask_root, TIMEPOINTS)
        }

        # collect all unique regions
         # ─── REGION MASK IMPACT ANALYSIS (end-to-end with FullModel) ───────────────
 

        # 1) wrap your trained models
        full_model = FullModel(model, ensemble_models[0]).to(device)
        full_model.eval()
        full_model.half()

        # 2) build a lookup dict for every (pid,tp,region)
        masked_paths = {
            (pid, tp, region): path
            for pid, tp, region, path in list_masked_paths(mask_root, TIMEPOINTS)
        }
        print(f"Found {len(masked_paths)} masked volumes total")

        # 3) call the new multi‐timepoint evaluator
        df_impacts = evaluate_region_impacts_multi_timepoint(
            full_model,
            data,
            mri_dict,
            masked_paths,
            TIMEPOINTS
        )

        if df_impacts.empty:
            print("⚠️ No region impacts computed…")
        else:
            # --- group, sort, and save summary only ---
            region_means = df_impacts.groupby("Region")["Delta"]\
                                    .mean()\
                                    .sort_values(ascending=True)
            summary = region_means.reset_index().rename(columns={"Delta":"MeanDelta"})
            print("Mean Δ‐logit by region (ascending):")
            print(summary)
            summary.to_csv(f"region_impacts_summary_run{run}.csv", index=False)
                # ─────────────────────────────────────────────────────────────────────────

       


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

        attn_model = ensemble_models[0]

    
       

        if len(train_attn_dataset) > 0:
            # Pick your explainer model (first ensemble member)
            attn_model_for_shap = ensemble_models[0].eval()

            # Grab a single sample to get T and E
            sample_embedding, _ = train_attn_dataset[0]   # shape (T, E)
            sample_input_flat = sample_embedding.view(-1).cpu().numpy()

            T, E = sample_embedding.shape  
            baseline = np.zeros((1, T * E), dtype=np.float32)

            def attention_wrapper(X):
                X_t = torch.from_numpy(X).half().to(device)   # ← use .half() here
                B = X_t.shape[0]
                X_t = X_t.view(B, T, E)
                with torch.no_grad():
                    logits = attn_model_for_shap(X_t)
                return logits.mean(dim=1).cpu().numpy()

            explainer    = shap.KernelExplainer(attention_wrapper, baseline)
            shap_values  = explainer.shap_values(
                np.expand_dims(sample_input_flat, axis=0),
                nsamples=1000
            )[0]

            # now split into five equal blocks of size rep*E
            rep       = T // 5
            block     = rep * E

            modality_indices = {
                "MRI":     slice(0*block,   1*block),
                "Micro":   slice(1*block,   2*block),
                "Biom":    slice(2*block,   3*block),
                "Other":   slice(3*block,   4*block),
                "Numeric": slice(4*block,   5*block),
            }

            print("Modality‐level SHAP importances:")
            for mod, sl in modality_indices.items():
                val = np.sum(np.abs(shap_values[sl]))
                print(f"  {mod}: {val:.4f}")
                shap_records[mod].append(val)

            overall_accuracies.append(overall_acc)
            test_accuracies.append(trial_accuracy)

    # summary
    # ===== SUMMARY OVER 20 RUNS =====
    print("\n\n===== SUMMARY OVER 20 RUNS =====")
    for i in range(num_runs):
        ov = overall_accuracies[i] * 100
        te = test_accuracies[i] * 100
        mri_sh = shap_records["MRI"][i]
        mic_sh = shap_records["Micro"][i]
        bio_sh = shap_records["Biom"][i]
        oth_sh = shap_records["Other"][i]
        num_sh = shap_records["Numeric"][i]
    print(
        f"Run {i+1:2d}: Overall={ov:5.2f}%   Test={te:5.2f}%   "
        f"SHAP(MRI={mri_sh:.4f}, Micro={mic_sh:.4f}, Biom={bio_sh:.4f}, "
        f"Other={oth_sh:.4f}, Numeric={num_sh:.4f})"
    )

    # compute means and std devs
    import numpy as np
    avg_ov, std_ov = np.mean(overall_accuracies)*100, np.std(overall_accuracies)*100
    avg_te, std_te = np.mean(test_accuracies)*100,    np.std(test_accuracies)*100

    print(f"\nAverage Overall Accuracy: {avg_ov:.2f}% (±{std_ov:.2f}%)")
    print(f"Average Test    Accuracy: {avg_te:.2f}% (±{std_te:.2f}%)\n")

    print("Average SHAP importances per modality (±std):")
    for mod, vals in shap_records.items():
        m, s = np.mean(vals), np.std(vals)
        print(f"  {mod}: {m:.4f} (±{s:.4f})")

if __name__ == "__main__":
    main()

