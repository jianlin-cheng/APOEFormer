#!/usr/bin/env python3
from torch.cuda.amp import GradScaler, autocast
import random
import numpy as np
import torch
import os

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
import wandb
MASKED_CACHE_DIR = "/bmlfast/tom/masked_mri_cache"
os.makedirs(MASKED_CACHE_DIR, exist_ok=True)

from torch.nn import DataParallel
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
print(torch.__version__)
torch.cuda.empty_cache()

import torch

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

import pandas as pd

import pandas as pd

def load_data():
    def load_and_filter(path, prefix=None, drop_apoe4=False):
        # 1) Load
        if 'drop_apoe4' in load_file.__code__.co_varnames:
            df = load_file(path, drop_apoe4=drop_apoe4)
        else:
            df = load_file(path)

        # 2) Standardize ID columns
        for col in ('Patient_ID', 'Timepoint'):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # 3) Drop unwanted columns if present
        for col in ('date', 'ID', 'Groups'):
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        # 4) Filter out Washout timepoints
        if 'Timepoint' in df.columns:
            df = df[df['Timepoint'] != 'Washout']

        # 5) Force remaining columns to numeric
        non_num = ['Patient_ID', 'Timepoint']
        num_cols = [c for c in df.columns if c not in non_num]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df.fillna(0, inplace=True)

        # 6) Add prefix to numeric columns
        if prefix:
            rename_map = {c: f"{prefix}_{c}" for c in num_cols}
            df.rename(columns=rename_map, inplace=True)

        return df

    # Load each CSV, skipping Washout
    micro_df   = load_and_filter('/bmlfast/tom/New1/Microbiome.csv',    prefix="Microbiome")
    bm1_df     = load_and_filter('/bmlfast/tom/New1/Blood_Metabolites.csv', prefix="Biomarker")
    bm2_df     = load_and_filter('/bmlfast/tom/New1/Sirolimus_Blood_Data.csv', prefix="Biomarker")
    inflam_df  = load_and_filter('/bmlfast/tom/New1/Sirolimus_inflammatory_markers.csv', prefix="Biomarker")
    other_df   = load_and_filter('/bmlfast/tom/New1/Other.csv',           prefix="Other", drop_apoe4=True)
    cbf_df     = load_and_filter('/bmlfast/tom/New1/Brain_CBF_Imaging.csv', prefix="mri_numeric")

    # Print warnings / shapes
    names = ["Microbiome", "Blood Metabolites", "Blood Data", 
             "Inflammatory Markers", "Other", "Brain CBF"]
    dfs   = [micro_df, bm1_df, bm2_df, inflam_df, other_df, cbf_df]
    for name, df in zip(names, dfs):
        if df.empty:
            print(f"⚠️ Warning: {name} (excluding Washout) is empty.")
        else:
            print(f"{name}: {df.shape[0]} rows × {df.shape[1]} cols")

    # Merge biomarker sub-modalities
    biomarker_data = bm1_df.merge(bm2_df, on=['Patient_ID','Timepoint'], how='outer')
    biomarker_data = biomarker_data.merge(inflam_df, on=['Patient_ID','Timepoint'], how='outer')
    biomarker_data.fillna(0, inplace=True)

    # Merge everything
    data = other_df.copy()
    for df2 in [micro_df, biomarker_data, cbf_df]:
        data = data.merge(df2, on=['Patient_ID','Timepoint'], how='outer')
    data.fillna(0, inplace=True)

    print(f"Final merged data (no Washout): {data.shape}")
    return data


###########################################
# 2) MRI Data Loading (for brain imaging)
###########################################
import os
import torch
import numpy as np
import nibabel as nib

def load_mri_data(
    root_dir,
    cache_dir=None,
    device="cpu",
    allowed_timepoints=("Baseline", "Post")
):
    """
    Only loads/caches those sub-folders whose name is in allowed_timepoints.
    """
    mri_dict = {}
    total_cached = total_computed = 0

    # 1) Load from cache
    if cache_dir and os.path.isdir(cache_dir):
        for fn in os.listdir(cache_dir):
            if not fn.endswith(".pt"):
                continue
            name = fn[:-3]  # strip .pt
            pid, tp = name.split("_", 1)
            if tp not in allowed_timepoints:
                continue
            path = os.path.join(cache_dir, fn)
            try:
                emb = torch.load(path, map_location="cpu")  # (1,H,W,D) or (1,D,E)
                mri_dict[(pid, tp)] = emb
                total_cached += 1
            except Exception as e:
                print(f"⚠️ Failed to load cache {path}: {e}")

    if total_cached:
        print(f"[INFO] Loaded {total_cached} MRI entries from cache.")

    # 2) Walk raw NIfTI tree for missing ones
    for pid in os.listdir(root_dir):
        pid_path = os.path.join(root_dir, pid)
        if not os.path.isdir(pid_path):
            continue

        for tp in os.listdir(pid_path):
            if tp not in allowed_timepoints:
                continue

            key = (pid, tp)
            if key in mri_dict:
                continue

            tp_path = os.path.join(pid_path, tp)
            if not os.path.isdir(tp_path):
                continue

            # find a .nii or .nii.gz
            nii_fn = next(
                (f for f in os.listdir(tp_path) 
                 if f.endswith(".nii") or f.endswith(".nii.gz")),
                None
            )
            if not nii_fn:
                print(f"  [WARN] No NIfTI in {tp_path}, skipping.")
                continue

            full_nii = os.path.join(tp_path, nii_fn)
            arr = nib.load(full_nii).get_fdata()
            # squeeze singleton 4th dim
            if arr.ndim == 4 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)

            # normalize per-volume
            mean, std = arr.mean(), arr.std()
            norm = (arr - mean) / (std + 1e-8)
            tensor = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0)

            mri_dict[key] = tensor
            total_computed += 1

            # cache the normalized tensor
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(cache_dir, f"{pid}_{tp}.pt")
                torch.save(tensor.cpu(), cache_path)

    if total_computed:
        print(f"[INFO] Computed & cached {total_computed} new MRI entries.")

    print(f"[INFO] Finished loading MRI data: {len(mri_dict)} entries.")
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
    def __init__(self, embed_dim=64, augment=False, dropout_p=0.3):
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
          # ——— fast‐path for cached embeddings ———
        if mri_batch.dim() == 3:
            # Already (B, D, E)
            return mri_batch
        if mri_batch.dim() == 4:
            # (B,1,D,E) → squeeze out the dummy channel
            return mri_batch.squeeze(1)

        # ——— original 5D volume pipeline follows below ———
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
    def __init__(self, input_dim, output_dim=64, hidden_dims=[256,256,256], 
                 dropout_p=0.3, use_gelu=True):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            self.layers.append(nn.LayerNorm(dims[i+1]))
            self.layers.append(nn.GELU() if use_gelu else nn.ReLU())
            self.layers.append(nn.Dropout(dropout_p))
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        h = x
        # Process blocks of 4 layers at a time
        for i in range(0, len(self.layers), 4):
            lin   = self.layers[i]     # Linear
            norm  = self.layers[i+1]   # LayerNorm
            act   = self.layers[i+2]   # Activation
            drop  = self.layers[i+3]   # Dropout

            # apply them in sequence
            fwd = lin(h)
            fwd = norm(fwd)
            fwd = act(fwd)
            fwd = drop(fwd)

            # add residual if same shape
            if fwd.shape == h.shape:
                h = h + fwd
            else:
                h = fwd

        # final normalization + L2
        h = self.out_norm(h)
        return F.normalize(h, p=2, dim=-1)



###########################################
# 5) CombinedContrastiveDataset
###########################################
import random
import torch
import numpy as np
from torch.utils.data import Dataset

class CombinedContrastiveDataset(Dataset):
    def __init__(
        self,
        data,
        mri_dict,
        negative_sample_fraction=1,
        positive_repeat=1,
        augment=False
    ):
        self.data = data.reset_index(drop=True)
        self.mri_dict = mri_dict
        self.N = len(self.data)
        self.augment = augment

        # 1) positive pairs: same sample repeated
        self.positive_samples = [
            (i, i, i, i, i)
            for i in range(self.N)
            for _ in range(positive_repeat)
        ]

        # 2) hard-negative pairs: differ in exactly one modality
        num_negatives = int(self.N * negative_sample_fraction)
        negative_samples = []
        for _ in range(num_negatives):
            anchor = random.randrange(self.N)
            # pick one modality index to replace
            mod_to_replace = random.randrange(5)
            neg = [anchor] * 5
            # sample a different index for that modality
            choices = list(range(self.N))
            choices.remove(anchor)
            neg[mod_to_replace] = random.choice(choices)
            negative_samples.append(tuple(neg))

        self.negative_samples = negative_samples

        # combine and label
        self.samples = (
            [(s, 1) for s in self.positive_samples]
            + [(s, 0) for s in self.negative_samples]
        )
        # optional: shuffle so positives/negatives are mixed
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        indices, label = self.samples[idx]
        i, j, k, l, m = indices

        # pull out IDs and timepoints
        pid_mri, tpt_mri     = map(str, self.data.loc[i, ["Patient_ID", "Timepoint"]])
        pid_micro, tpt_micro = map(str, self.data.loc[j, ["Patient_ID", "Timepoint"]])
        pid_biom, tpt_biom   = map(str, self.data.loc[k, ["Patient_ID", "Timepoint"]])
        pid_other, tpt_other = map(str, self.data.loc[l, ["Patient_ID", "Timepoint"]])
        pid_num, tpt_num     = map(str, self.data.loc[m, ["Patient_ID", "Timepoint"]])

        # load tensors
        mri_tensor = self.mri_dict.get(
            (pid_mri, tpt_mri),
            torch.zeros((1,128,128,128), dtype=torch.float32)
        )
        micro_tensor = torch.tensor(
            self.data.filter(like="Microbiome_").iloc[j].values.astype(np.float32)
        )
        biom_tensor = torch.tensor(
            self.data.filter(like="Biomarker_").iloc[k].values.astype(np.float32)
        )
        other_tensor = torch.tensor(
            self.data.filter(like="Other_").iloc[l].values.astype(np.float32)
        )
        numeric_tensor = torch.tensor(
            self.data.filter(like="mri_numeric_").iloc[m].values.astype(np.float32)
        )

        # augment positives if desired
        if self.augment and label == 1:
            micro_tensor   = augment_numeric(micro_tensor)
            biom_tensor    = augment_numeric(biom_tensor)
            other_tensor   = augment_numeric(other_tensor)
            numeric_tensor = augment_numeric(numeric_tensor)

        sample_label = f"{pid_mri}_{tpt_mri}"
        return {
            "mri":         mri_tensor,
            "micro":       micro_tensor,
            "biom":        biom_tensor,
            "other":       other_tensor,
            "mri_numeric": numeric_tensor,
            "sample_label": sample_label,
            "label":       torch.tensor(label, dtype=torch.float32),
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
def patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, e_numeric, sample_labels, tau=0.2):
    
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
        self.embed_dim = embed_dim
        self.mri_encoder = MRIClipEncoder(embed_dim=embed_dim, augment=augment)
        self.micro_encoder = MLPEncoder(input_dim=micro_dim, output_dim=embed_dim)
        self.biom_encoder = MLPEncoder(input_dim=biom_dim, output_dim=embed_dim)
        self.other_encoder = MLPEncoder(input_dim=other_dim, output_dim=embed_dim)
        self.numeric_encoder = MLPEncoder(input_dim=numeric_dim, output_dim=embed_dim)

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
    # ─── 1) unwrap DataParallel/DDP ──────────────────────────────────────────────
    base_model = model.module if hasattr(model, "module") else model

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
            # you can still call `model(...)` here or `base_model(...)`—
            # it doesn’t matter for forwarding
            e_mri, e_micro, e_biom, e_other, e_numeric = model(
                mri, micro, biom, other, mri_numeric
            )
            e_mri_mean = e_mri.mean(dim=1)
            loss = patient_contrastive_loss(
                e_mri_mean, e_micro, e_biom, e_other, e_numeric,
                sample_labels=sample_labels, tau=0.5
            ) / accum_steps

        # backward + gradient accumulation
        scaler.scale(loss).backward()
        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            # ─── NOTE: gradient clipping on the *base* model’s params ───────────
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += (loss * accum_steps).item()

    avg_loss = total_loss / len(loader)

    # ─── 2) use base_model when unfreezing ────────────────────────────────────
    if len(val_loss_history) > 0 and avg_loss >= max(val_loss_history[-patience_threshold:]):
        base_model.mri_encoder.gradually_unfreeze(
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class StackedAttentionClassifier(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            )
        self.dropout   = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)
        # fuse original + attended
        self.fuse      = nn.Linear(embed_dim * 2, embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, T, embed_dim)
        returns:    (B,)   one logit per sample
        """
        # keep the original token sequence
        residual = embeddings   # (B, T, E)
        out = embeddings        # (B, T, E)

        # stacked self-attention
        for attn in self.layers:
            attn_out, _ = attn(out, out, out)            # (B, T, E)
            out = self.layernorm(out + self.dropout(attn_out))

        # fuse original + attended representations
        fused = torch.cat([residual, out], dim=-1)        # (B, T, 2E)
        fused = F.relu(self.fuse(fused))                  # (B, T, E)

        # per-token logits
        token_logits = self.classifier(fused).squeeze(-1)  # (B, T)

        # mean-pool across tokens to get one logit per sample
        sample_logits = token_logits.mean(dim=1)           # (B,)
        return sample_logits


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

        # NEW—unpack DDP/DataParallel if needed
        base_model = emb_model.module if hasattr(emb_model, "module") else emb_model

        # now grab the embed dim from the raw model
        # if you defined embed_dim on MultiModalEmbeddingModel:
        E = base_model.embed_dim 

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


import os, zipfile

def list_masked_paths(mask_root, timepoints):
    paths = []
    for pid in os.listdir(mask_root):
        pid_dir = os.path.join(mask_root, pid)
        if not os.path.isdir(pid_dir): continue

        for tp in timepoints:
            tp_dir = os.path.join(pid_dir, tp)
            if not os.path.isdir(tp_dir): 
                print(f"  [SKIP] no folder {tp_dir}")
                continue

            for fn in os.listdir(tp_dir):
                full = os.path.join(tp_dir, fn)

                if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                    region = fn.rsplit(".nii",1)[0]
                    paths.append((pid, tp, region, full))

                elif fn.endswith(".nii.zip"):
                    region = fn.rsplit(".nii.zip",1)[0]
                    # extract into a temp file-like object
                    def loader():
                        with zipfile.ZipFile(full, "r") as z:
                            # assume there's exactly one file inside
                            inner = z.namelist()[0]
                            return z.open(inner)
                    paths.append((pid, tp, region, loader))
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
        logits = self.attn(seq)
        return logits

    


def evaluate_region_impacts_multi_timepoint(
    full_model,
    data_df: pd.DataFrame,
    mri_orig: dict[tuple[str,str], torch.Tensor],
    masked_cache_dir: str,
    timepoints: list[str]
):
    """
    For each patient & region, concatenates Baseline/Washout/Post into one long token
    sequence (orig vs. masked‐.pt), runs your attention head, and returns Δ‐logits
    (+ probs) per (Patient, Region).
    """
    import os, time, torch, pandas as pd
    from torch.utils.data import DataLoader

    full_model.eval()
    device = next(full_model.parameters()).device
    start = time.time()

    # 1) index all your .pt masks
    mask_index: dict[tuple[str,str,str], str] = {}
    for fn in os.listdir(masked_cache_dir):
        if not fn.endswith(".pt"):
            continue
        pid, tp, region = fn[:-3].split("_", 2)
        mask_index[(pid, tp, region)] = os.path.join(masked_cache_dir, fn)
    print(f"[DEBUG] Indexed {len(mask_index)} masked .pt volumes from '{masked_cache_dir}'")

    # 2) find all pid/region with all 3 tps
    patients = sorted({pid for pid, _ in mri_orig.keys()})
    regions  = sorted({reg for _,_,reg in mask_index.keys()})
    to_process = [
        (pid, region)
        for pid in patients
        for region in regions
        if all(((pid, tp) in mri_orig and (pid, tp, region) in mask_index)
               for tp in timepoints)
    ]
    print(f"[DEBUG] Found {len(to_process)} valid patient-region pairs: {to_process[:5]}{'...' if len(to_process)>5 else ''}")

    # 3) build your orig vs. masked sequences
    seq_pairs = []
    for idx, (pid, region) in enumerate(to_process, 1):
        print(f"[DEBUG] ({idx}/{len(to_process)}) Processing Patient={pid}, Region={region}")
        seqs_o, seqs_m = [], []
        for tp in timepoints:
            print(f"  [DEBUG] Timepoint={tp}")
            # original MRI embedding
            img_o = mri_orig[(pid, tp)].unsqueeze(0).to(device)
            print(f"    [DEBUG] Orig tensor shape: {img_o.shape}")

            # masked MRI tensor
            mask_path = mask_index[(pid, tp, region)]
            img_m = torch.load(mask_path, map_location="cpu").unsqueeze(0).to(device)
            print(f"    [DEBUG] Masked tensor loaded from {mask_path}, shape: {img_m.shape}")

            # static features
            row = data_df[(data_df.Patient_ID==pid)&(data_df.Timepoint==tp)]
            if row.empty:
                raise ValueError(f"Missing static data for {pid}/{tp}")
            row = row.iloc[0]

            def to_tensor(pref):
                vals = row.filter(like=pref).astype(float).fillna(0).values
                t = torch.from_numpy(vals.astype("float32")).unsqueeze(0).to(device)
                print(f"    [DEBUG] Static '{pref}' tensor shape: {t.shape}")
                return t

            micro = to_tensor("Microbiome_")
            biom   = to_tensor("Biomarker_")
            other  = to_tensor("Other_")
            num    = to_tensor("mri_numeric_")

            # embed both
            with torch.no_grad():
                e_o, e_micro, e_biom, e_other, e_num = full_model.emb(
                    img_o, micro, biom, other, num
                )
                e_m, *_ = full_model.emb(
                    img_m, micro, biom, other, num
                )
            print(f"    [DEBUG] Embeddings shapes: e_o={e_o.shape}, e_m={e_m.shape}")

            # tile and concat
            B, D, E = e_o.shape
            def tile(x): return x.unsqueeze(1).repeat(1, D, 1)
            seqs_o.append(torch.cat([e_o, tile(e_micro), tile(e_biom), tile(e_other), tile(e_num)], dim=1))
            seqs_m.append(torch.cat([e_m, tile(e_micro), tile(e_biom), tile(e_other), tile(e_num)], dim=1))
        
        total_len = seqs_o[0].shape[1] * len(seqs_o)
        print(f"  [DEBUG] Queued sequences for {pid}/{region}, total seq length per: {total_len}")

        seq_pairs.append((
            torch.cat(seqs_o, dim=1),
            torch.cat(seqs_m, dim=1),
            (pid, region)
        ))

    # 4) batch through attention head
    def collate_fn(batch):
        so = torch.cat([b[0] for b in batch], dim=0)
        sm = torch.cat([b[1] for b in batch], dim=0)
        metas = [b[2] for b in batch]
        return so, sm, metas

    loader = DataLoader(seq_pairs, batch_size=4, collate_fn=collate_fn)
    print(f"[DEBUG] Running attention on {len(loader)} batches (batch_size=4)")
    results = []
    for i, (so, sm, metas) in enumerate(loader, 1):
        print(f"[DEBUG] Batch {i}/{len(loader)}: so={so.shape}, sm={sm.shape}")
        so, sm = so.to(device), sm.to(device)
        with torch.no_grad():
            lo = full_model.attn(so).mean(dim=1)
            lm = full_model.attn(sm).mean(dim=1)

        for idx, (pid, region) in enumerate(metas):
            l_o, l_m = lo[idx].item(), lm[idx].item()
            p_o, p_m = torch.sigmoid(lo[idx]).item(), torch.sigmoid(lm[idx]).item()
            results.append({
                "Patient":    pid,
                "Region":     region,
                "Logit_orig": l_o,
                "Logit_mask": l_m,
                "Delta":      l_o - l_m,
                "Prob_orig":  p_o,
                "Prob_mask":  p_m,
                "Delta_prob": p_o - p_m
            })

    df = pd.DataFrame(results)
    print(f"[INFO] Completed in {time.time()-start:.1f}s — {len(df)} entries.")
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

# 1) Precompute embeddings once
def precompute_embeddings(mri_dict, mri_encoder, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    mri_encoder.eval()
    for (pid, tp), mri in mri_dict.items():
        emb_path = os.path.join(out_dir, f"{pid}_{tp}.pt")
        if os.path.exists(emb_path):
            continue
        with torch.no_grad():
            # mri: FloatTensor [1,H,W,D]
            e_mri = mri_encoder(mri.unsqueeze(0).to(device))  # → (1, D, E)
        torch.save(e_mri.squeeze(0).cpu(), emb_path)
        print(f"[DEBUG] Saved embedding: {emb_path}")

def load_precomputed(mri_keys, emb_dir):
    mri_emb = {}
    for pid, tp in mri_keys:
        path = os.path.join(emb_dir, f"{pid}_{tp}.pt")
        mri_emb[(pid, tp)] = torch.load(path)
    return mri_emb

def precompute_masked_mri(masked_paths, cache_dir, device="cpu"):
    """
    masked_paths: dict[(pid,tp,region)] -> either filepath or loader()
    cache_dir:     path where to save .pt files
    """
    os.makedirs(cache_dir, exist_ok=True)
    for (pid, tp, region), loader in masked_paths.items():
        cache_fn = f"{pid}_{tp}_{region}.pt"
        cache_path = os.path.join(cache_dir, cache_fn)
        if os.path.exists(cache_path):
            continue

        # load raw NIfTI (unzipping if needed)
        if callable(loader):
            with loader() as f:
                raw = f.read()
            with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
                tmp.write(raw); tmp.flush()
                arr = nib.load(tmp.name).get_fdata()
        else:
            arr = nib.load(loader).get_fdata()

        # same preprocessing as load_mri_data
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        mean, std = arr.mean(), arr.std()
        norm = (arr - mean) / (std + 1e-8)

        tensor = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0).to(device)
        torch.save(tensor.cpu(), cache_path)
        print(f"[CACHE] Saved masked MRI: {cache_path}")

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


def augment_embeddings(emb, noise_std=0.1):
    # emb: (B, T, E)
    noise = torch.randn_like(emb) * noise_std
    return emb + noise

class ClassBalancedBCE(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, device='cpu'):
        """
        samples_per_cls: list or tensor [n_neg, n_pos]
        beta: hyperparam close to 1.0
        """
        super().__init__()
        samples = torch.tensor(samples_per_cls, dtype=torch.float32, device=device)
        effective_num = 1.0 - torch.pow(beta, samples)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * 2.0   # normalize sum->2 (neg+pos)
        # weights = [w_neg, w_pos]
        self.register_buffer('class_weights', weights)

    def forward(self, logits, targets):
        # logits: (B,), targets: (B,)
        pos_weight = self.class_weights[1] / self.class_weights[0]
        # BCEWithLogits with pos_weight will weight positive examples
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pos_weight,
            reduction='mean'
        )

def main():
    import numpy as np
    import pandas as pd
    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader

    # ─── WANDB SETUP ─────────────────────────────────────────────────────────
    wandb.init(
        project="brain-region-prediction",
        config={
            "batch_size":32,
            "embed_dim": 64,
            "neg_frac": 50,
            "pos_repeat": 50,
            "lr_clip": 5e-3,
            "lr_proj": 5e-3,
            "lr_num_heads": 5e-3,
            "lr_mlp":  5e-3,
            "epochs_pre": 700,
            "patience_pre": 10,
            "epochs_attn": 700,
            "patience_attn": 3,
            "ensemble_size":3,
            "threshold": 0.5,
            "num_layer": 3,
            "drop_out":0.4,
            "num_run": 5
        },
    )
    config = wandb.config
    orig_hparams = dict(config) 


    all_runs = []

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from torch.utils.data import DataLoader
    
    scaler_pre = torch.cuda.amp.GradScaler()
    TIMEPOINTS = ["Baseline", "Post"]

    num_runs = config.num_run
    overall_accuracies = []
    test_accuracies = []
    shap_records = {mod: [] for mod in ["MRI", "Micro", "Biom", "Other", "Numeric"]}


    # ─── DISTRIBUTED SETUP ──────────────────────────────────────────
   

    for run in range(1, num_runs + 1):
        print(f"\n\n===== RUN {run}/{num_runs} =====")
        

        data = load_data()
        all_patients = np.unique(data["Patient_ID"])
        if len(all_patients) < 19:
            raise ValueError("Need at least 19 patients for pretraining.")
        
        from sklearn.model_selection import train_test_split

        # 0) Load the APOE4 labels once
        other_df = pd.read_csv('/bmlfast/tom/New1/Other.csv')
        other_df.rename(columns=lambda x: x.strip(), inplace=True)
        other_df['Patient_ID'] = other_df['Patient_ID'].astype(str).str.strip()
        patient_labels_df = other_df.groupby("Patient_ID")["APOE4"].max().reset_index()
        patient_apoe4 = {
            row["Patient_ID"]: int(row["APOE4"])
            for _, row in patient_labels_df.iterrows()
        }

        all_patients = list(patient_apoe4.keys())
        if len(all_patients) < 23:
            raise ValueError("Need at least 23 unique patients for stratified splitting.")

        # 1) Split off 4 patients for TEST, stratified by APOE4
        trainval_ids, test_ids = train_test_split(
            all_patients,
            test_size=4,
            stratify=[patient_apoe4[p] for p in all_patients],
            random_state=run
        )

        # 2) From the remaining, split off 3 patients for VAL
        train_ids, val_ids = train_test_split(
            trainval_ids,
            test_size=3,
            stratify=[patient_apoe4[p] for p in trainval_ids],
            random_state=run
        )

        print(f"Stratified split — Train: {train_ids}, Val: {val_ids}, Test: {test_ids}")

        train_data = (
            data[data["Patient_ID"].isin(train_ids)]
            .sort_values(["Patient_ID","Timepoint"])
            .groupby("Patient_ID").head(3)
            .reset_index(drop=True)
        )
        val_data = (
            data[data["Patient_ID"].isin(val_ids)]
            .sort_values(["Patient_ID","Timepoint"])
            .groupby("Patient_ID").head(3)
            .reset_index(drop=True)
        )
        print(f"Train rows: {len(train_data)}, Val rows: {len(val_data)}")

        # ────────────────────────────────────────────────────────────────────────
        # 3) Instantiate & wrap your model with DataParallel
        # ────────────────────────────────────────────────────────────────────────
        micro_dim = train_data.filter(like="Microbiome_").shape[1]
        biom_dim  = train_data.filter(like="Biomarker_").shape[1]
        other_dim = train_data.filter(like="Other_").shape[1]
        num_dim   = train_data.filter(like="mri_numeric_").shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # After you choose your device:

        print(f"[INFO] Assigned device: {device}")
        print(f"### DEBUG: config.embed_dim = {config.embed_dim}")

        model  = MultiModalEmbeddingModel(
            micro_dim, biom_dim, other_dim, num_dim,
            embed_dim=config.embed_dim, augment=True
        ).to(device)
        model  = torch.nn.DataParallel(model)
        print(f"### DEBUG: backbone.embed_dim = {model.module.embed_dim}")

        print(f"[INFO] DataParallel is using device_ids = {model.device_ids}")
        wandb.watch(model, log="all", log_freq=50)

        # ────────────────────────────────────────────────────────────────────────
        # 4) Precompute & cache CLIP embeddings once
        # ────────────────────────────────────────────────────────────────────────
        # ─── LOAD YOUR 64-DIM EMBEDDINGS ─────────────────────────────────
        # ─── 4) Load raw MRI volumes & normalize ─────────────────────────────
        from test_val_up import load_precomputed

        # Wherever you defined EMB_CACHE for your 64‐dim embeddings:
        EMB_CACHE = "/bmlfast/tom/mri_cache_embed64"

        # Build a list of all (Patient_ID, Timepoint) pairs present in your merged CSV:
        mri_keys = list(
            zip(
                data["Patient_ID"].astype(str).tolist(),
                data["Timepoint"].astype(str).tolist()
            )
        )

        # Load the .pt files you just generated:
        print(f"[INFO] Loading {len(mri_keys)} precomputed embeddings from {EMB_CACHE}")
        mri_dict = load_precomputed(mri_keys, EMB_CACHE)
        print(f"[INFO] Loaded {len(mri_dict)} MRI entries from EMB_CACHE")
                # ────────────────────────────────────────────────────────────────────────
        # 6) Optimizer, Scaler & Early-Stopping Setup
        # ────────────────────────────────────────────────────────────────────────
        optimizer     = torch.optim.AdamW([
            {'params': model.module.mri_encoder.clip_model.vision_model.encoder.parameters(), 'lr': config.lr_clip},
            {'params': model.module.mri_encoder.project.parameters(),                           'lr': config.lr_proj},
            {'params': model.module.micro_encoder.parameters(),                                 'lr': config.lr_mlp},
            {'params': model.module.biom_encoder.parameters(),                                  'lr': config.lr_mlp},
            {'params': model.module.other_encoder.parameters(),                                 'lr': config.lr_mlp},
            {'params': model.module.numeric_encoder.parameters(),                               'lr': config.lr_mlp},
        ], weight_decay=4)

        scheduler_pre = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',        # we want to reduce LR when val loss stops improving
            factor=0.5,        # LR ← LR * 0.5
            patience=2,        # wait 2 epochs with no improvement
            min_lr=1e-6
        )

        scaler_pre    = torch.cuda.amp.GradScaler()
        total_epochs  = config.epochs_pre
        patience_pre  = config.patience_pre
        best_val_loss = float('inf')
        no_improve    = 0
        val_history   = []

        from torch.utils.data import WeightedRandomSampler, DataLoader

        # 1) Build your datasets as before
        train_dataset = CombinedContrastiveDataset(
            train_data,
            mri_dict,
            negative_sample_fraction=config.neg_frac,
            positive_repeat=         config.pos_repeat,
            augment=True
        )
        val_dataset = CombinedContrastiveDataset(
            val_data,
            mri_dict,
            negative_sample_fraction=config.neg_frac,
            positive_repeat=config.pos_repeat,
            augment=False
        )

        # 2) Extract the labels for each pair (1=positive, 0=negative)
        all_labels = [label for _, label in train_dataset.samples]
        num_pos    = sum(all_labels)
        num_neg    = len(all_labels) - num_pos

        # 3) Compute a sampling weight so positives get drawn more often
        ratio = num_neg / (num_pos + 1e-8)
        sample_weights = [ratio if lbl == 1 else 1.0 for lbl in all_labels]

        # 4) Create the sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # 5) Use it in the train loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,          # ← swapped in place of shuffle=True
            num_workers=8,
            pin_memory=False,
            collate_fn=custom_collate
        )

        # 6) Validation loader stays the same (no sampling)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
            collate_fn=custom_collate
        )


        
        # ────────────────────────────────────────────────────────────────────────
        # 7) Pretraining Loop
        # ────────────────────────────────────────────────────────────────────────
        for ep in range(1, total_epochs+1):
            tr_loss = train_epoch(
                model, train_loader, optimizer, device,
                epoch=ep,
                val_loss_history=val_history,
                patience_threshold=patience_pre,
                scaler=scaler_pre
            )
            
            val_loss = validate_epoch(model, val_loader, device)
            val_history.append(val_loss)
            torch.cuda.empty_cache()

            scheduler_pre.step(val_loss)

            # Log to wandb
            wandb.log({
                "pretrain/train_loss": tr_loss,
                "pretrain/val_loss":   val_loss,
                "epoch":               ep
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                best_state    = model.state_dict()
            else:
                no_improve += 1

            print(f"Epoch {ep}/{total_epochs} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}")
            if no_improve >= patience_pre:
                print(f"→ Early stop at epoch {ep}, best val {best_val_loss:.4f}")
                break


        # reload best pretraining weights
        if best_val_loss < float('inf'):
            model.load_state_dict(best_state)
            
        # ---------------------
        # Attention Classification Phase
        # ---------------------
        import pandas as pd
        from torch.utils.data import DataLoader

        # 1) Prepare patient‐level labels (APOE4)
        other_df = pd.read_csv('/bmlfast/tom/New1/Other.csv')
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
        ensemble_size = config.ensemble_size
        max_epochs     = config.epochs_attn
        early_stop_pat = config.patience_attn

        for ens in range(ensemble_size):
            # a) instantiate your classifier
            attn_model = StackedAttentionClassifier(
                embed_dim=config.embed_dim, num_heads=4, num_layers=config.num_layer, dropout=config.drop_out
            ).to(device)

            print("Attention Phase using same split — Train:", train_ids,
            "Val:", val_ids, "Test:", test_ids)

            # b) build train/val datasets & loaders right here
            train_attn_ds = AttentionDatasetWithLabels(
                train_ids, data, mri_dict,
                model, attn_model, device, patient_apoe4, TIMEPOINTS
            )
            # … after you’ve constructed train_attn_ds …
            # Gather all train labels (0 or 1)
            all_train_labels = [lbl.item() for _, lbl in train_attn_ds]
            N_pos = sum(all_train_labels)
            N_neg = len(all_train_labels) - N_pos

            ratio = N_neg / (N_pos + 1e-8)
            sample_weights = [ratio if lbl == 1 else 1.0 for lbl in all_train_labels]
            # # pos_weight is ratio of negatives to positives
            # pos_weight = torch.tensor([N_neg / N_pos], device=device)

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            

            val_attn_ds = AttentionDatasetWithLabels(
                val_ids, data, mri_dict,
                model, attn_model, device, patient_apoe4, TIMEPOINTS
            )
            # after train_attn_ds is built:
            pids = [pid for pid, _ in train_attn_ds]           # e.g. ["35","13",...,"7",...]
            weights = [3.0 if pid == "7" else 1.0 for pid in pids]
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

            train_attn_loader = DataLoader(
                train_attn_ds,
                batch_size=64,
                sampler=sampler,
                collate_fn=attn_collate_with_labels,
                num_workers=0,
                pin_memory=False,
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
                batch_size=64,      # or whatever batch size you used for train/val
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=attn_collate_with_labels
            )

            val_attn_loader = DataLoader(
                val_attn_ds,
                batch_size=64,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=attn_collate_with_labels
            )

            train_labels = torch.tensor([lbl.item() for _, lbl in train_attn_ds], device=device)
            num_pos = train_labels.sum()
            num_neg = (train_labels.numel() - num_pos)
            
            pos_weight = (num_neg / (num_pos + 1e-8)) *3
            # Optional: clamp to avoid extreme values
            pos_weight = pos_weight.clamp(min=2.0, max=10.0)

            pos_weight_param = nn.Parameter(pos_weight.clone().detach(), requires_grad=True).to(device)

            # 3) Create weighted BCE loss
            criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

            # 2) now include its parameters in the optimizer
            cb_loss = ClassBalancedBCE(samples_per_cls=[num_neg, num_pos],
                           beta=0.9999, device=device).to(device)

            optimizer_attn = torch.optim.AdamW(
                attn_model.parameters(),
                lr=5e-4,
                weight_decay=5e-3
            )


            best_val_loss   = float('inf')
            epochs_no_improve = 0
            best_state = None

            # d) train / validate loop
            for epoch in range(1, max_epochs + 1):
                # — train —
                attn_model.train()
                total_tr = 0.0

                for emb, lbl in train_attn_loader:
                    emb, lbl = emb.to(device), lbl.to(device)

                    mask_pos = lbl.bool().unsqueeze(-1).unsqueeze(-1)   # (B,1,1)
                    emb_aug  = torch.where(
                        mask_pos,
                        augment_embeddings(emb, noise_std=0.2),  # noisy positives
                        emb                                       # untouched negatives
                    )

                    optimizer_attn.zero_grad()
                    with torch.cuda.amp.autocast():
                        logits = attn_model(emb_aug)              # (B,)
                        loss   = cb_loss(logits, lbl) # BCEWithLogitsLoss
                    
                    with torch.no_grad():
                        pos_weight_param.clamp_(1.0, 10.0)

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
                        logits   = attn_model(emb)
                        total_val += cb_loss(logits, lbl).item() * emb.size(0)

                val_loss = total_val / len(val_attn_ds)
                
                # early stopping logic, checkpointing, etc., follows here…

            
                # — check for improvement & early stop —
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = attn_model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epoch == 1 or epoch % 5 == 0:
                    print(f"Attention Head {ens+1}, Epoch {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f}")
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
        full_loader = DataLoader(full_attn_dataset, batch_size=64, shuffle=False, collate_fn=attn_collate_with_labels)


        


        #     # ─── Multi‐timepoint, per‐region masked evaluation ─────────────────────────
        # # build a dict for quick lookup
        # mask_root = "/bmlfast/tom/result"
   
        # masked_cache = "/bmlfast/tom/masked_mri_cache"


        # # collect all unique regions
        # # ─── REGION MASK IMPACT ANALYSIS (end-to-end with FullModel) ───────────────


        # # 1) wrap your trained models
        # full_model = FullModel(model, ensemble_models[0]).to(device)
        # full_model.eval()

       
     
        # data_df = load_data()
        # mri_orig = load_mri_data(              # dict[(pid,tp)] → CPU tensor [1,H,W,D]
        #     root_dir="/bmlfast/tom/T1",
        #     cache_dir="/bmlfast/tom/mri_cache",
        #     device=device
        # )
        # # 3) call the new multi‐timepoint evaluator
        # df_impacts = evaluate_region_impacts_multi_timepoint(
        #     full_model=full_model,      # your FullModel wrapper
        #     data_df = data_df,            # the merged DataFrame
        #     mri_orig = mri_orig,          # original MRIs loaded into memory
        #     masked_cache_dir = masked_cache,  # the dict of mask loaders
        #     timepoints=TIMEPOINTS,      # exactly 3 timepoints
        # )

        # # Save the raw per‐patient results
        # raw_fname = f"region_impacts_details_run{run}.csv"
        # df_impacts.to_csv(raw_fname, index=False)
        # print(f"[INFO] Saved detailed per‐patient region impacts to {raw_fname}")


        # if df_impacts.empty:
        #     print("⚠️ No region impacts computed…")
        # else:
        #     # 1) group raw logits and deltas by region
        #     region_logit_orig = df_impacts.groupby("Region")["Logit_orig"].mean()
        #     region_logit_mask = df_impacts.groupby("Region")["Logit_mask"].mean()
        #     signed_delta      = df_impacts.groupby("Region")["Delta"].mean()

        #     # 2) sort regions by absolute Δ‐logit descending
        #     abs_delta        = signed_delta.abs().sort_values(ascending=False)
        #     sorted_regions   = abs_delta.index

        #     # 3) compute mean probabilities from mean logits
        #     import numpy as np
        #     mean_logit_orig = region_logit_orig.loc[sorted_regions].values
        #     mean_logit_mask = region_logit_mask.loc[sorted_regions].values
        #     mean_prob_orig  = 1 / (1 + np.exp(-mean_logit_orig))
        #     mean_prob_mask  = 1 / (1 + np.exp(-mean_logit_mask))

        #     # 4) build the summary DataFrame
        #     summary = pd.DataFrame({
        #         "Region":             sorted_regions,
        #         "MeanLogit_orig":     mean_logit_orig,
        #         "MeanProb_orig":      mean_prob_orig,
        #         "MeanLogit_mask":     mean_logit_mask,
        #         "MeanProb_mask":      mean_prob_mask,
        #         "MeanDelta_logit":    signed_delta.loc[sorted_regions].values,
        #         "MeanAbsDelta_logit": abs_delta.values
        #     })

        #     # 5) print & save
        #     print("Region impact summary (sorted by |Δ‐logit|):")
        #     print(summary)
        #     summary.to_csv(f"region_impacts_summary_run{run}.csv", index=False)

        val_probs  = []
        val_labels = []
        with torch.no_grad():
            for emb, lbl in val_attn_loader:
                emb       = emb.to(device)
                lbl       = lbl.numpy()        # keep on CPU for sklearn
                logits    = attn_model(emb)    # or ensemble average
                probs_np  = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(probs_np)
                val_labels.append(lbl)

        val_probs  = np.concatenate(val_probs)   # shape (N_val,)
        val_labels = np.concatenate(val_labels)  # shape (N_val,)

        from sklearn.metrics import precision_recall_curve

        # val_labels, val_probs already collected on your validation set
        prec, rec, thr = precision_recall_curve(val_labels, val_probs)

        # compute F1 for each threshold
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)

        # pick the threshold that maximizes F1
        ix = np.nanargmax(f1_scores)
        best_thresh = thr[ix]           # note: thr has length len(prec)-1
        best_f1     = f1_scores[ix]

        print(f"→ Best threshold: {best_thresh:.3f}, F1 = {best_f1:.3f}")


        # ─── Overall accuracy (sigmoid + ensemble average) ─────────────────────────
        # Overall accuracy on full dataset
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for embeddings, labels in full_loader:
                embeddings = embeddings.to(device)
                labels     = labels.to(device).float()
                # 1) ensemble-average logits
                ensemble_logits = sum(m(embeddings) for m in ensemble_models) / len(ensemble_models)
                # 2) convert to probabilities
                probs = torch.sigmoid(ensemble_logits)
                # 3) threshold at best_thresh
                preds = (probs >= 0.5).float()

                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        overall_acc = total_correct / total_samples if total_samples else 0.0
        print(f"Overall Accuracy (prob ≥{best_thresh:.3f}): {overall_acc*100:.2f}%")

        # Per-patient test set evaluation
        # ─── right before you start evaluating on test_attn_loader ─────────────────
        total_correct_test = 0
        total_samples_test = 0
        run_records       = []      # ←─ initialize here

        with torch.no_grad():
            for embeddings, labels in test_attn_loader:
                embeddings = embeddings.to(device)
                labels     = labels.to(device).float()

                # ensemble‐average logits
                logits = sum(m(embeddings) for m in ensemble_models) / len(ensemble_models)
                probs  = torch.sigmoid(logits)
                preds  = (probs >= 0.5).float()

                # update your accuracy counters
                total_correct_test += (preds == labels).sum().item()
                total_samples_test += labels.size(0)

                # append one record per patient in this batch
                for i, pid in enumerate(test_ids):
                    run_records.append({
                        "Run":            run,
                        "Patient_ID":     pid,
                        "TrueLabel":      float(labels[i].item()),
                        "PredictedLabel": float(preds[i].item())
                    })

        # ─── after the loop, compute accuracy ──────────────────────────────────────
        test_acc = total_correct_test / total_samples_test
        print(f"Test Accuracy: {test_acc*100:.2f}%")

        # ─── now print your per‐patient lines from run_records ────────────────────
        print("\nTest‐set predictions:")
        for rec in run_records:
            print(
                f"[Run {rec['Run']}] "
                f"PID={rec['Patient_ID']}  "
                f"True={int(rec['TrueLabel'])}  "
                f"Pred={int(rec['PredictedLabel'])}  "
                f"(thr={best_thresh:.3f})"
            )


        all_runs.append(pd.DataFrame(run_records))

        test_acc = total_correct_test / total_samples_test if total_samples_test else 0.0
        print(f"Run {run} Test Accuracy (prob ≥{best_thresh:.3f}): {test_acc*100:.2f}%")

                

        import os

        # os.makedirs("saved_models", exist_ok=True)
        # for idx, head in enumerate(ensemble_models, start=1):
        #     path = f"saved_models/run_{run}_attn_head{idx}.pt"
        #     torch.save(head.state_dict(), path)
        #     print(f"[INFO] Saved attention head {idx} to {path}")

        # # (Optionally) Save the frozen embedding backbone too:
        # backbone_path = f"saved_models/run_{run}_embedding_backbone.pt"
        # torch.save(model.module.state_dict(), backbone_path)
        # print(f"[INFO] Saved embedding backbone to {backbone_path}")



        # # wandb.log({
        # #     "overall_accuracy": overall_acc,
        # #     "test_accuracy":    trial_accuracy,
        # #     **{f"shap/{mod}": np.mean(vals) for mod, vals in shap_records.items()}ls

        # # })

        # # SHAP analysis

        # attn_model = ensemble_models[0]

        
        #     # ─── MODALITY‑LEVEL SHAP ANALYSIS (KernelExplainer with wrapper) ─────────────
        # if len(train_attn_ds) > 0:
        #     # 1) Grab a single sample embedding
        #     sample_embedding, _ = train_attn_ds[0]   # (T, E)
        #     T, E = sample_embedding.shape            # T = total tokens, E = embed_dim

        #     # 2) Flatten to (T*E,) and prepare device & model
        #     sample_input_flat = sample_embedding.view(-1).cpu().numpy()
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     attn_model = ensemble_models[0].to(device).eval()

        #     # 3) Define the SHAP wrapper
        #     def attention_classifier_wrapper(X: np.ndarray) -> np.ndarray:
        #         """
        #         X: shape (B, T*E)
        #         Returns: shape (B,) of scalar logits from the attention classifier
        #         """
        #         X_torch = torch.from_numpy(X).float().to(device)   # (B, T*E)
        #         B = X_torch.size(0)
        #         X_torch = X_torch.view(B, T, E)                   # (B, T, E)
        #         with torch.no_grad():
        #             logits = attn_model(X_torch)                  # (B,)
        #         return logits.cpu().numpy()                       # (B,)

        #     # 4) Baseline for Kernel SHAP
        #     baseline = np.zeros((1, T * E), dtype=np.float32)

        #     # 5) Build explainer & compute SHAP values
        #     explainer   = shap.KernelExplainer(attention_classifier_wrapper, baseline)
        #     shap_vals   = explainer.shap_values(
        #         np.expand_dims(sample_input_flat, axis=0),     # (1, T*E)
        #         nsamples=1000
        #     )[0]  # (T*E,)

        #     # 6) Split into modality blocks and sum absolute SHAP scores
        #     D = T // 5
        #     modality_slices = {
        #         "MRI":     slice(0 * D * E, 1 * D * E),
        #         "Micro":   slice(1 * D * E, 2 * D * E),
        #         "Biom":    slice(2 * D * E, 3 * D * E),
        #         "Other":   slice(3 * D * E, 4 * D * E),
        #         "Numeric": slice(4 * D * E, 5 * D * E),
        #     }

        #     print("\nModality‑level SHAP importance (KernelExplainer):")
        #     for mod, sl in modality_slices.items():
        #         importance = np.sum(np.abs(shap_vals[sl]))
        #         print(f"  {mod}: {importance:.4f}")
        #         shap_records[mod].append(importance)

    
        #      # ─── MODALITY‐LEVEL SHAP ANALYSIS ─────────────────────────────────────
        

        overall_accuracies.append(overall_acc)
        test_accuracies.append(test_acc)

    # summary
    # ===== SUMMARY OVER 20 RUNS =====

        # 3) Concatenate all runs and save once
    df_all = pd.concat(all_runs, ignore_index=True)
    out_path = "all_runs_predictions.csv"
    df_all.to_csv(out_path, index=False)
    print(f"[INFO] Saved all‐runs predictions to {out_path}")
    print(df_all)

    # print("\n\n===== SUMMARY OVER 20 RUNS =====")
    # for i in range(num_runs):
    #     ov = overall_accuracies[i] * 100
    #     te = test_accuracies[i] * 100
    #     mri_sh = shap_records["MRI"][i]
    #     mic_sh = shap_records["Micro"][i]
    #     bio_sh = shap_records["Biom"][i]
    #     oth_sh = shap_records["Other"][i]
    #     num_sh = shap_records["Numeric"][i]
    # print(
    #     f"Run {i+1:2d}: Overall={ov:5.2f}%   Test={te:5.2f}%   "
    #     f"SHAP(MRI={mri_sh:.4f}, Micro={mic_sh:.4f}, Biom={bio_sh:.4f}, "
    #     f"Other={oth_sh:.4f}, Numeric={num_sh:.4f})"
    # )

    # compute means and std devs
    import numpy as np
    avg_ov, std_ov = np.mean(overall_accuracies)*100, np.std(overall_accuracies)*100
    avg_te, std_te = np.mean(test_accuracies)*100,    np.std(test_accuracies)*100

    print(f"\nAverage Overall Accuracy: {avg_ov:.2f}% (±{std_ov:.2f}%)")
    print(f"Average Test    Accuracy: {avg_te:.2f}% (±{std_te:.2f}%)\n")

    # print("Average SHAP importances per modality (±std):")
    # for mod, vals in shap_records.items():
    #     m, s = np.mean(vals), np.std(vals)
    #     print(f"  {mod}: {m:.4f} (±{s:.4f})")

if __name__ == "__main__":
    main()

