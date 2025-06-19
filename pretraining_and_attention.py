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

import os
import pandas as pd
import numpy as np
import torch
import nibabel as nib

###########################################
# UPDATED CSV LOADING & MERGING
###########################################
import os
import pandas as pd
import numpy as np

def load_data():
    def load_and_filter(path, prefix, drop_apoe4=False):
        df = load_file(path, drop_apoe4=drop_apoe4)
        # strip & standardize IDs
        for col in ('Patient_ID','Timepoint'):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        # drop unwanted
        for col in ('date','ID','Groups'):
            if col in df.columns:
                df.drop(columns=col, inplace=True)
        # exclude Washout
        if 'Timepoint' in df.columns:
            df = df[df['Timepoint'] != 'Washout']
        # coerce numeric
        non_num = ['Patient_ID','Timepoint']
        num_cols = [c for c in df.columns if c not in non_num]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df.fillna(0, inplace=True)
        # prefix
        rename_map = {c: f"{prefix}_{c}" for c in num_cols}
        df.rename(columns=rename_map, inplace=True)
        return df

    # -- static modalities --
    micro_df = load_and_filter(
        '/bmlfast/tom/New1/Microbiome.csv',
        prefix='Microbiome'
    )
    # split your three biomarker files:
    bioA_df = load_and_filter(
        '/bmlfast/tom/New1/Blood_Metabolites.csv',
        prefix='Biomarker_A'
    )
    bioB_df = load_and_filter(
        '/bmlfast/tom/New1/Sirolimus_inflammatory_markers.csv',
        prefix='Biomarker_B'
    )
    bioC_df = load_and_filter(
        '/bmlfast/tom/New1/Sirolimus_Blood_Data.csv',
        prefix='Biomarker_C'
    )
    other_df = load_and_filter(
        '/bmlfast/tom/New1/Other.csv',
        prefix='Other',
        drop_apoe4=True
    )
    cbf_df = load_and_filter(
        '/bmlfast/tom/New1/Brain_CBF_Imaging.csv',
        prefix='Brain_CBF_Imaging'
    )

    # report
    modalities = [
        ("Microbiome", micro_df),
        ("Biomarker_A", bioA_df),
        ("Biomarker_B", bioB_df),
        ("Biomarker_C", bioC_df),
        ("Other", other_df),
        ("Brain_CBF_Imaging", cbf_df),
    ]
    for name, df in modalities:
        if df.empty:
            print(f"⚠️ Warning: {name} is empty.")
        else:
            print(f"{name}: {df.shape[0]}×{df.shape[1]}")

    # merge on Patient_ID & Timepoint
    data = other_df.copy()
    for df2 in [micro_df, bioA_df, bioB_df, bioC_df, cbf_df]:
        data = data.merge(df2, on=['Patient_ID','Timepoint'], how='outer')
    data.fillna(0, inplace=True)

    print(f"Final merged data: {data.shape}")
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
        self.unfreeze_layers = 6
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

class CombinedContrastiveDataset(Dataset):
    def __init__(
        self,
        data,
        mri_dict,
        x_dict,
        negative_sample_fraction=1,
        positive_repeat=1,
        augment=False
    ):
        self.data     = data.reset_index(drop=True)
        self.mri_dict = mri_dict
        self.x_dict   = x_dict
        self.N        = len(self.data)
        self.augment  = augment

        M = 8  # MRI, X, Microbiome, Biomarker_A, Biomarker_B, Biomarker_C, Other, Brain_CBF_Imaging

        # positives: same index in every slot
        self.positive_samples = [
            (i,)*M
            for i in range(self.N)
            for _ in range(positive_repeat)
        ]

        # hard negatives: differ in exactly one modality
        num_neg = int(self.N * negative_sample_fraction)
        negative_samples = []
        for _ in range(num_neg):
            anchor = random.randrange(self.N)
            slot   = random.randrange(M)
            idxs   = [anchor]*M
            choices = list(range(self.N))
            choices.remove(anchor)
            idxs[slot] = random.choice(choices)
            negative_samples.append(tuple(idxs))

        self.samples = [(s,1) for s in self.positive_samples] + \
                       [(s,0) for s in negative_samples]
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (i_mri, i_x, i_micro, i_bioA, i_bioB, i_bioC, i_other, i_num), label = self.samples[idx]

        def get_pid_tp(i):
            row = self.data.loc[i, ["Patient_ID","Timepoint"]]
            return str(row["Patient_ID"]), str(row["Timepoint"])

        pid_mri, tp_mri   = get_pid_tp(i_mri)
        pid_x,   tp_x     = get_pid_tp(i_x)
        pid_micro, tp_micro = get_pid_tp(i_micro)
        pid_bioA,  tp_bioA = get_pid_tp(i_bioA)
        pid_bioB,  tp_bioB = get_pid_tp(i_bioB)
        pid_bioC,  tp_bioC = get_pid_tp(i_bioC)
        pid_other, tp_other = get_pid_tp(i_other)
        pid_num,   tp_num   = get_pid_tp(i_num)

        # load volumes
        mri_tensor = self.mri_dict.get((pid_mri, tp_mri),
                                       torch.zeros((1,128,128,128), dtype=torch.float32))
        x_tensor   = self.x_dict.get((pid_x, tp_x),
                                     torch.zeros((1,128,128,128), dtype=torch.float32))

        # load static vectors
        micro_tensor = torch.tensor(
            self.data.filter(like="Microbiome_").iloc[i_micro].values.astype(np.float32)
        )
        bioA_tensor = torch.tensor(
            self.data.filter(like="Biomarker_A_").iloc[i_bioA].values.astype(np.float32)
        )
        bioB_tensor = torch.tensor(
            self.data.filter(like="Biomarker_B_").iloc[i_bioB].values.astype(np.float32)
        )
        bioC_tensor = torch.tensor(
            self.data.filter(like="Biomarker_C_").iloc[i_bioC].values.astype(np.float32)
        )
        other_tensor = torch.tensor(
            self.data.filter(like="Other_").iloc[i_other].values.astype(np.float32)
        )
        num_tensor = torch.tensor(
            self.data.filter(like="Brain_CBF_Imaging_").iloc[i_num].values.astype(np.float32)
        )

        # optional augmentation of static
        if self.augment and label == 1:
            micro_tensor = augment_numeric(micro_tensor)
            bioA_tensor  = augment_numeric(bioA_tensor)
            bioB_tensor  = augment_numeric(bioB_tensor)
            bioC_tensor  = augment_numeric(bioC_tensor)
            other_tensor = augment_numeric(other_tensor)
            num_tensor   = augment_numeric(num_tensor)

        return {
            "mri":    mri_tensor,
            "x_img":  x_tensor,
            "micro":  micro_tensor,
            "biomarker_A": bioA_tensor,
            "biomarker_B": bioB_tensor,
            "biomarker_C": bioC_tensor,
            "other":  other_tensor,
            "numeric":num_tensor,
            "label":  torch.tensor(label, dtype=torch.float32),
        }
    
def custom_collate(batch):
    collated = {}
    # stack labels & static
    collated["label"] = torch.stack([d["label"] for d in batch])
    for key in ("micro","biomarker_A","biomarker_B","biomarker_C","other","numeric"):
        collated[key] = torch.stack([d[key] for d in batch])
    # leave images as lists
    collated["mri"]   = [d["mri"]   for d in batch]  # list of (1,H,W,D_i)
    collated["x_img"] = [d["x_img"] for d in batch]  # list of (1,H',W',D'_i)
    return collated



###########################################
# 6) Patient Contrastive Loss
###########################################
import torch
import numpy as np

def patient_contrastive_loss(
    e_mri, e_x, e_micro, e_bioA, e_bioB, e_bioC, e_other, e_num,
    sample_labels,
    tau=0.1
):
    B, E = e_mri.shape
    labels_list = list(sample_labels)
    all_labels  = labels_list * 8
    _, inv     = np.unique(all_labels, return_inverse=True)
    labels = torch.tensor(inv, device=e_mri.device)

    emb_all = torch.cat([
        e_mri, e_x, e_micro, e_bioA, e_bioB, e_bioC, e_other, e_num
    ], dim=0)  # (8B, E)

    sim = torch.matmul(emb_all, emb_all.t()) / tau
    N   = 8 * B
    diag = torch.eye(N, dtype=torch.bool, device=sim.device)

    exp_sim = torch.exp(sim)
    sum_pos = (exp_sim * ((labels.unsqueeze(0)==labels.unsqueeze(1)) & ~diag).float()).sum(dim=1)
    sum_all = (exp_sim * (~diag).float()).sum(dim=1)

    loss = -torch.log((sum_pos + 1e-8) / (sum_all + 1e-8))
    return loss.mean()

###########################################
# 7) MultiModalEmbeddingModel
###########################################
import torch.nn as nn

import torch
import torch.nn as nn
from torch.nn import LayerNorm

# Assume MRIClipEncoder and MLPEncoder are imported/provided elsewhere

class MultiModalEmbeddingModel(nn.Module):
    def __init__(
        self,
        micro_dim,
        biomarker_A_dim,
        biomarker_B_dim,
        biomarker_C_dim,
        other_dim,
        numeric_dim,
        embed_dim=64,
        augment=False
    ):
        super().__init__()

        # two image encoders
        self.mri_encoder = MRIClipEncoder(embed_dim=embed_dim, augment=augment)
        self.x_encoder   = MRIClipEncoder(embed_dim=embed_dim, augment=augment)

        # static encoders
        self.micro_encoder  = MLPEncoder(input_dim=micro_dim,      output_dim=embed_dim)
        self.bioA_encoder   = MLPEncoder(input_dim=biomarker_A_dim, output_dim=embed_dim)
        self.bioB_encoder   = MLPEncoder(input_dim=biomarker_B_dim, output_dim=embed_dim)
        self.bioC_encoder   = MLPEncoder(input_dim=biomarker_C_dim, output_dim=embed_dim)
        self.other_encoder  = MLPEncoder(input_dim=other_dim,       output_dim=embed_dim)
        self.num_encoder    = MLPEncoder(input_dim=numeric_dim,     output_dim=embed_dim)

        # maximum number of slices you expect in any volumet
        max_slices = 256
        # register a learnable weight per slice index for attention pooling
        self.slice_weights = nn.Parameter(torch.zeros(max_slices))

        # store embed_dim for downstream use
        self.embed_dim = embed_dim

    def forward(self, mri, micro, biomarker_A, biomarker_B, biomarker_C, other, numeric, x_img):
        e_mri  = self.mri_encoder(mri)       # (B, D, E)
        e_x    = self.x_encoder(x_img)       # (B, D, E)
        e_micro = self.micro_encoder(micro)  # (B, E)
        e_bioA = self.bioA_encoder(biomarker_A)
        e_bioB = self.bioB_encoder(biomarker_B)
        e_bioC = self.bioC_encoder(biomarker_C)
        e_other= self.other_encoder(other)
        e_num  = self.num_encoder(numeric)
        return e_mri, e_x, e_micro, e_bioA, e_bioB, e_bioC, e_other, e_num



###########################################
# 8) Training & Validation for Pretraining
###########################################
from torch import amp
import torch
import numpy as np

def embed_and_pad(vol_list, encoder, device):
    """
    vol_list: list of 3D tensors shape (1,H,W,D_i)
    encoder:  your MRIClipEncoder or x_encoder instance
    Returns a tensor of shape (B, D_max, E)
    """
    embeddings = []
    encoder.eval()
    with torch.no_grad():
        for vol in vol_list:
            # vol: (1, H, W, D_i)
            vol_in = vol.unsqueeze(0).to(device)        # (1,1,H,W,D_i)
            e = encoder(vol_in)                         # (1, D_i, E)
            embeddings.append(e.squeeze(0))             # (D_i, E)

    # pad/truncate to max D in batch
    D_max = max(e.shape[0] for e in embeddings)
    padded = []
    for e in embeddings:
        D, E = e.shape
        if D < D_max:
            pad = torch.zeros(D_max - D, E, device=e.device)
            e = torch.cat([e, pad], dim=0)
        else:
            e = e[:D_max]
        padded.append(e)
    return torch.stack(padded, dim=0)  # (B, D_max, E)


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    val_loss_history,
    patience_threshold=5,
    accum_steps=4,
    scaler=None
):
    """
    One training epoch over contrastive batches.
    """
    base = model.module if hasattr(model, "module") else model
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        # 1) embed & pad MRI and X volumes
        e_mri = embed_and_pad(batch["mri"],   base.mri_encoder, device)  # (B, D, E)
        e_x   = embed_and_pad(batch["x_img"], base.x_encoder,   device)  # (B, D, E)


        D_mri     = e_mri.size(1)
        w_mri     = base.slice_weights[:D_mri]                # (D_mri,)
        alpha_mri = F.softmax(w_mri, dim=0)                   # (D_mri,)
        emri_m    = (e_mri * alpha_mri[None,:,None]).sum(1)   # (B, E)

        # X-modality attention-pool
        D_x       = e_x.size(1)
        w_x       = base.slice_weights[:D_x]                  # (D_x,)
        alpha_x   = F.softmax(w_x, dim=0)                     # (D_x,)
        ex_m      = (e_x * alpha_x[None,:,None]).sum(1)       # (B, E)

        # 2) load static modalities
        micro = batch["micro"].to(device)
        bioA  = batch["biomarker_A"].to(device)
        bioB  = batch["biomarker_B"].to(device)
        bioC  = batch["biomarker_C"].to(device)
        other = batch["other"].to(device)
        num   = batch["numeric"].to(device)
        labels= batch["label"]

        # 3) encode static via MLPs
        e_micro = base.micro_encoder(micro)     # (B, E)
        e_bioA  = base.bioA_encoder(bioA)
        e_bioB  = base.bioB_encoder(bioB)
        e_bioC  = base.bioC_encoder(bioC)
        e_other = base.other_encoder(other)
        e_num   = base.num_encoder(num)

        ids   = torch.arange(emri_m.size(0), device=device)   # unique ID per sample

        # 4) compute contrastive loss under autocast
        with amp.autocast(device_type="cuda"):
            loss = patient_contrastive_loss(
                emri_m, ex_m,
                e_micro, e_bioA, e_bioB, e_bioC, e_other, e_num,
                sample_labels=labels,
                tau=0.1
            ) / accum_steps

        # 5) backward + step
        scaler.scale(loss).backward()
        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += (loss * accum_steps).item()

    avg_loss = total_loss / len(loader)

    # dynamic unfreeze if no improvement
    if val_loss_history and avg_loss >= max(val_loss_history[-patience_threshold:]):
        base.mri_encoder.gradually_unfreeze(patience_threshold, patience_threshold)
        base.x_encoder.gradually_unfreeze(patience_threshold, patience_threshold)

    return avg_loss


def validate_epoch(model, loader, device):
    """
    One validation epoch over contrastive batches.
    """
    base = model.module if hasattr(model, "module") else model
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
             # 1) embed & pad MRI and X volumes
            e_mri = embed_and_pad(batch["mri"],   base.mri_encoder, device)  # (B, D_mri, E)
            e_x   = embed_and_pad(batch["x_img"], base.x_encoder,   device)  # (B, D_x,   E)

            # MRI attention‐pool
            D_mri    = e_mri.size(1)
            w_mri    = base.slice_weights[:D_mri]               # (D_mri,)
            alpha_mri= F.softmax(w_mri, dim=0)                  # (D_mri,)
            emri_m   = (e_mri * alpha_mri[None,:,None]).sum(1)  # (B, E)

            # X‐modality attention‐pool
            D_x      = e_x.size(1)
            w_x      = base.slice_weights[:D_x]                # (D_x,)
            alpha_x  = F.softmax(w_x, dim=0)                   # (D_x,)
            ex_m     = (e_x * alpha_x[None,:,None]).sum(1)     # (B, E)

            micro = batch["micro"].to(device)
            bioA  = batch["biomarker_A"].to(device)
            bioB  = batch["biomarker_B"].to(device)
            bioC  = batch["biomarker_C"].to(device)
            other = batch["other"].to(device)
            num   = batch["numeric"].to(device)
            labels= batch["label"]

            e_micro = base.micro_encoder(micro)
            e_bioA  = base.bioA_encoder(bioA)
            e_bioB  = base.bioB_encoder(bioB)
            e_bioC  = base.bioC_encoder(bioC)
            e_other = base.other_encoder(other)
            e_num   = base.num_encoder(num)

            loss = patient_contrastive_loss(
                emri_m, ex_m,
                e_micro, e_bioA, e_bioB, e_bioC, e_other, e_num,
                sample_labels=labels,
                tau=0.1

            )
            total_loss += loss.item()

    return total_loss / len(loader)



###########################################
# 9) Stacked Multi-Head Attention Classifier
###########################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class StackedAttentionClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,        # will end up with num_layers+1
        dropout: float = 0.1
    ):
        super().__init__()
        # -- stacked self‐attention --
        self.layers = nn.ModuleList()
        total_layers = num_layers + 1
        for _ in range(total_layers):
            self.layers.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
        self.dropout   = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

        # -- learned pooling: score each token, then weighted sum --
        self.pool_fc = nn.Linear(embed_dim, 1)

        # -- deeper MLP after pooling --
        self.post_pool_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, T, embed_dim)
        returns:    (B,)   one logit per sample
        """
        # 1) stacked self‐attention
        out = embeddings
        for attn in self.layers:
            attn_out, _ = attn(out, out, out)
            out = self.layernorm(out + self.dropout(attn_out))

        # 2) learned pooling weights
        #    pool_scores: (B, T)
        pool_scores = self.pool_fc(out).squeeze(-1)
        pool_weights = F.softmax(pool_scores, dim=1)

        # 3) weighted sum to get one (B, embed_dim) vector
        #    -> (B, 1, T) @ (B, T, E) -> (B, E)
        rep = torch.bmm(pool_weights.unsqueeze(1), out).squeeze(1)

        # 4) deeper MLP to produce final logit
        logits = self.post_pool_mlp(rep).squeeze(-1)
        return logits


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

import torch
from torch.utils.data import Dataset
import numpy as np

class AttentionDatasetWithLabels(Dataset):
    def __init__(
        self,
        patient_ids,
        data,
        mri_dict,
        x_dict,
        emb_model,
        attn_model,
        device,
        patient_labels,
        timepoints
    ):
        super().__init__()
        self.device     = device
        self.emb_model  = emb_model.to(device).eval()
        self.attn_model = attn_model.to(device).eval()

        base = emb_model.module if hasattr(emb_model, "module") else emb_model
        self.samples = []

        for pid in patient_ids:
            seqs = []
            for tp in timepoints:
                # 1) static row
                row = data[(data.Patient_ID == pid) & (data.Timepoint == tp)].iloc[0]

                # 2) load precomputed embeddings
                img_mri = mri_dict[(pid, tp)].unsqueeze(0).to(device)  # (1, D_mri, E)
                img_x   = x_dict[(pid, tp)].unsqueeze(0).to(device)    # (1, D_x,   E)

                # 3) load static vectors
                def to_tensor(pref):
                    vals = row.filter(like=pref).astype(float).fillna(0).values
                    return torch.from_numpy(vals.astype("float32")).unsqueeze(0).to(device)

                micro = to_tensor("Microbiome_")              # (1, E)
                bioA  = to_tensor("Biomarker_A_")             # (1, E)
                bioB  = to_tensor("Biomarker_B_")             # (1, E)
                bioC  = to_tensor("Biomarker_C_")             # (1, E)
                other = to_tensor("Other_")                   # (1, E)
                num   = to_tensor("Brain_CBF_Imaging_")       # (1, E)

                # 4) embed with frozen backbone
                with torch.no_grad():
                    e_mri, e_x, e_micro, e_bioA, e_bioB, e_bioC, e_other, e_num = \
                        self.emb_model(
                            mri=img_mri,
                            x_img=img_x,
                            micro=micro,
                            biomarker_A=bioA,
                            biomarker_B=bioB,
                            biomarker_C=bioC,
                            other=other,
                            numeric=num
                        )
                # e_mri: (1, D_mri, E), e_x: (1, D_x, E)

                # 5) pad/truncate X embedding to match MRI slice count
                _, D_mri, E = e_mri.shape
                _, D_x, _   = e_x.shape
                if D_x < D_mri:
                    pad = torch.zeros((1, D_mri - D_x, E), device=e_x.device)
                    e_x = torch.cat([e_x, pad], dim=1)
                elif D_x > D_mri:
                    e_x = e_x[:, :D_mri, :]

                # 6) tile static embeddings to (1, D_mri, E)
                def tile(x):
                    return x.unsqueeze(1).repeat(1, D_mri, 1)

                # 7) concatenate all modalities along the token dimension
                seq = torch.cat([
                    e_mri,
                    e_x,
                    tile(e_micro),
                    tile(e_bioA),
                    tile(e_bioB),
                    tile(e_bioC),
                    tile(e_other),
                    tile(e_num),
                ], dim=1)  # (1, 8 * D_mri, E)

                seqs.append(seq.squeeze(0))  # (8 * D_mri, E)

            # 8) concatenate across timepoints
            full_seq = torch.cat(seqs, dim=0)  # (T_total, E)
            lbl = torch.tensor(patient_labels[pid], dtype=torch.float32)
            self.samples.append((full_seq, lbl))
    

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

class FocalLoss(nn.Module):
    def __init__(self,
                 init_alpha_pos: float = 0.8,
                 init_alpha_neg: float = 0.2,
                 gamma:         float = 4.0,
                 reduction:     str   = "mean"):
        """
        A focal loss where the positive‐class weight (alpha_pos)
        and negative-class weight (alpha_neg) are learned.

        init_alpha_pos: starting weight for true-1 examples
        init_alpha_neg: starting weight for true-0 examples
        gamma:          focusing parameter
        """
        super().__init__()
        # these two will now get gradients
        self.alpha_pos = nn.Parameter(torch.tensor(init_alpha_pos, dtype=torch.float32))
        self.alpha_neg = nn.Parameter(torch.tensor(init_alpha_neg, dtype=torch.float32))
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,) raw scores
        targets: (B,) floats in {0.0,1.0}
        """
        # 1) standard BCE part
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B,)

        # 2) probability of the true class
        probs = torch.sigmoid(logits)  # (B,)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)  # (B,)

        # 3) ensure alphas stay in (0,1)
        alpha_pos = torch.sigmoid(self.alpha_pos)  # scalar ∈ (0,1)
        alpha_neg = torch.sigmoid(self.alpha_neg)  # scalar ∈ (0,1)

        # 4) pick the correct alpha per example
        alpha_t = targets * alpha_pos + (1.0 - targets) * alpha_neg  # (B,)

        # 5) focal modulation
        mod_term = (1.0 - p_t) ** self.gamma  # (B,)

        # 6) combined loss
        loss = alpha_t * mod_term * ce_loss  # (B,)

        # 7) reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        
from torch.utils.data import Dataset

def main():
    import os
    import numpy as np
    import pandas as pd
    import torch
    from torch import amp
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve

    # ─── WANDB SETUP ─────────────────────────────────────────────────────────
    wandb.init(
        project="brain-region-prediction",
        config={
            "batch_size":64,
            "embed_dim": 64,
            "neg_frac": 20,
            "pos_repeat": 1,
            "lr_clip": 5e-3,
            "lr_proj": 5e-3,
            "lr_mlp":  5e-3,
            "epochs_pre": 100,
            "patience_pre": 3,
            "epochs_attn": 700,
            "patience_attn": 20,
            "ensemble_size":3,
            "threshold": 0.5,
            "num_layer": 2,
            "drop_out":0.4,
            "num_run": 5
        },
    )
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"  → GPU count: {torch.cuda.device_count()}")
        # if you only have one GPU you can do
        print(f"  → Using GPU #{torch.cuda.current_device()}: "
            f"{torch.cuda.get_device_name(torch.cuda.current_device())}")
        scaler_pre = amp.GradScaler()
    TIMEPOINTS = ["Baseline","Post"]

    # initialize run‐level accumulators
    all_runs           = []
    overall_accuracies = []
    test_accuracies    = []
    shap_records       = {m: [] for m in [
        "MRI","X","Microbiome","Biomarker_A","Biomarker_B",
        "Biomarker_C","Other","Numeric"
    ]}

    for run in range(1, config.num_run+1):
        print(f"\n===== RUN {run}/{config.num_run} =====")

        # 1) Load & merge CSVs
        data = load_data()

        # 2) Prepare patient‐level APOE4 labels & splits
        other_df = pd.read_csv('/bmlfast/tom/New1/Other.csv')
        other_df.rename(columns=lambda x: x.strip(), inplace=True)
        other_df.Patient_ID = other_df.Patient_ID.astype(str).str.strip()
        patient_apoe4 = dict(other_df.groupby("Patient_ID")["APOE4"].max())
        all_patients = list(patient_apoe4.keys())

        trainval_ids, test_ids = train_test_split(
            all_patients, test_size=4,
            stratify=[patient_apoe4[p] for p in all_patients],
            random_state=run
        )
        train_ids, val_ids = train_test_split(
            trainval_ids, test_size=3,
            stratify=[patient_apoe4[p] for p in trainval_ids],
            random_state=run
        )
        print(f"Train: {train_ids}, Val: {val_ids}, Test: {test_ids}")

        def select_rows(ids):
            return (data[data.Patient_ID.isin(ids)]
                    .sort_values(["Patient_ID","Timepoint"])
                    .groupby("Patient_ID").head(3)
                    .reset_index(drop=True))
        train_data = select_rows(train_ids)
        val_data   = select_rows(val_ids)

        # 3) Build multimodal embedding backbone
        dims = {
            "micro": train_data.filter(like="Microbiome_").shape[1],
            "bioA":  train_data.filter(like="Biomarker_A_").shape[1],
            "bioB":  train_data.filter(like="Biomarker_B_").shape[1],
            "bioC":  train_data.filter(like="Biomarker_C_").shape[1],
            "other": train_data.filter(like="Other_").shape[1],
            "num":   train_data.filter(like="Brain_CBF_Imaging_").shape[1],
        }
        model = MultiModalEmbeddingModel(
            dims["micro"], dims["bioA"], dims["bioB"], dims["bioC"],
            dims["other"], dims["num"],
            embed_dim=config.embed_dim,
            augment=True
        )
        model = torch.nn.DataParallel(model).to(device)
        wandb.watch(model, log="all")

        # 4) Precompute & load CLIP embeddings for MRI + X
        EMB_CACHE = "/bmlfast/tom/mri_cache_embed64"
        keys = list(zip(data.Patient_ID, data.Timepoint))
        mri_dict = load_precomputed(keys, EMB_CACHE)

        X_ROOT = "/bmlfast/tom/Perfusion_images"
        X_CACHE = "/bmlfast/tom/Perfusion_cache"
        x_raw_dict = load_mri_data(X_ROOT, cache_dir=X_CACHE)
        X_EMB_CACHE = "/bmlfast/tom/x_cache_embed64"
        precompute_embeddings(x_raw_dict, model.module.x_encoder, X_EMB_CACHE, device)
        x_dict = load_precomputed(keys, X_EMB_CACHE)

        # 5) Contrastive pretraining
        optimizer = torch.optim.AdamW(
            [p for enc in ("mri_encoder","x_encoder") for p in (
                {"params": getattr(model.module,enc).clip_model.vision_model.encoder.parameters(), "lr":config.lr_clip},
                {"params": getattr(model.module,enc).project.parameters(),                               "lr":config.lr_proj},
            )] + [
                {"params": getattr(model.module,name).parameters(), "lr":config.lr_mlp}
                for name in ("micro_encoder","bioA_encoder","bioB_encoder","bioC_encoder","other_encoder","num_encoder")
            ],
            weight_decay=4
        )
        scheduler_pre = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
        )

        train_ds = CombinedContrastiveDataset(
            train_data, mri_dict, x_dict,
            negative_sample_fraction=config.neg_frac,
            positive_repeat=config.pos_repeat,
            augment=True
        )
        val_ds = CombinedContrastiveDataset(
            val_data, mri_dict, x_dict,
            negative_sample_fraction=config.neg_frac,
            positive_repeat=config.pos_repeat,
            augment=False
        )
        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size,
            shuffle=True, num_workers=8, pin_memory=True,
            collate_fn=custom_collate
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size,
            shuffle=False, num_workers=8, pin_memory=True,
            collate_fn=custom_collate
        )

        best_val_loss = float('inf')
        val_history = []
        for pre_ep in range(1, config.epochs_pre+1):
            tr_loss = train_epoch(
                model, train_loader, optimizer, device, pre_ep,
                val_history, patience_threshold=config.patience_pre,
                accum_steps=4, scaler=scaler_pre
            )
            val_loss = validate_epoch(model, val_loader, device)
            val_history.append(val_loss)
            scheduler_pre.step(val_loss)
            wandb.log({
                "pretrain/train_loss": tr_loss,
                "pretrain/val_loss": val_loss,
                "epoch": pre_ep
            })

            print(f"[Pretrain] Epoch {pre_ep}/{config.epochs_pre}  "f"Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}")

            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = model.state_dict()
            if len(val_history) >= config.patience_pre and val_loss >= max(val_history[-config.patience_pre:]):
                print(f"Early stopping pretrain at epoch {pre_ep}")
                break

        if best_val_loss < float('inf'):
            model.load_state_dict(best_state)

        # 6) Attention phase: train ensemble of heads
        ensemble_models = []
        for ens in range(config.ensemble_size):
            attn_model = StackedAttentionClassifier(
                embed_dim=config.embed_dim,
                num_heads=2,
                num_layers=config.num_layer,
                dropout=config.drop_out
            ).to(device)

            # build datasets
            train_attn_ds = AttentionDatasetWithLabels(
                train_ids, data, mri_dict, x_dict,
                model, attn_model, device,
                patient_apoe4, TIMEPOINTS
            )
            val_attn_ds = AttentionDatasetWithLabels(
                val_ids, data, mri_dict, x_dict,
                model, attn_model, device,
                patient_apoe4, TIMEPOINTS
            )
            test_attn_ds = AttentionDatasetWithLabels(
                test_ids, data, mri_dict, x_dict,
                model, attn_model, device,
                patient_apoe4, TIMEPOINTS
            )

            # inside the for ens in range(config.ensemble_size): loop …
            labels    = [patient_apoe4[pid] for pid in train_ids]
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count

            # ------------------------------------------------------------------
            # ↓↓↓ new: BCEWithLogitsLoss with class balancing ↓↓↓
            pos_weight = torch.tensor([neg_count / max(pos_count, 1)],
                                    dtype=torch.float32,
                                    device=device)

            criterion_attn = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction="mean"
            )
            # ------------------------------------------------------------------

            # weighted sampler (unchanged – remains useful for variance reduction)
            weights = [pos_weight.item() if patient_apoe4[pid] else 1.0
                    for pid in train_ids]
            sampler = WeightedRandomSampler(weights,
                                            num_samples=len(weights),
                                            replacement=True)


            train_attn_loader = DataLoader(
                train_attn_ds, batch_size=64, sampler=sampler,
                collate_fn=attn_collate_with_labels
            )
            val_attn_loader = DataLoader(
                val_attn_ds, batch_size=64, shuffle=False,
                collate_fn=attn_collate_with_labels
            )
            test_attn_loader = DataLoader(
                test_attn_ds, batch_size=64, shuffle=False,
                collate_fn=attn_collate_with_labels
            )

            # optimizer & loss
        
            optimizer_attn = torch.optim.AdamW(
                [
                    {"params": attn_model.parameters(),      "lr": 1e-4},
                    
                ],
                weight_decay=5e-3
            )

            best_head_val = float('inf')
            no_improve    = 0
            for epoch in range(1, config.epochs_attn+1):
                # train head
                attn_model.train()
                tr_sum = 0.0
                for emb, lbl in train_attn_loader:
                    emb, lbl = emb.to(device), lbl.to(device)
                    # emb = emb + torch.randn_like(emb)*0.1
                    optimizer_attn.zero_grad()
                    with amp.autocast(device_type="cuda"):
                        logits = attn_model(emb)
                        with torch.no_grad():
                            probs = torch.sigmoid(logits)
                            print(f"  ↳ [train] min/max prob: {probs.min().item():.4f}/{probs.max().item():.4f}")
                  
                        loss   = criterion_attn(logits, lbl)
                        preds = (probs >= 0.5).float()
                        train_acc = (preds == lbl).float().mean().item()
                        print(f"  ↳ [train] Acc: {train_acc:.3f},  Prob-range: {probs.min():.3f}–{probs.max():.3f}")

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(attn_model.parameters(),1.0)
                    optimizer_attn.step()
                    tr_sum += loss.item()*emb.size(0)
                tr_loss = tr_sum / len(train_attn_ds)

                # validate head
                attn_model.eval()
                val_sum = 0.0
                with torch.no_grad():
                    for emb, lbl in val_attn_loader:
                        emb, lbl = emb.to(device), lbl.to(device)
                        logits = attn_model(emb)
                        val_sum += criterion_attn(logits, lbl).item()*emb.size(0)
                val_loss = val_sum / len(val_attn_ds)
                if ens == 0:
                    if epoch == 1 or epoch % 5 == 0:
                        print(
                            f"[Ensemble {ens+1}] Epoch {epoch}/{config.epochs_attn}  "
                            f"Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}"
                        )


                if val_loss < best_head_val:
                    best_head_val = val_loss
                    best_head_state = attn_model.state_dict()
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= config.patience_attn:
                    print(f"Early stopping ensemble {ens+1} at epoch {epoch}")
                    break

            if best_head_val < float('inf'):
                attn_model.load_state_dict(best_head_state)
            ensemble_models.append(attn_model)

        if not ensemble_models:
            raise RuntimeError("No attention heads were trained!")
        best_head = ensemble_models[0]

        # 7) Final evaluation & SHAP

        # rebuild datasets with best_head
        val_ds   = AttentionDatasetWithLabels(val_ids,   data, mri_dict, x_dict, model, best_head, device, patient_apoe4, TIMEPOINTS)
        test_ds  = AttentionDatasetWithLabels(test_ids,  data, mri_dict, x_dict, model, best_head, device, patient_apoe4, TIMEPOINTS)
        full_ds  = AttentionDatasetWithLabels(all_patients, data, mri_dict, x_dict, model, best_head, device, patient_apoe4, TIMEPOINTS)

        val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False, collate_fn=attn_collate_with_labels)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=attn_collate_with_labels)
        full_loader = DataLoader(full_ds, batch_size=64, shuffle=False, collate_fn=attn_collate_with_labels)

        # threshold selection on validation
        val_probs, val_labels = [], []
        with torch.no_grad():
            for emb, lbl in val_loader:
                emb = emb.to(device)
                lg = best_head(emb)
                val_probs .append(torch.sigmoid(lg).cpu().numpy())
                val_labels.append(lbl.numpy())
        val_probs  = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        prec, rec, thr = precision_recall_curve(val_labels, val_probs)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
        ix = np.nanargmax(f1_scores)
        best_thresh, best_f1 = thr[ix], f1_scores[ix]
        print(f"→ Best threshold: {best_thresh:.3f}, F1={best_f1:.3f}")
        # disable dropout in all heads
        for head in ensemble_models:
            head.eval()

        # overall accuracy on full dataset
        total_c, total_s = 0, 0
        with torch.no_grad():
            for emb, lbl in full_loader:
                emb, lbl = emb.to(device), lbl.to(device).float()
                logits = sum(m(emb) for m in ensemble_models) / len(ensemble_models)
                preds  = (torch.sigmoid(logits) >= 0.5).float()
                total_c += (preds == lbl).sum().item()
                total_s += lbl.size(0)
        overall_acc = total_c / total_s
        print(f"Overall Accuracy: {overall_acc*100:.2f}%")

        # per-patient test evaluation
        total_correct_test = 0
        total_samples_test = 0
        run_records = []

        with torch.no_grad():
            for emb, lbl in test_loader:
                emb, lbl = emb.to(device), lbl.to(device).float()
                # ensemble‐average logits
                logits = sum(m(emb) for m in ensemble_models) / len(ensemble_models)
                probs  = torch.sigmoid(logits)
                preds  = (probs >= 0.5).float()

                total_correct_test += (preds == lbl).sum().item()
                total_samples_test += lbl.size(0)

                # **Here**: loop over the batch and print each prediction with probability
                for i, pid in enumerate(test_ids):
                    true = int(lbl[i].item())
                    prob = float(probs[i].item())
                    pred = int(preds[i].item())
                    print(
                        f"Predicting Patient {pid}: "
                        f"True={true}, Pred={pred}, Prob={prob:.4f}"
                    )

                    run_records.append({
                        "Run":           run,
                        "Patient_ID":    pid,
                        "TrueLabel":     float(true),
                        "PredProb":      prob,
                        "PredictedLabel": float(pred)
                    })

        test_acc = total_correct_test / total_samples_test
        print(f"Test Accuracy: {test_acc*100:.2f}%")

        # accumulate results
        overall_accuracies.append(overall_acc)
        test_accuracies.append(test_acc)
        all_runs.append(pd.DataFrame(run_records))

        # Kernel SHAP modality‐level analysis
        # shap_ds = AttentionDatasetWithLabels(train_ids, data, mri_dict, x_dict, model, best_head, device, patient_apoe4, TIMEPOINTS)
        # if len(shap_ds) > 0:
        #     sample_emb, _ = shap_ds[0]
        #     T, E = sample_emb.shape
        #     sample_flat = sample_emb.view(-1).cpu().numpy()
        #     def wrapper(X: np.ndarray) -> np.ndarray:
        #         X_t = torch.from_numpy(X).float().to(device)
        #         B = X_t.size(0)
        #         X_t = X_t.view(B, T, E)
        #         with torch.no_grad():
        #             return best_head(X_t).cpu().numpy()
        #     baseline = np.zeros((1, T * E), dtype=np.float32)
        #     explainer = shap.KernelExplainer(wrapper, baseline)
        #     shap_vals = explainer.shap_values(sample_flat[None, :], nsamples=1000)[0]
        #     D = T // 8
        #     modality_slices = {
        #         "MRI":         slice(0*D*E, 1*D*E),
        #         "X":           slice(1*D*E, 2*D*E),
        #         "Microbiome":  slice(2*D*E, 3*D*E),
        #         "Biomarker_A": slice(3*D*E, 4*D*E),
        #         "Biomarker_B": slice(4*D*E, 5*D*E),
        #         "Biomarker_C": slice(5*D*E, 6*D*E),
        #         "Other":       slice(6*D*E, 7*D*E),
        #         "Numeric":     slice(7*D*E, 8*D*E),
        #     }
        #     print("Modality‐level SHAP importance:")
        #     for mod, sl in modality_slices.items():
        #         imp = np.sum(np.abs(shap_vals[sl]))
        #         print(f"  {mod}: {imp:.4f}")
        #         shap_records[mod].append(imp)

    # end for run
    # save and summarize across runs
    df_all = pd.concat(all_runs, ignore_index=True)
    df_all.to_csv("20_runs.csv", index=False)

    # 2) print overall summaries
    avg_ov, std_ov = np.mean(overall_accuracies)*100, np.std(overall_accuracies)*100
    avg_te, std_te = np.mean(test_accuracies)*100,    np.std(test_accuracies)*100
    print(f"\nAverage Overall Accuracy: {avg_ov:.2f}% (±{std_ov:.2f}%)")
    print(f"Average Test    Accuracy: {avg_te:.2f}% (±{std_te:.2f}%)\n")

    # 3) detailed per‐run breakdown
    print("Detailed test results by run:")
    for run_id, group in df_all.groupby("Run"):
        print(f"\n––– Run {run_id} –––")
        print(group[["Patient_ID","TrueLabel","PredictedLabel"]].to_string(index=False))

if __name__ == "__main__":
    main()
