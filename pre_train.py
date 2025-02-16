import os
import math
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

import torchvision.transforms as T

############################################################
# 1) CSV Data Loading & Merging
############################################################
def load_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        # Drop APOE4 column if present.
        if 'APOE4' in df.columns:
            df.drop(columns=['APOE4'], inplace=True)
        return df
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_path}")
        return pd.DataFrame()

def load_data():
    microbiome = load_file('/home/tmnthc/New/Microbiome.csv')
    blood_metabolites = load_file('/home/tmnthc/New/Blood_Metabolites.csv')
    inflammatory_markers = load_file('/home/tmnthc/New/Sirolimus_inflammatory_markers.csv')
    blood_data = load_file('/home/tmnthc/New/Sirolimus_Blood_Data.csv')
    other_data = load_file('/home/tmnthc/New/Other.csv')
    
    for name, df in zip(
        ["Microbiome", "Blood Metabolites", "Inflammatory Markers", "Blood Data", "Other"],
        [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data]
    ):
        if df.empty:
            print(f"⚠️ Warning: {name} data is empty or missing.")
        else:
            print(f"{name} data loaded with shape: {df.shape}")
    
    for df in [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data]:
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
    
    def add_prefix(df, prefix):
        df = df.copy()
        cols = [c for c in df.columns if c not in ['Patient_ID','Timepoint']]
        df.rename(columns={c: f"{prefix}_{c}" for c in cols}, inplace=True)
        return df
    
    micro_num  = add_prefix(microbiome, "Microbiome")
    blood_met_num  = add_prefix(blood_metabolites, "Biomarker")
    blood_data_num = add_prefix(blood_data, "Biomarker")
    inflam_num = add_prefix(inflammatory_markers, "Biomarker")
    
    biomarker_data = blood_met_num.merge(blood_data_num, on=['Patient_ID', 'Timepoint'], how='outer')
    biomarker_data = biomarker_data.merge(inflam_num, on=['Patient_ID', 'Timepoint'], how='outer')
    
    other_num  = add_prefix(other_data, "Other")
    
    data = other_num.copy()
    for df2 in [micro_num, biomarker_data]:
        data = data.merge(df2, on=['Patient_ID','Timepoint'], how='outer')
    data.fillna(0, inplace=True)
    print(f"Final merged data shape: {data.shape}")
    return data

############################################################
# 2) MRI Data Loading (for brain imaging)
############################################################
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
                    if len(mri_image.shape) == 4:
                        mri_image = np.mean(mri_image, axis=-1)
                    mri_tensor = torch.tensor(mri_image, dtype=torch.float32).unsqueeze(0)
                    mri_dict[(patient_id, timepoint_name)] = mri_tensor
    return mri_dict

############################################################
# Safe normalization and Augmentation Functions
############################################################
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

def advanced_numeric_augmentation(x, noise_std=0.05, scale_range=(0.9, 1.1)):
    scale = torch.empty(1).uniform_(*scale_range).item()
    noise = torch.randn_like(x) * noise_std
    return x * scale + noise

############################################################
# 4) MRIClipEncoder with Multi-Slice Aggregation
############################################################
class MRIClipEncoder(nn.Module):
    def __init__(self, embed_dim=64, augment=False, dropout_p=0.0):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # Freeze all layers initially, then unfreeze last 2 layers.
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self._unfreeze_last_n_layers(2)
        
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Use LayerNorm (instead of BatchNorm1d) and no dropout.
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
    
    def forward(self, mri_batch):
        """
        For each MRI volume, extract three slices: center, one above, and one below.
        Compute CLIP features for each slice, project them, and average the embeddings.
        """
        device = mri_batch.device
        B = mri_batch.size(0)
        all_proj = []
        for i in range(B):
            vol = mri_batch[i, 0]
            if vol.ndim != 3:
                vol = torch.zeros((128,128,128), dtype=torch.float32, device=device)
            D, H, W = vol.shape
            center = D // 2
            idxs = [center]
            if center - 1 >= 0:
                idxs.append(center - 1)
            if center + 1 < D:
                idxs.append(center + 1)
            slice_features = []
            for idx in idxs:
                slice_2d = vol[idx].cpu().numpy()
                rng = slice_2d.max() - slice_2d.min()
                slice_2d = (slice_2d - slice_2d.min()) / (rng + 1e-8)
                slice_2d = (slice_2d * 255.0).astype(np.uint8)
                if slice_2d.ndim != 2:
                    slice_2d = np.zeros((128,128), dtype=np.uint8)
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
            avg_proj = torch.mean(torch.cat(slice_features, dim=0), dim=0, keepdim=True)
            all_proj.append(avg_proj)
        return torch.cat(all_proj, dim=0)

############################################################
# 5) MLPEncoder for Numeric Data with Increased Capacity
############################################################
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=64, augment=False, dropout_p=0.0):
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

############################################################
# 6) CombinedContrastiveDataset for 4 Modalities
############################################################
class CombinedContrastiveDataset(Dataset):
    def __init__(self, data, mri_dict, negative_sample_fraction=1, positive_repeat=10, augment=False):
        self.data = data.reset_index(drop=True)
        self.mri_dict = mri_dict
        self.N = len(self.data)
        self.augment = augment
        self.positive_samples = [(i, i, i, i) for i in range(self.N) for _ in range(positive_repeat)]
        num_negatives = int(self.N * negative_sample_fraction)
        negative_samples = []
        while len(negative_samples) < num_negatives:
            sample = [random.choice(range(self.N)) for _ in range(4)]
            if len(set(sample)) > 1:
                negative_samples.append(tuple(sample))
        self.negative_samples = negative_samples
        self.samples = [(s, 1) for s in self.positive_samples] + [(s, 0) for s in self.negative_samples]
        print(f"Total combined samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        indices, label = self.samples[idx]
        i, j, k, l = indices
        pid_mri, tpt_mri = str(self.data.loc[i, "Patient_ID"]), str(self.data.loc[i, "Timepoint"])
        pid_micro, tpt_micro = str(self.data.loc[j, "Patient_ID"]), str(self.data.loc[j, "Timepoint"])
        pid_biom, tpt_biom = str(self.data.loc[k, "Patient_ID"]), str(self.data.loc[k, "Timepoint"])
        pid_other, tpt_other = str(self.data.loc[l, "Patient_ID"]), str(self.data.loc[l, "Timepoint"])
        
        mri_tensor = self.mri_dict.get((pid_mri, tpt_mri), torch.zeros((1,128,128,128), dtype=torch.float32))
        micro_tensor = torch.tensor(self.data.filter(like="Microbiome_").iloc[j].values.astype(np.float32))
        biom_tensor = torch.tensor(self.data.filter(like="Biomarker_").iloc[k].values.astype(np.float32))
        other_tensor = torch.tensor(self.data.filter(like="Other_").iloc[l].values.astype(np.float32))
        
        if self.augment and label == 1:
            micro_tensor = advanced_numeric_augmentation(micro_tensor)
            biom_tensor = advanced_numeric_augmentation(biom_tensor)
            other_tensor = advanced_numeric_augmentation(other_tensor)
        
        patient_ids = torch.tensor([int(pid_mri), int(pid_micro), int(pid_biom), int(pid_other)], dtype=torch.long)
        return {
            "mri": mri_tensor,
            "micro": micro_tensor,
            "biom": biom_tensor,
            "other": other_tensor,
            "patient_ids": patient_ids,
            "label": torch.tensor(label, dtype=torch.float32)
        }

############################################################
# 8) Original Patient Contrastive Loss (unchanged)
############################################################
def patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, patient_ids, tau=0.03):
    B, D = e_mri.shape
    labels = torch.cat([
        patient_ids[:, 0],
        patient_ids[:, 1],
        patient_ids[:, 2],
        patient_ids[:, 3]
    ], dim=0)
    emb_all = torch.cat([e_mri, e_micro, e_biom, e_other], dim=0)
    sim_matrix = torch.matmul(emb_all, emb_all.t()) / tau
    diag_mask = torch.eye(4 * B, dtype=torch.bool, device=sim_matrix.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~diag_mask)
    neg_mask = ~pos_mask & (~diag_mask)
    sim_pos = sim_matrix * pos_mask.float()
    sim_neg = sim_matrix * neg_mask.float()
    eps = 1e-8
    sum_pos = torch.exp(sim_pos).sum(dim=1)
    sum_neg = torch.exp(sim_neg).sum(dim=1)
    loss = -torch.log((sum_pos + eps) / (sum_pos + sum_neg + eps))
    return loss.mean()

############################################################
# 10) Training & Validation with AdamW, Warmup + Cosine Annealing,
# and Progressive Unfreezing of the CLIP Encoder.
############################################################
def train_epoch(model, loader, optimizer, device, current_epoch):
    model.train()
    total_loss = 0
    # Progressive unfreezing: at epoch 30, unfreeze last 4 layers; at epoch 60, unfreeze all layers.
    if current_epoch == 30:
        model.mri_encoder._unfreeze_last_n_layers(4)
    if current_epoch == 60:
        model.mri_encoder._unfreeze_last_n_layers(12)  # assuming total 12 layers
    for batch in loader:
        mri = batch["mri"].to(device)
        micro = batch["micro"].to(device)
        biom = batch["biom"].to(device)
        other = batch["other"].to(device)
        pids = batch["patient_ids"].to(device)
        optimizer.zero_grad()
        e_mri, e_micro, e_biom, e_other = model(mri, micro, biom, other)
        loss = patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, patient_ids=pids, tau=0.03)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            mri = batch["mri"].to(device)
            micro = batch["micro"].to(device)
            biom = batch["biom"].to(device)
            other = batch["other"].to(device)
            pids = batch["patient_ids"].to(device)
            e_mri, e_micro, e_biom, e_other = model(mri, micro, biom, other)
            loss = patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, pids, tau=0.03)
            total_loss += loss.item()
    return total_loss / len(loader)

############################################################
# 11) Plotting Helpers & Embedding Analysis Functions
############################################################
def plot_metrics(train_losses, val_losses, epochs):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved loss.png")

def knn_accuracy(embs, labels, k=1):
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(embs, labels)
    preds = knn.predict(embs)
    return (preds == labels).mean()

def tsne_visualization(embeds, group_labels, full_labels, title='t-SNE'):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1500, random_state=42)
    e2d = tsne.fit_transform(embeds)
    unique_groups = np.unique(group_labels)
    cmap = plt.get_cmap('tab10', len(unique_groups))
    group_color = {g: cmap(i) for i, g in enumerate(unique_groups)}
    marker_dict = {"mri": "o", "micro": "s", "biom": "D", "other": "^"}
    plt.figure(figsize=(10,8))
    plotted = {}
    for i, (x, y) in enumerate(e2d):
        group = group_labels[i]
        mod = full_labels[i].split('_')[-1].lower()
        marker = marker_dict.get(mod, 'o')
        legend_label = f"{group}_{mod}"
        if legend_label not in plotted:
            plt.scatter(x, y, c=group_color[group], marker=marker, s=80, label=legend_label)
            plotted[legend_label] = True
        else:
            plt.scatter(x, y, c=group_color[group], marker=marker, s=80)
    plt.title(title)
    plt.xlabel("t-SNE dim1")
    plt.ylabel("t-SNE dim2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("tsne_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved tsne_plot.png")

def umap_visualization(embeds, group_labels, full_labels, title='UMAP'):
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.01, random_state=42)
    e2d = reducer.fit_transform(embeds)
    unique_groups = np.unique(group_labels)
    cmap = plt.get_cmap('tab10', len(unique_groups))
    group_color = {g: cmap(i) for i, g in enumerate(unique_groups)}
    marker_dict = {"mri": "o", "micro": "s", "biom": "D", "other": "^"}
    plt.figure(figsize=(10,8))
    plotted = {}
    for i, (x, y) in enumerate(e2d):
        group = group_labels[i]
        mod = full_labels[i].split('_')[-1].lower()
        marker = marker_dict.get(mod, 'o')
        legend_label = f"{group}_{mod}"
        if legend_label not in plotted:
            plt.scatter(x, y, c=group_color[group], marker=marker, s=80, label=legend_label)
            plotted[legend_label] = True
        else:
            plt.scatter(x, y, c=group_color[group], marker=marker, s=80)
    plt.title(title)
    plt.xlabel("UMAP dim1")
    plt.ylabel("UMAP dim2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("umap_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved umap_plot.png")

############################################################
# 9) MultiModalEmbeddingModel combining all encoders
############################################################
class MultiModalEmbeddingModel(nn.Module):
    def __init__(self, micro_dim, biom_dim, other_dim, embed_dim=64, augment=False):
        super().__init__()
        self.mri_encoder = MRIClipEncoder(embed_dim=embed_dim, augment=augment)
        self.micro_encoder = MLPEncoder(micro_dim, output_dim=embed_dim, augment=augment)
        self.biom_encoder = MLPEncoder(biom_dim, output_dim=embed_dim, augment=augment)
        self.other_encoder = MLPEncoder(other_dim, output_dim=embed_dim, augment=augment)
    
    def forward(self, mri, micro, biom, other):
        e_mri = self.mri_encoder(mri)
        e_micro = self.micro_encoder(micro)
        e_biom = self.biom_encoder(biom)
        e_other = self.other_encoder(other)
        return e_mri, e_micro, e_biom, e_other

############################################################
# 13) Main function
############################################################
def main():
    data = load_data()
    mri_dict = load_mri_data("/home/tmnthc/CBF_imaging")
    
    all_pats = np.unique(data["Patient_ID"])
    if len(all_pats) < 19:
        raise ValueError("Need at least 19 patients for this example.")
    
    selected_pats = np.random.choice(all_pats, size=19, replace=False)
    train_pats = selected_pats[:16]
    val_pats = selected_pats[16:]
    
    print("Training Patient IDs:", train_pats)
    print("Validation Patient IDs:", val_pats)
    
    train_data = data[data["Patient_ID"].isin(train_pats)]
    val_data = data[data["Patient_ID"].isin(val_pats)]
    
    train_data = train_data.sort_values(["Patient_ID", "Timepoint"]).groupby("Patient_ID").head(3).reset_index(drop=True)
    val_data = val_data.sort_values(["Patient_ID", "Timepoint"]).groupby("Patient_ID").head(3).reset_index(drop=True)
    
    print(f"Training data rows: {len(train_data)}")
    print(f"Validation data rows: {len(val_data)}")
    
    train_set = CombinedContrastiveDataset(train_data, mri_dict, negative_sample_fraction=8, positive_repeat=2, augment=True)
    val_set = CombinedContrastiveDataset(val_data, mri_dict, negative_sample_fraction=8, positive_repeat=2, augment=False)
    
    print(f"Number of training combined samples: {len(train_set)}")
    print(f"Number of validation combined samples: {len(val_set)}")
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
    
    micro_dim = train_data.filter(like='Microbiome_').shape[1]
    biom_dim = train_data.filter(like='Biomarker_').shape[1]
    other_dim = train_data.filter(like='Other_').shape[1]
    print(f"Micro={micro_dim}, Biom={biom_dim}, Other={other_dim}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiModalEmbeddingModel(micro_dim, biom_dim, other_dim, embed_dim=64, augment=True).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.mri_encoder.clip_model.vision_model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.mri_encoder.project.parameters(), 'lr': 5e-3},
        {'params': model.micro_encoder.parameters(), 'lr': 1e-3},
        {'params': model.biom_encoder.parameters(), 'lr': 1e-3},
        {'params': model.other_encoder.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-6)
    
    total_epochs = 120
    warmup_epochs = 10
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    train_losses = []
    val_losses = []
    for ep in range(1, total_epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device, current_epoch=ep)
        val_loss = validate_epoch(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {ep}/{total_epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
    
    plot_metrics(train_losses, val_losses, total_epochs)
    torch.save(model.state_dict(), "model_checkpoint.pth")
    print("Checkpoint saved as model_checkpoint.pth")
    
    # Embedding Analysis
    class PositiveModalityDataset(Dataset):
        def __init__(self, data, mri_dict):
            self.data = data.reset_index(drop=True)
            self.mri_dict = mri_dict
            self.modalities = ["mri", "micro", "biom", "other"]
        def __len__(self):
            return len(self.data) * len(self.modalities)
        def __getitem__(self, idx):
            row_idx = idx // len(self.modalities)
            mod_idx = idx % len(self.modalities)
            mod = self.modalities[mod_idx]
            row = self.data.iloc[row_idx]
            pid = str(row["Patient_ID"])
            tpt = str(row["Timepoint"])
            group_label = f"{pid}_{tpt}"
            if mod == "mri":
                dp = self.mri_dict.get((pid, tpt), torch.zeros((1,128,128,128), dtype=torch.float32))
            elif mod == "micro":
                dp = torch.tensor(self.data.filter(like="Microbiome_").iloc[row_idx].values.astype(np.float32))
            elif mod == "biom":
                dp = torch.tensor(self.data.filter(like="Biomarker_").iloc[row_idx].values.astype(np.float32))
            elif mod == "other":
                dp = torch.tensor(self.data.filter(like="Other_").iloc[row_idx].values.astype(np.float32))
            else:
                raise ValueError(f"Unknown modality {mod}")
            full_label = f"{group_label}_{mod}"
            return {"data": dp, "group_label": group_label, "modality_label": full_label}
    
    val_ds = PositiveModalityDataset(val_data, mri_dict)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    all_embs = []
    all_groups = []
    all_mods = []
    
    model.eval()
    with torch.no_grad():
        for sample in val_ld:
            dp = sample["data"].to(device)
            group_label = sample["group_label"][0]
            mod_label = sample["modality_label"][0]
            mod_type = mod_label.split('_')[-1]
            if mod_type == "mri":
                emb = model.mri_encoder(dp.unsqueeze(0))
            elif mod_type == "micro":
                emb = model.micro_encoder(dp)
            elif mod_type == "biom":
                emb = model.biom_encoder(dp)
            elif mod_type == "other":
                emb = model.other_encoder(dp)
            else:
                raise ValueError(f"Unknown modality: {mod_type}")
            all_embs.append(emb.cpu().numpy())
            all_groups.append(group_label)
            all_mods.append(mod_label)
    
    all_embs = np.concatenate(all_embs, axis=0)
    scaler = StandardScaler()
    all_embs = scaler.fit_transform(all_embs)
    if all_embs.shape[0] > 50:
        pca = PCA(n_components=50, whiten=True)
        all_embs = pca.fit_transform(all_embs)
    all_embs = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
    
    acc = knn_accuracy(all_embs, all_groups, k=1)
    print(f"KNN accuracy (k=1) on val embeddings: {acc*100:.2f}%")
    
    tsne_visualization(all_embs, all_groups, all_mods, title="t-SNE of Validation")
    umap_visualization(all_embs, all_groups, all_mods, title="UMAP of Validation")

if __name__ == "__main__":
    main()
