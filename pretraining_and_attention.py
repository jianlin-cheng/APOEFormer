import random
import numpy as np
import torch

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
# For more reproducible results (may impact performance)
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

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# For scaling and PCA in embedding analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torchvision.transforms as T

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
    microbiome = load_file('/home/tmnthc/New1/Microbiome.csv')
    blood_metabolites = load_file('/home/tmnthc/New1/Blood_Metabolites.csv')
    inflammatory_markers = load_file('/home/tmnthc/New1/Sirolimus_inflammatory_markers.csv')
    blood_data = load_file('/home/tmnthc/New1/Sirolimus_Blood_Data.csv')
    # For pretraining, drop APOE4
    other_data = load_file('/home/tmnthc/New1/Other.csv', drop_apoe4=True)

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
                    if len(mri_image.shape) == 4:
                        mri_image = np.mean(mri_image, axis=-1)
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
# 4) MRIClipEncoder with Multi-Slice Aggregation (LayerNorm, no dropout)
###########################################
class MRIClipEncoder(nn.Module):
    def __init__(self, embed_dim=64, augment=False, dropout_p=0.1):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self._unfreeze_last_n_layers(2)
        
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
    
    def forward(self, mri_batch):
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

###########################################
# 5) MLPEncoder for Numeric Data with Increased Capacity (LayerNorm, no dropout)
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
# 6) CombinedContrastiveDataset for 4 Modalities
###########################################
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
            micro_tensor = augment_numeric(micro_tensor)
            biom_tensor = augment_numeric(biom_tensor)
            other_tensor = augment_numeric(other_tensor)
        
        sample_label = f"{pid_mri}_{tpt_mri}"
        return {
            "mri": mri_tensor,
            "micro": micro_tensor,
            "biom": biom_tensor,
            "other": other_tensor,
            "sample_label": sample_label,
            "label": torch.tensor(label, dtype=torch.float32)
        }

###########################################
# Custom collate function for CombinedContrastiveDataset
###########################################
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
# 8) Patient Contrastive Loss (NT-Xent style)
###########################################
def patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, sample_labels, tau=0.5):
    B, D = e_mri.shape
    all_labels = sample_labels + sample_labels + sample_labels + sample_labels
    unique_labels, inverse = np.unique(all_labels, return_inverse=True)
    labels = torch.tensor(inverse, device=e_mri.device)
    
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

###########################################
# 10) Training & Validation for Pretraining
###########################################
def train_epoch(model, loader, optimizer, device, current_epoch):
    model.train()
    total_loss = 0
    if current_epoch == 30:
        model.mri_encoder._unfreeze_last_n_layers(4)
    if current_epoch == 60:
        model.mri_encoder._unfreeze_last_n_layers(12)
    for batch in loader:
        mri = batch["mri"].to(device)
        micro = batch["micro"].to(device)
        biom = batch["biom"].to(device)
        other = batch["other"].to(device)
        sample_labels = batch["sample_label"]
        optimizer.zero_grad()
        e_mri, e_micro, e_biom, e_other = model(mri, micro, biom, other)
        loss = patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, sample_labels=sample_labels, tau=0.5)
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
            sample_labels = batch["sample_label"]
            e_mri, e_micro, e_biom, e_other = model(mri, micro, biom, other)
            loss = patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, sample_labels=sample_labels, tau=0.5)
            total_loss += loss.item()
    return total_loss / len(loader)

###########################################
# 11) MultiModalEmbeddingModel
###########################################
class MultiModalEmbeddingModel(nn.Module):
    def __init__(self, micro_dim, biom_dim, other_dim, embed_dim=64, augment=False):
        super().__init__()
        self.mri_encoder = MRIClipEncoder(embed_dim=embed_dim, augment=augment)
        self.micro_encoder = MLPEncoder(input_dim=micro_dim, output_dim=embed_dim, augment=augment)
        self.biom_encoder = MLPEncoder(input_dim=biom_dim, output_dim=embed_dim, augment=augment)
        self.other_encoder = MLPEncoder(input_dim=other_dim, output_dim=embed_dim, augment=augment)
    def forward(self, mri, micro, biom, other):
        e_mri = self.mri_encoder(mri)
        e_micro = self.micro_encoder(micro)
        e_biom = self.biom_encoder(biom)
        e_other = self.other_encoder(other)
        return e_mri, e_micro, e_biom, e_other

###########################################
# NEW: Attention Layer Classification Head
###########################################
class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim=64, num_modalities=4):
        """
        Takes four modality embeddings (shape: [B, num_modalities, embed_dim]),
        computes attention weights, aggregates them, and outputs a binary logit.
        """
        super(AttentionClassifier, self).__init__()
        self.attention_fc = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, 1)
    
    def forward(self, embeddings):
        # embeddings: [B, num_modalities, embed_dim]
        attn_scores = self.attention_fc(embeddings)       # [B, num_modalities, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)   # [B, num_modalities, 1]
        weighted_emb = torch.sum(attn_weights * embeddings, dim=1)  # [B, embed_dim]
        logits = self.classifier(weighted_emb)              # [B, 1]
        return logits.squeeze(-1)                          # [B]

###########################################
# Custom Dataset for Attention Classification with Labels
###########################################
class AttentionDatasetWithLabels(Dataset):
    def __init__(self, patient_ids, data, mri_dict, model, device, patient_labels):
        """
        For each patient in patient_ids, select one sample and compute the embeddings.
        patient_labels: dictionary mapping patient_id -> label.
        """
        self.samples = []
        self.device = device
        self.model = model
        for pid in patient_ids:
            patient_data = data[data["Patient_ID"] == pid]
            if len(patient_data) == 0:
                continue
            row = patient_data.iloc[0]
            mri_tensor = mri_dict.get((str(row["Patient_ID"]), str(row["Timepoint"])),
                                      torch.zeros((1,128,128,128), dtype=torch.float32))
            micro_tensor = torch.tensor(row.filter(like="Microbiome_").values.astype(np.float32))
            biom_tensor = torch.tensor(row.filter(like="Biomarker_").values.astype(np.float32))
            other_tensor = torch.tensor(row.filter(like="Other_").values.astype(np.float32))
            mri_tensor = mri_tensor.to(device).unsqueeze(0)
            micro_tensor = micro_tensor.to(device).unsqueeze(0)
            biom_tensor = biom_tensor.to(device).unsqueeze(0)
            other_tensor = other_tensor.to(device).unsqueeze(0)
            with torch.no_grad():
                e_mri, e_micro, e_biom, e_other = model(mri_tensor, micro_tensor, biom_tensor, other_tensor)
            embeddings = torch.stack([e_mri.squeeze(0), e_micro.squeeze(0), e_biom.squeeze(0), e_other.squeeze(0)], dim=0)
            label = patient_labels.get(str(pid), 0)
            self.samples.append((embeddings, torch.tensor(label, dtype=torch.float32)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def attn_collate_with_labels(batch):
    embeddings_list, label_list = zip(*batch)
    embeddings = torch.stack(embeddings_list, dim=0)  # [B, 4, embed_dim]
    labels = torch.stack(label_list, dim=0)             # [B]
    return embeddings, labels

###########################################
# Main Function: Pretraining and Attention Classification
###########################################
def main():
    # Pretraining Phase
    data = load_data()
    mri_dict = load_mri_data("/home/tmnthc/CBF_imaging")
    
    all_pats = np.unique(data["Patient_ID"])
    if len(all_pats) < 19:
        raise ValueError("Need at least 19 patients for this example.")
    
    # Randomly select 19 patients and split into 16 train and 3 validation for pretraining
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
    
    train_set = CombinedContrastiveDataset(train_data, mri_dict, negative_sample_fraction=6, positive_repeat=2, augment=True)
    val_set = CombinedContrastiveDataset(val_data, mri_dict, negative_sample_fraction=6, positive_repeat=2, augment=False)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, collate_fn=custom_collate)
    
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
    
    total_epochs = 100
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
        print(f"Pretraining Epoch {ep}/{total_epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
    
    plot_metrics(train_losses, val_losses, total_epochs, filename="pretraining_loss.png")
    torch.save(model.state_dict(), "model_checkpoint.pth")
    print("Pretraining checkpoint saved as model_checkpoint.pth")
    
    # Attention Classification for APOE4
    # Load original Other.csv to get APOE4 labels.
    other_df = pd.read_csv('/home/tmnthc/New1/Other.csv')
    other_df.rename(columns=lambda x: x.strip(), inplace=True)
    other_df['Patient_ID'] = other_df['Patient_ID'].astype(str).str.strip()
    other_df['Timepoint'] = other_df['Timepoint'].astype(str).str.strip()
    patient_labels_df = other_df.groupby("Patient_ID")["APOE4"].max().reset_index()
    patient_apoe4 = {row["Patient_ID"]: int(row["APOE4"]) for _, row in patient_labels_df.iterrows()}
    
    # For attention classifier, we use the same training and validation patients as pretraining.
    # The remaining patients will be used as the test set.
    train_attn_dataset = AttentionDatasetWithLabels(train_pats, data, mri_dict, model, device, patient_apoe4)
    val_attn_dataset = AttentionDatasetWithLabels(val_pats, data, mri_dict, model, device, patient_apoe4)
    all_patient_ids = set(other_df["Patient_ID"].unique())
    test_ids = list(all_patient_ids - set(train_pats.tolist()) - set(val_pats.tolist()))
    print("Attention Classification: Test Patient IDs:", test_ids)
    test_attn_dataset = AttentionDatasetWithLabels(test_ids, data, mri_dict, model, device, patient_apoe4)
    
    train_attn_loader = DataLoader(train_attn_dataset, batch_size=4, shuffle=True, collate_fn=attn_collate_with_labels)
    val_attn_loader = DataLoader(val_attn_dataset, batch_size=4, shuffle=False, collate_fn=attn_collate_with_labels)
    test_attn_loader = DataLoader(test_attn_dataset, batch_size=4, shuffle=False, collate_fn=attn_collate_with_labels)
    
    attn_classifier = AttentionClassifier(embed_dim=64, num_modalities=4).to(device)
    attn_optimizer = torch.optim.Adam(attn_classifier.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    num_attn_epochs = 50
    attn_train_losses = []
    attn_val_losses = []
    for epoch in range(num_attn_epochs):
        # Training loop for attention classifier
        attn_classifier.train()
        total_loss = 0.0
        for embeddings, labels in train_attn_loader:
            embeddings = embeddings.to(device)  # [B, 4, 64]
            labels = labels.to(device).float()    # [B]
            attn_optimizer.zero_grad()
            logits = attn_classifier(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            attn_optimizer.step()
            total_loss += loss.item() * embeddings.size(0)
        avg_train_loss = total_loss / len(train_attn_dataset)
        
        # Validation loop for attention classifier
        attn_classifier.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for embeddings, labels in val_attn_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device).float()
                logits = attn_classifier(embeddings)
                loss = criterion(logits, labels)
                total_val_loss += loss.item() * embeddings.size(0)
        avg_val_loss = total_val_loss / len(val_attn_dataset)
        
        attn_train_losses.append(avg_train_loss)
        attn_val_losses.append(avg_val_loss)
        print(f"Attention Classifier Epoch {epoch+1}/{num_attn_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    plot_metrics(attn_train_losses, attn_val_losses, num_attn_epochs, filename="attention_loss.png")
    
    # Evaluate attention classifier on test set
    attn_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings, labels in test_attn_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device).float()
            logits = attn_classifier(embeddings)
            preds = (logits >= 0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if total > 0:
        print(f"Attention Classifier Test Accuracy: {correct/total*100:.2f}%")
    else:
        print("No test samples available for attention classification.")

if __name__ == "__main__":
    main()
