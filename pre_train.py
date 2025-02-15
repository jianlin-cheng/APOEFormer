import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random

import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Additional imports for scaling and PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Sklearn metrics (we compute recall and AUC)
from sklearn.metrics import recall_score, roc_auc_score

#############################################
# DATA LOADING FUNCTIONS
#############################################
def load_data():
    def load_file(file_path):
        try:
            df = pd.read_csv(file_path)
            df.rename(columns=lambda x: x.strip(), inplace=True)
            return df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return pd.DataFrame()
    # Update paths as needed
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
            print(f"Warning: {name} data is empty or missing.")
        else:
            print(f"{name} data loaded with shape: {df.shape}")

    # Do not drop APOE4 because it is used downstream.
    for df in [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data]:
        if 'Patient_ID' in df.columns:
            df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
        if 'Timepoint' in df.columns:
            df['Timepoint'] = df['Timepoint'].astype(str).str.strip()
    
    def filter_numeric(df):
        return df.select_dtypes(include=['number']).copy()
    
    def add_prefix(df, prefix):
        df = df.copy()
        cols = [col for col in df.columns if col not in ['Patient_ID', 'Timepoint']]
        df.rename(columns={col: f"{prefix}_{col}" for col in cols}, inplace=True)
        return df
    
    microbiome_numeric = add_prefix(filter_numeric(microbiome), "Microbiome")
    blood_metabolites_numeric = add_prefix(filter_numeric(blood_metabolites), "Biomarker")
    inflammatory_markers_numeric = add_prefix(filter_numeric(inflammatory_markers), "Inflammatory")
    blood_data_numeric = add_prefix(filter_numeric(blood_data), "Blood")
    other_numeric = add_prefix(other_data.select_dtypes(include=[np.number]), "Other")
    if not other_data.empty and 'Patient_ID' in other_data.columns and 'Timepoint' in other_data.columns:
        other_numeric[['Patient_ID', 'Timepoint']] = other_data[['Patient_ID', 'Timepoint']]
    
    for numeric_df, original_df in zip(
        [microbiome_numeric, blood_metabolites_numeric, inflammatory_markers_numeric, blood_data_numeric],
        [microbiome, blood_metabolites, inflammatory_markers, blood_data]
    ):
        if not original_df.empty and 'Patient_ID' in original_df.columns and 'Timepoint' in original_df.columns:
            numeric_df[['Patient_ID', 'Timepoint']] = original_df[['Patient_ID', 'Timepoint']]
    
    data = other_numeric.copy()
    for numeric_df in [microbiome_numeric, blood_metabolites_numeric, inflammatory_markers_numeric, blood_data_numeric]:
        data = data.merge(numeric_df, on=['Patient_ID', 'Timepoint'], how='outer')
    
    data.fillna(0, inplace=True)
    print(f"Final merged data shape: {data.shape}")
    return data

def load_mri_data(root_dir):
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
                        mri_image = np.mean(mri_image, axis=-1)
                    mri_tensor = torch.tensor(mri_image, dtype=torch.float32).unsqueeze(0)
                    mri_dict[(patient_id, timepoint_name)] = mri_tensor
    return mri_dict

#############################################
# ENCODERS
#############################################
class MRIClipEncoder(nn.Module):
    """
    MRI Encoder: Extracts a middle 2D slice from a 3D MRI, converts it to a PIL image,
    processes it through CLIP, and projects the resulting 512-d embedding to embed_dim.
    """
    def __init__(self, embed_dim=32, device="cpu"):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        self.project = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, mri_batch):
        b_size = mri_batch.size(0)
        embeddings = []
        for i in range(b_size):
            volume_3d = mri_batch[i, 0]  # (D, H, W)
            D = volume_3d.shape[0]
            mid_slice_idx = D // 2
            slice_2d = volume_3d[mid_slice_idx]
            arr_np = slice_2d.cpu().numpy()
            rng = np.ptp(arr_np)
            arr_np = (arr_np - arr_np.min()) / (rng + 1e-8)
            arr_np = (arr_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(arr_np, mode='L').convert("RGB")
            inputs = self.processor(
                images=pil_img,
                text=None,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            with torch.no_grad():
                img_feats = self.clip_model.get_image_features(**inputs)
                img_feats = F.normalize(img_feats, p=2, dim=-1)
            out = self.project(img_feats)
            out = F.normalize(out, p=2, dim=-1)
            embeddings.append(out)
        return torch.cat(embeddings, dim=0)

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
        return F.normalize(self.encoder(x), p=2, dim=-1)

#############################################
# COMBINED DATASET FOR SELF-SUPERVISED PRETRAINING
#############################################
class CombinedContrastiveDataset(Dataset):
    """
    Precompute a list of combined samples.
    
    Positive samples: For each row, (i, i, i, i) meaning all modalities from the same row. Label = 1.
    
    Negative samples: For each negative sample, we generate a tuple of 4 indices
    (with possible repetitions) such that not all 4 are the same.
    This allows, for example, 2 indices from one patient and 2 from another.
    """
    def __init__(self, data, mri_dict, negative_sample_fraction=1.0):
        self.data = data.reset_index(drop=True)
        self.mri_dict = mri_dict
        self.N = len(self.data)
        
        # Positive samples: (i, i, i, i) for each row.
        self.positive_samples = [(i, i, i, i) for i in range(self.N)]
        
        # Negative samples: generate a fixed number of negatives by random sampling.
        num_negatives = int(self.N * negative_sample_fraction)
        negative_samples = []
        while len(negative_samples) < num_negatives:
            sample = [random.choice(range(self.N)) for _ in range(4)]
            # Ensure not all indices are identical (to avoid being a positive sample)
            if len(set(sample)) > 1:
                negative_samples.append(tuple(sample))
        self.negative_samples = negative_samples
        
        self.samples = [(sample, 1) for sample in self.positive_samples] + [(sample, 0) for sample in self.negative_samples]
        print("Total combined samples:", len(self.samples))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        indices, label = self.samples[index]
        i, j, k, l = indices
        patient_id_i = str(self.data.loc[i, 'Patient_ID'])
        timepoint_i = str(self.data.loc[i, 'Timepoint'])
        mri_tensor = self.mri_dict.get((patient_id_i, timepoint_i),
                                       torch.zeros((1, 128, 128, 128), dtype=torch.float32))
        micro_tensor = torch.tensor(self.data.filter(like='Microbiome_').iloc[j].values.astype(np.float32))
        biom_tensor = torch.tensor(self.data.filter(like='Biomarker_').iloc[k].values.astype(np.float32))
        other_tensor = torch.tensor(self.data.filter(like='Other_').iloc[l].values.astype(np.float32))
        return {
            "mri": mri_tensor,
            "micro": micro_tensor,
            "biom": biom_tensor,
            "other": other_tensor,
            "label": torch.tensor(label, dtype=torch.float32)
        }

#############################################
# EVALUATION DATASET: POSITIVE SAMPLES ONLY
#############################################
class PositiveIndividualModalityDataset(Dataset):
    """
    This dataset splits each positive row into individual modality samples.
    Each sample has a group_label (Patient_ID+Timepoint) and a full_label (including modality).
    """
    def __init__(self, data, mri_dict):
        self.data = data.reset_index(drop=True)
        self.mri_dict = mri_dict
        self.modalities = ["mri", "micro", "biom", "other"]
    
    def __len__(self):
        return len(self.data) * len(self.modalities)
    
    def __getitem__(self, idx):
        row_idx = idx // len(self.modalities)
        modality_idx = idx % len(self.modalities)
        modality = self.modalities[modality_idx]
        patient_id = str(self.data.loc[row_idx, 'Patient_ID'])
        timepoint = str(self.data.loc[row_idx, 'Timepoint'])
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
        full_label = f"{group_label}_{modality}"
        return {
            "data": data_point,
            "group_label": group_label,
            "modality_label": modality,
            "full_label": full_label
        }

# Custom collate function for variable shapes
def custom_collate_fn(batch):
    out = {}
    for key in batch[0]:
        out[key] = [d[key] for d in batch]
    return out

#############################################
# PLOTTING HELPERS
#############################################
def plot_metrics(train_metrics, val_metrics, epochs):
    # Unpack metrics (loss, recall, AUC)
    train_losses = [m[0] for m in train_metrics]
    train_recs   = [m[1] for m in train_metrics]
    train_aucs   = [m[2] for m in train_metrics]
    
    val_losses = [m[0] for m in val_metrics]
    val_recs   = [m[1] for m in val_metrics]
    val_aucs   = [m[2] for m in val_metrics]
    
    epochs_range = range(1, epochs+1)

    # Loss Plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.savefig('loss.png', dpi=300, bbox_inches="tight")
    plt.close()
    
    # Recall Plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_recs, label='Train Recall')
    plt.plot(epochs_range, val_recs, label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training & Validation Recall')
    plt.legend()
    plt.savefig('recall.png', dpi=300, bbox_inches="tight")
    plt.close()
    
    # AUC Plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_aucs, label='Train AUC')
    plt.plot(epochs_range, val_aucs, label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training & Validation AUC')
    plt.legend()
    plt.savefig('auc.png', dpi=300, bbox_inches="tight")
    plt.close()

#############################################
# SELF-SUPERVISED PRETRAINING FUNCTIONS
#############################################
def train_pretrain_epoch(model_dict, dataloader, optimizer, device="cpu"):
    mri_encoder = model_dict["mri_encoder"]
    micro_encoder = model_dict["micro_encoder"]
    biom_encoder = model_dict["biom_encoder"]
    other_encoder = model_dict["other_encoder"]
    
    mri_encoder.train()
    micro_encoder.train()
    biom_encoder.train()
    other_encoder.train()
    
    bce_loss = nn.BCELoss()
    running_loss = 0.0
    
    all_labels = []
    all_preds = []
    
    for batch in dataloader:
        mri = batch["mri"].to(device)
        micro = batch["micro"].to(device)
        biom = batch["biom"].to(device)
        other = batch["other"].to(device)
        labels = batch["label"].to(device)
        
        e_mri = mri_encoder(mri)
        e_micro = micro_encoder(micro)
        e_biom = biom_encoder(biom)
        e_other = other_encoder(other)
        
        e_tabular = (e_micro + e_biom + e_other) / 3
        cos_sim = F.cosine_similarity(e_mri, e_tabular, dim=-1)
        pred = (cos_sim + 1) / 2  # range [0, 1]
        
        loss = bce_loss(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(pred.detach().cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    try:
        epoch_rec = recall_score(all_labels, binary_preds, zero_division=0)
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_rec = epoch_auc = 0.0
    
    return epoch_loss, epoch_rec, epoch_auc

def validate_pretrain_epoch(model_dict, dataloader, device="cpu"):
    bce_loss = nn.BCELoss()
    running_loss = 0.0
    
    all_labels = []
    all_preds = []
    
    mri_encoder = model_dict["mri_encoder"]
    micro_encoder = model_dict["micro_encoder"]
    biom_encoder = model_dict["biom_encoder"]
    other_encoder = model_dict["other_encoder"]
    
    mri_encoder.eval()
    micro_encoder.eval()
    biom_encoder.eval()
    other_encoder.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            mri = batch["mri"].to(device)
            micro = batch["micro"].to(device)
            biom = batch["biom"].to(device)
            other = batch["other"].to(device)
            labels = batch["label"].to(device)
            
            e_mri = mri_encoder(mri)
            e_micro = micro_encoder(micro)
            e_biom = biom_encoder(biom)
            e_other = other_encoder(other)
            
            e_tabular = (e_micro + e_biom + e_other) / 3
            cos_sim = F.cosine_similarity(e_mri, e_tabular, dim=-1)
            pred = (cos_sim + 1) / 2
            
            loss = bce_loss(pred, labels)
            running_loss += loss.item()
            
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(pred.detach().cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    try:
        epoch_rec = recall_score(all_labels, binary_preds, zero_division=0)
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_rec = epoch_auc = 0.0
    
    return epoch_loss, epoch_rec, epoch_auc

def pretrain_model(data, mri_dict, embed_dim=32, epochs=5, batch_size=3, lr=1e-4):
    # Select training and validation patients:
    all_patients = np.unique(data['Patient_ID'])
    if len(all_patients) < 19:
        raise ValueError("Not enough patients. Need at least 19 patients.")
    train_patient_ids = all_patients[:16]
    val_patient_ids = all_patients[16:19]
    print("Training Patient IDs:", train_patient_ids)
    print("Validation Patient IDs:", val_patient_ids)
    
    train_data = data[data['Patient_ID'].isin(train_patient_ids)]
    val_data = data[data['Patient_ID'].isin(val_patient_ids)]
    
    # For each patient, take the first 3 timepoints (sorted)
    train_data = train_data.sort_values(['Patient_ID', 'Timepoint']).groupby('Patient_ID').head(3).reset_index(drop=True)
    val_data = val_data.sort_values(['Patient_ID', 'Timepoint']).groupby('Patient_ID').head(3).reset_index(drop=True)
    
    print(f"Training data rows: {len(train_data)} (expected 16*3 = 48)")
    print(f"Validation data rows: {len(val_data)} (expected 3*3 = 9)")
    
    # Create datasets
    train_dataset = CombinedContrastiveDataset(train_data, mri_dict, negative_sample_fraction=100)
    val_dataset = CombinedContrastiveDataset(val_data, mri_dict, negative_sample_fraction=100)
    
    print(f"Number of training combined samples: {len(train_dataset)}")
    print(f"Number of validation combined samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    micro_dim = train_data.filter(like='Microbiome_').shape[1]
    biom_dim = train_data.filter(like='Biomarker_').shape[1]
    other_dim = train_data.filter(like='Other_').shape[1]
    micro_encoder = MLPEncoder(micro_dim, embed_dim).to(device)
    biom_encoder = MLPEncoder(biom_dim, embed_dim).to(device)
    other_encoder = MLPEncoder(other_dim, embed_dim).to(device)
    mri_encoder = MRIClipEncoder(embed_dim=embed_dim, device=device).to(device)
    
    model_dict = {
        "mri_encoder": mri_encoder,
        "micro_encoder": micro_encoder,
        "biom_encoder": biom_encoder,
        "other_encoder": other_encoder
    }
    
    params = list(mri_encoder.parameters()) + list(micro_encoder.parameters()) + \
             list(biom_encoder.parameters()) + list(other_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    
    train_metrics = []
    val_metrics = []
    
    for epoch in range(1, epochs + 1):
        (train_loss, train_rec, train_auc) = train_pretrain_epoch(model_dict, train_loader, optimizer, device=device)
        (val_loss, val_rec, val_auc) = validate_pretrain_epoch(model_dict, val_loader, device=device)
        
        train_metrics.append((train_loss, train_rec, train_auc))
        val_metrics.append((val_loss, val_rec, val_auc))
        
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Rec: {train_rec:.4f} AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} Rec: {val_rec:.4f} AUC: {val_auc:.4f}")
    
    # Save metric plots
    plot_metrics(train_metrics, val_metrics, epochs)
    
    return mri_encoder, micro_encoder, biom_encoder, other_encoder

#############################################
# EVALUATION FUNCTIONS
#############################################
def knn_accuracy(embeddings, labels, k=5):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(embeddings, labels)
    predictions = knn.predict(embeddings)
    return np.mean(predictions == labels)

def extract_individual_embeddings(model, dataloader, device="cpu"):
    model.eval()
    embeddings = []
    group_labels = []
    full_labels = []
    with torch.no_grad():
        for batch in dataloader:
            for i in range(len(batch["data"])):
                x = batch["data"][i].to(device)
                group_label = batch["group_label"][i]
                full_label = batch["full_label"][i]
                modality = batch["modality_label"][i].lower()
                if modality == "mri":
                    emb = model.mri_encoder(x.unsqueeze(0))
                elif modality == "micro":
                    emb = model.micro_encoder(x.unsqueeze(0))
                elif modality == "biom":
                    emb = model.biom_encoder(x.unsqueeze(0))
                elif modality == "other":
                    emb = model.other_encoder(x.unsqueeze(0))
                else:
                    raise ValueError(f"Unknown modality: {modality}")
                embeddings.append(emb.cpu().numpy())
                group_labels.append(group_label)
                full_labels.append(full_label)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, group_labels, full_labels

def tsne_visualization_individual(embeddings, group_labels, full_labels, title="t-SNE Visualization"):
    from sklearn.manifold import TSNE
    # Improved t-SNE parameters
    tsne = TSNE(n_components=2, perplexity=20, learning_rate=500, n_iter=1500, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    unique_groups = np.unique(group_labels)
    cmap = plt.get_cmap('tab10', len(unique_groups))
    group_color = {group: cmap(i) for i, group in enumerate(unique_groups)}
    marker_dict = {"mri": "o", "micro": "s", "biom": "D", "other": "^"}
    
    plt.figure(figsize=(10,8))
    plotted = {}
    for i in range(len(tsne_embeddings)):
        group = group_labels[i]
        modality = full_labels[i].split('_')[-1].lower()
        marker = marker_dict.get(modality, "o")
        label = f"{group}_{modality}"
        if label not in plotted:
            plt.scatter(tsne_embeddings[i,0], tsne_embeddings[i,1],
                        color=group_color[group], marker=marker, s=80, label=label)
            plotted[label] = True
        else:
            plt.scatter(tsne_embeddings[i,0], tsne_embeddings[i,1],
                        color=group_color[group], marker=marker, s=80)
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(title="Patient+Timepoint_Modality", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig("tsne_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

def umap_visualization_individual(embeddings, group_labels, full_labels, title="UMAP Visualization"):
    import umap.umap_ as umap
    # Improved UMAP parameters
    umap_model = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.05, random_state=42)
    umap_embeddings = umap_model.fit_transform(embeddings)
    
    unique_groups = np.unique(group_labels)
    cmap = plt.get_cmap('tab10', len(unique_groups))
    group_color = {group: cmap(i) for i, group in enumerate(unique_groups)}
    marker_dict = {"mri": "o", "micro": "s", "biom": "D", "other": "^"}
    
    plt.figure(figsize=(10,8))
    plotted = {}
    for i in range(len(umap_embeddings)):
        group = group_labels[i]
        modality = full_labels[i].split('_')[-1].lower()
        marker = marker_dict.get(modality, "o")
        label = f"{group}_{modality}"
        if label not in plotted:
            plt.scatter(umap_embeddings[i,0], umap_embeddings[i,1],
                        color=group_color[group], marker=marker, s=80, label=label)
            plotted[label] = True
        else:
            plt.scatter(umap_embeddings[i,0], umap_embeddings[i,1],
                        color=group_color[group], marker=marker, s=80)
    plt.title(title)
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.legend(title="Patient+Timepoint_Modality", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig("umap_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

def run_analysis_individual(model, dataloader, device="cpu", k=1):
    embeddings, group_labels, full_labels = extract_individual_embeddings(model, dataloader, device)
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    if embeddings_scaled.shape[0] >= 50:
        pca = PCA(n_components=50, whiten=True)
        reduced_embeddings = pca.fit_transform(embeddings_scaled)
    else:
        reduced_embeddings = embeddings_scaled
    # Further normalize each embedding to unit norm
    normalized_embeddings = reduced_embeddings / np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)
    
    acc = knn_accuracy(normalized_embeddings, group_labels, k)
    print(f"KNN Accuracy (grouped by patient+timepoint) with k={k}: {acc*100:.2f}%")
    tsne_visualization_individual(normalized_embeddings, group_labels, full_labels,
                                  title="t-SNE Visualization of Individual Embeddings")
    umap_visualization_individual(normalized_embeddings, group_labels, full_labels,
                                  title="UMAP Visualization of Individual Embeddings")

#############################################
# MAIN TRAINING AND EVALUATION
#############################################
if __name__ == "__main__":
    data = load_data()
    mri_dict = load_mri_data("/home/tmnthc/CBF_imaging")
    
    print("\n===== SELF-SUPERVISED PRETRAINING =====")
    # Pretrain on 16 patients (each with 3 timepoints) and validate on 3 patients (each with 3 timepoints)
    all_patients = np.unique(data['Patient_ID'])
    if len(all_patients) < 19:
        raise ValueError("Not enough patients in data. Need at least 19 patients.")
    train_patient_ids = all_patients[:16]
    val_patient_ids = all_patients[16:19]
    print("Training Patient IDs:", train_patient_ids)
    print("Validation Patient IDs:", val_patient_ids)
    
    train_data = data[data['Patient_ID'].isin(train_patient_ids)]
    val_data = data[data['Patient_ID'].isin(val_patient_ids)]
    
    # Sort by Patient_ID and Timepoint, take first 3 timepoints per patient
    train_data = train_data.sort_values(['Patient_ID', 'Timepoint']).groupby('Patient_ID').head(3).reset_index(drop=True)
    val_data = val_data.sort_values(['Patient_ID', 'Timepoint']).groupby('Patient_ID').head(3).reset_index(drop=True)
    
    print(f"Training data rows: {len(train_data)} (expected 16*3 = 48)")
    print(f"Validation data rows: {len(val_data)} (expected 3*3 = 9)")
    
    train_dataset = CombinedContrastiveDataset(train_data, mri_dict, negative_sample_fraction=100)
    val_dataset = CombinedContrastiveDataset(val_data, mri_dict, negative_sample_fraction=100)
    
    print(f"Number of training combined samples: {len(train_dataset)}")
    print(f"Number of validation combined samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    micro_dim = train_data.filter(like='Microbiome_').shape[1]
    biom_dim = train_data.filter(like='Biomarker_').shape[1]
    other_dim = train_data.filter(like='Other_').shape[1]
    micro_encoder = MLPEncoder(micro_dim, 32).to(device)
    biom_encoder = MLPEncoder(biom_dim, 32).to(device)
    other_encoder = MLPEncoder(other_dim, 32).to(device)
    mri_encoder = MRIClipEncoder(embed_dim=32, device=device).to(device)
    
    model_dict = {
        "mri_encoder": mri_encoder,
        "micro_encoder": micro_encoder,
        "biom_encoder": biom_encoder,
        "other_encoder": other_encoder
    }
    
    params = list(mri_encoder.parameters()) + list(micro_encoder.parameters()) + \
             list(biom_encoder.parameters()) + list(other_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)
    
    train_metrics = []
    val_metrics = []
    epochs = 80
    for epoch in range(1, epochs + 1):
        (train_loss, train_rec, train_auc) = train_pretrain_epoch(model_dict, train_loader, optimizer, device=device)
        (val_loss, val_rec, val_auc) = validate_pretrain_epoch(model_dict, val_loader, device=device)
        
        train_metrics.append((train_loss, train_rec, train_auc))
        val_metrics.append((val_loss, val_rec, val_auc))
        
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Rec: {train_rec:.4f} AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} Rec: {val_rec:.4f} AUC: {val_auc:.4f}")
    
    # Save metric plots
    plot_metrics(train_metrics, val_metrics, epochs)
    
    # EVALUATION
    # For evaluation, restrict to positive samples from 4 patients.
    eval_patient_ids = all_patients[:4]
    print("Evaluation Patient IDs:", eval_patient_ids)
    eval_data = data[data['Patient_ID'].isin(eval_patient_ids)]
    eval_dataset = PositiveIndividualModalityDataset(eval_data, mri_dict)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    print("\n===== EMBEDDING ANALYSIS (KNN, t-SNE, UMAP) ON POSITIVE SAMPLES =====")
    class SelfSupervisedModel(nn.Module):
        def __init__(self, mri_encoder, micro_encoder, biom_encoder, other_encoder):
            super().__init__()
            self.mri_encoder = mri_encoder
            self.micro_encoder = micro_encoder
            self.biom_encoder = biom_encoder
            self.other_encoder = other_encoder
        def forward(self, mri, micro, biom, other):
            e_mri = self.mri_encoder(mri)
            e_micro = self.micro_encoder(micro)
            e_biom = self.biom_encoder(biom)
            e_other = self.other_encoder(other)
            return torch.cat([e_mri, e_micro, e_biom, e_other], dim=-1)
    
    model = SelfSupervisedModel(mri_encoder, micro_encoder, biom_encoder, other_encoder)
    run_analysis_individual(model, eval_loader, device=device, k=1)
