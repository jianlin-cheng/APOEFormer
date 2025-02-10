import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

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

    # Do not drop APOE4 because it is used as the supervised label.
    
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
        data = data.merge(numeric_df, on=['Patient_ID', 'Timepoint'], how='inner')
    
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
    Precompute a list of "combined" samples.
    
    Positive samples: For each observation (row) use (i, i, i, i) meaning that all modalities
    are taken from the same row. Label = 1.
    
    Negative samples: Create combinations where each modality comes from a different observation.
    For example, (i, j, k, l) where i, j, k, l are distinct indices with distinct Patient_IDs.
    Label = 0.
    
    The __getitem__ uses these precomputed indices to load data from the merged DataFrame and MRI dictionary.
    """
    def __init__(self, data, mri_dict, negative_sample_fraction=1.0):
        self.data = data.reset_index(drop=True)
        self.mri_dict = mri_dict
        self.N = len(self.data)
        
        # Positive samples: (i, i, i, i) for each observation.
        self.positive_samples = [(i, i, i, i) for i in range(self.N)]
        
        # Negative samples: precompute all combinations of 4 distinct indices where patient IDs are distinct.
        all_indices = list(range(self.N))
        negative_samples = []
        for comb in itertools.combinations(all_indices, 4):
            # Check that the Patient_IDs for these indices are all distinct.
            p_ids = [self.data.loc[idx, 'Patient_ID'] for idx in comb]
            if len(set(p_ids)) == 4:
                negative_samples.append(comb)
        print("Total negative combinations (before sampling):", len(negative_samples))
        if negative_sample_fraction < 1.0:
            np.random.shuffle(negative_samples)
            new_len = int(len(negative_samples) * negative_sample_fraction)
            negative_samples = negative_samples[:new_len]
            print("Negative combinations after subsampling:", len(negative_samples))
        self.negative_samples = negative_samples
        
        self.samples = [(sample, 1) for sample in self.positive_samples] + [(sample, 0) for sample in self.negative_samples]
        print("Total combined samples:", len(self.samples))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        indices, label = self.samples[index]
        i, j, k, l = indices
        # Load MRI from observation i.
        patient_id_i = str(self.data.loc[i, 'Patient_ID'])
        timepoint_i = str(self.data.loc[i, 'Timepoint'])
        mri_tensor = self.mri_dict.get((patient_id_i, timepoint_i), 
                                       torch.zeros((1, 128, 128, 128), dtype=torch.float32))
        # Load tabular data for modalities from rows j, k, l.
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
# SUPERVISED DATASET FOR APOE4 PREDICTION
#############################################
class SupervisedDataset(Dataset):
    def __init__(self, data, mri_dict):
        self.data = data.reset_index(drop=True)
        self.mri_dict = mri_dict
        self.N = len(self.data)
        self.micro_data = data.filter(like='Microbiome_').astype(np.float32).values
        self.biom_data = data.filter(like='Biomarker_').astype(np.float32).values
        self.other_data = data.filter(like='Other_').astype(np.float32).values
        self.labels = data['APOE4'].values.astype(np.float32)
        print("Supervised dataset size:", self.N)
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        patient_id = str(row['Patient_ID'])
        timepoint = str(row['Timepoint'])
        mri_tensor = self.mri_dict.get((patient_id, timepoint),
                                       torch.zeros((1, 128, 128, 128), dtype=torch.float32))
        micro_tensor = torch.tensor(self.micro_data[index], dtype=torch.float32)
        biom_tensor = torch.tensor(self.biom_data[index], dtype=torch.float32)
        other_tensor = torch.tensor(self.other_data[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return {
            "mri": mri_tensor,
            "micro": micro_tensor,
            "biom": biom_tensor,
            "other": other_tensor,
            "label": label
        }

#############################################
# MODEL: MULTI-MODAL ATTENTION PREDICTOR (SUPERVISED HEAD)
#############################################
class MultiModalAttentionPredictor(nn.Module):
    def __init__(self, mri_encoder, micro_encoder, biom_encoder, other_encoder,
                 embed_dim=32, hidden_dim=64, output_dim=1):
        super().__init__()
        self.mri_encoder = mri_encoder
        self.micro_encoder = micro_encoder
        self.biom_encoder = biom_encoder
        self.other_encoder = other_encoder
        
        self.attention = nn.Linear(embed_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, mri, micro, biom, other):
        e_mri = self.mri_encoder(mri)         # (B, embed_dim)
        e_micro = self.micro_encoder(micro)   # (B, embed_dim)
        e_biom = self.biom_encoder(biom)      # (B, embed_dim)
        e_other = self.other_encoder(other)   # (B, embed_dim)
        modalities = torch.stack([e_mri, e_micro, e_biom, e_other], dim=1)  # (B, 4, embed_dim)
        attn_scores = self.attention(modalities)  # (B, 4, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused_embedding = torch.sum(attn_weights * modalities, dim=1)  # (B, embed_dim)
        logits = self.classifier(fused_embedding)
        return logits, attn_weights

#############################################
# SELF-SUPERVISED PRETRAINING FUNCTIONS
#############################################
def train_pretrain_epoch(model_dict, dataloader, optimizer, device="cpu"):
    # Extract the modules from the dictionary.
    mri_encoder = model_dict["mri_encoder"]
    micro_encoder = model_dict["micro_encoder"]
    biom_encoder = model_dict["biom_encoder"]
    other_encoder = model_dict["other_encoder"]
    
    # Set each to train mode.
    mri_encoder.train()
    micro_encoder.train()
    biom_encoder.train()
    other_encoder.train()
    
    running_loss = 0.0
    bce_loss = nn.BCELoss()
    
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_pretrain_epoch(model_dict, dataloader, device="cpu"):
    mri_encoder = model_dict["mri_encoder"]
    micro_encoder = model_dict["micro_encoder"]
    biom_encoder = model_dict["biom_encoder"]
    other_encoder = model_dict["other_encoder"]
    
    mri_encoder.eval()
    micro_encoder.eval()
    biom_encoder.eval()
    other_encoder.eval()
    
    running_loss = 0.0
    bce_loss = nn.BCELoss()
    
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
    return running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

#############################################
# SUPERVISED FINETUNING FUNCTIONS
#############################################
def train_supervised_epoch(model, dataloader, optimizer, device="cpu"):
    model.train()
    running_loss = 0.0
    bce_loss = nn.BCEWithLogitsLoss()
    for batch in dataloader:
        mri = batch["mri"].to(device)
        micro = batch["micro"].to(device)
        biom = batch["biom"].to(device)
        other = batch["other"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)
        logits, attn_weights = model(mri, micro, biom, other)
        loss = bce_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_supervised_epoch(model, dataloader, device="cpu"):
    model.eval()
    running_loss = 0.0
    bce_loss = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in dataloader:
            mri = batch["mri"].to(device)
            micro = batch["micro"].to(device)
            biom = batch["biom"].to(device)
            other = batch["other"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            logits, attn_weights = model(mri, micro, biom, other)
            loss = bce_loss(logits, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

#############################################
# MAIN TRAINING FUNCTIONS
#############################################
def pretrain_model(data, mri_dict, embed_dim=32, epochs=5, batch_size=3, lr=1e-4):
    # Use the CombinedContrastiveDataset that precomputes positive and negative samples.
    dataset = CombinedContrastiveDataset(data, mri_dict, negative_sample_fraction=1.0)
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    print(f"Pretraining - Training set size: {train_size} | Validation set size: {val_size}")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create encoders using the merged data dimensions.
    micro_dim = data.filter(like='Microbiome_').shape[1]
    biom_dim = data.filter(like='Biomarker_').shape[1]
    other_dim = data.filter(like='Other_').shape[1]
    micro_encoder = MLPEncoder(micro_dim, embed_dim).to(device)
    biom_encoder = MLPEncoder(biom_dim, embed_dim).to(device)
    other_encoder = MLPEncoder(other_dim, embed_dim).to(device)
    mri_encoder = MRIClipEncoder(embed_dim=embed_dim, device=device).to(device)
    
    model_dict = {"mri_encoder": mri_encoder,
                  "micro_encoder": micro_encoder,
                  "biom_encoder": biom_encoder,
                  "other_encoder": other_encoder}
    
    params = list(mri_encoder.parameters()) + list(micro_encoder.parameters()) + \
             list(biom_encoder.parameters()) + list(other_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    
    train_losses, val_losses = [], []
    patience = 3
    for epoch in range(1, epochs + 1):
        train_loss = train_pretrain_epoch(model_dict, train_loader, optimizer, device=device)
        val_loss = validate_pretrain_epoch(model_dict, val_loader, device=device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Pretraining Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if epoch >= patience and min(val_losses[-patience:]) > (min(val_losses[:-patience], default=float('inf')) - 1e-4):
            print(f"Early stopping triggered at pretraining epoch {epoch}.")
            break
    return mri_encoder, micro_encoder, biom_encoder, other_encoder

def supervised_finetune(data, mri_dict, mri_encoder, micro_encoder, biom_encoder, other_encoder,
                         embed_dim=32, epochs=100, batch_size=3, lr=1e-4):
    dataset = SupervisedDataset(data, mri_dict)
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    print(f"Supervised Finetuning - Training set size: {train_size} | Validation set size: {val_size}")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiModalAttentionPredictor(mri_encoder, micro_encoder, biom_encoder, other_encoder,
                                          embed_dim=embed_dim, hidden_dim=64, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    patience = 5
    for epoch in range(1, epochs + 1):
        train_loss = train_supervised_epoch(model, train_loader, optimizer, device=device)
        val_loss = validate_supervised_epoch(model, val_loader, device=device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Supervised Finetuning Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if epoch >= patience and min(val_losses[-patience:]) > (min(val_losses[:-patience], default=float('inf')) - 1e-4):
            print(f"Early stopping triggered at supervised finetuning epoch {epoch}.")
            break
    
    plt.figure(figsize=(7, 5))
    x_axis = range(1, len(train_losses) + 1)
    plt.plot(x_axis, train_losses, '-o', label='Train Loss')
    plt.plot(x_axis, val_losses, '-x', label='Val Loss')
    plt.title("Supervised Finetuning with Attention Fusion")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("supervised_train_val_loss.png")
    plt.close()
    print("Saved supervised training and validation loss to 'supervised_train_val_loss.png'.")
    return model

#############################################
# MAIN
#############################################
if __name__ == "__main__":
    data = load_data()
    mri_dict = load_mri_data("/home/tmnthc/CBF_imaging")
    
    print("\n===== SELF-SUPERVISED PRETRAINING =====")
    mri_encoder, micro_encoder, biom_encoder, other_encoder = pretrain_model(
        data, mri_dict, embed_dim=32, epochs=5, batch_size=3, lr=1e-4
    )
    
    print("\n===== SUPERVISED FINETUNING =====")
    model = supervised_finetune(
        data, mri_dict,
        mri_encoder, micro_encoder, biom_encoder, other_encoder,
        embed_dim=32, epochs=100, batch_size=3, lr=1e-4
    )
