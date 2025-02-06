import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ==============================
# 🚀 LOAD STRUCTURED DATA
# ==============================
def load_data():
    def load_file(file_path):
        try:
            df = pd.read_csv(file_path)
            df.rename(columns=lambda x: x.strip(), inplace=True)
            return df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return pd.DataFrame()

    microbiome = load_file('/Users/thongnguyen/Downloads/New/Microbiome.csv')
    blood_metabolites = load_file('/Users/thongnguyen/Downloads/New/Blood_Metabolites.csv')
    inflammatory_markers = load_file('/Users/thongnguyen/Downloads/New/Sirolimus_inflammatory_markers.csv')
    blood_data = load_file('/Users/thongnguyen/Downloads/New/Sirolimus_Blood_Data.csv')
    other_data = load_file('/Users/thongnguyen/Downloads/New/Other.csv')

    for name, df in zip(
        ["Microbiome", "Blood Metabolites", "Inflammatory Markers", "Blood Data", "Other"],
        [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data]
    ):
        if df.empty:
            print(f"Warning: {name} data is empty or missing.")
        else:
            print(f"{name} data loaded with shape: {df.shape}")

    if 'APOE4' in other_data.columns:
        other_data.drop(columns=['APOE4'], inplace=True)

    # Convert IDs and Timepoints to string
    for df in [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data]:
        if 'Patient_ID' in df.columns:
            df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
        if 'Timepoint' in df.columns:
            df['Timepoint'] = df['Timepoint'].astype(str).str.strip()

    def filter_numeric(df):
        return df.select_dtypes(include=['number']).copy()

    # Numeric subsets
    microbiome_numeric           = filter_numeric(microbiome)
    blood_metabolites_numeric    = filter_numeric(blood_metabolites)
    inflammatory_markers_numeric = filter_numeric(inflammatory_markers)
    blood_data_numeric           = filter_numeric(blood_data)

    # Add ID/time back
    for numeric_df, original_df in zip(
        [microbiome_numeric, blood_metabolites_numeric, inflammatory_markers_numeric, blood_data_numeric],
        [microbiome, blood_metabolites, inflammatory_markers, blood_data]
    ):
        numeric_df[['Patient_ID','Timepoint']] = original_df[['Patient_ID','Timepoint']]

    # Merge into other_data
    data = other_data
    for numeric_df in [microbiome_numeric, blood_metabolites_numeric, inflammatory_markers_numeric, blood_data_numeric]:
        data = data.merge(numeric_df, on=['Patient_ID','Timepoint'], how='inner')

    data.fillna(0, inplace=True)
    print(f"Final merged data shape: {data.shape}")
    return data

# ==============================
# 🔥 LOAD MRI DATA
# ==============================
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
                    # If 4D, average last dim
                    if len(mri_image.shape) == 4:
                        mri_image = np.mean(mri_image, axis=-1)

                    # shape => (1, D, H, W) for consistency
                    mri_tensor = torch.tensor(mri_image, dtype=torch.float32).unsqueeze(0)
                    mri_dict[(patient_id, timepoint_name)] = mri_tensor
    return mri_dict

# ==============================
# 📌 PATIENT DATASET
# ==============================
class PatientDataset(Dataset):
    def __init__(self, data, mri_dict):
        super().__init__()
        self.data = data
        self.mri_dict = mri_dict

        # Convert patient ID strings to numeric codes
        self.patient_ids = torch.tensor(
            data['Patient_ID'].astype('category').cat.codes.values, 
            dtype=torch.long
        )
        self.timepoints = torch.tensor(
            data['Timepoint'].astype('category').cat.codes.values,
            dtype=torch.long
        )

        # Tabular data
        self.micro_data = data.filter(like='Microbiome_').values
        self.biom_data  = data.filter(like='Biomarker_').values
        self.other_data = data.filter(like='Other_').values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = str(row['Patient_ID'])
        timepoint  = str(row['Timepoint'])

        mri_tensor = self.mri_dict.get(
            (patient_id, timepoint),
            torch.zeros((1,128,128,128), dtype=torch.float32)  # fallback
        )

        micro_tensor = torch.tensor(self.micro_data[idx],  dtype=torch.float32)
        biom_tensor  = torch.tensor(self.biom_data[idx],   dtype=torch.float32)
        other_tensor = torch.tensor(self.other_data[idx],  dtype=torch.float32)
        pid_code     = self.patient_ids[idx]
        t_code       = self.timepoints[idx]

        return mri_tensor, micro_tensor, biom_tensor, other_tensor, pid_code, t_code

# ==============================
# 🚀 CLIP-BASED MRI ENCODER
# ==============================
class MRIClipEncoder(nn.Module):
    """
    Demonstration: Takes (batch, 1, D, H, W) MRI, extracts a 2D slice,
    runs it through CLIP, then projects 512-d -> 32-d.
    """
    def __init__(self, embed_dim=32, device="cpu"):
        super().__init__()
        # Load CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device

        # Extra linear to reduce 512 -> embed_dim
        self.project = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)  # optionally keep it simpler or do nothing
        )

    def forward(self, mri_batch):
        """
        1) Extract middle slice
        2) Convert to PIL
        3) CLIP image_features => shape (batch, 512)
        4) self.project => shape (batch, embed_dim=32)
        """
        b_size = mri_batch.size(0)
        embeddings = []

        for i in range(b_size):
            volume_3d = mri_batch[i, 0]  # shape => (D,H,W)
            D = volume_3d.shape[0]
            mid_slice_idx = D // 2

            slice_2d = volume_3d[mid_slice_idx]  # (H,W)

            # Normalize to [0,255]
            arr_np = slice_2d.cpu().numpy()
            rng = np.ptp(arr_np)
            arr_np = (arr_np - arr_np.min()) / (rng + 1e-8)
            arr_np = (arr_np * 255).astype(np.uint8)

            # Convert to PIL
            pil_img = Image.fromarray(arr_np, mode='L').convert("RGB")  

            # Preprocess for CLIP
            inputs = self.processor(
                images=pil_img,
                text=None,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                img_feats = self.clip_model.get_image_features(**inputs)  # shape (1, 512)
                img_feats = F.normalize(img_feats, p=2, dim=-1)  # L2 norm

            # Now reduce 512 -> embed_dim
            out = self.project(img_feats)          # shape => (1, embed_dim=32)
            out = F.normalize(out, p=2, dim=-1)    # optional final normalization

            embeddings.append(out)

        return torch.cat(embeddings, dim=0)  # shape => (b_size, embed_dim=32)

# ==============================
# 🚀 MLP ENCODERS FOR TABULAR
# ==============================
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=-1)

# ==============================
# 🚀 PATIENT CONTRASTIVE LOSS
# ==============================
def explicit_patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, patient_ids, tau=0.1):
    """
    Explicitly computes positive & negative pairs:

    1) Stack [e_mri, e_micro, e_biom, e_other] => shape (B*4, embed_dim)
    2) Pairwise similarities => shape (B*4, B*4)
    3) same_mask => True if same patient (positive), neg_mask => different patient (negative)
    4) final loss = - log( sum(positive_sim) / sum(positive_sim + negative_sim) )
    """

    B, D = e_mri.shape  # e_mri => (batch_size, embed_dim)

    # Concatenate embeddings along batch dimension => (B*4, embed_dim)
    embeddings = torch.cat([e_mri, e_micro, e_biom, e_other], dim=0)

    # Pairwise similarity => (B*4, B*4)
    sim_matrix = torch.matmul(embeddings, embeddings.t()) / tau  # multiply by 1/tau

    # Exponentiate for a softmax-like approach
    sim_exp = torch.exp(sim_matrix)  # (B*4, B*4)

    # Build masks for positives vs. negatives
    # repeated_ids => shape (B*4,)
    repeated_ids = patient_ids.repeat(4)
    diag_mask = torch.eye(B*4, dtype=torch.bool, device=sim_matrix.device)

    # Positive => same patient & not diagonal
    same_mask = (repeated_ids.unsqueeze(0) == repeated_ids.unsqueeze(1)) & (~diag_mask)
    # Negative => different patient & not diagonal
    diff_mask = (repeated_ids.unsqueeze(0) != repeated_ids.unsqueeze(1)) & (~diag_mask)

    # Sum of positive similarities per anchor
    pos_sim = sim_exp * same_mask.float()  # (B*4, B*4)
    sum_pos = pos_sim.sum(dim=1)          # (B*4,)

    # Sum of negative similarities per anchor
    neg_sim = sim_exp * diff_mask.float()  # (B*4, B*4)
    sum_neg = neg_sim.sum(dim=1)           # (B*4,)

    # final => ratio of sum_pos / (sum_pos + sum_neg)
    # i.e. positives / (positives + negatives)
    eps = 1e-8
    loss = -torch.log((sum_pos + eps) / (sum_pos + sum_neg + eps))
    return loss.mean()


# ==============================
# 🚀 TRAINING LOOP
# ==============================
def train_one_epoch(
    mri_encoder, micro_encoder, biom_encoder, other_encoder,
    dataloader, optimizer, device="cpu", tau=0.1
):
    mri_encoder.train()
    micro_encoder.train()
    biom_encoder.train()
    other_encoder.train()

    running_loss = 0.0
    for batch in dataloader:
        mri, micro, biom, other, pids, _ = [x.to(device) for x in batch]

        e_mri   = mri_encoder(mri)
        e_micro = micro_encoder(micro)
        e_biom  = biom_encoder(biom)
        e_other = other_encoder(other)

        loss = patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, pids, tau=tau)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate_one_epoch(
    mri_encoder, micro_encoder, biom_encoder, other_encoder,
    dataloader, device="cpu", tau=0.1
):
    mri_encoder.eval()
    micro_encoder.eval()
    biom_encoder.eval()
    other_encoder.eval()

    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            mri, micro, biom, other, pids, _ = [x.to(device) for x in batch]
            e_mri   = mri_encoder(mri)
            e_micro = micro_encoder(micro)
            e_biom  = biom_encoder(biom)
            e_other = other_encoder(other)

            loss = patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, pids, tau=tau)
            running_loss += loss.item()

    return running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

# ==============================
# 🚀 SIMPLE EARLY STOPPING FUNCTION
# ==============================
def early_stopping_check(val_loss_history, patience=3, min_delta=1e-4):
    """
    Returns True if there's no improvement in 'val_loss' for 'patience' epochs 
    beyond a threshold 'min_delta'. 
    """
    if len(val_loss_history) < patience:
        return False

    # Compare the last 'patience' epochs to the best prior epoch
    recent_losses = val_loss_history[-patience:]
    best_in_recent = min(recent_losses)

    # If best in the last 'patience' epochs is still higher (by min_delta) 
    # than the overall best prior to that window, we consider it no improvement
    overall_best_before = min(val_loss_history[:-patience], default=float('inf'))

    return (best_in_recent > (overall_best_before - min_delta))

# ==============================
# 🚀 TRAIN MODEL WITH EARLY STOPPING
# ==============================
def train_model(data, mri_dict, embed_dim=32, epochs=30, batch_size=3, lr=1e-4):
    # 1) Dataset & 80:20 Split
    dataset = PatientDataset(data, mri_dict)
    total_size = len(dataset)
    val_size   = int(0.2 * total_size)
    train_size = total_size - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    # 2) Encoders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    micro_dim = dataset.micro_data.shape[1]
    biom_dim  = dataset.biom_data.shape[1]
    other_dim = dataset.other_data.shape[1]

    micro_encoder = MLPEncoder(micro_dim, embed_dim).to(device)
    biom_encoder  = MLPEncoder(biom_dim,  embed_dim).to(device)
    other_encoder = MLPEncoder(other_dim, embed_dim).to(device)

    # CLIP-based MRI encoder
    mri_encoder   = MRIClipEncoder(embed_dim=embed_dim, device=device).to(device)

    # 3) Optimizer
    params = (list(micro_encoder.parameters()) 
            + list(biom_encoder.parameters())  
            + list(other_encoder.parameters()) 
            + list(mri_encoder.parameters()))
    optimizer = torch.optim.Adam(params, lr=lr)

    # 4) Train & Validate with Early Stopping
    train_losses, val_losses = [], []

    patience = 3   # stop if no improvement for 3 epochs
    for epoch in range(1, epochs+1):
        print("{epochs;}")
        train_loss = train_one_epoch(
            mri_encoder, micro_encoder, biom_encoder, other_encoder,
            train_loader, optimizer, device=device, tau=0.1
        )
        val_loss = validate_one_epoch(
            mri_encoder, micro_encoder, biom_encoder, other_encoder,
            val_loader, device=device, tau=0.1
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping check
        if early_stopping_check(val_losses, patience=patience, min_delta=1e-4):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # 5) Plot both (only up to final epoch if we stopped early)
    final_epoch = len(train_losses)
    x_axis = range(1, final_epoch+1)
    plt.figure(figsize=(7,5))
    plt.plot(x_axis, train_losses, '-o', label='Train Loss')
    plt.plot(x_axis, val_losses,   '-x', label='Val Loss')
    plt.title("Multi-Modal Contrastive (CLIP + Tabular) with Early Stopping")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_val_loss.png")
    plt.close()
    print("Saved training and validation loss to 'train_val_loss.png'.")


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    data = load_data()
    mri_dict = load_mri_data("/Users/thongnguyen/Downloads/CBF_imaging")
    train_model(data, mri_dict, embed_dim=32, epochs=100, batch_size=3, lr=1e-4)
