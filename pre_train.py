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
# 🚀 LOAD STRUCTURED DATA WITH PREFIXES
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

    # Load CSV files
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

    # Drop problematic columns if necessary
    if 'APOE4' in other_data.columns:
        other_data.drop(columns=['APOE4'], inplace=True)

    # Convert IDs and Timepoints to strings
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

    # Create numeric subsets with prefixes
    microbiome_numeric = add_prefix(filter_numeric(microbiome), "Microbiome")
    blood_metabolites_numeric = add_prefix(filter_numeric(blood_metabolites), "Biomarker")
    inflammatory_markers_numeric = add_prefix(filter_numeric(inflammatory_markers), "Inflammatory")
    blood_data_numeric = add_prefix(filter_numeric(blood_data), "Blood")
    # For Other data, select numeric columns and then add Patient_ID and Timepoint back.
    other_numeric = add_prefix(other_data.select_dtypes(include=[np.number]), "Other")
    other_numeric[['Patient_ID', 'Timepoint']] = other_data[['Patient_ID', 'Timepoint']]

    # Add Patient_ID and Timepoint back to the other numeric dataframes
    for numeric_df, original_df in zip(
        [microbiome_numeric, blood_metabolites_numeric, inflammatory_markers_numeric, blood_data_numeric],
        [microbiome, blood_metabolites, inflammatory_markers, blood_data]
    ):
        numeric_df[['Patient_ID', 'Timepoint']] = original_df[['Patient_ID', 'Timepoint']]

    # Merge into other_numeric
    data = other_numeric.copy()
    for numeric_df in [microbiome_numeric, blood_metabolites_numeric, inflammatory_markers_numeric, blood_data_numeric]:
        data = data.merge(numeric_df, on=['Patient_ID', 'Timepoint'], how='inner')

    data.fillna(0, inplace=True)
    print(f"Final merged data shape: {data.shape}")
    return data


# ==============================
# 🔥 LOAD MRI DATA (unchanged)
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
                    # If 4D, average over the last dimension
                    if len(mri_image.shape) == 4:
                        mri_image = np.mean(mri_image, axis=-1)
                    # Ensure shape is (1, D, H, W)
                    mri_tensor = torch.tensor(mri_image, dtype=torch.float32).unsqueeze(0)
                    mri_dict[(patient_id, timepoint_name)] = mri_tensor
    return mri_dict

# ==============================
# 📌 DATASET WITH POSITIVE & NEGATIVE PAIRS
# ==============================
class PatientContrastiveDataset(Dataset):
    """
    For each sample, return a tuple:
      (positive_sample, negative_sample)
      
    - Positive sample: All modalities (MRI, Microbiome, Biomarkers, Other) from the same patient.
    - Negative sample: Uses the same MRI (rotated by 90° along the H and W dims)
      paired with tabular data from a different patient.
    """
    def __init__(self, data, mri_dict):
        self.data = data.reset_index(drop=True)
        self.mri_dict = mri_dict

        # Convert Patient_ID and Timepoint into numeric codes (if needed for loss)
        self.patient_ids = torch.tensor(
            data['Patient_ID'].astype('category').cat.codes.values, 
            dtype=torch.long
        )
        self.timepoints = torch.tensor(
            data['Timepoint'].astype('category').cat.codes.values,
            dtype=torch.long
        )

        # Convert filtered columns to float32 so they can be turned into tensors
        self.micro_data = data.filter(like='Microbiome_').astype(np.float32).values
        self.biom_data  = data.filter(like='Biomarker_').astype(np.float32).values
        self.other_data = data.filter(like='Other_').astype(np.float32).values

        # Print dataset sizes and shapes
        print("Original dataset size (CSV rows):", len(self.data))
        print("Dataset size after adding positive and negative pairs: ", len(self.data))
        print("Micro data shape:", self.micro_data.shape)
        print("Biom data shape:", self.biom_data.shape)
        print("Other data shape:", self.other_data.shape)
        self._printed_example = False

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # --------- Positive sample ---------
        row = self.data.iloc[idx]
        patient_id = str(row['Patient_ID'])
        timepoint  = str(row['Timepoint'])
        mri_tensor = self.mri_dict.get(
            (patient_id, timepoint),
            torch.zeros((1,128,128,128), dtype=torch.float32)  # fallback if MRI not found
        )
        micro_tensor = torch.tensor(self.micro_data[idx], dtype=torch.float32)
        biom_tensor  = torch.tensor(self.biom_data[idx], dtype=torch.float32)
        other_tensor = torch.tensor(self.other_data[idx], dtype=torch.float32)
        pid_code = self.patient_ids[idx]
        t_code   = self.timepoints[idx]
        positive = (mri_tensor, micro_tensor, biom_tensor, other_tensor, pid_code, t_code)
        
        # --------- Negative sample ---------
        # Choose a row from a different patient
        neg_idx = idx
        while True:
            neg_idx = np.random.randint(0, len(self.data))
            if str(self.data.iloc[neg_idx]['Patient_ID']) != patient_id:
                break
        # Use the same MRI but apply a 90° rotation (rotate along the H and W dimensions)
        neg_mri = torch.rot90(mri_tensor, k=1, dims=(2,3))
        neg_micro = torch.tensor(self.micro_data[neg_idx], dtype=torch.float32)
        neg_biom  = torch.tensor(self.biom_data[neg_idx], dtype=torch.float32)
        neg_other = torch.tensor(self.other_data[neg_idx], dtype=torch.float32)
        neg_pid   = self.patient_ids[neg_idx]
        neg_time  = self.timepoints[neg_idx]
        negative = (neg_mri, neg_micro, neg_biom, neg_other, neg_pid, neg_time)
        
        # Print an example pair (only once for the first sample)
        if idx == 0 and not self._printed_example:
            print("\n--- Example Pair from Dataset Index 0 ---")
            print("Positive sample:")
            print(" MRI tensor shape:", positive[0].shape)
            print(" Micro data sample:", positive[1])
            print(" Biomarker data sample:", positive[2])
            print(" Other data sample:", positive[3])
            print("Negative sample (MRI rotated):")
            print(" MRI tensor shape:", negative[0].shape)
            print(" Micro data sample:", negative[1])
            print(" Biomarker data sample:", negative[2])
            print(" Other data sample:", negative[3])
            print("----------------------------------------\n")
            self._printed_example = True

        return positive, negative

# ==============================
# 🚀 CLIP-BASED MRI ENCODER (unchanged)
# ==============================
class MRIClipEncoder(nn.Module):
    """
    Takes (batch, 1, D, H, W) MRI, extracts a 2D slice,
    runs it through CLIP, then projects 512-d -> embed_dim.
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
            volume_3d = mri_batch[i, 0]  # shape: (D, H, W)
            D = volume_3d.shape[0]
            mid_slice_idx = D // 2
            slice_2d = volume_3d[mid_slice_idx]  # (H, W)
            # Normalize to [0,255]
            arr_np = slice_2d.cpu().numpy()
            rng = np.ptp(arr_np)
            arr_np = (arr_np - arr_np.min()) / (rng + 1e-8)
            arr_np = (arr_np * 255).astype(np.uint8)
            # Convert to PIL image (RGB)
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

# ==============================
# 🚀 MLP ENCODER FOR TABULAR DATA (unchanged)
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
# 🚀 PATIENT CONTRASTIVE LOSS (for positive pairs)
# ==============================
def explicit_patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, patient_ids, tau=0.1):
    B, D = e_mri.shape  # (batch, embed_dim)
    embeddings = torch.cat([e_mri, e_micro, e_biom, e_other], dim=0)  # shape: (B*4, embed_dim)
    sim_matrix = torch.matmul(embeddings, embeddings.t()) / tau
    sim_exp = torch.exp(sim_matrix)
    repeated_ids = patient_ids.repeat(4)
    diag_mask = torch.eye(B*4, dtype=torch.bool, device=sim_matrix.device)
    same_mask = (repeated_ids.unsqueeze(0) == repeated_ids.unsqueeze(1)) & (~diag_mask)
    diff_mask = (repeated_ids.unsqueeze(0) != repeated_ids.unsqueeze(1)) & (~diag_mask)
    pos_sim = sim_exp * same_mask.float()
    sum_pos = pos_sim.sum(dim=1)
    neg_sim = sim_exp * diff_mask.float()
    sum_neg = neg_sim.sum(dim=1)
    eps = 1e-8
    loss = -torch.log((sum_pos + eps) / (sum_pos + sum_neg + eps))
    return loss.mean()

# ==============================
# 🚀 TRAINING LOOP WITH NEGATIVE PAIRS INCLUDED
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
    margin = 0.2
    cos_sim = nn.CosineSimilarity(dim=-1)

    for batch in dataloader:
        # Each batch item is a tuple: (positive, negative)
        pos_batch, neg_batch = batch  # unpack positive and negative lists
        pos_mri, pos_micro, pos_biom, pos_other, pos_pids, _ = zip(*pos_batch)
        pos_mri   = torch.stack(pos_mri).to(device)
        pos_micro = torch.stack(pos_micro).to(device)
        pos_biom  = torch.stack(pos_biom).to(device)
        pos_other = torch.stack(pos_other).to(device)
        pos_pids  = torch.stack(pos_pids).to(device)

        neg_mri, neg_micro, neg_biom, neg_other, neg_pids, _ = zip(*neg_batch)
        neg_mri   = torch.stack(neg_mri).to(device)
        neg_micro = torch.stack(neg_micro).to(device)
        neg_biom  = torch.stack(neg_biom).to(device)
        neg_other = torch.stack(neg_other).to(device)
        neg_pids  = torch.stack(neg_pids).to(device)

        # Compute embeddings for positive samples:
        e_mri_pos   = mri_encoder(pos_mri)
        e_micro_pos = micro_encoder(pos_micro)
        e_biom_pos  = biom_encoder(pos_biom)
        e_other_pos = other_encoder(pos_other)
        pos_loss = explicit_patient_contrastive_loss(e_mri_pos, e_micro_pos, e_biom_pos, e_other_pos, pos_pids, tau=tau)

        # Compute embeddings for negative samples:
        e_mri_neg   = mri_encoder(neg_mri)
        e_micro_neg = micro_encoder(neg_micro)
        e_biom_neg  = biom_encoder(neg_biom)
        e_other_neg = other_encoder(neg_other)
        sim_mri_micro = cos_sim(e_mri_neg, e_micro_neg)
        sim_mri_biom  = cos_sim(e_mri_neg, e_biom_neg)
        sim_mri_other = cos_sim(e_mri_neg, e_other_neg)
        neg_loss = (F.relu(sim_mri_micro - margin).mean() +
                    F.relu(sim_mri_biom  - margin).mean() +
                    F.relu(sim_mri_other - margin).mean()) / 3

        loss = pos_loss + neg_loss

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
            pos_batch, _ = batch  # Only use positive pairs for validation
            pos_mri, pos_micro, pos_biom, pos_other, pos_pids, _ = zip(*pos_batch)
            pos_mri   = torch.stack(pos_mri).to(device)
            pos_micro = torch.stack(pos_micro).to(device)
            pos_biom  = torch.stack(pos_biom).to(device)
            pos_other = torch.stack(pos_other).to(device)
            pos_pids  = torch.stack(pos_pids).to(device)
            e_mri   = mri_encoder(pos_mri)
            e_micro = micro_encoder(pos_micro)
            e_biom  = biom_encoder(pos_biom)
            e_other = other_encoder(pos_other)
            loss = explicit_patient_contrastive_loss(e_mri, e_micro, e_biom, e_other, pos_pids, tau=tau)
            running_loss += loss.item()

    return running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

# ==============================
# 🚀 TRAIN MODEL WITH EARLY STOPPING
# ==============================
def train_model(data, mri_dict, embed_dim=32, epochs=100, batch_size=3, lr=1e-4):
    print("CSV data size (before pairing):", len(data))
    # Create our dataset that returns (positive, negative) pairs
    dataset = PatientContrastiveDataset(data, mri_dict)
    print("Dataset size (after pairing):", len(dataset))
    
    total_size = len(dataset)
    val_size   = int(0.2 * total_size)
    train_size = total_size - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f"Training set size: {train_size} | Validation set size: {val_size}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=lambda b: list(zip(*b)))
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=lambda b: list(zip(*b)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    micro_dim = dataset.micro_data.shape[1]
    biom_dim  = dataset.biom_data.shape[1]
    other_dim = dataset.other_data.shape[1]

    micro_encoder = MLPEncoder(micro_dim, embed_dim).to(device)
    biom_encoder  = MLPEncoder(biom_dim,  embed_dim).to(device)
    other_encoder = MLPEncoder(other_dim, embed_dim).to(device)
    mri_encoder   = MRIClipEncoder(embed_dim=embed_dim, device=device).to(device)

    params = (list(micro_encoder.parameters()) +
              list(biom_encoder.parameters())  +
              list(other_encoder.parameters()) +
              list(mri_encoder.parameters()))
    optimizer = torch.optim.Adam(params, lr=lr)

    train_losses, val_losses = [], []
    patience = 3
    for epoch in range(1, epochs+1):
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
        if len(val_losses) >= patience and min(val_losses[-patience:]) > (min(val_losses[:-patience], default=float('inf')) - 1e-4):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    final_epoch = len(train_losses)
    x_axis = range(1, final_epoch+1)
    plt.figure(figsize=(7,5))
    plt.plot(x_axis, train_losses, '-o', label='Train Loss')
    plt.plot(x_axis, val_losses,   '-x', label='Val Loss')
    plt.title("Multi-Modal Contrastive (CLIP + Tabular) with Explicit Negative Pairs")
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
