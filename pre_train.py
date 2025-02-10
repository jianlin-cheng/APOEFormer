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

    # Drop a problematic column if present
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
    # For Other, select numeric columns then add back Patient_ID and Timepoint
    other_numeric = add_prefix(other_data.select_dtypes(include=[np.number]), "Other")
    other_numeric[['Patient_ID', 'Timepoint']] = other_data[['Patient_ID', 'Timepoint']]
    
    # Add Patient_ID and Timepoint back for the other dataframes
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
# DATASET CLASS WITH MULTIPLE NEGATIVE SAMPLES
#############################################
class PatientContrastiveDatasetSeparate(Dataset):
    """
    For each original observation (a row in the merged data), this dataset generates
    (num_negatives + 1) samples:
      - 1 positive sample (label = 1): all modalities come from that observation.
      - num_negatives negative samples (label = 0): the MRI (rotated) from that observation
        is paired with tabular data that is a mixture (averaged from num_to_mix different rows)
        from different observations.
    
    Total dataset size = (# original rows) * (1 + num_negatives)
    """
    def __init__(self, data, mri_dict, num_negatives=300, num_to_mix=3):
        self.data = data.reset_index(drop=True)
        self.mri_dict = mri_dict
        self.N = len(self.data)
        self.num_negatives = num_negatives
        self.num_to_mix = num_to_mix
        
        # Convert tabular data columns to float32 arrays (using the prefixes)
        self.micro_data = data.filter(like='Microbiome_').astype(np.float32).values
        self.biom_data  = data.filter(like='Biomarker_').astype(np.float32).values
        self.other_data = data.filter(like='Other_').astype(np.float32).values

        # Optionally store patient/time info (for debugging)
        self.patient_ids = data['Patient_ID'].values
        self.timepoints = data['Timepoint'].values

        print("CSV data size (before pairing):", self.N)
        total_samples = self.N * (1 + self.num_negatives)
        print("Dataset size (after separating positive and negative samples):", total_samples)
        print("Micro data shape:", self.micro_data.shape)
        print("Biom data shape:", self.biom_data.shape)
        print("Other data shape:", self.other_data.shape)
        self._printed_example = False

    def __len__(self):
        return self.N * (self.num_negatives + 1)

    def __getitem__(self, index):
        # Determine which original row and which sample type.
        row_index = index // (self.num_negatives + 1)
        sample_type = index % (self.num_negatives + 1)
        row = self.data.iloc[row_index]
        patient_id = str(row['Patient_ID'])
        timepoint = str(row['Timepoint'])
        mri_tensor = self.mri_dict.get(
            (patient_id, timepoint),
            torch.zeros((1, 128, 128, 128), dtype=torch.float32)
        )
        # Get the tabular data from the current row
        micro_tensor = torch.tensor(self.micro_data[row_index], dtype=torch.float32)
        biom_tensor  = torch.tensor(self.biom_data[row_index], dtype=torch.float32)
        other_tensor = torch.tensor(self.other_data[row_index], dtype=torch.float32)
        
        if sample_type == 0:
            # Positive sample: all modalities come from the same observation.
            label = 1
            sample_mri = mri_tensor
            sample_micro = micro_tensor
            sample_biom = biom_tensor
            sample_other = other_tensor
        else:
            # Negative sample: use the rotated MRI from the current row,
            # but mix tabular data from num_to_mix different random rows.
            label = 0
            sample_mri = torch.rot90(mri_tensor, k=1, dims=(2,3))
            neg_micro_vals = []
            neg_biom_vals = []
            neg_other_vals = []
            for _ in range(self.num_to_mix):
                neg_row = row_index
                while neg_row == row_index:
                    neg_row = np.random.randint(0, self.N)
                neg_micro_vals.append(self.micro_data[neg_row])
                neg_biom_vals.append(self.biom_data[neg_row])
                neg_other_vals.append(self.other_data[neg_row])
            neg_micro_avg = np.mean(neg_micro_vals, axis=0)
            neg_biom_avg = np.mean(neg_biom_vals, axis=0)
            neg_other_avg = np.mean(neg_other_vals, axis=0)
            sample_micro = torch.tensor(neg_micro_avg, dtype=torch.float32)
            sample_biom = torch.tensor(neg_biom_avg, dtype=torch.float32)
            sample_other = torch.tensor(neg_other_avg, dtype=torch.float32)
        
        # Optionally print an example from the first observation's samples.
        if index < (self.num_negatives + 1) and not self._printed_example:
            sample_type_str = "Positive" if label == 1 else "Negative"
            print("\n--- Example from Dataset ---")
            print(f"{sample_type_str} sample (label={label}):")
            print(" MRI tensor shape:", sample_mri.shape)
            print(" Micro data sample:", sample_micro)
            print(" Biomarker data sample:", sample_biom)
            print(" Other data sample:", sample_other)
            print("----------------------------------------\n")
            self._printed_example = True

        return {
            "mri": sample_mri,
            "micro": sample_micro,
            "biom": sample_biom,
            "other": sample_other,
            "label": torch.tensor(label, dtype=torch.float32)
        }

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
            volume_3d = mri_batch[i, 0]  # shape: (D, H, W)
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
# TRAINING AND VALIDATION FUNCTIONS
#############################################
def train_one_epoch_separate(mri_encoder, micro_encoder, biom_encoder, other_encoder,
                             dataloader, optimizer, device="cpu"):
    mri_encoder.train()
    micro_encoder.train()
    biom_encoder.train()
    other_encoder.train()

    running_loss = 0.0
    bce_loss = nn.BCELoss()

    for batch in dataloader:
        mri_batch = batch["mri"].to(device)
        micro_batch = batch["micro"].to(device)
        biom_batch = batch["biom"].to(device)
        other_batch = batch["other"].to(device)
        labels = batch["label"].to(device)

        e_mri = mri_encoder(mri_batch)
        e_micro = micro_encoder(micro_batch)
        e_biom = biom_encoder(biom_batch)
        e_other = other_encoder(other_batch)
        # Average tabular embeddings
        e_tabular = (e_micro + e_biom + e_other) / 3

        cos_sim = F.cosine_similarity(e_mri, e_tabular, dim=-1)
        pred = (cos_sim + 1) / 2

        loss = bce_loss(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate_one_epoch_separate(mri_encoder, micro_encoder, biom_encoder, other_encoder,
                                dataloader, device="cpu"):
    mri_encoder.eval()
    micro_encoder.eval()
    biom_encoder.eval()
    other_encoder.eval()

    running_loss = 0.0
    bce_loss = nn.BCELoss()

    with torch.no_grad():
        for batch in dataloader:
            mri_batch = batch["mri"].to(device)
            micro_batch = batch["micro"].to(device)
            biom_batch = batch["biom"].to(device)
            other_batch = batch["other"].to(device)
            labels = batch["label"].to(device)

            e_mri = mri_encoder(mri_batch)
            e_micro = micro_encoder(micro_batch)
            e_biom = biom_encoder(biom_batch)
            e_other = other_encoder(other_batch)
            e_tabular = (e_micro + e_biom + e_other) / 3

            cos_sim = F.cosine_similarity(e_mri, e_tabular, dim=-1)
            pred = (cos_sim + 1) / 2

            loss = bce_loss(pred, labels)
            running_loss += loss.item()
    
    return running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

#############################################
# TRAINING FUNCTION
#############################################
def train_model(data, mri_dict, embed_dim=32, epochs=100, batch_size=3, lr=1e-4, num_negatives=300):
    print("CSV data size (before pairing):", len(data))
    dataset = PatientContrastiveDatasetSeparate(data, mri_dict, num_negatives=num_negatives, num_to_mix=3)
    print("Dataset size (after separation):", len(dataset))
    
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    print(f"Training set size: {train_size} | Validation set size: {val_size}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    micro_dim = dataset.micro_data.shape[1]
    biom_dim = dataset.biom_data.shape[1]
    other_dim = dataset.other_data.shape[1]

    micro_encoder = MLPEncoder(micro_dim, embed_dim).to(device)
    biom_encoder = MLPEncoder(biom_dim, embed_dim).to(device)
    other_encoder = MLPEncoder(other_dim, embed_dim).to(device)
    mri_encoder = MRIClipEncoder(embed_dim=embed_dim, device=device).to(device)

    params = list(mri_encoder.parameters()) + list(micro_encoder.parameters()) + \
             list(biom_encoder.parameters()) + list(other_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    train_losses, val_losses = [], []
    patience = 3

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_separate(mri_encoder, micro_encoder, biom_encoder, other_encoder,
                                              train_loader, optimizer, device=device)
        val_loss = validate_one_epoch_separate(mri_encoder, micro_encoder, biom_encoder, other_encoder,
                                               val_loader, device=device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if epoch >= patience and min(val_losses[-patience:]) > (min(val_losses[:-patience], default=float('inf')) - 1e-4):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    final_epoch = len(train_losses)
    x_axis = range(1, final_epoch + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(x_axis, train_losses, '-o', label='Train Loss')
    plt.plot(x_axis, val_losses, '-x', label='Val Loss')
    plt.title("Contrastive Training with Multiple Negative Samples")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_val_loss.png")
    plt.close()
    print("Saved training and validation loss to 'train_val_loss.png'.")

#############################################
# MAIN
#############################################
if __name__ == "__main__":
    data = load_data()
    mri_dict = load_mri_data("/Users/thongnguyen/Downloads/CBF_imaging")
    # Adjust num_negatives as desired; here, num_negatives=300 will yield a large dataset.
    train_model(data, mri_dict, embed_dim=32, epochs=100, batch_size=3, lr=1e-4, num_negatives=300)
