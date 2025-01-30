import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import nibabel as nib

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, Callback

import matplotlib.pyplot as plt  # For the chart

# ------------------------------
# 1) CSV Data Loading & Merging
# ------------------------------
def load_data():
    """
    Load and merge all CSV files, including Brain_CBF_Imaging.csv.
    Rename numeric columns to have 'Microbiome_', 'Biomarker_', 'Other_', 'CBF_' prefixes.
    """
    def load_file(file_path):
        try:
            df = pd.read_csv(file_path)
            df.rename(columns=lambda x: x.strip(), inplace=True)
            return df
        except FileNotFoundError:
            print(f"⚠️ File not found: {file_path}")
            return pd.DataFrame()

    # Adjust these paths to your local CSVs
    microbiome           = load_file('/Users/thongnguyen/Downloads/New/Microbiome.csv')
    blood_metabolites    = load_file('/Users/thongnguyen/Downloads/New/Blood_Metabolites.csv')
    inflammatory_markers = load_file('/Users/thongnguyen/Downloads/New/Sirolimus_inflammatory_markers.csv')
    blood_data           = load_file('/Users/thongnguyen/Downloads/New/Sirolimus_Blood_Data.csv')
    other_data           = load_file('/Users/thongnguyen/Downloads/New/Other.csv')
    brain_cbf_imaging    = load_file('/Users/thongnguyen/Downloads/New/Brain_CBF_Imaging.csv')  # <--- NEW FILE

    for name, df in zip(
        ["Microbiome", "Blood Metabolites", "Inflammatory Markers", "Blood Data", "Other", "Brain CBF Imaging"],
        [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data, brain_cbf_imaging]
    ):
        if df.empty:
            print(f"⚠️ Warning: {name} data is empty or missing.")
        else:
            print(f"{name} data loaded with shape: {df.shape}")

    # Drop APOE4 if present
    if 'APOE4' in other_data.columns:
        other_data.drop(columns=['APOE4'], inplace=True)

    # Convert 'Patient_ID' & 'Timepoint' to string format
    for df in [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data, brain_cbf_imaging]:
        if 'Patient_ID' in df.columns:
            df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
        if 'Timepoint' in df.columns:
            df['Timepoint'] = df['Timepoint'].astype(str).str.strip()

    def filter_numeric(df):
        return df.select_dtypes(include=['number']).copy()

    # Extract numeric features
    microbiome_numeric           = filter_numeric(microbiome)
    blood_metabolites_numeric    = filter_numeric(blood_metabolites)
    inflammatory_markers_numeric = filter_numeric(inflammatory_markers)
    blood_data_numeric           = filter_numeric(blood_data)
    other_data_numeric           = filter_numeric(other_data)
    brain_cbf_numeric            = filter_numeric(brain_cbf_imaging)

    # Helper to restore 'Patient_ID' and 'Timepoint'
    def add_id_time(df, original_df):
        if 'Patient_ID' in original_df.columns and 'Timepoint' in original_df.columns:
            df['Patient_ID'] = original_df['Patient_ID']
            df['Timepoint']  = original_df['Timepoint']
        return df

    microbiome_numeric           = add_id_time(microbiome_numeric,           microbiome)
    blood_metabolites_numeric    = add_id_time(blood_metabolites_numeric,    blood_metabolites)
    inflammatory_markers_numeric = add_id_time(inflammatory_markers_numeric, inflammatory_markers)
    blood_data_numeric           = add_id_time(blood_data_numeric,           blood_data)
    other_data_numeric           = add_id_time(other_data_numeric,           other_data)
    brain_cbf_numeric            = add_id_time(brain_cbf_numeric,            brain_cbf_imaging)

    # Prefix column names
    def add_prefix_except(df, prefix, skip=('Patient_ID', 'Timepoint')):
        return df.rename(columns={
            col: f"{prefix}{col}" 
            for col in df.columns if col not in skip
        })

    microbiome_numeric           = add_prefix_except(microbiome_numeric,           "Microbiome_")
    blood_metabolites_numeric    = add_prefix_except(blood_metabolites_numeric,    "Biomarker_")
    inflammatory_markers_numeric = add_prefix_except(inflammatory_markers_numeric, "Biomarker_")
    blood_data_numeric           = add_prefix_except(blood_data_numeric,           "Biomarker_")
    other_data_numeric           = add_prefix_except(other_data_numeric,           "Other_")
    brain_cbf_numeric            = add_prefix_except(brain_cbf_numeric,            "CBF_")

    # Merge all on 'Patient_ID' and 'Timepoint'
    data = other_data_numeric
    for numeric_df in [
        microbiome_numeric,
        blood_metabolites_numeric,
        inflammatory_markers_numeric,
        blood_data_numeric,
        brain_cbf_numeric  # <--- Include Brain CBF Imaging data
    ]:
        data = data.merge(numeric_df, on=['Patient_ID', 'Timepoint'], how='inner')

    data.fillna(0, inplace=True)
    print(f"✅ Final merged data shape: {data.shape}")
    return data


# -----------------------------
# 2) Load MRI Data (.nii Files)
# -----------------------------
def load_mri_data(root_dir):
    """
    Load .nii (3D or 4D) files. If 4D, we average over the last dimension.
    Returns a dict keyed by (patient_id, timepoint_name) -> torch.Tensor
    """
    mri_dict = {}
    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        if os.path.isdir(patient_path):
            for timepoint_file in os.listdir(patient_path):
                if timepoint_file.endswith('.nii'):
                    timepoint_name = timepoint_file.split('.')[0]
                    timepoint_path = os.path.join(patient_path, timepoint_file)

                    mri_image = nib.load(timepoint_path).get_fdata()
                    # If 4D, reduce to 3D
                    if len(mri_image.shape) == 4:
                        print(f"Reducing 4D MRI for {patient_id} at {timepoint_name} to 3D...")
                        mri_image = np.mean(mri_image, axis=-1)

                    mri_tensor = torch.tensor(mri_image, dtype=torch.float32).unsqueeze(0)
                    mri_dict[(patient_id, timepoint_name)] = mri_tensor
    return mri_dict


# -----------------------------
# 3) PatientDataset Definition
# -----------------------------
class PatientDataset(Dataset):
    """
    Returns (mri, microbiome_data, biomarker_data, other_data, cbf_data, patient_id_code).
    """
    def __init__(self, data, mri_dict):
        super().__init__()
        self.data = data
        self.mri_dict = mri_dict

        # Turn patient ID strings into numeric codes
        self.patient_ids = torch.tensor(
            data['Patient_ID'].astype('category').cat.codes.values, dtype=torch.long
        )

        # Each subset of columns
        self.microbiome_data = data.filter(like='Microbiome_').values
        self.biomarker_data  = data.filter(like='Biomarker_').values
        self.other_data      = data.filter(like='Other_').values
        self.cbf_data        = data.filter(like='CBF_').values  # <--- new

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pat_id = str(row['Patient_ID'])
        timept = str(row['Timepoint'])

        # MRI => shape: (1, D, H, W), or zero if missing
        mri_tensor = self.mri_dict.get(
            (pat_id, timept),
            torch.zeros((1,128,128,128), dtype=torch.float32)
        )

        micro_tensor = torch.tensor(self.microbiome_data[idx], dtype=torch.float32)
        biom_tensor  = torch.tensor(self.biomarker_data[idx],  dtype=torch.float32)
        other_tensor = torch.tensor(self.other_data[idx],      dtype=torch.float32)
        cbf_tensor   = torch.tensor(self.cbf_data[idx],        dtype=torch.float32)
        pid_code     = self.patient_ids[idx]

        return mri_tensor, micro_tensor, biom_tensor, other_tensor, cbf_tensor, pid_code


# -----------------------------
# 4) DataModule
# -----------------------------
class AlzheimerDataModule(pl.LightningDataModule):
    def __init__(self, data, mri_dict, batch_size=4):
        super().__init__()
        self.data = data
        self.mri_dict = mri_dict
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = PatientDataset(self.data, self.mri_dict)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


# -----------------------------
# 5) Encoders & Contrastive Loss
# -----------------------------
class CLIPEncoder(nn.Module):
    """A simple 3D CNN that encodes MRI volumes to an embedding."""
    def __init__(self, input_shape=(128,128,128), embed_dim=32):
        super(CLIPEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Figure out flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            out   = self.conv(dummy)
            self.flat_dim = out.view(1, -1).shape[1]

        self.fc = nn.Linear(self.flat_dim, embed_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=-1)


class MLPEncoder(nn.Module):
    """An MLP to encode numeric data (microbiome, biomarker, other, CBF) into same embed_dim."""
    def __init__(self, input_dim, embed_dim=32):
        super(MLPEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return F.normalize(z, p=2, dim=-1)


def patient_contrastive_loss(
    e_mri, e_microbiome, e_biomarker, e_other, e_cbf, patient_ids, tau=0.1
):
    """
    Computes a multi-modal contrastive (CLIP-like) loss over a single batch:
      - Flatten (batch_size*5, embed_dim) 
      - Cross-sample similarity
      - InfoNCE variant with multiple positives (same patient)
    """
    B, D = e_mri.shape  # batch_size, embed_dim

    # Stack embeddings: shape => (B*5, embed_dim)
    embeddings = torch.cat([e_mri, e_microbiome, e_biomarker, e_other, e_cbf], dim=0)

    # Pairwise similarity => (B*5, B*5)
    sim_matrix = torch.matmul(embeddings, embeddings.t()) / tau

    # Expand patient_ids for each modality
    repeated_ids = patient_ids.repeat(5)  # shape => (B*5,)

    # Mask out diagonal
    diag_mask = torch.eye(B*5, dtype=torch.bool, device=sim_matrix.device)
    positive_mask = (repeated_ids.unsqueeze(0) == repeated_ids.unsqueeze(1)) & (~diag_mask)

    sim_exp = torch.exp(sim_matrix)
    all_sum = sim_exp.sum(dim=1)
    pos_sum = (sim_exp * positive_mask).sum(dim=1)

    eps = 1e-8
    loss = -torch.log((pos_sum + eps) / (all_sum + eps))
    return loss.mean()


# -----------------------------
# 6) LightningModule
# -----------------------------
class MultiModalEmbeddingModule(pl.LightningModule):
    """
    Encodes each modality (MRI, Microbiome, Biomarker, Other, CBF) to embed_dim,
    then performs a contrastive loss in training_step.
    """
    def __init__(self, micro_dim, biom_dim, other_dim, cbf_dim, embed_dim=32, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Encoders
        self.mri_encoder       = CLIPEncoder(input_shape=(128,128,128), embed_dim=embed_dim)
        self.microbe_encoder   = MLPEncoder(micro_dim, embed_dim)
        self.biomarker_encoder = MLPEncoder(biom_dim, embed_dim)
        self.other_encoder     = MLPEncoder(other_dim, embed_dim)
        self.cbf_encoder       = MLPEncoder(cbf_dim, embed_dim)

        self.lr = lr

    def forward(self, mri, micro, biom, other, cbf):
        e_mri        = self.mri_encoder(mri)
        e_microbe    = self.microbe_encoder(micro)
        e_biomarker  = self.biomarker_encoder(biom)
        e_other      = self.other_encoder(other)
        e_cbf        = self.cbf_encoder(cbf)
        return e_mri, e_microbe, e_biomarker, e_other, e_cbf

    def training_step(self, batch, batch_idx):
        # batch => (mri, micro, biom, other, cbf, patient_ids)
        mri, micro, biom, other, cbf, pids = batch
        e_mri, e_micro, e_biom, e_oth, e_cbf = self(mri, micro, biom, other, cbf)

        loss = patient_contrastive_loss(
            e_mri, e_micro, e_biom, e_oth, e_cbf, pids, tau=0.1
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# -----------------------------
# 7) Custom Callback for Chart
# -----------------------------
class ChartLoggerCallback(Callback):
    """
    Collects train_loss after each epoch and plots a chart at the end.
    """
    def __init__(self):
        super().__init__()
        self.epoch_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.epoch_losses.append(loss.item())

    def on_fit_end(self, trainer, pl_module):
        # Plot the training loss vs. epoch
        epochs = range(1, len(self.epoch_losses) + 1)
        plt.figure(figsize=(6,4))
        plt.plot(epochs, self.epoch_losses, marker='o', label='Train Loss')
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig("training_loss_plot1.png")
        plt.close()
        print("Saved training loss plot to training_loss_plot1.png")


# -----------------------------
# 8) Main Script
# -----------------------------
def main():
    # 1) Load structured CSV data (including new Brain_CBF_Imaging.csv)
    data = load_data()

    # 2) Load MRI data
    mri_root = "/Users/thongnguyen/Downloads/CBF_imaging"
    mri_dict = load_mri_data(mri_root)

    # 3) Count feature dims
    micro_cols = data.filter(like='Microbiome_').columns
    biom_cols  = data.filter(like='Biomarker_').columns
    other_cols = data.filter(like='Other_').columns
    cbf_cols   = data.filter(like='CBF_').columns

    micro_dim = len(micro_cols)
    biom_dim  = len(biom_cols)
    other_dim = len(other_cols)
    cbf_dim   = len(cbf_cols)

    print(f"📊 Microbiome feature dim = {micro_dim}")
    print(f"📊 Biomarker feature dim  = {biom_dim}")
    print(f"📊 Other feature dim      = {other_dim}")
    print(f"📊 CBF Imaging feature dim = {cbf_dim}")

    # 4) Build DataModule
    batch_size = 3
    dm = AlzheimerDataModule(data, mri_dict, batch_size=batch_size)
    dm.setup()

    # 5) Build Model (with 5 modalities: MRI, Microbiome, Biomarker, Other, CBF)
    embed_dim = 32
    model = MultiModalEmbeddingModule(
        micro_dim=micro_dim,
        biom_dim=biom_dim,
        other_dim=other_dim,
        cbf_dim=cbf_dim,
        embed_dim=embed_dim,
        lr=1e-4
    )

    # 6) Logger + Callbacks
    tb_logger = TensorBoardLogger("lightning_logs", name="alzheimer_multimodal")
    chart_logger = ChartLoggerCallback()
    early_stopping = EarlyStopping(
        monitor="train_loss",
        patience=3,
        mode="min"
    )

    # 7) Trainer
    trainer = pl.Trainer(
        max_epochs=20,
        logger=tb_logger,
        callbacks=[chart_logger, early_stopping],
        # accelerator="gpu", devices=1,  # Uncomment if you have a GPU
    )

    # 8) Fit Model
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
