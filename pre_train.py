import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def load_data():
    # Load data files
    microbiome = pd.read_csv('/Users/thongnguyen/Downloads/New/Microbiome.csv')
    blood_metabolites = pd.read_csv('/Users/thongnguyen/Downloads/New/Blood_Metabolites.csv')
    inflammatory_markers = pd.read_csv('/Users/thongnguyen/Downloads/New/Sirolimus_inflammatory_markers.csv')
    blood_data = pd.read_csv('/Users/thongnguyen/Downloads/New/Sirolimus_Blood_Data.csv')
    other_data = pd.read_csv('/Users/thongnguyen/Downloads/New/Other.csv')

    # Drop the 'APOE4' column from other_data, if it exists
    if 'APOE4' in other_data.columns:
        other_data = other_data.drop(columns=['APOE4'])

    # Standardize and clean column names
    for df in [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data]:
        df.rename(columns=lambda x: x.strip(), inplace=True)

    # Ensure 'Patient_ID' and 'Timepoint' columns are strings for consistent merging
    for df in [microbiome, blood_metabolites, inflammatory_markers, blood_data, other_data]:
        if 'Patient_ID' in df.columns:
            df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
        if 'Timepoint' in df.columns:
            df['Timepoint'] = df['Timepoint'].astype(str).str.strip()

    # Retain only numeric columns
    def filter_numeric(df):
        return df.select_dtypes(include=['number']).copy()

    microbiome_numeric = filter_numeric(microbiome)
    blood_metabolites_numeric = filter_numeric(blood_metabolites)
    inflammatory_markers_numeric = filter_numeric(inflammatory_markers)
    blood_data_numeric = filter_numeric(blood_data)

    # Add back 'Patient_ID' and 'Timepoint' for merging
    microbiome_numeric[['Patient_ID', 'Timepoint']] = microbiome[['Patient_ID', 'Timepoint']]
    blood_metabolites_numeric[['Patient_ID', 'Timepoint']] = blood_metabolites[['Patient_ID', 'Timepoint']]
    inflammatory_markers_numeric[['Patient_ID', 'Timepoint']] = inflammatory_markers[['Patient_ID', 'Timepoint']]
    blood_data_numeric[['Patient_ID', 'Timepoint']] = blood_data[['Patient_ID', 'Timepoint']]

    # Merge datasets on Patient_ID and Timepoint
    data = other_data.merge(microbiome_numeric, on=['Patient_ID', 'Timepoint'], how='inner')
    data = data.merge(blood_metabolites_numeric, on=['Patient_ID', 'Timepoint'], how='inner')
    data = data.merge(inflammatory_markers_numeric, on=['Patient_ID', 'Timepoint'], how='inner')
    data = data.merge(blood_data_numeric, on=['Patient_ID', 'Timepoint'], how='inner')

    # Handle missing values
    data = data.fillna(0)

    return data

# Dataset class
class PatientDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.patient_ids = torch.tensor(data['Patient_ID'].astype('category').cat.codes.values, dtype=torch.long)
        self.timepoints = torch.tensor(data['Timepoint'].astype('category').cat.codes.values, dtype=torch.long)
        self.microbiome_data = data.filter(like='Microbiome_').values
        self.biomarker_data = data.filter(like='Biomarker_').values
        self.other_data = data.filter(like='Other_').values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.microbiome_data[idx], dtype=torch.float32),
            torch.tensor(self.biomarker_data[idx], dtype=torch.float32),
            torch.tensor(self.other_data[idx], dtype=torch.float32),
            self.patient_ids[idx],
            self.timepoints[idx]
        )

# Encoders
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        x = self.encoder(x)
        return F.normalize(x, p=2, dim=-1)  # L2 normalization

def forward_pass(microbiome_data, biomarker_data, other_data, microbiome_encoder, biomarker_encoder, other_encoder):
    # Encode all data types
    e_microbiome = microbiome_encoder(microbiome_data)
    e_biomarker = biomarker_encoder(biomarker_data)
    e_other = other_encoder(other_data)

    # Normalize all embeddings
    e_microbiome = F.normalize(e_microbiome, p=2, dim=-1)
    e_biomarker = F.normalize(e_biomarker, p=2, dim=-1)
    e_other = F.normalize(e_other, p=2, dim=-1)

    return e_microbiome, e_biomarker, e_other

def patient_contrastive_loss(e_microbiome, e_biomarker, e_other, patient_ids, tau=0.1):
    """
    Computes the patient contrastive loss with multiple embeddings.

    Parameters:
    - e_microbiome, e_biomarker, e_other: Encoded representations of different data modalities.
    - patient_ids: Tensor of patient IDs.
    - tau: Temperature scaling factor.

    Returns:
    - Mean contrastive loss for the batch.
    """
    # Concatenate all embeddings into a single matrix
    embeddings = torch.stack([e_microbiome, e_biomarker, e_other], dim=1)  # Shape: (batch_size, 3, embed_dim)

    # Compute pairwise similarities
    sim_matrix = torch.einsum('bmd,bnd->bmn', embeddings, embeddings)  # Shape: (batch_size, 3, 3)
    sim_matrix = sim_matrix / tau  # Apply temperature scaling

    # Create a mask to identify positive pairs (same patient)
    patient_mask = patient_ids[:, None] == patient_ids[None, :]  # Shape: (batch_size, batch_size)

    # Expand the patient_mask to align with sim_matrix
    patient_mask = patient_mask.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3, -1)  # Shape: (batch_size, 3, 3, batch_size)

    # Expand sim_matrix for broadcasting
    sim_matrix = sim_matrix.unsqueeze(-1)  # Shape: (batch_size, 3, 3, 1)

    # Compute positive and total pairs
    positive_pairs = sim_matrix * patient_mask.float()  # Shape: (batch_size, 3, 3, batch_size)
    total_pairs = sim_matrix  # Shape: (batch_size, 3, 3, batch_size)

    # Sum over relevant dimensions
    positive_sum = torch.sum(positive_pairs, dim=(-1, 2))  # Shape: (batch_size, 3)
    total_sum = torch.sum(total_pairs, dim=(-1, 2))  # Shape: (batch_size, 3)

    # Compute contrastive loss
    loss = -torch.log(torch.sum(positive_sum) / torch.sum(sim_matrix))
    return torch.mean(loss)


def train_model(data, patience=10, batch_size=3, output_dim=32):
    """
    Trains the model with early stopping.
    
    Parameters:
    - data: Merged dataset.
    - patience: Number of epochs to wait for improvement before stopping.
    - batch_size: Batch size for training.
    - output_dim: Output dimension of the encoders.
    """
    dataset = PatientDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    microbiome_encoder = MLPEncoder(input_dim=data.filter(like='Microbiome_').shape[1], output_dim=output_dim)
    biomarker_encoder = MLPEncoder(input_dim=data.filter(like='Biomarker_').shape[1], output_dim=output_dim)
    other_encoder = MLPEncoder(input_dim=data.filter(like='Other_').shape[1], output_dim=output_dim)

    optimizer = torch.optim.Adam(
        list(microbiome_encoder.parameters()) +
        list(biomarker_encoder.parameters()) +
        list(other_encoder.parameters()), lr=1e-4
    )

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, 1001):  # Arbitrary high number; will stop early if needed
        epoch_loss = 0
        for batch in dataloader:
            microbiome_data, biomarker_data, other_data, patient_ids, timepoints = batch

            # Forward pass
            e_microbiome, e_biomarker, e_other = forward_pass(
                microbiome_data, biomarker_data, other_data,
                microbiome_encoder, biomarker_encoder, other_encoder
            )

            # Compute loss
            loss = patient_contrastive_loss(e_microbiome, e_biomarker, e_other, patient_ids)
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute average epoch loss
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0  # Reset counter if improvement
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch}. Best loss: {best_loss:.4f}")
            break


# Load data and train the model
data = load_data()
train_model(data, patience=5, batch_size=16, output_dim=32)

