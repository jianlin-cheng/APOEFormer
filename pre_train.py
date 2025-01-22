import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define MLP Encoder using PyTorch
class MLPEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

# Define Linear Projection Layer using PyTorch
class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProjection, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# NT-Xent Contrastive Loss Function
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # Normalize the representations
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate positive pairs
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        sim = torch.mm(z, z.T) / self.temperature
        sim_exp = torch.exp(sim)

        # Debug: Print similarity matrix
        print("Similarity matrix:", sim.cpu().detach().numpy())

        # Create mask to exclude self-similarities
        batch_size = z_i.size(0)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)

        # Compute NT-Xent Loss
        positive_sim = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)])
        positive_sim = torch.exp(positive_sim / self.temperature)

        # Debug: Print positive similarities
        print("Positive similarities:", positive_sim.cpu().detach().numpy())

        denominator = sim_exp.sum(dim=1) - sim_exp.diagonal()

        # Debug: Print denominator values
        print("Denominator values:", denominator.cpu().detach().numpy())

        loss = -torch.log(positive_sim / denominator)

        # Debug: Print individual loss values
        print("Individual loss values:", loss.cpu().detach().numpy())

        return loss.mean()

# Function to load and preprocess .nii MRI images
def preprocess_mri_images(root_dir):
    """
    Load and preprocess .nii MRI images organized in folders by patient ID.
    Args:
        root_dir (str): Path to the root directory containing patient folders.
    Returns:
        dict: A dictionary where keys are patient IDs and values are preprocessed image tensors.
    """
    patient_data = {}
    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)
        if os.path.isdir(patient_path):  # Ensure it's a directory
            images = {}
            for file_name in os.listdir(patient_path):
                file_path = os.path.join(patient_path, file_name)
                if file_name.lower().endswith('.nii'):
                    # Identify timepoint from file name
                    if "base" in file_name.lower():
                        timepoint = "Base"
                    elif "post" in file_name.lower():
                        timepoint = "Post"
                    elif "washout" in file_name.lower():
                        timepoint = "Washout"
                    else:
                        continue  # Skip files without a known timepoint

                    # Load and preprocess the .nii file
                    nii_image = nib.load(file_path)
                    image_data = nii_image.get_fdata()  # Extract voxel data
                    image_data = np.nan_to_num(image_data)  # Replace NaNs with zeros
                    image_data = (image_data - image_data.mean()) / image_data.std()  # Normalize voxel values
                    image_data = torch.tensor(image_data, dtype=torch.float32)  # Convert to tensor
                    image_data = image_data.unsqueeze(0)  # Add channel dimension (e.g., for CNNs)

                    images[timepoint] = image_data

            # Store the processed images for the patient
            if len(images) == 3:  # Ensure all three timepoints are present
                patient_data[patient_folder] = images
    return patient_data

# Function to preprocess other data files
def preprocess_data_custom(file_path, file_name):
    df = pd.read_csv(file_path)
    timepoints = ["Base", "Post", "Washout"]
    if "Timepoint" in df.columns:
        df = df[df["Timepoint"].isin(timepoints)]
    else:
        raise ValueError(f"The file '{file_name}' does not contain a 'Timepoint' column.")
    drop_columns = ["date", "ID", "sample", "Subject ID Full", "Subject ID", "Unnamed: 0"]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")
    for col in df.select_dtypes(include=["object"]).columns:
        if col not in ["Timepoint", "Patient_ID"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    if "Patient_ID" in df.columns and "Timepoint" in df.columns:
        df_pivot = df.pivot(index="Patient_ID", columns="Timepoint")
        df_pivot.columns = ["_".join(col).strip() for col in df_pivot.columns.values]
        df_pivot = df_pivot.dropna()
    else:
        raise ValueError(f"The file '{file_name}' is missing required columns 'Patient_ID' or 'Timepoint'.")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_pivot)
    return torch.tensor(data_scaled, dtype=torch.float32)

# Main script
if __name__ == "__main__":
    # Set root directory for MRI images
    root_directory = "/Users/thongnguyen/Downloads/CBF_imaging"  # Update this to your actual root directory containing patient folders

    # Load and preprocess images
    patient_images = preprocess_mri_images(root_directory)

    # Process other data files
    file_paths = {
        "Sirolimus_Blood_Data": "/Users/thongnguyen/Downloads/New/Sirolimus_Blood_Data.csv",
        "Microbiome": "/Users/thongnguyen/Downloads/New/Microbiome.csv",
        "Blood_Metabolites": "/Users/thongnguyen/Downloads/New/Blood_Metabolites.csv",
        "Brain_CBF_Imaging": "/Users/thongnguyen/Downloads/New/Brain_CBF_Imaging.csv",
        "Other": "/Users/thongnguyen/Downloads/New/Other.csv",
        "Sirolimus_Inflammatory_Markers": "/Users/thongnguyen/Downloads/New/Sirolimus_Inflammatory_Markers.csv",
    }

    models = {}
    for name, path in file_paths.items():
        try:
            data_tensor = preprocess_data_custom(path, name)
            input_dim = data_tensor.shape[1]
            hidden_dim = 64
            encoded_dim = 32
            output_dim = 10

            mlp_model = MLPEncoder(input_dim, hidden_dim, encoded_dim)
            projection_layer = LinearProjection(encoded_dim, output_dim)
            contrastive_loss = NTXentLoss(temperature=0.5)

            # Generate augmented views
            aug_view1 = data_tensor + torch.randn_like(data_tensor) * 0.1
            aug_view2 = data_tensor + torch.randn_like(data_tensor) * 0.1

            # Pass through encoder and projection
            encoded1 = mlp_model(aug_view1)
            encoded2 = mlp_model(aug_view2)
            projected1 = projection_layer(encoded1)
            projected2 = projection_layer(encoded2)

            # Compute contrastive loss
            loss = contrastive_loss(projected1, projected2)
            print(f"Contrastive Loss for {name}: {loss.item()}")

            models[name] = {"mlp": mlp_model, "projection": projection_layer}
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Process MRI images through MLP and projection in batches
    batch_size = 16
    patient_ids = list(patient_images.keys())

    for i in range(0, len(patient_ids), batch_size):
        batch_patients = patient_ids[i:i + batch_size]
        batch_data = []

        for patient_id in batch_patients:
            images = patient_images[patient_id]
            combined_image = torch.cat([images["Base"], images["Post"], images["Washout"]], dim=0)
            flattened_image = combined_image.view(1, -1)  # Flatten for MLP input
            batch_data.append(flattened_image)

        batch_data = torch.cat(batch_data, dim=0)  # Create a batch

        try:
            input_dim = batch_data.size(1)
            hidden_dim = 64
            encoded_dim = 32
            output_dim = 10

            mlp_model = MLPEncoder(input_dim, hidden_dim, encoded_dim)
            projection_layer = LinearProjection(encoded_dim, output_dim)
            contrastive_loss = NTXentLoss(temperature=0.5)

            # Generate augmented views
            aug_view1 = batch_data + torch.randn_like(batch_data) * 0.1
            aug_view2 = batch_data + torch.randn_like(batch_data) * 0.1

            # Pass through encoder and projection
            encoded1 = mlp_model(aug_view1)
            encoded2 = mlp_model(aug_view2)
            projected1 = projection_layer(encoded1)
            projected2 = projection_layer(encoded2)

            # Compute contrastive loss
            loss = contrastive_loss(projected1, projected2)
            print(f"Batch {i // batch_size + 1}: Contrastive Loss = {loss.item()}")

        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")

