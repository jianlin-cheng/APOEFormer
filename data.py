import os
import math
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import torchvision.transforms as T
import random

def build_paths(base_path: str):
    paths = {
        "MRI_ROOT": os.path.join(base_path, "T1"),
        "MRI_CACHE": os.path.join(base_path, "mri_raw_cache"),

        "X_ROOT": os.path.join(base_path, "Perfusion_images"),
        "X_CACHE": os.path.join(base_path, "Perfusion_cache"),

        "EMB_MRI_CACHE": os.path.join(base_path, "mri_embed_cache"),
        "EMB_X_CACHE": os.path.join(base_path, "x_embed_cache"),
    }

    for k, v in paths.items():
        if k.endswith("_CACHE"):
            os.makedirs(v, exist_ok=True)

    return paths

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
    
def load_file(p, drop=True):
    try:
        d = pd.read_csv(p)
        d.rename(columns=lambda x: x.strip(), inplace=True)
        if drop and 'APOE4' in d.columns: d.drop(columns='APOE4', inplace=True)
        return d
    except: 
        return pd.DataFrame()

def load_data(base_path):
    def f(p, pref, drop=False):
        d = load_file(p, drop)
        for c in ('Patient_ID', 'Timepoint'):
            if c in d.columns:
                d[c] = d[c].astype(str).str.strip().str.title()
        for c in ('date', 'ID', 'Groups'):
            if c in d.columns:
                d.drop(columns=c, inplace=True)
        d = d[d.get('Timepoint', '') != 'Washout']

        k = ['Patient_ID', 'Timepoint']
        v = [c for c in d.columns if c not in k]
        for c in v:
            d[c] = pd.to_numeric(d[c], errors='coerce')

        s = [c for c in v if (';' in c) or c.startswith('d__')]
        if s and set(k).issubset(d.columns):
            d = d.sort_values(k)
            d[s] = d.groupby('Patient_ID', sort=False)[s].transform(lambda x: x.iloc[0])

        d[v] = d[v].fillna(0)
        d.rename(columns={c: f"{pref}_{c}" for c in v}, inplace=True)
        return d

    a = f(os.path.join(base_path, 'Micro.csv'), 'Microbiome')
    b = f(os.path.join(base_path, 'Blood_Metabolites.csv'), 'Biomarker_A')
    c = f(os.path.join(base_path, 'Sirolimus_inflammatory_markers.csv'), 'Biomarker_B')
    d = f(os.path.join(base_path, 'Sirolimus_Blood_Data.csv'), 'Biomarker_C')
    e = f(os.path.join(base_path, 'Other.csv'), 'Other', drop=True)
    g = f(os.path.join(base_path, 'Brain_CBF_Imaging.csv'), 'Brain_CBF_Imaging')

    xs = [x for x in (a, b, c, d, e, g) if not x.empty]

    z = (
        pd.concat([x[['Patient_ID', 'Timepoint']] for x in xs], ignore_index=True)
        .drop_duplicates()
        if xs else pd.DataFrame(columns=['Patient_ID', 'Timepoint'])
    )

    for x in (a, b, c, d, e, g):
        if not x.empty:
            z = z.merge(x, on=['Patient_ID', 'Timepoint'], how='outer')

    z.fillna(0, inplace=True)
    return z


def load_mri_data(
    root_dir,
    cache_dir=None,
    device="cpu",
    allowed_timepoints=("Baseline", "Post")
    ):

    mri_dict = {}
    total_cached = total_computed = 0


    if cache_dir and os.path.isdir(cache_dir):
        for fn in os.listdir(cache_dir):
            if not fn.endswith(".pt"):
                continue
            name = fn[:-3] 
            pid, tp = name.split("_", 1)
            if tp not in allowed_timepoints:
                continue
            path = os.path.join(cache_dir, fn)
            try:
                emb = torch.load(path, map_location="cpu")  
                mri_dict[(pid, tp)] = emb
                total_cached += 1
            except Exception as e:
                print(f"⚠️ Failed to load cache {path}: {e}")

   

    for pid in os.listdir(root_dir):
        pid_path = os.path.join(root_dir, pid)
        if not os.path.isdir(pid_path):
            continue

        for tp in os.listdir(pid_path):
            if tp not in allowed_timepoints:
                continue

            key = (pid, tp)
            if key in mri_dict:
                continue

            tp_path = os.path.join(pid_path, tp)
            if not os.path.isdir(tp_path):
                continue

            nii_fn = next(
                (f for f in os.listdir(tp_path)
                    if f.endswith(".nii") or f.endswith(".nii.gz")),
                None
            )
            if not nii_fn:
                print(f"  [WARN] No NIfTI in {tp_path}, skipping.")
                continue

            full_nii = os.path.join(tp_path, nii_fn)
            arr = nib.load(full_nii).get_fdata()

            if arr.ndim == 4 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)

            mean, std = arr.mean(), arr.std()
            norm = (arr - mean) / (std + 1e-8)
            tensor = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0)


            mri_dict[key] = tensor
            total_computed += 1

            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(cache_dir, f"{pid}_{tp}.pt")
                torch.save(tensor.cpu(), cache_path)

    return mri_dict

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

def augment_numeric(x, noise_std=0.1):
    noise = torch.randn_like(x) * noise_std
    return x + noise

class CombinedContrastiveDataset(Dataset):
    def __init__(
        self,
        data,
        mri_dict,
        x_dict,
        negative_sample_fraction=1,
        positive_repeat=1,
        augment=False
    ):
        self.data     = data.reset_index(drop=True)
        self.mri_dict = mri_dict
        self.x_dict   = x_dict
        self.N        = len(self.data)
        self.augment  = augment


        M = 8  

        self.positive_samples = [
            (i,)*M
            for i in range(self.N)
            for _ in range(positive_repeat)
        ]

        num_neg = int(self.N * negative_sample_fraction)
        negative_samples = []
        for _ in range(num_neg):
            anchor = random.randrange(self.N)
            slot   = random.randrange(M)
            idxs   = [anchor]*M
            choices = list(range(self.N))
            choices.remove(anchor)
            idxs[slot] = random.choice(choices)
            negative_samples.append(tuple(idxs))

        self.samples = [(s,1) for s in self.positive_samples] + \
                        [(s,0) for s in negative_samples]
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (i_mri, i_x, i_micro, i_bioA, i_bioB, i_bioC, i_other, i_num), label = self.samples[idx]

        def get_pid_tp(i):
           row = self.data.loc[i, ["Patient_ID","Timepoint"]]
           return str(row["Patient_ID"]), str(row["Timepoint"])
        
        pid_mri, tp_mri   = get_pid_tp(i_mri)
        pid_x,   tp_x     = get_pid_tp(i_x)
        pid_micro, tp_micro = get_pid_tp(i_micro)
        pid_bioA,  tp_bioA = get_pid_tp(i_bioA)
        pid_bioB,  tp_bioB = get_pid_tp(i_bioB)
        pid_bioC,  tp_bioC = get_pid_tp(i_bioC)
        pid_other, tp_other = get_pid_tp(i_other)
        pid_num,   tp_num   = get_pid_tp(i_num)

        mri_tensor = self.mri_dict.get((pid_mri, tp_mri),
                                        torch.zeros((1,128,128,128), dtype=torch.float32))
        x_tensor   = self.x_dict.get((pid_x, tp_x),
                                        torch.zeros((1,128,128,128), dtype=torch.float32))

        micro_tensor = torch.tensor(
            self.data.filter(like="Microbiome_").iloc[i_micro].values.astype(np.float32)
        )
        bioA_tensor = torch.tensor(
            self.data.filter(like="Biomarker_A_").iloc[i_bioA].values.astype(np.float32)
        )
        bioB_tensor = torch.tensor(
            self.data.filter(like="Biomarker_B_").iloc[i_bioB].values.astype(np.float32)
        )
        bioC_tensor = torch.tensor(
            self.data.filter(like="Biomarker_C_").iloc[i_bioC].values.astype(np.float32)
        )
        other_tensor = torch.tensor(
            self.data.filter(like="Other_").iloc[i_other].values.astype(np.float32)
        )
        num_tensor = torch.tensor(
            self.data.filter(like="Brain_CBF_Imaging_").iloc[i_num].values.astype(np.float32)
        )

        if self.augment and label == 1:
            micro_tensor = augment_numeric(micro_tensor)
            bioA_tensor  = augment_numeric(bioA_tensor)
            bioB_tensor  = augment_numeric(bioB_tensor)
            bioC_tensor  = augment_numeric(bioC_tensor)
            other_tensor = augment_numeric(other_tensor)
            num_tensor   = augment_numeric(num_tensor)

        return {
            "mri":    mri_tensor,
            "x_img":  x_tensor,
            "micro":  micro_tensor,
            "biomarker_A": bioA_tensor,
            "biomarker_B": bioB_tensor,
            "biomarker_C": bioC_tensor,
            "other":  other_tensor,
            "numeric":num_tensor,
            "label":  torch.tensor(label, dtype=torch.float32),
        }

class AttentionDatasetWithLabels(Dataset):
    def __init__(
       self,
       patient_ids,
       data,
       mri_dict,
       x_dict,
       emb_model,
       attn_model,
       device,
       patient_labels,
       timepoints
   ):
        super().__init__()
        self.device     = device
        self.emb_model  = emb_model.to(device).eval()
        self.attn_model = attn_model.to(device).eval()

        base = emb_model.module if hasattr(emb_model, "module") else emb_model
        self.samples = []

        for pid in patient_ids:
            seqs = []
            for tp in timepoints:
                row = data[(data.Patient_ID == pid) & (data.Timepoint == tp)].iloc[0]

                img_mri = mri_dict[(pid, tp)].unsqueeze(0).to(device)  # (1, D_mri, E)
                img_x   = x_dict[(pid, tp)].unsqueeze(0).to(device)    # (1, D_x,   E)


                def to_tensor(pref):
                    vals = row.filter(like=pref).astype(float).fillna(0).values
                    return torch.from_numpy(vals.astype("float32")).unsqueeze(0).to(device)

                micro = to_tensor("Microbiome_")              # (1, E)
                bioA  = to_tensor("Biomarker_A_")             # (1, E)
                bioB  = to_tensor("Biomarker_B_")             # (1, E)
                bioC  = to_tensor("Biomarker_C_")             # (1, E)
                other = to_tensor("Other_")                   # (1, E)
                num   = to_tensor("Brain_CBF_Imaging_")       # (1, E)

                with torch.no_grad():
                    e_mri, e_x, e_micro, e_bioA, e_bioB, e_bioC, e_other, e_num = \
                        self.emb_model(
                            mri=img_mri,
                            x_img=img_x,
                            micro=micro,
                            biomarker_A=bioA,
                            biomarker_B=bioB,
                            biomarker_C=bioC,
                            other=other,
                            numeric=num
                        )

                _, D_mri, E = e_mri.shape
                _, D_x, _   = e_x.shape
                if D_x < D_mri:
                    pad = torch.zeros((1, D_mri - D_x, E), device=e_x.device)
                    e_x = torch.cat([e_x, pad], dim=1)
                elif D_x > D_mri:
                    e_x = e_x[:, :D_mri, :]

                def tile(x):
                    return x.unsqueeze(1).repeat(1, D_mri, 1)

                seq = torch.cat([
                    e_mri,
                    e_x,
                    tile(e_micro),
                    tile(e_bioA),
                    tile(e_bioB),
                    tile(e_bioC),
                    tile(e_other),
                    tile(e_num),
                ], dim=1) 


                seqs.append(seq.squeeze(0))  


            full_seq = torch.cat(seqs, dim=0) 
            lbl = torch.tensor(patient_labels[pid], dtype=torch.float32)
            self.samples.append((full_seq, lbl))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        return self.samples[idx]

def custom_collate(batch):
    collated = {}
    collated["label"] = torch.stack([d["label"] for d in batch])
    for key in ("micro","biomarker_A","biomarker_B","biomarker_C","other","numeric"):
        collated[key] = torch.stack([d[key] for d in batch])
    collated["mri"]   = [d["mri"]   for d in batch]  
    collated["x_img"] = [d["x_img"] for d in batch] 
    return collated

def attn_collate_with_labels(batch):

    embeddings_list, label_list = zip(*batch)
    embeddings = torch.stack(embeddings_list, dim=0)  
    labels = torch.stack(label_list, dim=0)         
    return embeddings, labels
