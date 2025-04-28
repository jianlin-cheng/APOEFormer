#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import torch
import gzip, shutil

# ─── User‑editable settings ─────────────────────────────────────────────
mri_root    = "/home/tmnthc/T1"
mask_root   = "/home/tmnthc/mri"
output_root = "/home/tmnthc/result"
TIMEPOINTS  = ["Baseline", "Washout", "Post"]
# ────────────────────────────────────────────────────────────────────────
def load_mri_data(root_dir):
    """
    Returns dict[(pid, timepoint)] = {
      'tensor': FloatTensor [1, D, H, W],
      'affine':   4×4 numpy array,
      'header':   NIfTI header
    }
    Expects files in root_dir/pid/<timepoint>/*.nii(.gz)
    """
    mri_dict = {}

    for pid in sorted(os.listdir(root_dir)):
        pdir = os.path.join(root_dir, pid)
        if not os.path.isdir(pdir):
            continue

        print(f"Scanning patient: {pid}")
        for tp in TIMEPOINTS:
            tpdir = os.path.join(pdir, tp)
            if not os.path.isdir(tpdir):
                print(f"  ⚠️  Missing folder: {pid}/{tp}")
                continue

            # load every .nii / .nii.gz in that folder
            loaded = 0
            for fname in sorted(os.listdir(tpdir)):
                low = fname.lower()
                if not (low.endswith(".nii") or low.endswith(".nii.gz")):
                    continue

                path = os.path.join(tpdir, fname)
                print(f"    ✓ loading {pid}/{tp} ← {fname}")
                img = nib.load(path)
                arr = img.get_fdata()

                # drop singleton 4th dim
                if arr.ndim == 4 and arr.shape[3] == 1:
                    arr = arr[..., 0]
                # bring depth first if needed
                if arr.ndim == 3 and arr.shape[2] not in (arr.shape[0], arr.shape[1]):
                    arr = arr.transpose(2, 0, 1)

                # normalize
                m, s = arr.mean(), arr.std()
                arr = (arr - m) / (s if s > 0 else 1.0)

                tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
                # if there are multiple files, this will overwrite previous;
                # if you want to keep all, consider storing a list instead
                mri_dict[(pid, tp)] = {
                    "tensor": tensor,
                    "affine": img.affine,
                    "header": img.header
                }
                loaded += 1

            if loaded == 0:
                print(f"  ⚠️  No NIfTI files found in {pid}/{tp}")
            else:
                print(f"  → Loaded {loaded} file(s) for {pid}/{tp}")

    print(f"\n✔️ Loaded {len(mri_dict)} MRI volumes in total.\n")
    return mri_dict
def load_region_masks(mask_root):
    """
    Now expects mask files living in mask_root/<patient_id>/<timepoint>/*.nii(.gz)
    Returns a dict keyed by (pid, timepoint, region_name) → Boolean mask Tensor.
    """
    mask_dict = {}
    for pid in sorted(os.listdir(mask_root)):
        pdir = os.path.join(mask_root, pid)
        if not os.path.isdir(pdir):
            continue

        for tp in TIMEPOINTS:
            tpdir = os.path.join(pdir, tp)
            if not os.path.isdir(tpdir):
                print(f"⚠️  Missing masks folder for {pid}/{tp}")
                continue

            # first, decompress any .nii.gz in place
            for fname in sorted(os.listdir(tpdir)):
                if fname.endswith(".nii.gz"):
                    gz_path = os.path.join(tpdir, fname)
                    nii_name = fname[:-3]
                    nii_path = os.path.join(tpdir, nii_name)
                    with gzip.open(gz_path, "rb") as f_in, open(nii_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(gz_path)
                    print(f"⟳  Decompressed mask: {pid}/{tp}/{nii_name}")

            # now load all .nii files as masks
            for fname in sorted(os.listdir(tpdir)):
                if not fname.endswith(".nii"):
                    continue
                region = os.path.splitext(fname)[0]
                path   = os.path.join(tpdir, fname)
                arr    = nib.load(path).get_fdata()

                # if there's a singleton 4th dim
                if arr.ndim == 4 and arr.shape[3] == 1:
                    arr = arr[..., 0]
                # bring depth axis to front if needed
                if arr.ndim == 3 and arr.shape[2] not in (arr.shape[0], arr.shape[1]):
                    arr = arr.transpose(2, 0, 1)

                mask = torch.from_numpy((arr > 0).astype(bool))
                mask_dict[(pid, tp, region)] = mask
                print(f"✔️  Loaded mask `{region}` for {pid}/{tp}")

    print(f"\n✔️  Loaded {len(mask_dict)} total masks.\n")
    return mask_dict

def apply_region_mask(volume, mask):
    out = volume.clone()
    out[mask] = 0.0
    return out

def main():
    os.makedirs(output_root, exist_ok=True)
    print(f"📂 Ensured output root: {output_root}\n")

    mri_dict  = load_mri_data(mri_root)
    mask_dict = load_region_masks(mask_root)

    for (pid, tp), info in mri_dict.items():
        print(f"\n🔨 Processing {pid}/{tp}")
        orig_vol = info['tensor'].squeeze(0)
        affine   = info['affine']
        header   = info['header']

        regions = [r for (p, t, r) in mask_dict if p==pid and t==tp]
        out_dir = os.path.join(output_root, pid, tp)
        os.makedirs(out_dir, exist_ok=True)
        print(f"📂 Created/verified {out_dir}")

        if not regions:
            print(f"⚠️  No masks to apply for {pid}/{tp}.")
            continue

        for region in regions:
            mask       = mask_dict[(pid, tp, region)]
            masked_vol = apply_region_mask(orig_vol, mask)
            data_np    = masked_vol.numpy()
            out_fname  = f"{region}.nii"
            out_path   = os.path.join(out_dir, out_fname)
            nib.save(nib.Nifti1Image(data_np, affine, header), out_path)
            print(f"  • Saved masked `{region}` → {out_path}")

        print(f"✅ Done ∙ {len(regions)} masks applied for {pid}/{tp}")

if __name__ == "__main__":
    main()
