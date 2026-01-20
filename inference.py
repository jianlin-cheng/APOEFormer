import os
import random
import warnings
import argparse 
import numpy as np
import pandas as pd
import torch

from torch.cuda import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from data import *
from model import *
from train import *
from losses import FocalLoss, patient_contrastive_loss
from torch.cuda.amp import GradScaler, autocast

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
   torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
TIMEPOINTS = ["Baseline", "Post"]

def select_rows(data: pd.DataFrame, ids):
    """
    Keep same patient filtering/sorting logic, just moved out of main.
    """
    return (
        data[data.Patient_ID.isin(ids)]
        .sort_values(["Patient_ID", "Timepoint"])
        .groupby("Patient_ID")
        .head(3)
        .reset_index(drop=True)
    )

def parse_args():
    p = argparse.ArgumentParser(
        description="Brain-region prediction: load → pretrain → finetune → evaluate"
    )
    # REQUIRED (only thing you must provide)
    p.add_argument("data_path", help="Path to dataset root directory")

    # OPTIONAL overrides (everything has defaults)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--embed_dim",     type=int,   default=128)
    p.add_argument("--neg_frac",      type=int,   default=40)
    p.add_argument("--pos_repeat",    type=int,   default=1)

    p.add_argument("--lr_clip",       type=float, default=5e-3)
    p.add_argument("--lr_proj",       type=float, default=5e-3)
    p.add_argument("--lr_mlp",        type=float, default=5e-3)

    p.add_argument("--epochs_pre",    type=int,   default=200)
    p.add_argument("--patience_pre",  type=int,   default=3)
    p.add_argument("--epochs_attn",   type=int,   default=1000)
    p.add_argument("--patience_attn", type=int,   default=500)

    p.add_argument("--ensemble_size", type=int,   default=1)
    p.add_argument("--threshold",     type=float, default=0.5)
    p.add_argument("--num_layer",     type=int,   default=3)
    p.add_argument("--drop_out",      type=float, default=0.3)
    p.add_argument("--num_run",       type=int,   default=10)
    p.add_argument("--ce_weight",     type=float, default=0.05)

    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()

def main():
    args = parse_args()
    config = args

    device = torch.device(config.device)
    print(f"Selected device: {device}")

    scaler_pre = amp.GradScaler()

    if device.type == "cuda":
        print(f"  → GPU count: {torch.cuda.device_count()}")
        print(
            f"  → Using GPU #{torch.cuda.current_device()}: "
            f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
        )

    paths = build_paths(config.data_path)

    TIMEPOINTS = ["Baseline", "Post"]

    # -------------------------
    # Output accumulators
    # -------------------------
    all_runs = []
    test_accuracies = []

    # =========================
    # Checkpoints live inside data_path
    # =========================
    CKPT_DIR = os.path.join(config.data_path, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)
    CKPT_NAME_FMT = "full_run{run}.pt" 

    print(f"[INFO] Checkpoints will be saved to: {CKPT_DIR}")


    # -------------------------
    # RUN loop (unchanged structure)
    # -------------------------
    for run in range(1, config.num_run + 1):
        print(f"\n===== RUN {run}/{config.num_run} =====")

        # ---- Load tabular data (unchanged)
        data = load_data(base_path=config.data_path)  

        other_df = pd.read_csv(os.path.join(config.data_path, "Other.csv")) 

        other_df.rename(columns=lambda x: x.strip(), inplace=True)
        other_df.Patient_ID = other_df.Patient_ID.astype(str).str.strip()
        patient_apoe4 = dict(other_df.groupby("Patient_ID")["APOE4"].max())
        all_patients = list(patient_apoe4.keys())

        positive_patients = [p for p in all_patients if patient_apoe4[p] == 1]
        negative_patients = [p for p in all_patients if patient_apoe4[p] == 0]

        # ---- split logic (unchanged)
        random.seed(run)
        val_pos = random.sample(positive_patients, 1)
        val_neg = random.sample(negative_patients, 2)
        val_ids = val_pos + val_neg

        remaining_patients = [p for p in all_patients if p not in val_ids]
        trainval_ids, test_ids = train_test_split(
            remaining_patients,
            test_size=4,
            stratify=[patient_apoe4[p] for p in remaining_patients],
            random_state=run
        )
        train_ids = trainval_ids

        print(f"Train: {train_ids}, Val: {val_ids}, Test: {test_ids}")

        def select_rows(ids):
            return (data[data.Patient_ID.isin(ids)]
                    .sort_values(["Patient_ID", "Timepoint"])
                    .groupby("Patient_ID").head(3)
                    .reset_index(drop=True))

        train_data = select_rows(train_ids)
        val_data   = select_rows(val_ids)

        # -------------------------
        # Build model objects (same as original)
        # -------------------------
        dims = {
            "micro": train_data.filter(like="Microbiome_").shape[1],
            "bioA":  train_data.filter(like="Biomarker_A_").shape[1],
            "bioB":  train_data.filter(like="Biomarker_B_").shape[1],
            "bioC":  train_data.filter(like="Biomarker_C_").shape[1],
            "other": train_data.filter(like="Other_").shape[1],
            "num":   train_data.filter(like="Brain_CBF_Imaging_").shape[1],
        }

        model = MultiModalEmbeddingModel(
            dims["micro"], dims["bioA"], dims["bioB"], dims["bioC"],
            dims["other"], dims["num"],
            embed_dim=config.embed_dim,
            augment=True
        )

        # keep original as much as possible
        model = torch.compile(model)
        model = torch.nn.DataParallel(model).to(device)

        # -------------------------
        # Load MRI/X embeddings (same flow as your code)
        # -------------------------
        MRI_ROOT   = "/bmlfast/tom/T1"
        MRI_CACHE  = "/bmlfast/tom/mri_raw_cache"
        X_ROOT     = "/bmlfast/tom/Perfusion_images"
        X_CACHE    = "/bmlfast/tom/Perfusion_cache"

        print("Loading MRI raw volumes (with cache)…")
        mri_dict = load_mri_data(
            root_dir=MRI_ROOT,
            cache_dir=MRI_CACHE,
            device=device,
            allowed_timepoints=TIMEPOINTS
        )

        print("Loading X raw volumes (with cache)…")
        x_dict = load_mri_data(
            root_dir=X_ROOT,
            cache_dir=X_CACHE,
            device=device,
            allowed_timepoints=TIMEPOINTS
        )

        EMB_MRI_CACHE = "/bmlfast/tom/mri_embed_cache"
        EMB_X_CACHE   = "/bmlfast/tom/x_embed_cache"

        # If your caches already exist, you can skip precompute by commenting these out.
        mri_encoder = MRIClipEncoder(embed_dim=config.embed_dim, augment=False, dropout_p=config.drop_out).to(device)
        x_encoder   = MRIClipEncoder(embed_dim=config.embed_dim, augment=False, dropout_p=config.drop_out).to(device)
        mri_encoder.eval()
        x_encoder.eval()

        precompute_embeddings(mri_dict, mri_encoder, EMB_MRI_CACHE, device)

        precompute_embeddings(x_dict, x_encoder, EMB_X_CACHE, device)

        # Reload precomputed (as your code)
        mri_dict = load_precomputed(list(zip(data.Patient_ID, data.Timepoint)), EMB_MRI_CACHE)
        x_dict   = load_precomputed(list(zip(data.Patient_ID, data.Timepoint)), EMB_X_CACHE)

        # -------------------------
        # LOAD FULL CHECKPOINT and SKIP TRAINING
        # -------------------------
        ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME_FMT.format(run=run))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Make sure you copied full_run{run}.pt from Machine A to Machine B."
            )

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)


        # load stage-1 backbone weights
        model.module.load_state_dict(ckpt["backbone_state"], strict=True)

        # build + load stage-2 head
        best_head = StackedAttentionClassifier(
            embed_dim=config.embed_dim,
            num_heads=2,
            num_layers=config.num_layer,
            dropout=config.drop_out
        ).to(device)
        best_head.load_state_dict(ckpt["attn_state"], strict=True)

        model.eval()
        best_head.eval()
        ensemble_models = [best_head]

        # Optional: force identical splits used when saving
        train_ids = ckpt.get("train_ids", train_ids)
        val_ids   = ckpt.get("val_ids", val_ids)
        test_ids  = ckpt.get("test_ids", test_ids)

        print(f"[CKPT] Loaded {ckpt_path}. Skipping training; running evaluation only.")

        # -------------------------
        # Evaluation (same as your original structure)
        # -------------------------
        val_ds  = AttentionDatasetWithLabels(val_ids,   data, mri_dict, x_dict, model, best_head, device, patient_apoe4, TIMEPOINTS)
        test_ds = AttentionDatasetWithLabels(test_ids,  data, mri_dict, x_dict, model, best_head, device, patient_apoe4, TIMEPOINTS)
        full_ds = AttentionDatasetWithLabels(all_patients, data, mri_dict, x_dict, model, best_head, device, patient_apoe4, TIMEPOINTS)

        val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False, collate_fn=attn_collate_with_labels)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=attn_collate_with_labels)
        full_loader = DataLoader(full_ds, batch_size=64, shuffle=False, collate_fn=attn_collate_with_labels)


        # Per-run test evaluation
        total_correct_test = 0
        total_samples_test = 0
        run_records = []

        with torch.no_grad():
            for emb, lbl in test_loader:
                emb = emb.to(device).float()
                lbl = lbl.to(device).float()
                logits = sum(m(emb) for m in ensemble_models) / len(ensemble_models)
                probs  = torch.sigmoid(logits)
                preds  = (probs >= 0.5).float()

                total_correct_test += (preds == lbl).sum().item()
                total_samples_test += lbl.size(0)

                # NOTE: your original code prints by iterating test_ids, but that's risky if batching != len(test_ids).
                # To keep "as original as possible" but correct, we print batch-wise without assuming alignment.
                for i in range(lbl.size(0)):
                    pred_prob  = float(probs[i].item())
                    pred_logit = float(logits[i].item())
                    true_lbl   = float(lbl[i].item())
                    pred_lbl   = float(preds[i].item())

                    # PRINT to console
                    print(
                        f"[RUN {run}] "
                        f"Logit={pred_logit:.6f}  "
                        f"Prob={pred_prob:.6f}  "
                        f"Pred={int(pred_lbl)}  "
                        f"True={int(true_lbl)}"
                    )

                    # SAVE to dataframe
                    run_records.append({
                        "Run":            run,
                        "TrueLabel":      true_lbl,
                        "PredLogit":      pred_logit,
                        "PredProb":       pred_prob,
                        "PredictedLabel": pred_lbl,
                    })



        test_acc = total_correct_test / total_samples_test
        print(f"Test Accuracy: {test_acc*100:.2f}%")

        test_accuracies.append(test_acc)
        all_runs.append(pd.DataFrame(run_records))
    
    df_all = pd.concat(all_runs, ignore_index=True)

    avg_te, std_te = np.mean(test_accuracies)*100,    np.std(test_accuracies)*100
    print(f"Average Test    Accuracy: {avg_te:.2f}% (±{std_te:.2f}%)\n")



    print("Detailed test results by run:")
    for run_id, group in df_all.groupby("Run"):
        print(f"\n––– Run {run_id} –––")
        cols = ["TrueLabel", "PredProb", "PredictedLabel"]

        if "Patient_ID" in group.columns:
            cols = ["Patient_ID"] + cols
        print(group[cols].to_string(index=False))


   


if __name__ == "__main__":
    main()
