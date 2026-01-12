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

    all_runs = []
    overall_accuracies = []
    test_accuracies = []
    shap_records = {m: [] for m in [
        "MRI", "X", "Microbiome", "Biomarker_A", "Biomarker_B",
        "Biomarker_C", "Other", "Numeric"
    ]}

    for run in range(1, config.num_run + 1):
        print(f"\n===== RUN {run}/{config.num_run} =====")

        data = load_data(base_path=config.data_path)  

        other_df = pd.read_csv(os.path.join(config.data_path, "Other.csv")) 

        other_df.rename(columns=lambda x: x.strip(), inplace=True)
        other_df.Patient_ID = other_df.Patient_ID.astype(str).str.strip()
        patient_apoe4 = dict(other_df.groupby("Patient_ID")["APOE4"].max())
        all_patients = list(patient_apoe4.keys())

        positive_patients = [p for p in all_patients if patient_apoe4[p] == 1]
        negative_patients = [p for p in all_patients if patient_apoe4[p] == 0]

        random.seed(run)  

        val_pos = random.sample(positive_patients, 1)
        val_neg = random.sample(negative_patients, 2)
        val_ids = val_pos + val_neg

        remaining_patients = [p for p in all_patients if p not in val_ids]

        trainval_ids, test_ids = train_test_split(
            remaining_patients, test_size=4,
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
        val_data = select_rows(val_ids)

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
        model = torch.compile(model)


        model = torch.nn.DataParallel(model).to(device)

        MRI_ROOT       = paths["MRI_ROOT"]
        MRI_CACHE      = paths["MRI_CACHE"]
        X_ROOT         = paths["X_ROOT"]
        X_CACHE        = paths["X_CACHE"]
        EMB_MRI_CACHE  = paths["EMB_MRI_CACHE"]
        EMB_X_CACHE    = paths["EMB_X_CACHE"]

        mri_dict = load_mri_data(
            root_dir=MRI_ROOT,
            cache_dir=MRI_CACHE,
            device=device,
            allowed_timepoints=TIMEPOINTS
        )

        x_dict = load_mri_data(
            root_dir=X_ROOT,
            cache_dir=X_CACHE,
            device=device,
            allowed_timepoints=TIMEPOINTS
        )

        mri_encoder = MRIClipEncoder(embed_dim=config.embed_dim, augment=False, dropout_p=config.drop_out).to(device)
        x_encoder   = MRIClipEncoder(embed_dim=config.embed_dim, augment=False, dropout_p=config.drop_out).to(device)
        mri_encoder.eval()
        x_encoder.eval()

        precompute_embeddings(mri_dict, mri_encoder, EMB_MRI_CACHE, device)
        precompute_embeddings(x_dict, x_encoder, EMB_X_CACHE, device)

        mri_dict = load_precomputed(list(zip(data.Patient_ID, data.Timepoint)), EMB_MRI_CACHE)
        x_dict   = load_precomputed(list(zip(data.Patient_ID, data.Timepoint)), EMB_X_CACHE)
        param_groups = []

        for enc, lr_clip, lr_proj in [
            ("mri_encoder", config.lr_clip, config.lr_proj),
            ("x_encoder",  config.lr_clip, config.lr_proj),
        ]:
            param_groups.append({
                "params": getattr(model.module, enc).clip_model.vision_model.encoder.parameters(),
                "lr":      lr_clip
            })
            param_groups.append({
                "params": getattr(model.module, enc).project.parameters(),
                "lr":      lr_proj
            })

        for name in ("micro_encoder","bioA_encoder","bioB_encoder",
                    "bioC_encoder","other_encoder","num_encoder"):
            param_groups.append({
                "params": getattr(model.module, name).parameters(),
                "lr":      config.lr_mlp
            })

        param_groups.append({
            "params": model.module.probe_head.parameters(),
            "lr":      config.lr_mlp
        })

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=4
        )

        scheduler_pre = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
        )

        train_ds = CombinedContrastiveDataset(
            train_data, mri_dict, x_dict,
            negative_sample_fraction=config.neg_frac,
            positive_repeat=config.pos_repeat,
            augment=True
        )
        val_ds = CombinedContrastiveDataset(
            val_data, mri_dict, x_dict,
            negative_sample_fraction=config.neg_frac,
            positive_repeat=config.pos_repeat,
            augment=False
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,          
            pin_memory=True,
            prefetch_factor=2,      
            persistent_workers=True,  
            collate_fn=custom_collate
        )

        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size,
            shuffle=False, num_workers=8, pin_memory=True,
            collate_fn=custom_collate
        )

        best_val_loss = float('inf')
        val_history = []
        for pre_ep in range(1, config.epochs_pre+1):
            tr_loss = train_epoch(
                model, train_loader, optimizer, device,
                pre_ep, val_history,
                config.ce_weight,          
                patience_threshold=config.patience_pre,
                accum_steps=4,
                scaler=scaler_pre
            )

            val_loss = validate_epoch(
                model, val_loader, device,
                config.ce_weight           
            )

            val_history.append(val_loss)
            scheduler_pre.step(val_loss)

            print(f"[Pretrain] Epoch {pre_ep}/{config.epochs_pre}  "f"Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = model.state_dict()
            if len(val_history) >= config.patience_pre and val_loss >= max(val_history[-config.patience_pre:]):
                print(f"Early stopping pretrain at epoch {pre_ep}")
                break


        if best_val_loss < float('inf'):
            model.load_state_dict(best_state)

        ensemble_models = []
        for ens in range(config.ensemble_size):
            attn_model = StackedAttentionClassifier(
                embed_dim=config.embed_dim,
                num_heads=2,
                num_layers=config.num_layer,
                dropout=config.drop_out
            ).to(device)

            train_attn_ds = AttentionDatasetWithLabels(
                train_ids, data, mri_dict, x_dict,
                model, attn_model, device,
                patient_apoe4, TIMEPOINTS
            )
            val_attn_ds = AttentionDatasetWithLabels(
                val_ids, data, mri_dict, x_dict,
                model, attn_model, device,
                patient_apoe4, TIMEPOINTS
            )
            test_attn_ds = AttentionDatasetWithLabels(
                test_ids, data, mri_dict, x_dict,
                model, attn_model, device,
                patient_apoe4, TIMEPOINTS
            )

            labels    = [patient_apoe4[pid] for pid in train_ids]
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count

            punish_factor = 1.0  
            pos_weight = torch.tensor([punish_factor * (neg_count / pos_count)], dtype=torch.float32).to(device)

            criterion_attn = FocalLoss(
                init_alpha_pos=0.9,
                init_alpha_neg=0.1,
                gamma=7.0,
                reduction="mean"
            ).to(device)

            weights = [pos_weight.item() if patient_apoe4[pid] else 1.0
                    for pid in train_ids]
            sampler = WeightedRandomSampler(weights,
                                            num_samples=len(weights),
                                            replacement=True)


            train_attn_loader = DataLoader(
                train_attn_ds, batch_size=64, shuffle=True,
                collate_fn=attn_collate_with_labels
            )
            val_attn_loader = DataLoader(
                val_attn_ds, batch_size=64, shuffle=True,
                collate_fn=attn_collate_with_labels
            )
            test_attn_loader = DataLoader(
                test_attn_ds, batch_size=64, shuffle=False,
                collate_fn=attn_collate_with_labels
            )

            optimizer_attn = torch.optim.AdamW(
                [
                    {"params": attn_model.parameters(),      "lr": 1e-4},
                    
                ],
                weight_decay=5e-3
            )

            best_head_val = float('inf')
            no_improve    = 0
            for epoch in range(1, config.epochs_attn+1):

                attn_model.train()
            
                emb0, lbl0 = next(iter(train_attn_loader))

                noise_std = 0.2
                emb0 = emb0.to(device)
                emb0 = emb0 + torch.randn_like(emb0) * noise_std
                lbl0 = lbl0.to(device)

                with torch.no_grad():
                    logits0 = attn_model(emb0.to(device))
                    loss0 = criterion_attn(logits0, lbl0.to(device))
                scaler_attn = amp.GradScaler()
                tr_sum = 0.0
                for emb, lbl in train_attn_loader:
                    emb, lbl = emb.to(device), lbl.to(device)
                    optimizer_attn.zero_grad()
                    emb = emb + torch.randn_like(emb) * noise_std

                    with autocast():
                        logits = attn_model(emb)
                        probs = torch.sigmoid(logits)                      
                        loss   = criterion_attn(logits, lbl)
                        preds  = (probs >= 0.5).float()
                        train_acc = (preds == lbl).float().mean().item()                    
                    scaler_attn.scale(loss).backward()
                    scaler_attn.unscale_(optimizer_attn)
                    torch.nn.utils.clip_grad_norm_(attn_model.parameters(), 1.0)
                    scaler_attn.step(optimizer_attn)
                    scaler_attn.update()
                    tr_sum += loss.item() * emb.size(0)

                tr_loss = tr_sum / len(train_attn_ds)

                attn_model.eval()
                val_sum = 0.0
                with torch.no_grad():
                    for emb, lbl in val_attn_loader:
                        emb, lbl = emb.to(device), lbl.to(device)
                        logits = attn_model(emb)
                        val_sum += criterion_attn(logits, lbl).item()*emb.size(0)
                val_loss = val_sum / len(val_attn_ds)
                if ens == 0:
                    if epoch == 1 or epoch % 5 == 0:
                        print(
                            f"[Ensemble {ens+1}] Epoch {epoch}/{config.epochs_attn}  "
                            f"Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}"
                        )

                if val_loss < best_head_val:
                    best_head_val = val_loss
                    best_head_state = attn_model.state_dict()
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= config.patience_attn:
                    print(f"Early stopping ensemble {ens+1} at epoch {epoch}")
                    break

            if best_head_val < float('inf'):
                attn_model.load_state_dict(best_head_state)
            ensemble_models.append(attn_model)


        if not ensemble_models:
            raise RuntimeError("No attention heads were trained!")
        best_head = ensemble_models[0]

        val_ds   = AttentionDatasetWithLabels(val_ids,   data, mri_dict, x_dict, model, best_head, device, patient_apoe4, TIMEPOINTS)
        test_ds  = AttentionDatasetWithLabels(test_ids,  data, mri_dict, x_dict, model, best_head, device, patient_apoe4, TIMEPOINTS)
        full_ds  = AttentionDatasetWithLabels(all_patients, data, mri_dict, x_dict, model, best_head, device, patient_apoe4, TIMEPOINTS)


        val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False, collate_fn=attn_collate_with_labels)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=attn_collate_with_labels)
        full_loader = DataLoader(full_ds, batch_size=64, shuffle=False, collate_fn=attn_collate_with_labels)

        val_probs, val_labels = [], []
        with torch.no_grad():
            for emb, lbl in val_loader:
                emb = emb.to(device)
                lg = best_head(emb)
                val_probs .append(torch.sigmoid(lg).cpu().numpy())
                val_labels.append(lbl.numpy())
        val_probs  = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        prec, rec, thr = precision_recall_curve(val_labels, val_probs)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
        ix = np.nanargmax(f1_scores)
        best_thresh, best_f1 = thr[ix], f1_scores[ix]

        for head in ensemble_models:
            head.eval()

        total_c, total_s = 0, 0
        with torch.no_grad():
            for emb, lbl in full_loader:
                emb, lbl = emb.to(device), lbl.to(device).float()
                logits = sum(m(emb) for m in ensemble_models) / len(ensemble_models)
                preds  = (torch.sigmoid(logits) >= 0.5).float()
                total_c += (preds == lbl).sum().item()
                total_s += lbl.size(0)
        overall_acc = total_c / total_s
     

        total_correct_test = 0
        total_samples_test = 0
        run_records = []

        with torch.no_grad():
            for emb, lbl in test_loader:
                emb, lbl = emb.to(device), lbl.to(device).float()
                logits = sum(m(emb) for m in ensemble_models) / len(ensemble_models)
                probs  = torch.sigmoid(logits)
                preds  = (probs >= 0.5).float()


                total_correct_test += (preds == lbl).sum().item()
                total_samples_test += lbl.size(0)

                for i, pid in enumerate(test_ids):
                    true = int(lbl[i].item())
                    prob = float(probs[i].item())
                    pred = int(preds[i].item())
                    print(
                        f"Predicting Patient {pid}: "
                        f"True={true}, Pred={pred}, Prob={prob:.4f}"
                    )

                    run_records.append({
                        "Run":           run,
                        "Patient_ID":    pid,
                        "TrueLabel":     float(true),
                        "PredProb":      prob,
                        "PredictedLabel": float(pred)
                    })

        test_acc = total_correct_test / total_samples_test
        print(f"Test Accuracy: {test_acc*100:.2f}%")
  
        overall_accuracies.append(overall_acc)
        test_accuracies.append(test_acc)
        all_runs.append(pd.DataFrame(run_records))

    df_all = pd.concat(all_runs, ignore_index=True)
    df_all.to_csv("20_runs.csv", index=False)


    # 2) print overall summaries
    avg_ov, std_ov = np.mean(overall_accuracies)*100, np.std(overall_accuracies)*100
    avg_te, std_te = np.mean(test_accuracies)*100,    np.std(test_accuracies)*100
    print(f"\nAverage Overall Accuracy: {avg_ov:.2f}% (±{std_ov:.2f}%)")
    print(f"Average Test    Accuracy: {avg_te:.2f}% (±{std_te:.2f}%)\n")

    print("Detailed test results by run:")
    for run_id, group in df_all.groupby("Run"):
        print(f"\n––– Run {run_id} –––")
        print(group[["Patient_ID","TrueLabel","PredictedLabel"]].to_string(index=False))


if __name__ == "__main__":
    main()
