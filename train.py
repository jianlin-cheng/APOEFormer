import os

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CombinedContrastiveDataset, AttentionDatasetWithLabels
from model import *
from losses import patient_contrastive_loss, FocalLoss

from torch.cuda import amp


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    val_loss_history,
    ce_weight,
    patience_threshold=5,
    accum_steps=4,
    scaler=None,
):
    base = model.module if hasattr(model, "module") else model
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        e_mri = embed_and_pad(batch["mri"], base.mri_encoder, device)
        e_x = embed_and_pad(batch["x_img"], base.x_encoder, device)

        alpha_mri = F.softmax(base.slice_weights[: e_mri.size(1)], dim=0)
        emri_m = (e_mri * alpha_mri[None, :, None]).sum(1)
        alpha_x = F.softmax(base.slice_weights[: e_x.size(1)], dim=0)
        ex_m = (e_x * alpha_x[None, :, None]).sum(1)

        micro = batch["micro"].to(device)
        bioA = batch["biomarker_A"].to(device)
        bioB = batch["biomarker_B"].to(device)
        bioC = batch["biomarker_C"].to(device)
        other = batch["other"].to(device)
        num = batch["numeric"].to(device)
        labels = batch["label"].to(device)

        e_micro = base.micro_encoder(micro)
        e_bioA = base.bioA_encoder(bioA)
        e_bioB = base.bioB_encoder(bioB)
        e_bioC = base.bioC_encoder(bioC)
        e_other = base.other_encoder(other)
        e_num = base.num_encoder(num)

        with amp.autocast():
            contr_loss = patient_contrastive_loss(
                emri_m,
                ex_m,
                e_micro,
                e_bioA,
                e_bioB,
                e_bioC,
                e_other,
                e_num,
                sample_labels=labels,
                tau=0.1,
            )
            probe_logits = base.probe_head(emri_m).squeeze(-1)
            ce_loss = F.binary_cross_entropy_with_logits(probe_logits, labels)
            loss = (contr_loss + ce_weight * ce_loss) / accum_steps

        scaler.scale(loss).backward()
        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += (loss * accum_steps).item()

    avg_loss = total_loss / len(loader)

    if val_loss_history and avg_loss >= max(val_loss_history[-patience_threshold:]):
        base.mri_encoder.gradually_unfreeze(patience_threshold, patience_threshold)
        base.x_encoder.gradually_unfreeze(patience_threshold, patience_threshold)

    return avg_loss


def validate_epoch(model, loader, device, ce_weight):
    base = model.module if hasattr(model, "module") else model
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            e_mri = embed_and_pad(batch["mri"], base.mri_encoder, device)
            e_x = embed_and_pad(batch["x_img"], base.x_encoder, device)

            alpha_mri = F.softmax(base.slice_weights[: e_mri.size(1)], dim=0)
            emri_m = (e_mri * alpha_mri[None, :, None]).sum(1)
            alpha_x = F.softmax(base.slice_weights[: e_x.size(1)], dim=0)
            ex_m = (e_x * alpha_x[None, :, None]).sum(1)

            micro = batch["micro"].to(device)
            bioA = batch["biomarker_A"].to(device)
            bioB = batch["biomarker_B"].to(device)
            bioC = batch["biomarker_C"].to(device)
            other = batch["other"].to(device)
            num = batch["numeric"].to(device)
            labels = batch["label"].to(device)

            e_micro = base.micro_encoder(micro)
            e_bioA = base.bioA_encoder(bioA)
            e_bioB = base.bioB_encoder(bioB)
            e_bioC = base.bioC_encoder(bioC)
            e_other = base.other_encoder(other)
            e_num = base.num_encoder(num)

            contr_loss = patient_contrastive_loss(
                emri_m,
                ex_m,
                e_micro,
                e_bioA,
                e_bioB,
                e_bioC,
                e_other,
                e_num,
                sample_labels=labels,
                tau=0.1,
            )
            probe_logits = base.probe_head(emri_m).squeeze(-1)
            ce_loss = F.binary_cross_entropy_with_logits(probe_logits, labels)
            loss = contr_loss + ce_weight * ce_loss

            total_loss += loss.item()

    return total_loss / len(loader)


def precompute_embeddings(mri_dict, mri_encoder, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    mri_encoder.eval()
    for (pid, tp), mri in mri_dict.items():
        emb_path = os.path.join(out_dir, f"{pid}_{tp}.pt")
        if os.path.exists(emb_path):
            continue
        with torch.no_grad():
            e_mri = mri_encoder(mri.unsqueeze(0).to(device))
        torch.save(e_mri.squeeze(0).cpu(), emb_path)


def load_precomputed(mri_keys, emb_dir):
    mri_emb = {}
    for pid, tp in mri_keys:
        path = os.path.join(emb_dir, f"{pid}_{tp}.pt")
        mri_emb[(pid, tp)] = torch.load(path)
    return mri_emb
