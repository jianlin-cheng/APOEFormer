
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel


def patient_contrastive_loss(
    e_mri, e_x, e_micro, e_bioA, e_bioB, e_bioC, e_other, e_num,
    sample_labels,
    tau=0.1
):
    labels = sample_labels.detach().long().to(e_mri.device)
    labels_all = labels.repeat(8)

    emb_all = torch.cat(
        [e_mri, e_x, e_micro, e_bioA, e_bioB, e_bioC, e_other, e_num],
        dim=0
    )

    sim = torch.matmul(emb_all, emb_all.t()) / tau
    N = labels_all.size(0)
    diag = torch.eye(N, dtype=torch.bool, device=sim.device)

    exp_sim = torch.exp(sim)

    pos_mask = (labels_all.unsqueeze(0) == labels_all.unsqueeze(1)) & ~diag
    sum_pos = (exp_sim * pos_mask.float()).sum(dim=1)
    sum_all = (exp_sim * (~diag).float()).sum(dim=1)

    loss = -torch.log((sum_pos + 1e-8) / (sum_all + 1e-8))
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        init_alpha_pos: float = 0.8,
        init_alpha_neg: float = 0.2,
        gamma: float = 4.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha_pos = nn.Parameter(torch.tensor(init_alpha_pos, dtype=torch.float32))
        self.alpha_neg = nn.Parameter(torch.tensor(init_alpha_neg, dtype=torch.float32))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        alpha_pos = torch.sigmoid(self.alpha_pos)
        alpha_neg = torch.sigmoid(self.alpha_neg)

        alpha_t = targets * alpha_pos + (1.0 - targets) * alpha_neg

        mod_term = (1.0 - p_t) ** self.gamma
        loss = alpha_t * mod_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
