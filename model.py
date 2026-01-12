import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from data import advanced_image_augmentation


class MRIClipEncoder(nn.Module):
    def __init__(self, embed_dim=64, augment=False, dropout_p=0.3):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.unfreeze_layers = 6
        self._unfreeze_last_n_layers(self.unfreeze_layers)

        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = True

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.project = nn.Sequential(
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.augment = augment
        if self.augment:
            self.augmentation = advanced_image_augmentation()

    def _unfreeze_last_n_layers(self, n):
        for layer in self.clip_model.vision_model.encoder.layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"Unfroze last {n} layers of CLIP vision encoder.")

    def gradually_unfreeze(self, current_patience, max_patience, increment=2, max_layers=12):
        if current_patience >= max_patience and self.unfreeze_layers < max_layers:
            self.unfreeze_layers = min(self.unfreeze_layers + increment, max_layers)
            self._unfreeze_last_n_layers(self.unfreeze_layers)

    def forward(self, mri_batch):
        if mri_batch.dim() == 3:
            return mri_batch
        if mri_batch.dim() == 4:
            return mri_batch.squeeze(1)

        device = mri_batch.device
        B, _, H, W, D = mri_batch.shape

        pil_images = []
        for i in range(B):
            vol = mri_batch[i, 0]
            if vol.ndim != 3:
                vol = torch.zeros((H, W, D), dtype=torch.float32, device=device)

            vol = vol.permute(2, 0, 1)
            for slice2d in vol:
                arr = slice2d.cpu().numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    minv = np.nanmin(arr)
                    maxv = np.nanmax(arr)
                    rng = maxv - minv
                    if rng < 1e-6:
                        img8 = np.zeros_like(arr, dtype=np.uint8)
                    else:
                        norm = (arr - minv) / rng
                        norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
                        img8 = (norm * 255.0).astype(np.uint8)

                pil = Image.fromarray(img8, mode="L").convert("RGB")
                if self.training and self.augment:
                    pil = self.augmentation(pil)
                pil_images.append(pil)

        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        feats = self.clip_model.get_image_features(**inputs)
        feats = safe_normalize(feats, p=2, dim=-1)

        projs = self.project(feats)
        projs = safe_normalize(projs, p=2, dim=-1)

        projs = projs.view(B, D, -1)
        return projs


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=64,
        hidden_dims=[256, 256, 256],
        dropout_p=0.3,
        use_gelu=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.layers.append(nn.LayerNorm(dims[i + 1]))
            self.layers.append(nn.GELU() if use_gelu else nn.ReLU())
            self.layers.append(nn.Dropout(dropout_p))
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        h = x
        for i in range(0, len(self.layers), 4):
            lin = self.layers[i]
            norm = self.layers[i + 1]
            act = self.layers[i + 2]
            drop = self.layers[i + 3]

            fwd = lin(h)
            fwd = norm(fwd)
            fwd = act(fwd)
            fwd = drop(fwd)

            if fwd.shape == h.shape:
                h = h + fwd
            else:
                h = fwd

        h = self.out_norm(h)
        return F.normalize(h, p=2, dim=-1)


def embed_and_pad(vol_list, encoder, device):
    embeddings = []
    encoder.eval()
    with torch.no_grad():
        for vol in vol_list:
            vol_in = vol.unsqueeze(0).to(device)
            e = encoder(vol_in)
            embeddings.append(e.squeeze(0))

    D_max = max(e.shape[0] for e in embeddings)
    padded = []
    for e in embeddings:
        D, E = e.shape
        if D < D_max:
            pad = torch.zeros(D_max - D, E, device=e.device)
            e = torch.cat([e, pad], dim=0)
        else:
            e = e[:D_max]
        padded.append(e)

    return torch.stack(padded, dim=0)


class MultiModalEmbeddingModel(nn.Module):
    def __init__(
        self,
        micro_dim,
        biomarker_A_dim,
        biomarker_B_dim,
        biomarker_C_dim,
        other_dim,
        numeric_dim,
        embed_dim=64,
        augment=False,
    ):
        super().__init__()
        self.mri_encoder = MRIClipEncoder(embed_dim=embed_dim, augment=augment)
        self.x_encoder = MRIClipEncoder(embed_dim=embed_dim, augment=augment)

        self.micro_encoder = MLPEncoder(input_dim=micro_dim, output_dim=embed_dim)
        self.bioA_encoder = MLPEncoder(input_dim=biomarker_A_dim, output_dim=embed_dim)
        self.bioB_encoder = MLPEncoder(input_dim=biomarker_B_dim, output_dim=embed_dim)
        self.bioC_encoder = MLPEncoder(input_dim=biomarker_C_dim, output_dim=embed_dim)
        self.other_encoder = MLPEncoder(input_dim=other_dim, output_dim=embed_dim)
        self.num_encoder = MLPEncoder(input_dim=numeric_dim, output_dim=embed_dim)

        max_slices = 256
        self.slice_weights = nn.Parameter(torch.zeros(max_slices))
        self.probe_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )

        self.embed_dim = embed_dim

    def forward(self, mri, micro, biomarker_A, biomarker_B, biomarker_C, other, numeric, x_img):
        e_mri = self.mri_encoder(mri)
        e_x = self.x_encoder(x_img)
        e_micro = self.micro_encoder(micro)
        e_bioA = self.bioA_encoder(biomarker_A)
        e_bioB = self.bioB_encoder(biomarker_B)
        e_bioC = self.bioC_encoder(biomarker_C)
        e_other = self.other_encoder(other)
        e_num = self.num_encoder(numeric)
        return e_mri, e_x, e_micro, e_bioA, e_bioB, e_bioC, e_other, e_num


class StackedAttentionClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        total_layers = num_layers + 1
        for _ in range(total_layers):
            self.layers.append(
                nn.MultiheadAttention(
                    embed_dim,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

        self.pool_fc = nn.Linear(embed_dim, 1)

        self.post_pool_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        out = embeddings
        for attn in self.layers:
            attn_out, _ = attn(out, out, out)
            out = self.layernorm(out + self.dropout(attn_out))

        pool_scores = self.pool_fc(out).squeeze(-1)
        pool_weights = F.softmax(pool_scores, dim=1)

        rep = torch.bmm(pool_weights.unsqueeze(1), out).squeeze(1)

        logits = self.post_pool_mlp(rep).squeeze(-1)
        return logits
