# ADFormer

![Demo image](images/modelarchitecture.jpg)

a two-stage multimodal transformer framework that inte-
grates volumetric magnetic resonance imaging (MRI) with
diverse non-imaging biomarkers to enable unified repre-
sentation learning for Alzheimer’s disease analysis. In the
first stage, modality-specific encoders are pretrained using
contrastive learning to align heterogeneous data sources in
a shared latent space. Three-dimensional MRI volumes are
encoded using a CLIP-based vision encoder, while hetero-
geneous non-imaging modalities—including microstruc-
tural features, biomarker panels, and numeric clinical vari-
ables—are encoded using multilayer perceptron–based en-
coders. A contrastive objective encourages consistent rep-
resentations across modalities while preserving subject-
level correspondence between imaging and non-imaging
data. In the second stage, the pretrained embeddings are
used as inputs to a multimodal transformer that integrates
information across modalities to produce subject-level pre-
dictions. 

---
## 📦 Download Required Supporting Files

Google Drive: https://mailmissouri-my.sharepoint.com/:f:/g/personal/tmnthc_umsystem_edu/IgA346DkGmR3R6Qzo8JCEv3dAcMy6JaUTNGZIyT_8reWSks?e=dfun40

---

## 🚀 Train From Scratch (main.py)

`main.py` loads data, performs the train/val/test split, trains the model, and runs final evaluation.

```bash
python main.py \
  --data_path /path/to/Data1
```
---
⚡ Inference Only (pretrained weights)

Use this when you already have trained checkpoints and just want predictions/evaluation.

```bash
python inference.py 

```

---

