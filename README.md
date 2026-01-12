# ADFormer
A multi-modal transformer model for studying Alzheimer's disease

---
## 📦 Download Required Supporting Files

Google Drive: https://mailmissouri-my.sharepoint.com/:f:/g/personal/tmnthc_umsystem_edu/IgA346DkGmR3R6Qzo8JCEv3dAcMy6JaUTNGZIyT_8reWSks?e=dfun40

---

## 🚀 How to Run

Run the main script and provide the input paths.  

The `main.py` will automatically load the data, split edges into 16 patients for training, 3 patients for validation and 4 patient for testing, train the model, and then run final evaluation on the test split.

```bash
python main.py \
  --data_path /path/to/Data1
```
---


