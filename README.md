# NeurIPS 2025 – Polymer Property Prediction 🚀

This repository contains our solution for the **NeurIPS 2025 Open Polymer Prediction Challenge**.  
We use a **multi-modal gated fusion model** that combines:

- **ChemBERTa-77M-MTR** (Transformer-based SMILES embeddings)  
- **RNN with Attention** (character-level SMILES encoder)  
- **RDKit Descriptors** (molecular property features)  

with **Optuna-tuned hyperparameters** for robust performance on polymer property prediction.

---

## 📂 Project Structure

```text
.
├── train.csv
├── test.csv
├── train_supplement/
│   ├── dataset1.csv   # Tc supplementary
│   ├── dataset3.csv   # Tg supplementary
│   ├── dataset4.csv   # FFV supplementary
├── sample_submission.csv
├── main.py            # Training & inference pipeline
└── submission.csv     # Final Kaggle output
```
---

## ⚙️ Requirements

The code is designed for **Kaggle** with RDKit wheels and ChemBERTa checkpoint preloaded.  
Installed at runtime:

- `rdkit-pypi==2022.9.5`  
- `torch`, `transformers`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `optuna`  
- `xgboost`, `lightgbm`, `catboost` (optional ensembling)

---

## 🧪 Data

We train on:

- **Main training set** → `train.csv`  
- **Supplementary sets**:  
  - `dataset3.csv` → Tg values  
  - `dataset4.csv` → FFV values  
  - `dataset1.csv` → Tc values  

All supplementary datasets are merged into a single `train_full`.  
Missing targets are marked as `NaN` and handled with **masked loss**.  

**Targets:** `Tg`, `FFV`, `Tc`, `Density`, `Rg`.

---

## 🏗️ Model Architecture

**FusionModel** = `ChemBERTa + RNN + RDKit` with **gated fusion**:

1. **ChemBERTa-MTR** → SMILES transformer embeddings.  
2. **BiGRU + Attention** → character-level SMILES encoder.  
3. **RDKit MLP** → MolWt, LogP, TPSA, HDonors, HAcceptors.  
4. **Gate mechanism** blends RNN+ChemBERTa with RDKit.  
5. **Final MLP head** → predict 5 polymer properties.  

---

## 🔥 Training

- Optimizer: **Adam**  
- Learning rate: **0.00095**  
- Loss: **Masked MSE** (ignores missing targets)  
- Epochs: **40**  
- Batch size: **16**  
- Best checkpoint saved as `best_model.pt`.

---

## 📊 Inference

1. Load best checkpoint.  
2. Predict on `test.csv`.  
3. Inverse transform predictions back to original units.  
4. Save `submission.csv`.  

---

## ▶️ Usage

```bash
python main.py
