# 🚀 Quick Start Guide

Get the predictive maintenance dashboard running in **5 minutes**.

---

## Prerequisites

- Python 3.9+
- NASA C-MAPS dataset downloaded

---

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2: Download Dataset

1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
2. Download **"Turbofan Engine Degradation Simulation Data Set"**
3. Extract these files to `data/` folder:
   - `train_FD001.txt`
   - `test_FD001.txt`
   - `RUL_FD001.txt`

Your structure should be:
```
data/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

---

## Step 3: Train Model

```bash
python src/model.py
```

**Expected output:**
```
=== Loading NASA C-MAPS FD001 Dataset ===
✓ Loaded training data: 20631 records from 100 engines

=== Feature Engineering ===
✓ Removed 7 low-variance features
✓ Added 112 rolling window features
✓ Normalized 126 features

=== Training XGBoost Model ===
✓ Training complete!
  Training RMSE: XX.XX cycles
  Training R² Score: 0.XXXX

✓ Model saved to model.pkl
```

---

## Step 4: Evaluate Model (Optional)

```bash
python src/evaluate.py
```

This generates:
- `results_plot.png` - prediction accuracy visualization
- `degradation_curves.png` - sample engine health curves

---

## Step 5: Launch Dashboard

```bash
streamlit run app.py
```

Dashboard opens at: **http://localhost:8501**

---

## 🎯 Using the Dashboard

1. **Select an engine** from the sidebar dropdown
2. **View health status**: Green (Healthy), Orange (Monitor), Red (Critical)
3. **Analyze degradation curve** showing predicted vs actual RUL
4. **Check metrics**: RMSE, R², prediction accuracy

---

## 📊 Test Individual Modules

```bash
# Test data loader
python src/data_loader.py

# Test feature engineering
python src/feature_engineering.py

# Test model training
python src/model.py

# Full evaluation
python src/evaluate.py
```

---

## 🐛 Troubleshooting

### "FileNotFoundError: train_FD001.txt"
- Ensure dataset files are in `data/` folder
- Check filenames match exactly (case-sensitive)

### "ModuleNotFoundError"
- Run `pip install -r requirements.txt`
- Ensure virtual environment is activated

### "Model not found"
- Run `python src/model.py` to train first

---

## 📚 Next Steps

- Explore `notebooks/exploration.ipynb` for EDA
- Customize dashboard in `app.py`
- Experiment with model hyperparameters in `src/model.py`
- Try multi-condition datasets (FD002-FD004)

---

**Questions?** Open an issue on GitHub or contact: rayhane.nouri1@gmail.com
