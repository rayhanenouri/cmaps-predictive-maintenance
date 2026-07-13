# First Time Setup

Welcome to your NASA C-MAPS Predictive Maintenance project!

---

## 📋 Prerequisites Checklist

Before starting, make sure you have:

- [ ] Python 3.9 or higher installed (`python --version`)
- [ ] Git installed (`git --version`)
- [ ] Internet connection (for downloading dataset and packages)
- [ ] 500MB free disk space

---

## 🚀 Step-by-Step Setup

### 1. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install packages
pip install -r requirements.txt
```

**Expected time**: 2-3 minutes

---

### 2. Download NASA C-MAPS Dataset

#### Option A: Direct Download
1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
2. Find **"Turbofan Engine Degradation Simulation Data Set"**
3. Download and extract the ZIP file
4. Copy these files to `data/` folder:
   - `train_FD001.txt`
   - `test_FD001.txt`
   - `RUL_FD001.txt`

#### Option B: Command Line (if available)
```bash
# Using wget (Linux/Mac)
cd data/
wget https://ti.arc.nasa.gov/c/6/
# Extract and copy FD001 files
cd ..
```

**Verify dataset**:
```bash
ls data/
# Should show: train_FD001.txt  test_FD001.txt  RUL_FD001.txt
```

---

### 3. Train Your First Model

```bash
python src/model.py
```

**What happens**:
- Loads 20,631 training records from 100 engines
- Engineers 126+ features from 21 raw sensors
- Trains XGBoost model (300 trees)
- Saves `model.pkl` (ready for dashboard)

**Expected output**:
```
=== Loading NASA C-MAPS FD001 Dataset ===
✓ Loaded training data: 20631 records from 100 engines

=== Feature Engineering ===
✓ Removed 7 low-variance features
✓ Added 112 rolling window features
✓ Normalized 126 features

=== Training XGBoost Model ===
Training samples: 20631
Features: 126

✓ Training complete!
  Training RMSE: 14.52 cycles
  Training R² Score: 0.8234

✓ Model saved to model.pkl
```

**Time**: 30-60 seconds

---

### 4. Evaluate Model Performance

```bash
python src/evaluate.py
```

**Outputs**:
- Terminal: Test RMSE, R², MAE metrics
- `results_plot.png` - Scatter plot (predicted vs actual)
- `degradation_curves.png` - Sample engine trajectories

---

### 5. Launch Interactive Dashboard

```bash
streamlit run app.py
```

**Dashboard opens at**: http://localhost:8501

**Features**:
- Select any of 100 test engines
- View health status (Healthy/Monitor/Critical)
- See degradation curve over time
- Compare predicted vs actual RUL
- Overall model metrics

---

## 🧪 Optional: Run Tests

```bash
python -m unittest discover tests -v
```

Tests validate:
- Data loading logic
- RUL calculation accuracy
- Column naming consistency

---

## 🐛 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'xgboost'"
**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: train_FD001.txt"
**Solution**: 
- Ensure dataset files are in `data/` folder
- Check filenames are exactly: `train_FD001.txt` (case-sensitive)

### Issue: "Streamlit command not found"
**Solution**:
```bash
pip install streamlit
# OR
python -m streamlit run app.py
```

### Issue: Dashboard loads but shows error
**Solution**: Train model first!
```bash
python src/model.py
```

---

## 📊 What You Should See

After successful setup:

```
cmaps-predictive-maintenance/
├── data/
│   ├── train_FD001.txt ✅
│   ├── test_FD001.txt ✅
│   └── RUL_FD001.txt ✅
├── model.pkl ✅ (after training)
├── results_plot.png ✅ (after evaluation)
└── degradation_curves.png ✅ (after evaluation)
```

---

## 🎓 Next Steps

1. ✅ Explore the Jupyter notebook: `jupyter notebook notebooks/exploration.ipynb`
2. ✅ Customize dashboard colors/layout in `app.py`
3. ✅ Experiment with hyperparameters in `src/model.py`
4. ✅ Take screenshots of your dashboard
5. ✅ Push to GitHub and share on LinkedIn!

---

## 📞 Need Help?

- Read: `QUICKSTART.md` for condensed guide
- Check: `README.md` for detailed documentation
- Review: `PROJECT_SUMMARY.md` for architecture overview
- Contact: rayhane.nouri1@gmail.com

---

**Ready to impress recruiters!** 🚀
