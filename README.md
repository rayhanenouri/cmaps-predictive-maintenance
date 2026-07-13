# ✈️ Aircraft Engine Predictive Maintenance

> **Production-grade Remaining Useful Life (RUL) prediction for turbofan engines using NASA C-MAPS dataset and XGBoost**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Why This Project Matters

In the aviation MRO (Maintenance, Repair, and Overhaul) industry, **predictive maintenance saves millions** by:
- **Preventing catastrophic failures** through early detection of engine degradation
- **Reducing unscheduled downtime** by 30-50% (Boeing, 2023)
- **Optimizing maintenance schedules** based on actual component health, not fixed intervals
- **Lowering operational costs** by avoiding premature part replacement

This project demonstrates real-world predictive maintenance capabilities using industry-standard machine learning techniques applied to NASA's turbofan engine degradation dataset — the same principles used by **Lufthansa Technik**, **Safran**, and **Rolls-Royce** for fleet health monitoring.

---

## 📊 Dataset: NASA C-MAPS FD001

The **Commercial Modular Aero-Propulsion System Simulation (C-MAPS)** dataset is a widely-used benchmark for prognostics research, developed by NASA Ames Research Center.

### Dataset Characteristics:
- **100 training engines** running until failure
- **100 test engines** with partial lifecycle data
- **21 sensor measurements** per cycle (temperature, pressure, vibration, etc.)
- **3 operational settings** (altitude, Mach number, throttle resolver angle)
- **Single failure mode** (FD001 subset)

### Sensor Data Includes:
- Fan inlet/outlet temperatures and pressures
- LPC (Low Pressure Compressor) outlet pressure/temperature
- HPC (High Pressure Compressor) outlet pressure/temperature
- HPT (High Pressure Turbine) coolant bleed
- Bypass and core flow ratios
- And 14+ additional sensor readings

**Download:** [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Raw Sensor Data (train/test FD001)                         │
│           ↓                                                  │
│  [Data Loader] → Parse, calculate RUL, cap at 125 cycles    │
│           ↓                                                  │
│  [Feature Engineering]                                       │
│     • Remove low-variance sensors (14/21 retained)           │
│     • Rolling mean/std (5, 10 cycle windows)                 │
│     • MinMax normalization                                   │
│           ↓                                                  │
│  [XGBoost Regressor]                                         │
│     • 300 estimators, depth=6, lr=0.05                       │
│     • Subsample=0.8 for regularization                       │
│           ↓                                                  │
│  RUL Predictions (capped at 0, saved to model.pkl)           │
│           ↓                                                  │
│  [Streamlit Dashboard] → Real-time health monitoring         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- Git

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/cmaps-predictive-maintenance.git
cd cmaps-predictive-maintenance

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NASA C-MAPS dataset
# Place train_FD001.txt, test_FD001.txt, and RUL_FD001.txt in data/
```

### Download Dataset
1. Visit [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
2. Download **C-MAPS Dataset**
3. Extract `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`
4. Place files in `data/` directory

---

## 📈 Usage

### 1. Train the Model

```bash
# Train XGBoost model and save to model.pkl
python src/model.py
```

**Expected Output:**
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

### 2. Evaluate on Test Set

```bash
# Evaluate model and generate performance plots
python src/evaluate.py
```

**Outputs:**
- `results_plot.png` - Predicted vs Actual RUL scatter + error distribution
- `degradation_curves.png` - Sample engine health trajectories

### 3. Launch Interactive Dashboard

```bash
# Start Streamlit app
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`

---

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Test RMSE** | `XX.XX` cycles |
| **Test R² Score** | `0.XXXX` |
| **MAE** | `XX.XX` cycles |
| **Predictions within ±20 cycles** | `XX.X%` |

> **Note:** Results will vary based on training run. Typical RMSE for C-MAPS FD001 with XGBoost: 15-20 cycles.

### Key Findings
- **Rolling window features** (5/10 cycles) significantly improve RUL tracking
- **Early lifecycle predictions** less accurate (high RUL variance)
- **End-of-life predictions** more precise as degradation patterns stabilize
- **Sensor correlation analysis** critical — 7/21 sensors provide minimal information

---

## 🖼️ Screenshots

### Dashboard Overview
*[Screenshot placeholder - run `streamlit run app.py` and capture main view]*

### Engine Degradation Curve
*[Screenshot placeholder - predicted vs actual RUL over operational cycles]*

### Evaluation Plots
*[Screenshot placeholder - results_plot.png and degradation_curves.png]*

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **ML Framework:** XGBoost 1.7.6
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Web Framework:** Streamlit 1.25.0
- **Model Persistence:** Joblib

---

## 📂 Project Structure

```
cmaps-predictive-maintenance/
├── data/                          # Dataset files (not tracked in Git)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── src/                           # Core modules
│   ├── data_loader.py             # NASA C-MAPS data parsing
│   ├── feature_engineering.py     # Sensor selection & rolling features
│   ├── model.py                   # XGBoost training & prediction
│   └── evaluate.py                # Test set evaluation & plotting
├── notebooks/                     # Jupyter notebooks for exploration
│   └── exploration.ipynb
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🔬 Future Enhancements

- [ ] **LSTM/Transformer models** for sequence-based RUL prediction
- [ ] **Multi-condition datasets** (FD002-FD004) for robustness
- [ ] **Real-time data streaming** simulation with Apache Kafka
- [ ] **Uncertainty quantification** using conformal prediction
- [ ] **Docker containerization** for deployment
- [ ] **CI/CD pipeline** with GitHub Actions

---

## 📚 References

1. Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames Prognostics Data Repository.
2. Zheng, S., et al. (2017). "Long Short-Term Memory Network for Remaining Useful Life estimation". *ICPHM*.
3. Li, X., et al. (2018). "Remaining useful life estimation in prognostics using deep convolution neural networks". *Reliability Engineering*.

---

## 👤 Author

**Rayhane Nouri**  
Electrical Engineering Student  
ENSIT (École Nationale des Sciences de l'Ingénieur de Tunis), Tunisia

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/yourusername)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat&logo=gmail)](mailto:rayhane.nouri1@gmail.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NASA Ames Research Center for the C-MAPS dataset
- Open-source contributors to XGBoost, Streamlit, and Scikit-learn
- Aviation MRO community for domain knowledge validation

---

<div align="center">
  <strong>Built with ❤️ for the future of aviation maintenance</strong>
</div>
