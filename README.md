# 🔧predictive-maintenance-dashboard
Streamlit dashboard for predictive maintenance using NASA turbofan data
An interactive, real‑time predictive maintenance simulator for aircraft engines. The dashboard ingests sensor data from the NASA C‑MAPSS dataset, predicts remaining useful life (RUL) using quantile regression, and displays live gauges with configurable alert thresholds. It also provides sophisticated failure mode classification, multi‑engine comparison, and downloadable reports.
## ✨ Features

- **Live sensor monitoring** – Temperature, vibration, and pressure gauges update as you step through cycles.
- **Quantile regression confidence intervals** – Real 90% prediction intervals using XGBoost quantile models (not just ±10%).
- **Configurable alerts** – Set your own thresholds; warnings appear when a sensor exceeds the limit.
- **Sophisticated failure mode classification** – Identifies specific issues like HPT/LPT degradation, bearing wear, compressor fouling, and more, with confidence levels.
- **Multi‑engine comparison** – Compare up to six engines side‑by‑side with trend charts, current status cards, and detailed data export.
- **Easy deployment** – One‑click deploy to Streamlit Community Cloud.
- **Export reports** – Download comparison data as CSV.
- **Auto‑refresh simulation** – Watch the engine age in real time (adjustable interval).


---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** – frontend framework
- **XGBoost** – quantile regression models
- **Scikit‑learn** – Random Forest model
- **Pandas / NumPy** – data handling
- **Plotly** – interactive gauges and charts
- **Joblib** – model serialisation
- **Git** – version control

---

## 📊 Dataset

The project uses the **NASA C‑MAPSS** turbofan engine degradation dataset ([publicly available](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)).  
Only the **FD001** subset is used:
- `train_FD001.txt` – training data (21 sensors + 3 operational settings)
- `test_FD001.txt` – test data for simulation
- `RUL_FD001.txt` – true RUL values for test engines

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account (for deployment)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/[your-username]/predictive-maintenance-dashboard.git
   cd predictive-maintenance-dashboard
   
2. **Create and activate a virtual environment**
  ```bash
  python -m venv venv

  #**Windows**
  venv\Scripts\activate

  #**Mac / Linux**
  source venv/bin/activate

3. **Install dependencies** 
  ```bash
  pip install -r requirements.txt

4. **Download the dataset**
Place the three text files (train_FD001.txt, test_FD001.txt, RUL_FD001.txt) inside a folder named data/ in the project root.

5. **Train the model**
  ```bash
   python train_model.py
   This creates rul_model.pkl.

6. **Run the dashboard locally**
  ```bash
  streamlit run app.py
