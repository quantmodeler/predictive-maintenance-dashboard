import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# -------------------------------
# 1. Load and prepare training data
# -------------------------------
# Use regex '\s+' to handle multiple spaces/tabs
train = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None)

# The FD001 dataset has 26 columns: engine id, cycle, 3 settings, 21 sensors
# If the file has more than 26 columns due to trailing spaces, keep only the first 26.
if train.shape[1] > 26:
    train = train.iloc[:, :26]

# Assign column names
columns = ['engine', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1,22)]
train.columns = columns

# -------------------------------
# 2. Compute Remaining Useful Life (RUL)
# -------------------------------
# For each engine, find the maximum cycle (last recorded cycle)
max_cycles = train.groupby('engine')['cycle'].max().reset_index()
max_cycles.columns = ['engine', 'max_cycle']

# Merge max cycle back to the training data
train = train.merge(max_cycles, on='engine')

# RUL = max_cycle - current cycle
train['RUL'] = train['max_cycle'] - train['cycle']

# -------------------------------
# 3. Feature selection
# -------------------------------
# Use all 21 sensor readings as features (you could also include settings)
feature_columns = [f'sensor{i}' for i in range(1,22)]
X = train[feature_columns]
y = train['RUL']

# -------------------------------
# 4. Train a Random Forest model
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# -------------------------------
# 5. Save the model
# -------------------------------
joblib.dump(model, 'rul_model.pkl')
print("Model saved as rul_model.pkl")
print(f"Training data shape: {train.shape}")
print(f"Features used: {feature_columns}")