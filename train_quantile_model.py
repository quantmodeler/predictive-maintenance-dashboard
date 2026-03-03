import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

print("🚀 Starting quantile model training...")

# Load and prepare data
print("📊 Loading training data...")
try:
    train = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None)
    print(f"✅ Data loaded successfully")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

if train.shape[1] > 26:
    train = train.iloc[:, :26]

# Assign column names
columns = ['engine', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1,22)]
train.columns = columns
print(f"✅ Data shape: {train.shape[0]} rows, {train.shape[1]} columns")

# Compute RUL
print("🧮 Computing RUL...")
max_cycles = train.groupby('engine')['cycle'].max().reset_index()
max_cycles.columns = ['engine', 'max_cycle']
train = train.merge(max_cycles, on='engine')
train['RUL'] = train['max_cycle'] - train['cycle']
print(f"✅ RUL computed. Range: {train['RUL'].min():.0f} - {train['RUL'].max():.0f} cycles")

# Features
feature_columns = [f'sensor{i}' for i in range(1,22)]
X = train[feature_columns]
y = train['RUL']

# Train three models: median (0.5), lower bound (0.1), upper bound (0.9)
models = {}
quantiles = [0.1, 0.5, 0.9]
quantile_names = ['q10', 'q50', 'q90']

for quantile, name in zip(quantiles, quantile_names):
    print(f"🎯 Training quantile {quantile} ({name})...")
    print(f"    This may take a few minutes...")
    model = XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=quantile,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X, y)
    models[name] = model
    print(f"✅ {name} model trained")

# Save all models
print("💾 Saving models...")
joblib.dump(models, 'quantile_models.pkl')
print("✅ Quantile models saved as quantile_models.pkl")
print("🎉 Training complete!")