"""
RCAF Engine: RGB-Depth Context-Aware Anomaly Fusion
Paper 1 Core Contribution
"""

import numpy as np
import json
import time
import pickle
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA_DIR    = Path("/home/sentinel/sentinel_iot/paper1/data")
MODEL_DIR   = Path("/home/sentinel/sentinel_iot/paper1/models")
RESULTS_DIR = Path("/home/sentinel/sentinel_iot/paper1/results")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("RCAF ENGINE TRAINING PIPELINE")
print("="*60)

# ----------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------
print("\n[1/4] Loading preprocessed data...")
X_train = np.load(str(DATA_DIR / "rcaf_X_train.npy"))
X_test  = np.load(str(DATA_DIR / "rcaf_X_test.npy"))
y_train = np.load(str(DATA_DIR / "rcaf_y_train.npy"))
y_test  = np.load(str(DATA_DIR / "rcaf_y_test.npy"))
print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"  Features: {X_train.shape[1]} (32 depth x2 weight + 16 RGB)")

# ----------------------------------------------------------
# 2. Isolation Forest
# ----------------------------------------------------------
print("\n[2/4] Training Isolation Forest...")
anomaly_frac = float(np.mean(y_train == 1))
contamination = max(0.01, min(0.49, anomaly_frac))

t0 = time.time()
iso = IsolationForest(
    n_estimators=100,
    contamination=contamination,
    random_state=42,
    n_jobs=2
)
iso.fit(X_train)
print(f"  Train time: {time.time()-t0:.2f}s")
print(f"  Contamination: {contamination:.3f}")

with open(str(MODEL_DIR / "rcaf_isoforest.pkl"), "wb") as f:
    pickle.dump(iso, f)
print(f"  Saved: rcaf_isoforest.pkl")

# ----------------------------------------------------------
# 3. One-Class SVM
# ----------------------------------------------------------
print("\n[3/4] Training One-Class SVM...")
X_normal = X_train[y_train == 0]
if len(X_normal) > 2000:
    idx = np.random.choice(len(X_normal), 2000, replace=False)
    X_normal_s = X_normal[idx]
else:
    X_normal_s = X_normal

t0 = time.time()
ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
ocsvm.fit(X_normal_s)
print(f"  Train time: {time.time()-t0:.2f}s")
print(f"  Trained on {len(X_normal_s)} normal samples")

with open(str(MODEL_DIR / "rcaf_ocsvm.pkl"), "wb") as f:
    pickle.dump(ocsvm, f)
print(f"  Saved: rcaf_ocsvm.pkl")

# ----------------------------------------------------------
# 4. Ensemble + Evaluate
# ----------------------------------------------------------
print("\n[4/4] Ensemble Evaluation...")

# Isolation Forest predictions
if_raw  = iso.predict(X_test)
if_pred = (if_raw == -1).astype(int)

# One-Class SVM predictions
svm_raw  = ocsvm.predict(X_test)
svm_pred = (svm_raw == -1).astype(int)

# Weighted ensemble (IF=0.6, SVM=0.4)
ensemble_score = if_pred * 0.6 + svm_pred * 0.4
y_pred = (ensemble_score >= 0.5).astype(int)

# Metrics
acc  = float(accuracy_score(y_test, y_pred))
prec = float(precision_score(y_test, y_pred, zero_division=0))
rec  = float(recall_score(y_test, y_pred, zero_division=0))
f1   = float(f1_score(y_test, y_pred, zero_division=0))

try:
    auc = float(roc_auc_score(y_test, y_pred))
except Exception:
    auc = 0.0

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,0)
fpr = float(fp/(fp+tn+1e-9))
dr  = float(tp/(tp+fn+1e-9))

# Latency benchmark
t0 = time.perf_counter()
for i in range(min(200, len(X_test))):
    x = X_test[i:i+1]
    s1 = iso.predict(x)[0]
    s2 = ocsvm.predict(x)[0]
lat_ms = ((time.perf_counter()-t0) / min(200,len(X_test))) * 1000

print(f"\n  === RCAF EVALUATION RESULTS ===")
print(f"  Accuracy:            {acc:.4f}")
print(f"  Precision:           {prec:.4f}")
print(f"  Recall (DR):         {rec:.4f}")
print(f"  F1 Score:            {f1:.4f}")
print(f"  ROC-AUC:             {auc:.4f}")
print(f"  Detection Rate:      {dr:.4f}")
print(f"  False Positive Rate: {fpr:.4f}")
print(f"  Avg Latency:         {lat_ms:.3f} ms/sample")
print(f"  ================================")

# Save results
results = {
    "model": "RCAF-IoT Ensemble",
    "paper": "Paper 1",
    "hardware": "Raspberry Pi 4B ARM Cortex-A72",
    "sensor": "Microsoft Kinect Xbox 360 Model 1473",
    "feature_dimensions": {
        "depth_features": 32,
        "rgb_features": 16,
        "fused_total": 48,
        "depth_context_weight": 2.0
    },
    "ensemble_weights": {
        "isolation_forest": 0.6,
        "ocsvm": 0.4
    },
    "evaluation_metrics": {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "detection_rate": dr,
        "false_positive_rate": fpr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "avg_latency_ms_per_sample": lat_ms
    },
    "hardware_benchmark": {
        "platform": "Raspberry Pi 4 ARM Cortex-A72 @ 1.8GHz",
        "single_sample_latency_ms": {
            "mean": lat_ms
        },
        "throughput_samples_per_sec": float(1000/lat_ms)
    },
    "dataset_info": {
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0])
    }
}

out = RESULTS_DIR / "rcaf_evaluation_results.json"
with open(str(out), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {out}")

print("\n" + "="*60)
print("RCAF ENGINE BUILD COMPLETE")
print("="*60)
print(f"Detection Rate:   {dr:.4f}")
print(f"False Pos Rate:   {fpr:.4f}")
print(f"F1 Score:         {f1:.4f}")
print(f"Avg Latency:      {lat_ms:.3f} ms/sample")
print("="*60)
