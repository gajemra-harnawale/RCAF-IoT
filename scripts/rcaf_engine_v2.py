"""
RCAF Engine v2 — Supervised Hybrid Ensemble
Targets F1 > 0.95, FPR < 0.05
"""
import numpy as np
import json
import time
import pickle
from pathlib import Path
from sklearn.ensemble import (IsolationForest,
                              RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

DATA    = Path("/home/sentinel/sentinel_iot/paper1/data")
MODELS  = Path("/home/sentinel/sentinel_iot/paper1/models")
RESULTS = Path("/home/sentinel/sentinel_iot/paper1/results")
MODELS.mkdir(parents=True, exist_ok=True)

X_train = np.load(str(DATA/"rcaf_X_train.npy"))
X_test  = np.load(str(DATA/"rcaf_X_test.npy"))
y_train = np.load(str(DATA/"rcaf_y_train.npy"))
y_test  = np.load(str(DATA/"rcaf_y_test.npy"))

print("="*60)
print("RCAF ENGINE v2 — SUPERVISED HYBRID ENSEMBLE")
print("="*60)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Train attack ratio: {np.mean(y_train):.2%}")
print(f"Test  attack ratio: {np.mean(y_test):.2%}")

contamination = max(0.01, min(0.49, float(np.mean(y_train))))

# Component 1: Random Forest
print("\n[1/3] Random Forest (supervised)...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=2,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
rf_time = time.time() - t0
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
print(f"  Train time: {rf_time:.2f}s")
print(f"  RF F1:  {f1_score(y_test, rf_pred):.4f}")
print(f"  RF AUC: {roc_auc_score(y_test, rf_prob):.4f}")

# Component 2: Gradient Boosting
print("\n[2/3] Gradient Boosting (supervised)...")
t0 = time.time()
gb = GradientBoostingClassifier(
    n_estimators=60,
    max_depth=5,
    learning_rate=0.15,
    random_state=42
)
gb.fit(X_train, y_train)
gb_time = time.time() - t0
gb_pred = gb.predict(X_test)
gb_prob = gb.predict_proba(X_test)[:, 1]
print(f"  Train time: {gb_time:.2f}s")
print(f"  GB F1:  {f1_score(y_test, gb_pred):.4f}")
print(f"  GB AUC: {roc_auc_score(y_test, gb_prob):.4f}")

# Component 3: Isolation Forest (zero-day detection)
print("\n[3/3] Isolation Forest (unsupervised)...")
t0 = time.time()
iso = IsolationForest(
    n_estimators=100,
    contamination=contamination,
    random_state=42,
    n_jobs=2
)
iso.fit(X_train)
iso_time = time.time() - t0
iso_pred = (iso.predict(X_test) == -1).astype(float)
print(f"  Train time: {iso_time:.2f}s")
print(f"  IF F1:  {f1_score(y_test, iso_pred):.4f}")

# Weighted ensemble
print("\n  Ensemble: RF=0.45 + GB=0.45 + IF=0.10")
ens_prob = (0.45 * rf_prob +
            0.45 * gb_prob +
            0.10 * iso_pred)
y_pred = (ens_prob >= 0.5).astype(int)

# Metrics
acc  = float(accuracy_score(y_test, y_pred))
prec = float(precision_score(y_test, y_pred, zero_division=0))
rec  = float(recall_score(y_test, y_pred, zero_division=0))
f1   = float(f1_score(y_test, y_pred, zero_division=0))
auc  = float(roc_auc_score(y_test, ens_prob))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = float(fp / (fp + tn + 1e-9))
dr  = float(tp / (tp + fn + 1e-9))

# Latency benchmark
t0 = time.perf_counter()
for i in range(200):
    x = X_test[i:i+1]
    _ = (0.45 * rf.predict_proba(x)[0, 1] +
         0.45 * gb.predict_proba(x)[0, 1] +
         0.10 * float(iso.predict(x)[0] == -1))
lat = ((time.perf_counter() - t0) / 200) * 1000

print("\n  === RCAF v2 RESULTS ===")
print(f"  Accuracy:   {acc:.4f}")
print(f"  Precision:  {prec:.4f}")
print(f"  Recall/DR:  {rec:.4f}")
print(f"  F1 Score:   {f1:.4f}")
print(f"  ROC-AUC:    {auc:.4f}")
print(f"  FPR:        {fpr:.4f}")
print(f"  Latency:    {lat:.3f} ms/sample")
print("  =======================")

# Save models
for name, model in [("rf", rf), ("gb", gb), ("iso", iso)]:
    with open(str(MODELS / f"rcaf_v2_{name}.pkl"), "wb") as f:
        pickle.dump(model, f)
print(f"\n  Models saved to: {MODELS}")

# Save results
results = {
    "model": "RCAF-IoT v2 Hybrid Ensemble",
    "version": 2,
    "components": {
        "random_forest": {"weight": 0.45, "n_estimators": 100},
        "gradient_boosting": {"weight": 0.45, "n_estimators": 60},
        "isolation_forest": {"weight": 0.10, "n_estimators": 100}
    },
    "evaluation_metrics": {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "detection_rate": dr,
        "false_positive_rate": fpr,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "avg_latency_ms_per_sample": lat
    },
    "hardware": "Raspberry Pi 4B ARM Cortex-A72 @ 1.8GHz",
    "sensor": "Microsoft Kinect Xbox 360 Model 1473"
}

out = RESULTS / "rcaf_v2_results.json"
with open(str(out), "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {out}")

print("\n" + "="*60)
print("RCAF v2 COMPLETE")
print("="*60)
print(f"  F1:      {f1:.4f}")
print(f"  DR:      {dr:.4f}")
print(f"  FPR:     {fpr:.4f}")
print(f"  ROC-AUC: {auc:.4f}")
print(f"  Latency: {lat:.3f} ms")
print("="*60)
