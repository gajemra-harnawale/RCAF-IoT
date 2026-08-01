"""
5-Fold Cross Validation for RCAF-IoT v2
Reports mean +/- std for all metrics
Required for statistical validity in IEEE papers
"""
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              IsolationForest)
from sklearn.metrics import (f1_score, recall_score,
                             precision_score, roc_auc_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

DATA    = Path("/home/sentinel/sentinel_iot/paper1/data")
RESULTS = Path("/home/sentinel/sentinel_iot/paper1/results")

X = np.vstack([
    np.load(str(DATA/"rcaf_X_train.npy")),
    np.load(str(DATA/"rcaf_X_test.npy"))
])
y = np.concatenate([
    np.load(str(DATA/"rcaf_y_train.npy")),
    np.load(str(DATA/"rcaf_y_test.npy"))
])

print("="*60)
print("5-FOLD CROSS VALIDATION — RCAF-IoT v2")
print(f"Total samples: {len(X)}")
print(f"Attack ratio:  {np.mean(y):.2%}")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n  Fold {fold}/5...", end=" ", flush=True)

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    contamination = max(0.01, min(0.49, float(np.mean(y_tr))))

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=12,
        random_state=fold, n_jobs=2,
        class_weight="balanced"
    )
    rf.fit(X_tr, y_tr)

    gb = GradientBoostingClassifier(
        n_estimators=60, max_depth=5,
        learning_rate=0.15, random_state=fold
    )
    gb.fit(X_tr, y_tr)

    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=fold, n_jobs=2
    )
    iso.fit(X_tr)

    ens = (0.45 * rf.predict_proba(X_te)[:, 1] +
           0.45 * gb.predict_proba(X_te)[:, 1] +
           0.10 * (iso.predict(X_te) == -1).astype(float))
    yp = (ens >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, yp).ravel()
    m = {
        "fold":      fold,
        "f1":        float(f1_score(y_te, yp, zero_division=0)),
        "precision": float(precision_score(y_te, yp, zero_division=0)),
        "recall":    float(recall_score(y_te, yp, zero_division=0)),
        "auc":       float(roc_auc_score(y_te, ens)),
        "fpr":       float(fp / (fp + tn + 1e-9)),
        "dr":        float(tp / (tp + fn + 1e-9)),
    }
    fold_metrics.append(m)
    print(f"F1={m['f1']:.4f}  "
          f"DR={m['dr']:.4f}  "
          f"FPR={m['fpr']:.4f}  "
          f"AUC={m['auc']:.4f}")

# Summary statistics
print("\n" + "="*60)
print("CROSS-VALIDATION SUMMARY (mean +/- std)")
print("="*60)

summary_stats = {}
for metric in ["f1", "precision", "recall", "auc", "fpr", "dr"]:
    vals = [m[metric] for m in fold_metrics]
    mean = float(np.mean(vals))
    std  = float(np.std(vals))
    mn   = float(np.min(vals))
    mx   = float(np.max(vals))
    summary_stats[metric] = {
        "mean": mean, "std": std,
        "min": mn, "max": mx
    }
    print(f"  {metric.upper():<12} "
          f"{mean:.4f} +/- {std:.4f}  "
          f"[{mn:.4f} - {mx:.4f}]")

print("="*60)

# Save
output = {
    "method": "RCAF-IoT v2",
    "validation": "5-fold stratified cross-validation",
    "total_samples": int(len(X)),
    "attack_ratio": float(np.mean(y)),
    "fold_results": fold_metrics,
    "summary": summary_stats
}

out = RESULTS / "cross_validation_results.json"
with open(str(out), "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out}")
print("="*60)
