"""
Baseline comparison for Paper 1.
Compares RCAF-IoT v2 against standard anomaly detectors.
"""
import numpy as np
import json
import time
from pathlib import Path
from sklearn.ensemble import (IsolationForest,
                              RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (f1_score, accuracy_score,
                             precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

DATA    = Path("/home/sentinel/sentinel_iot/paper1/data")
RESULTS = Path("/home/sentinel/sentinel_iot/paper1/results")

X_train = np.load(str(DATA/"rcaf_X_train.npy"))
X_test  = np.load(str(DATA/"rcaf_X_test.npy"))
y_train = np.load(str(DATA/"rcaf_y_train.npy"))
y_test  = np.load(str(DATA/"rcaf_y_test.npy"))

print("="*60)
print("BASELINE COMPARISON — RCAF-IoT v2 vs Baselines")
print("="*60)
print(f"Test samples: {len(X_test)}")
print(f"Attack ratio: {np.mean(y_test):.2%}")

contamination = max(0.01, min(0.49, float(np.mean(y_train))))
results = []

def evaluate(name, y_pred, y_prob, lat_ms):
    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    f1   = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = 0.0
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = float(fp / (fp + tn + 1e-9))
    dr  = float(tp / (tp + fn + 1e-9))
    print(f"  {name:<30} "
          f"F1={f1:.4f}  DR={dr:.4f}  "
          f"FPR={fpr:.4f}  AUC={auc:.4f}  "
          f"Lat={lat_ms:.2f}ms")
    return {
        "method": name,
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "roc_auc": auc, "detection_rate": dr,
        "false_positive_rate": fpr,
        "latency_ms": lat_ms
    }

# Baseline 1: Isolation Forest only
print("\n[1] Isolation Forest (standalone)...")
iso = IsolationForest(n_estimators=100,
                      contamination=contamination,
                      random_state=42, n_jobs=2)
iso.fit(X_train)
t0 = time.perf_counter()
ip = (iso.predict(X_test) == -1).astype(int)
lat = ((time.perf_counter()-t0)/len(X_test))*1000
results.append(evaluate("Isolation Forest", ip, ip.astype(float), lat))

# Baseline 2: One-Class SVM
print("\n[2] One-Class SVM...")
X_norm = X_train[y_train == 0][:2000]
svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
svm.fit(X_norm)
t0 = time.perf_counter()
sp = (svm.predict(X_test) == -1).astype(int)
lat = ((time.perf_counter()-t0)/len(X_test))*1000
results.append(evaluate("One-Class SVM", sp, sp.astype(float), lat))

# Baseline 3: Local Outlier Factor
print("\n[3] Local Outlier Factor...")
lof = LocalOutlierFactor(n_neighbors=20,
                          contamination=contamination,
                          novelty=True, n_jobs=2)
lof.fit(X_train[y_train == 0][:3000])
t0 = time.perf_counter()
lp = (lof.predict(X_test) == -1).astype(int)
lat = ((time.perf_counter()-t0)/len(X_test))*1000
results.append(evaluate("Local Outlier Factor", lp,
                         lp.astype(float), lat))

# Baseline 4: Random Forest only
print("\n[4] Random Forest (supervised)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42,
                             n_jobs=2, class_weight="balanced")
rf.fit(X_train, y_train)
t0 = time.perf_counter()
rp = rf.predict(X_test)
rprob = rf.predict_proba(X_test)[:, 1]
lat = ((time.perf_counter()-t0)/len(X_test))*1000
results.append(evaluate("Random Forest", rp, rprob, lat))

# Baseline 5: Gradient Boosting only
print("\n[5] Gradient Boosting (supervised)...")
gb = GradientBoostingClassifier(n_estimators=60, max_depth=5,
                                 learning_rate=0.15, random_state=42)
gb.fit(X_train, y_train)
t0 = time.perf_counter()
gp = gb.predict(X_test)
gprob = gb.predict_proba(X_test)[:, 1]
lat = ((time.perf_counter()-t0)/len(X_test))*1000
results.append(evaluate("Gradient Boosting", gp, gprob, lat))

# RCAF-IoT v2 (This Work)
print("\n[6] RCAF-IoT v2 (This Work)...")
iso2 = IsolationForest(n_estimators=100,
                        contamination=contamination,
                        random_state=42, n_jobs=2)
iso2.fit(X_train)
t0 = time.perf_counter()
ens_prob = (0.45 * rf.predict_proba(X_test)[:, 1] +
            0.45 * gb.predict_proba(X_test)[:, 1] +
            0.10 * (iso2.predict(X_test) == -1).astype(float))
ep = (ens_prob >= 0.5).astype(int)
lat = ((time.perf_counter()-t0)/len(X_test))*1000
results.append(evaluate("RCAF-IoT v2 (This Work)",
                         ep, ens_prob, lat))

# Summary table
print("\n" + "="*70)
print("COMPARISON TABLE")
print("="*70)
print(f"  {'Method':<30} {'F1':>7} {'DR':>7} "
      f"{'FPR':>7} {'AUC':>7} {'ms':>7}")
print("  " + "-"*65)
for r in results:
    marker = " ←" if "This Work" in r["method"] else ""
    print(f"  {r['method']:<30} "
          f"{r['f1']:>7.4f} "
          f"{r['detection_rate']:>7.4f} "
          f"{r['false_positive_rate']:>7.4f} "
          f"{r['roc_auc']:>7.4f} "
          f"{r['latency_ms']:>7.3f}"
          f"{marker}")
print("="*70)

out = RESULTS / "baseline_comparison.json"
with open(str(out), "w") as f:
    json.dump({"comparison": results}, f, indent=2)
print(f"\nSaved: {out}")
