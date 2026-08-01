"""
Dataset Preprocessor for Paper 1 (RCAF-IoT)
Produces 48-dim fused feature vectors (32 depth + 16 network)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE     = Path("/home/sentinel/sentinel_iot")
KDD_PATH = BASE / "shared/datasets/kdd99/kddcup99.csv"
DEPTH_DIR= BASE / "shared/datasets/simulation_frames"
OUT_DIR  = BASE / "paper1/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("RCAF DATASET PREPROCESSOR")
print("="*60)

# ----------------------------------------------------------
# 1. Load KDD99
# ----------------------------------------------------------
print("\n[1/3] Loading KDD99 dataset...")

COL_NAMES = [
    "duration","protocol_type","service","flag",
    "src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label"
]

df = pd.read_csv(KDD_PATH, header=None, names=COL_NAMES)
print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

# Encode categorical columns by NAME (not integer index)
for col in ["protocol_type", "service", "flag"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Binary label
df["binary_label"] = (df["label"] != "normal.").astype(int)

feature_cols = COL_NAMES[:-1]   # all except "label"
X_kdd = df[feature_cols].astype(np.float32).values
y_kdd = df["binary_label"].values

print(f"  KDD99 shape:  {X_kdd.shape}")
print(f"  Attack ratio: {np.mean(y_kdd):.2%}")

# Take first 16 columns as RGB/network representation
X_kdd_16 = X_kdd[:, :16]

# ----------------------------------------------------------
# 2. Load Depth Simulation Frames
# ----------------------------------------------------------
print("\n[2/3] Loading synthetic depth frames...")

def extract_depth_features(depth_array):
    """Extract 32 statistical depth features from one frame."""
    feats = []
    feats.append(float(np.mean(depth_array)))
    feats.append(float(np.std(depth_array)))
    feats.append(float(np.min(depth_array)))
    feats.append(float(np.max(depth_array)))
    for p in [5,10,25,50,75,90,95]:
        feats.append(float(np.percentile(depth_array, p)))
    # Quadrant means
    h, w = depth_array.shape
    feats.append(float(np.mean(depth_array[:h//2, :w//2])))
    feats.append(float(np.mean(depth_array[:h//2, w//2:])))
    feats.append(float(np.mean(depth_array[h//2:, :w//2])))
    feats.append(float(np.mean(depth_array[h//2:, w//2:])))
    # Near/far pixel counts
    near = float(np.sum(depth_array < 1500))
    far  = float(np.sum(depth_array > 3000))
    feats.append(near)
    feats.append(far)
    feats.append(near / (depth_array.size + 1e-9))
    # Gradient magnitude mean
    gx = np.diff(depth_array.astype(np.float32), axis=1)
    gy = np.diff(depth_array.astype(np.float32), axis=0)
    feats.append(float(np.mean(np.abs(gx))))
    feats.append(float(np.mean(np.abs(gy))))
    feats.append(float(np.std(gx)))
    feats.append(float(np.std(gy)))
    # Pad to exactly 32
    while len(feats) < 32:
        feats.append(0.0)
    return np.array(feats[:32], dtype=np.float32)

depth_features = []
depth_labels   = []

for label_name, label_val in [("normal", 0), ("anomalous", 1)]:
    folder = DEPTH_DIR / label_name
    files  = sorted(folder.glob("depth_*.npy"))
    for f in files:
        depth = np.load(str(f))
        depth_features.append(extract_depth_features(depth))
        depth_labels.append(label_val)
    print(f"  {label_name}: {len(files)} frames loaded")

X_depth = np.array(depth_features, dtype=np.float32)
y_depth = np.array(depth_labels,   dtype=np.int32)
print(f"  Depth feature matrix: {X_depth.shape}")

# ----------------------------------------------------------
# 3. Fuse Features (depth 32 + network 16 = 48)
# ----------------------------------------------------------
print("\n[3/3] Fusing features (48-dim)...")

# Repeat depth frames to match KDD99 size
n_kdd   = len(X_kdd_16)
n_depth = len(X_depth)

# Tile depth data to match KDD size
repeats    = (n_kdd // n_depth) + 1
X_depth_r  = np.tile(X_depth, (repeats, 1))[:n_kdd]
y_depth_r  = np.tile(y_depth, repeats)[:n_kdd]

# Final label: anomaly if EITHER source says anomaly
y_final = np.logical_or(y_depth_r, y_kdd).astype(np.int32)

# Fuse: depth(32) weighted x2 as per RCAF design
X_depth_w = X_depth_r * 2.0   # depth context weight
X_fused   = np.hstack([X_depth_w, X_kdd_16])

print(f"  Fused shape:   {X_fused.shape}  (32 depth + 16 network)")
print(f"  Attack ratio:  {np.mean(y_final):.2%}")

# Scale
scaler  = StandardScaler()
X_fused = scaler.fit_transform(X_fused).astype(np.float32)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_fused, y_final, test_size=0.2, random_state=42, stratify=y_final
)

print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
print(f"  Train attack ratio: {np.mean(y_train):.2%}")
print(f"  Test  attack ratio: {np.mean(y_test):.2%}")

# Save
for fname, arr in [
    ("rcaf_X_train.npy", X_train),
    ("rcaf_X_test.npy",  X_test),
    ("rcaf_y_train.npy", y_train),
    ("rcaf_y_test.npy",  y_test),
]:
    np.save(str(OUT_DIR / fname), arr)
    print(f"  Saved: {OUT_DIR / fname}")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE")
print("="*60)
