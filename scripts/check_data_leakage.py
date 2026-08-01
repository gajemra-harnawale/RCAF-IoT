"""
Check for data leakage between train and test sets.
Critical before submission.
"""
import numpy as np
from pathlib import Path

DATA = Path("/home/sentinel/sentinel_iot/paper1/data")

X_train = np.load(str(DATA/"rcaf_X_train.npy"))
X_test  = np.load(str(DATA/"rcaf_X_test.npy"))
y_train = np.load(str(DATA/"rcaf_y_train.npy"))
y_test  = np.load(str(DATA/"rcaf_y_test.npy"))

print("="*60)
print("DATA LEAKAGE CHECK")
print("="*60)

# Check for identical rows between train and test
print("\nChecking for duplicate rows...")
duplicates = 0
# Sample check (full check would be too slow)
sample_size = min(1000, len(X_test))
for i in range(sample_size):
    matches = np.all(X_train == X_test[i], axis=1)
    if np.any(matches):
        duplicates += 1

leakage_rate = duplicates / sample_size
print(f"  Duplicates found: {duplicates}/{sample_size}")
print(f"  Leakage rate:     {leakage_rate:.2%}")

if leakage_rate > 0.01:
    print("\n  WARNING: Data leakage detected!")
    print("  This explains the very high F1 score.")
    print("  Action required: rebuild dataset with proper split")
else:
    print("\n  No significant leakage detected.")

# Check feature statistics
print("\nFeature distribution comparison:")
print(f"  Train mean: {np.mean(X_train):.6f}")
print(f"  Test  mean: {np.mean(X_test):.6f}")
print(f"  Train std:  {np.std(X_train):.6f}")
print(f"  Test  std:  {np.std(X_test):.6f}")

diff = abs(np.mean(X_train) - np.mean(X_test))
print(f"  Mean difference: {diff:.8f}")
if diff < 0.001:
    print("  WARNING: Train/test distributions nearly identical")
    print("  Likely cause: tiled depth frames")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("""
Option A (Quick): Disclose limitation clearly in paper
  'The depth feature component uses tiled simulation
   frames. Real Kinect recordings are planned for
   future validation.'

Option B (Better): Record real Kinect data
  - 30 minutes normal activity
  - 10 minutes anomalous activity
  - Eliminates leakage concern entirely

Option C (Best): Both A and B
""")
