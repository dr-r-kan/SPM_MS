"""Test script to inspect MetaMaps_2023_06.set structure"""

from scipy.io import loadmat
import numpy as np
from pathlib import Path

set_file = Path("e:/EEGs/SPM_MS/MetaMaps_2023_06.set")

print(f"Loading: {set_file}")
print(f"File exists: {set_file.exists()}")
print(f"File size: {set_file.stat().st_size} bytes")

# Load the .set file
mat_data = loadmat(str(set_file), squeeze_me=False)

print("\n=== Top-level keys in .set file ===")
for key in mat_data.keys():
    if not key.startswith('__'):
        print(f"{key}: {type(mat_data[key])}, shape: {getattr(mat_data[key], 'shape', 'N/A')}")

# Extract the data directly (it's at top level, not in EEG sub-structure)
data = mat_data['data']
nbchan = mat_data['nbchan'][0, 0]
pnts = mat_data['pnts'][0, 0]
chanlocs = mat_data['chanlocs']

print(f"\n=== Data Analysis ===")
print(f"data shape: {data.shape}")
print(f"data type: {data.dtype}")
print(f"This represents: {data.shape[0]} channels x {data.shape[1]} samples")
print(f"nbchan: {nbchan}")
print(f"pnts (n_samples/templates): {pnts}")
print(f"\nBased on user info, samples = microstate templates")
print(f"Template structure: K=4,5,6,7 means 4+5+6+7 = 22 total templates")
print(f"Last 7 templates (indices -7:) are the K=7 templates (ABCDEFG)")

# Extract the last 7 templates
last_7 = data[:, -7:]
print(f"\n=== Last 7 templates (K=7, ABCDEFG) ===")
print(f"Shape: {last_7.shape}")
print(f"Mean absolute value per template:")
for i in range(7):
    label = chr(65 + i)  # A, B, C, D, E, F, G
    mean_abs = np.mean(np.abs(last_7[:, i]))
    max_abs = np.max(np.abs(last_7[:, i]))
    print(f"  {label}: mean={mean_abs:.4f}, max={max_abs:.4f}")

# Check chanlocs
print(f"\n=== Channel locations ===")
print(f"chanlocs shape: {chanlocs.shape}")
if chanlocs.size > 0:
    print(f"chanlocs dtype: {chanlocs.dtype.names}")
    
    # Extract channel labels
    labels = []
    for i in range(min(5, chanlocs.shape[1])):  # First 5 channels
        if 'labels' in chanlocs.dtype.names:
            label = chanlocs[0, i]['labels']
            if isinstance(label, np.ndarray) and label.size > 0:
                label = label[0] if label.ndim > 0 else str(label)
            labels.append(str(label))
    print(f"First 5 channel labels: {labels}")

print("\n=== Done ===")
