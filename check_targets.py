import pandas as pd
import numpy as np
import os

file_path = "Quantum_Results_Full.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Compute Target again to be sure (it should be in file though)
if 'Target' in df.columns:
    y = df['Target'].values
    print(f"Full Target unique values: {np.unique(y)}")
    print(f"Full Target counts: {np.bincount(y)}")
    
    # Check sliding window logic manually
    train_size = 500
    found = False
    for start_idx in range(0, len(y) - train_size, 100):
        subset = y[start_idx : start_idx + train_size]
        if len(np.unique(subset)) > 1:
            counts = np.bincount(subset)
            if np.min(counts) > 10:
                print(f"Found valid window at {start_idx}: {counts}")
                found = True
                break
    if not found:
        print("No valid window found in entire dataset!")
else:
    print("Target column not found.")
