import pandas as pd
import os

file_path = "Quantum_Results_Full.xlsx"
if os.path.exists(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # Check for duplicates
    if df.columns.duplicated().any():
        print("Duplicate columns found!")
        print(df.columns[df.columns.duplicated()].tolist())
        
        # Remove duplicates (keep first)
        df = df.loc[:, ~df.columns.duplicated()]
        print("Duplicates removed.")
        
        df.to_excel(file_path, index=False)
        print("Saved fixed file.")
    else:
        print("No duplicate columns found.")
        # But maybe 'RSI_9' and 'RSI_9 ' (whitespace)?
        # Strip whitespace was done in load_and_prepare, but saving might preserve?
        # Let's check matching names
        cols = df.columns.tolist()
        rsi_cols = [c for c in cols if 'RSI_9' in c]
        print(f"RSI related columns: {rsi_cols}")
        
else:
    print("File not found.")
