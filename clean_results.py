import pandas as pd
import os

file_path = "Quantum_Results_Full.xlsx"
if os.path.exists(file_path):
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    
    # Check if Q_RSA_Signal exists and if it has valid data
    if 'Q_RSA_Signal' in df.columns:
        print("Q_RSA_Signal found.")
        print("Unique values:", df['Q_RSA_Signal'].unique())
        
        # If it looks like placeholder (e.g. all 0 or NaN), we should drop it
        # But wait, 0 and 1 are valid. 
        # If it was never run, maybe it's not there? 
        # But the script said "Skipping Q_RSA (Already computed)".
        # So it MUST be there.
    
    # Check Q_4_Ensemble_Signal
    if 'Q_4_Ensemble_Signal' in df.columns:
        print("Q_4_Ensemble_Signal found.")
        print("Unique values:", df['Q_4_Ensemble_Signal'].unique())

    # We want to keep: 
    # Q_WaveTrend_Signal, Q_CCI_Signal, Q_ADX_Signal
    # And drop: Q_RSA_Signal, Q_4_Ensemble_Signal, Q_Master_9_Signal 
    # (and related columns like _Strategy_Ret, _Cum_Ret)
    
    models_to_reset = ['Q_RSA', 'Q_4_Ensemble', 'Q_Master_9']
    cols_to_drop = []
    
    for model in models_to_reset:
        cols_to_drop.extend([c for c in df.columns if c.startswith(model)])
        
    print(f"Dropping columns: {cols_to_drop}")
    
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        df.to_excel(file_path, index=False)
        print("Saved cleaned file.")
    else:
        print("No columns to drop.")

else:
    print("File not found.")
