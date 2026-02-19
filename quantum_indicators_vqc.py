import numpy as np
import pandas as pd
import ta
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class QuantumIndicatorSystem:
    def __init__(self, data_path: str, future_bars=10):
        """
        Initialize the Quantum Indicator System
        """
        self.data_path = data_path
        self.future_bars = future_bars
        self.df = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_and_prepare_data(self):
        """
        Load data and compute classical indicators
        """
        # Load Data
        import os
        results_file = "Quantum_Results_Full.xlsx"
        
        if os.path.exists(results_file):
            print(f"Found existing results file: {results_file}. Loading...")
            self.df = pd.read_excel(results_file, engine='openpyxl')
            
            # Force numeric conversion for all columns (except maybe Timestamp?)
            # Identify numeric cols
            cols = self.df.columns
            # Exclude timestamp if present (usually index or named Timestamp)
            # Assuming all columns are numeric for VQC
            self.df = self.df.apply(pd.to_numeric, errors='coerce')
            
            # Ensure no NaNs/Infs anywhere
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Ensure no NaNs/Infs anywhere
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.df.dropna(inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            
            # Ensure index reset if needed
            if 'Timestamp' in self.df.columns:
                 # Check if we need to parse dates
                 pass
        else:
            if self.data_path.endswith('.xlsx'):
                self.df = pd.read_excel(self.data_path, engine='openpyxl')
            elif self.data_path.endswith('.csv'):
                self.df = pd.read_csv(self.data_path)
                
            # Clean column names
            self.df.columns = self.df.columns.str.strip()
            
            # Ensure timestamp handling if needed, but we focus on numerical cols
            # Compute Classical Indicators
            
            # 1. WaveTrend (WT1, WT2)
            # Approximate WaveTrend calculation
            n1 = 10
            n2 = 21
            ap = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
            esa = ap.ewm(span=n1, adjust=False).mean()
            d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
            ci = (ap - esa) / (0.015 * d)
            self.df['WT1'] = ci.ewm(span=n2, adjust=False).mean()
            self.df['WT2'] = self.df['WT1'].rolling(4).mean()
            
            # 2. CCI (20)
            self.df['CCI_20'] = ta.trend.cci(self.df['High'], self.df['Low'], self.df['Close'], window=20)
            
            # 3. ADX (20), +DI, -DI
            self.df['ADX_20'] = ta.trend.adx(self.df['High'], self.df['Low'], self.df['Close'], window=20)
            self.df['+DI_20'] = ta.trend.adx_pos(self.df['High'], self.df['Low'], self.df['Close'], window=20)
            self.df['-DI_20'] = ta.trend.adx_neg(self.df['High'], self.df['Low'], self.df['Close'], window=20)
            
            # 4. RSI (9) for Q-RSA (and 14 for Ensemble)
            self.df['RSI_9'] = ta.momentum.rsi(self.df['Close'], window=9)
            self.df['RSI_14'] = ta.momentum.rsi(self.df['Close'], window=14)
            
            # Volatility (for Ensemble) - using ATR or simple volatility
            self.df['Volatility'] = (self.df['High'] - self.df['Low']) / self.df['Close']
            
            # Compute Target: 1 if Price[t+10] > Price[t], else 0
            self.df['Future_Return'] = self.df['Close'].shift(-self.future_bars) - self.df['Close']
            self.df['Target'] = (self.df['Future_Return'] > 0).astype(int)
            
            # Drop NaNs created by indicators
            self.df.dropna(inplace=True)
            self.df.reset_index(drop=True, inplace=True)
        
        return self.df

    def build_vqc_model(self, num_features):
        """
        Construct the Variable Quantum Classifier
        """
        # Feature Map: ZFeatureMap for 1 feature, ZZFeatureMap for >1
        if num_features == 1:
            feature_map = ZFeatureMap(feature_dimension=num_features, reps=2)
        else:
            feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
            
        # Ansatz
        ansatz = RealAmplitudes(num_qubits=num_features, reps=2)
        
        # Optimizer
        optimizer = COBYLA(maxiter=50) # Reduced iter for speed, 75 is okay but 50 is faster validation
        
        # Sampler
        # Try to use Aer for speed if available
        try:
            from qiskit_aer.primitives import Sampler as AerSampler
            sampler = AerSampler(run_options={"shots": 256})
            print("Using AerSampler for faster outcomes (Shots: 256).")
        except ImportError:
            print("Aer not found, using Reference Sampler (Slow).")
            sampler = Sampler()

        # VQC
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=sampler
        )
        
        return vqc, feature_map, ansatz

    def train_and_predict(self, feature_cols, model_name="Model", train_size=500):
        """
        Train on first `train_size` samples, predict on ALL samples.
        """
        # Check if already computed
        if f'{model_name}_Signal' in self.df.columns:
            print(f"Skipping {model_name} (Already computed).")
            return None, None

        print(f"\n--- Processing {model_name} ---")
        X = self.df[feature_cols].values
        # Force integer type for Classification
        y = self.df['Target'].astype(int).values
        
        # Scale Data (Crucial for Quantum)
        X_scaled = self.scaler.fit_transform(X)
        
        # Find a valid training window with both classes
        X_train = None
        y_train = None
        
        for start_idx in range(0, len(y) - train_size, 100):
            subset_y = y[start_idx : start_idx + train_size]
            if len(np.unique(subset_y)) > 1:
                # Check for at least minimal balance (e.g., > 10 samples of minority class)
                counts = np.bincount(subset_y)
                if np.min(counts) > 10:
                    print(f"Found valid training window at index {start_idx} (Counts: {counts})")
                    X_train = X_scaled[start_idx : start_idx + train_size]
                    y_train = subset_y
                    break
        
        if X_train is None:
            print("Warning: Could not find acceptable training window with both classes. Using first window (may fail).")
            X_train = X_scaled[:train_size]
            y_train = y[:train_size]
        
        X_all = X_scaled # Predict on everything
        
        # Build Model
        num_features = len(feature_cols)
        print(f"Building VQC for {model_name} with {num_features} features.")
        vqc, fm, ansatz = self.build_vqc_model(num_features)
        
        # Train
        print(f"Training VQC on {train_size} samples...")
        # print(f"X_train shape: {X_train.shape}")
        import time
        start_time = time.time()
        try:
            vqc.fit(X_train, y_train)
        except Exception as e:
            print(f"Fit Error: {e}")
            # If fit fails, we skip this model to allow others to run
            return None, None
        print(f"Training completed in {time.time() - start_time:.2f}s")
        
        # Predict on ALL data
        print(f"Predicting on full dataset ({len(X_all)} samples)...")
        
        try:
            # Predict in chunks to avoid memory issues/timeouts and show progress
            chunk_size = 2000 
            predictions = []
            total_chunks = (len(X_all) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(X_all), chunk_size):
                 chunk = X_all[i:i+chunk_size]
                 print(f"  Predicting chunk {i//chunk_size + 1}/{total_chunks}...", end='\r')
                 chunk_preds = vqc.predict(chunk)
                 predictions.extend(chunk_preds)
            
            print(f"\nPrediction completed.")
            y_pred = np.array(predictions)
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None, None
            
        # Calculate Strategy Returns
        self.df[f'{model_name}_Signal'] = y_pred
        
        # Return = (Close[t+1] - Close[t])/Close[t] * Signal[t]
        self.df['Log_Ret'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df[f'{model_name}_Strategy_Ret'] = self.df['Log_Ret'] * self.df[f'{model_name}_Signal'].shift(1)
        
        # Cumulative Returns
        self.df[f'{model_name}_Cum_Ret'] = self.df[f'{model_name}_Strategy_Ret'].cumsum().apply(np.exp)
        
        # Metrics using the 'Total' logic
        total_ret = self.df[f'{model_name}_Strategy_Ret'].sum()
        net_ret_pct = (np.exp(total_ret) - 1) * 100
        
        print(f"{model_name} Net Return: {net_ret_pct:.2f}%")
        
        # Save Intermediate Results
        self.df.to_excel("Quantum_Results_Full.xlsx", index=False)
        print(f"Saved intermediate results to Quantum_Results_Full.xlsx")
        
        return vqc, fm

    def run_all_models(self):
        self.load_and_prepare_data()
        
        # 1. Q-WaveTrend (2 features)
        self.train_and_predict(['WT1', 'WT2'], "Q_WaveTrend")
        
        # 2. Q-CCI (1 feature)
        self.train_and_predict(['CCI_20'], "Q_CCI")
        
        # 3. Q-ADX (3 features)
        self.train_and_predict(['ADX_20', '+DI_20', '-DI_20'], "Q_ADX")
        
        # 4. Q-RSA (1 feature - RSI 9)
        self.train_and_predict(['RSI_9'], "Q_RSA")
        
        # 5. 4-Feature Ensemble
        ensemble_features = ['RSI_14', 'Volatility', 'CCI_20', 'ADX_20'] # Approx mapping
        self.train_and_predict(ensemble_features, "Q_4_Ensemble")
        
        # 6. 9-Feature Master Model
        # All unique features used so far
        master_features = ['WT1', 'WT2', 'CCI_20', 'ADX_20', '+DI_20', '-DI_20', 'RSI_9', 'RSI_14', 'Volatility']
        self.train_and_predict(master_features, "Q_Master_9")
        
        # Save results
        self.df.to_excel("Quantum_Results_Full.xlsx")
        print("Results saved to Quantum_Results_Full.xlsx")

if __name__ == "__main__":
    # Adjust filename if needed
    system = QuantumIndicatorSystem('Sidebar_Merged (1).xlsx')
    system.run_all_models()
