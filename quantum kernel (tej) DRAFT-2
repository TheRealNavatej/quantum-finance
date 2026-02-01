import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class LorentzianKNN:
    def __init__(self, k_neighbors=8, lookback=500, sample_interval=4, 
                 regression_window=250, future_bars=4, use_lasso=True):
        """
        Initialize Lorentzian KNN Trading Algorithm with Quantum Kernels
        
        Parameters:
        - k_neighbors: Number of nearest neighbors (K)
        - lookback: Historical window for KNN search
        - sample_interval: Sampling interval for historical data
        - regression_window: Rolling window for regression
        - future_bars: Number of bars ahead for prediction
        - use_lasso: Placeholder for compatibility (quantum methods used instead)
        """
        self.k = k_neighbors
        self.lookback = lookback
        self.sample_interval = sample_interval
        self.regression_window = regression_window
        self.future_bars = future_bars
        
        # Feature scaler for quantum encoding
        self.scaler = StandardScaler()
        
        # Quantum Feature Maps
        self.n_qubits = 5  # One qubit per feature
        
        # ZZ Feature Map for classification (creates entanglement)
        self.feature_map_classifier = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=2,
            entanglement='linear'
        )
        
        # Pauli Feature Map for regression (different encoding strategy)
        self.feature_map_regressor = PauliFeatureMap(
            feature_dimension=self.n_qubits,
            reps=2,
            paulis=['Z', 'ZZ'],
            entanglement='linear'
        )
        
        # Quantum Kernels using Aer Simulator for efficiency
        self.backend = AerSimulator()
        self.sampler = Sampler()
        
        self.quantum_kernel_classifier = FidelityQuantumKernel(
            feature_map=self.feature_map_classifier,
            fidelity=ComputeUncompute(sampler=self.sampler)
        )
        
        self.quantum_kernel_regressor = FidelityQuantumKernel(
            feature_map=self.feature_map_regressor,
            fidelity=ComputeUncompute(sampler=self.sampler)
        )
        
        # Quantum Kernel SVM for classification (replaces XGBoost)
        self.knn_classifier = SVC(
            kernel='precomputed',
            C=1.0,
            probability=True,
            random_state=42
        )
        
        # Quantum Kernel SVR for regression (replaces quantile regression)
        # We'll use 3 separate models for quantiles
        self.regression_model = SVR(kernel='precomputed', C=1.0, epsilon=0.1)
        self.regression_model_q10 = SVR(kernel='precomputed', C=1.0, epsilon=0.05)
        self.regression_model_q90 = SVR(kernel='precomputed', C=1.0, epsilon=0.05)
        
        # Cache for quantum kernel matrices (expensive to compute)
        self.kernel_cache_classifier = {}
        self.kernel_cache_regressor = {}
        
        # Training data storage for kernel computation
        self.X_train_classifier = None
        self.X_train_regressor = None
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 5-dimensional feature vector from OHLCV data
        """
        data = df.copy()
        
        # Feature 1: Price Change (within bar)
        data['price_change'] = (data['Close'] - data['Open']) / data['Open']
        
        # Feature 2: Bar Volatility (range as % of close)
        data['bar_volatility'] = (data['High'] - data['Low']) / data['Close']
        
        # Feature 3: 1-Bar Return
        data['return_1bar'] = data['Close'].pct_change(1)
        
        # Feature 4: 2-Bar Return
        data['return_2bar'] = data['Close'].pct_change(2)
        
        # Feature 5: Volume Change Ratio
        data['volume_ratio'] = data['Volume'] / data['Volume'].shift(1)
        
        return data
    
    def compute_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute prediction targets
        """
        data = df.copy()
        
        # Continuous Target: 4-bar future return
        data['continuous_target'] = (data['Close'].shift(-self.future_bars) / data['Close']) - 1
        
        # Directional Target: 1 if up, -1 if down
        data['directional_target'] = np.where(
            data['Close'].shift(-self.future_bars) > data['Close'], 1, -1
        )
        
        return data
    
    def lorentzian_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate quantum kernel similarity (replaces Lorentzian distance)
        Uses quantum state fidelity as similarity measure
        """
        x1 = np.asarray(x1, dtype=np.float64).reshape(1, -1)
        x2 = np.asarray(x2, dtype=np.float64).reshape(1, -1)
        
        # Normalize features for quantum encoding
        x1_normalized = (x1 - x1.mean()) / (x1.std() + 1e-8)
        x2_normalized = (x2 - x2.mean()) / (x2.std() + 1e-8)
        
        # Clip to prevent encoding errors
        x1_normalized = np.clip(x1_normalized, -np.pi, np.pi)
        x2_normalized = np.clip(x2_normalized, -np.pi, np.pi)
        
        try:
            # Compute quantum kernel (fidelity between quantum states)
            X_pair = np.vstack([x1_normalized, x2_normalized])
            kernel_matrix = self.quantum_kernel_classifier.evaluate(X_pair)
            
            # Fidelity is in [0,1], convert to distance in [0,2]
            # Higher fidelity = lower distance
            fidelity = kernel_matrix[0, 1]
            distance = 2.0 * (1.0 - fidelity)
            
            return distance
        except:
            # Fallback to classical Lorentzian if quantum fails
            return np.sum(np.log(1 + np.abs(x1 - x2)))
    
    def knn_predict(self, current_features: np.ndarray, historical_data: pd.DataFrame, 
                    current_idx: int) -> int:
        """
        Generate prediction using Quantum Kernel SVM (replaces KNN)
        """
        # Define lookback window
        start_idx = max(0, current_idx - self.lookback)
        end_idx = current_idx
        
        # Sample historical data
        sample_indices = list(range(start_idx, end_idx, self.sample_interval))
        if not sample_indices:
            return 0
        
        feature_cols = ['price_change', 'bar_volatility', 'return_1bar', 
                       'return_2bar', 'volume_ratio']
        
        # Collect training data
        X_train = []
        y_train = []
        
        for idx in sample_indices:
            if idx >= len(historical_data) or pd.isna(historical_data.iloc[idx][feature_cols]).any():
                continue
            if pd.isna(historical_data.iloc[idx]['directional_target']):
                continue
                
            hist_features = historical_data.iloc[idx][feature_cols].values
            X_train.append(hist_features)
            y_train.append(historical_data.iloc[idx]['directional_target'])
        
        if len(X_train) < self.k:
            return 0
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Normalize features for quantum encoding
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = np.clip(X_train_scaled, -np.pi, np.pi)
        
        current_features_scaled = self.scaler.transform(current_features.reshape(1, -1))
        current_features_scaled = np.clip(current_features_scaled, -np.pi, np.pi)
        
        # Convert -1/1 labels to 0/1 for binary classification
        y_train_binary = ((y_train + 1) / 2).astype(int)
        
        try:
            # Compute quantum kernel matrix for training data
            K_train = self.quantum_kernel_classifier.evaluate(X_train_scaled)
            
            # Compute quantum kernel between test point and training data
            X_combined = np.vstack([X_train_scaled, current_features_scaled])
            K_combined = self.quantum_kernel_classifier.evaluate(X_combined)
            K_test = K_combined[-1, :-1].reshape(1, -1)
            
            # Train Quantum Kernel SVM
            self.knn_classifier.fit(K_train, y_train_binary)
            
            # Predict using quantum kernel
            pred_proba = self.knn_classifier.predict_proba(K_test)[0]
            
            # Convert back to -1/0/1 format
            if pred_proba[1] > 0.55:  # Strong positive
                return 1
            elif pred_proba[0] > 0.55:  # Strong negative
                return -1
            else:
                return 0
        except Exception as e:
            # Fallback to simple majority vote if quantum computation fails
            if np.mean(y_train) > 0.1:
                return 1
            elif np.mean(y_train) < -0.1:
                return -1
            return 0
    
    def kernel_regression(self, y: np.ndarray, bandwidth: float, 
                         kernel_type: str = 'gaussian') -> np.ndarray:
        """
        HMM-based probabilistic smoothing (replaces Nadaraya-Watson)
        """
        n = len(y)
        yhat = np.zeros(n)
        
        # Use HMM for probabilistic smoothing
        n_states = min(3, max(2, int(bandwidth / 2)))
        
        for i in range(n):
            if i < bandwidth:
                yhat[i] = y[i]
                continue
            
            # Fit HMM on window
            window_size = min(i, int(bandwidth * 2))
            y_window = y[max(0, i - window_size):i+1].reshape(-1, 1)
            
            try:
                hmm = GaussianHMM(
                    n_components=n_states,
                    covariance_type='diag',
                    n_iter=50,
                    random_state=42
                )
                hmm.fit(y_window)
                
                # Predict smoothed value using HMM
                posterior = hmm.predict_proba(y_window)[-1]
                means = hmm.means_.flatten()
                yhat[i] = np.dot(posterior, means)
            except:
                # Fallback to simple moving average
                yhat[i] = np.mean(y[max(0, i - int(bandwidth)):i+1])
        
        return yhat
    
    def regression_filter(self, data: pd.DataFrame, current_idx: int) -> float:
        """
        Quantum Kernel SVR for regression (replaces Lasso/Ridge)
        Implements quantile regression using multiple SVR models
        """
        start_idx = max(0, current_idx - self.regression_window)
        
        if current_idx - start_idx < 50:
            return 0
        
        feature_cols = ['price_change', 'bar_volatility', 'return_1bar', 
                       'return_2bar', 'volume_ratio']
        
        train_data = data.iloc[start_idx:current_idx]
        train_data = train_data.dropna(subset=feature_cols + ['continuous_target'])
        
        if len(train_data) < 30:
            return 0
        
        X_train = train_data[feature_cols].values
        y_train = train_data['continuous_target'].values
        
        # Limit training size for quantum kernel efficiency
        if len(X_train) > 100:
            sample_idx = np.linspace(0, len(X_train)-1, 100, dtype=int)
            X_train = X_train[sample_idx]
            y_train = y_train[sample_idx]
        
        try:
            # Normalize features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train_scaled = np.clip(X_train_scaled, -np.pi, np.pi)
            
            current_features = data.iloc[current_idx][feature_cols].values.reshape(1, -1)
            current_features_scaled = self.scaler.transform(current_features)
            current_features_scaled = np.clip(current_features_scaled, -np.pi, np.pi)
            
            # Compute quantum kernel matrix
            K_train = self.quantum_kernel_regressor.evaluate(X_train_scaled)
            
            # Compute kernel between test and training
            X_combined = np.vstack([X_train_scaled, current_features_scaled])
            K_combined = self.quantum_kernel_regressor.evaluate(X_combined)
            K_test = K_combined[-1, :-1].reshape(1, -1)
            
            # Train quantum kernel SVR (median prediction)
            self.regression_model.fit(K_train, y_train)
            pred_median = self.regression_model.predict(K_test)[0]
            
            # Simulate quantile predictions using modified targets
            # Q10: Train on lower values (pessimistic)
            y_train_q10 = y_train - np.abs(y_train) * 0.3
            self.regression_model_q10.fit(K_train, y_train_q10)
            pred_q10 = self.regression_model_q10.predict(K_test)[0]
            
            # Q90: Train on upper values (optimistic)
            y_train_q90 = y_train + np.abs(y_train) * 0.3
            self.regression_model_q90.fit(K_train, y_train_q90)
            pred_q90 = self.regression_model_q90.predict(K_test)[0]
            
            # Uncertainty-adjusted prediction
            uncertainty = pred_q90 - pred_q10
            confidence_factor = 1.0 / (1.0 + uncertainty * 10)
            
            prediction = pred_median * confidence_factor
            return prediction
            
        except Exception as e:
            # Fallback to simple mean if quantum fails
            return np.mean(y_train)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate final trading signals with quantum kernel filters
        """
        data = self.compute_features(df)
        data = self.compute_targets(data)
        
        # Apply HMM-based probabilistic smoothing
        bandwidth_rq = 8
        bandwidth_gauss = 6
        
        data['yhat_rq'] = self.kernel_regression(
            data['Close'].values, bandwidth_rq, 'rational_quadratic'
        )
        data['yhat_gauss'] = self.kernel_regression(
            data['Close'].values, bandwidth_gauss, 'gaussian'
        )
        
        # Calculate average range for volatility filter
        data['bar_range'] = data['High'] - data['Low']
        data['avg_range_14'] = data['bar_range'].rolling(14).mean()
        
        # Initialize signal columns
        data['knn_prediction'] = 0
        data['regression_prediction'] = 0.0
        data['final_signal'] = 0
        
        feature_cols = ['price_change', 'bar_volatility', 'return_1bar', 
                       'return_2bar', 'volume_ratio']
        
        # Generate signals for each bar
        for i in range(self.lookback + self.regression_window, len(data) - self.future_bars):
            # Skip if NaN features
            if pd.isna(data.iloc[i][feature_cols]).any():
                continue
            
            # Quantum Kernel Classification
            current_features = data.iloc[i][feature_cols].values
            knn_pred = self.knn_predict(current_features, data, i)
            data.iloc[i, data.columns.get_loc('knn_prediction')] = knn_pred
            
            # Quantum Kernel Regression
            reg_pred = self.regression_filter(data, i)
            data.iloc[i, data.columns.get_loc('regression_prediction')] = reg_pred
            
            # Apply final entry filters
            close_current = data.iloc[i]['Close']
            close_50_ago = data.iloc[i - 50]['Close']
            bar_range = data.iloc[i]['bar_range']
            avg_range = data.iloc[i]['avg_range_14']
            yhat_gauss = data.iloc[i]['yhat_gauss']
            yhat_rq = data.iloc[i]['yhat_rq']
            
            # Long Signal
            if (knn_pred == 1 and 
                reg_pred > 0 and 
                close_current > close_50_ago and 
                bar_range > avg_range and 
                yhat_gauss >= yhat_rq):
                data.iloc[i, data.columns.get_loc('final_signal')] = 1
            
            # Short Signal
            elif (knn_pred == -1 and 
                  reg_pred < 0 and 
                  close_current < close_50_ago and 
                  bar_range > avg_range and 
                  yhat_gauss <= yhat_rq):
                data.iloc[i, data.columns.get_loc('final_signal')] = -1
        
        return data
    
    def backtest(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Backtest the strategy with entry/exit logic
        """
        data = self.generate_signals(df)
        
        trades = []
        position = 0
        entry_price = 0
        entry_bar = 0
        bars_in_trade = 0
        
        for i in range(len(data) - 1):
            signal = data.iloc[i]['final_signal']
            
            # Check if in position
            if position != 0:
                bars_in_trade += 1
                
                # Exit conditions
                exit_trade = False
                exit_reason = ''
                
                # Time-based exit (4 bars)
                if bars_in_trade >= self.future_bars:
                    exit_trade = True
                    exit_reason = 'time_exit'
                
                # Early flip exit
                elif signal != position and signal != 0:
                    exit_trade = True
                    exit_reason = 'flip_exit'
                
                if exit_trade:
                    exit_price = data.iloc[i + 1]['Open']
                    pnl = (exit_price - entry_price) * position
                    pnl_pct = ((exit_price / entry_price) - 1) * position * 100
                    
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i + 1,
                        'direction': 'Long' if position == 1 else 'Short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'bars_held': bars_in_trade,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    bars_in_trade = 0
            
            # Entry logic (if not in position and signal exists)
            if position == 0 and signal != 0:
                position = signal
                entry_price = data.iloc[i + 1]['Open']
                entry_bar = i + 1
                bars_in_trade = 0
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            wins = len(trades_df[trades_df['pnl'] > 0])
            losses = len(trades_df[trades_df['pnl'] <= 0])
            win_rate = (wins / len(trades_df)) * 100
            total_pnl = trades_df['pnl'].sum()
            avg_pnl = trades_df['pnl'].mean()
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0
            
            metrics = {
                'total_trades': len(trades_df),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else 0
            }
        else:
            metrics = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        return trades_df, metrics


# Example usage
if __name__ == "__main__":
    # Load the OHLCV data
    df = pd.read_excel('Sidebar_Merged.xlsx')
    
    # Remove any whitespace column names
    df.columns = df.columns.str.strip()
    
    # Drop rows with missing data
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Sort chronologically
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # 80:20 split
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Initialize quantum strategy
    strategy = LorentzianKNN()
    
    # Run backtest
    trades_df, metrics = strategy.backtest(full_df)
    
    # Filter test set trades
    test_start_bar = split_index
    test_end_bar = len(full_df)-1
    if not trades_df.empty:
        test_trades = trades_df[
            (trades_df['entry_bar'] >= test_start_bar) & 
            (trades_df['exit_bar'] <= test_end_bar)
        ]
    else:
        test_trades = pd.DataFrame()
    
    print("="*60)
    print("QUANTUM ML BACKTEST RESULTS (80:20)")
    print("="*60)
    print(f"Test Trades: {len(test_trades)}")
    
    if len(test_trades) > 0:
        wins = len(test_trades[test_trades['pnl'] > 0])
        losses = len(test_trades[test_trades['pnl'] <= 0])
        win_rate = (wins / len(test_trades)) * 100
        total_pnl = test_trades['pnl'].sum()
        avg_pnl = test_trades['pnl'].mean()
        avg_win = test_trades[test_trades['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = test_trades[test_trades['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0
        profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else 0
        
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total P&L: {total_pnl:.2f}")
        print(f"Average P&L per Trade: {avg_pnl:.2f}")
        print(f"Average Win: {avg_win:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print("\nFirst 10 Test Trades:")
        print(test_trades.head(10).to_string(index=False))
    else:
        print("No test trades executed.")
