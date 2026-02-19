import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RealAmplitudes
from qiskit import QuantumCircuit

def visualize_results(file_path):
    try:
        df = pd.read_excel(file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    # Set style
    plt.style.use('dark_background')
    
    # 1. Plot Cumulative Returns Comparison
    plt.figure(figsize=(12, 6))
    
    models = ['Q_WaveTrend', 'Q_CCI', 'Q_ADX', 'Q_RSA', 'Q_4_Ensemble', 'Q_Master_9']
    colors = ['cyan', 'orange', 'yellow', 'magenta', 'lime', 'white']
    
    # Calculate Buy & Hold
    df['Buy_Hold_Ret'] = np.log(df['Close'] / df['Close'].shift(1)).cumsum().apply(np.exp)
    plt.plot(df.index, df['Buy_Hold_Ret'], label='Buy & Hold', color='gray',  linestyle='--', alpha=0.6)
    
    for i, model in enumerate(models):
        col_name = f'{model}_Cum_Ret'
        if col_name in df.columns:
            plt.plot(df.index, df[col_name], label=model, color=colors[i], linewidth=1.5)
            
    plt.title('Quantum Strategy Cumulative Returns (Full Dataset)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('Cumulative_Returns_Comparison.png')
    print("Saved Cumulative_Returns_Comparison.png")
    
    # Calculate Metrics Function
    def calculate_metrics(df, strategy_col):
        # Daily returns approximation (assuming 1-min data treated as sequential trade opportunities)
        # Actually, let's just use the trade pnl series
        rets = df[strategy_col]
        
        total_return = rets.sum()
        net_return_pct = (np.exp(total_return) - 1) * 100
        
        # Win Rate (approx based on positive return bars)
        winning_bars = len(rets[rets > 0])
        losing_bars = len(rets[rets < 0])
        total_trades = winning_bars + losing_bars
        win_rate = (winning_bars / total_trades * 100) if total_trades > 0 else 0
        
        # Profit Factor
        gross_profit = rets[rets > 0].sum()
        gross_loss = abs(rets[rets < 0].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else 0
        
        # Sharpe (Annualized, assuming 252*375 mins/year? Let's just do simple sharpe)
        # 1-min data has high noise. Let's just do mean/std.
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252 * 375) if rets.std() != 0 else 0
        
        # Max Drawdown
        cum_ret = rets.cumsum().apply(np.exp)
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        return {
            "Net Return %": f"{net_return_pct:.2f}%",
            "Win Rate": f"{win_rate:.2f}%",
            "Profit Factor": f"{profit_factor:.2f}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_drawdown:.2f}%"
        }

    # Generate Metrics Report
    metrics_data = []
    
    for model in models:
        strat_col = f'{model}_Strategy_Ret'
        if strat_col in df.columns:
            m = calculate_metrics(df, strat_col)
            m['Model'] = model
            metrics_data.append(m)
            
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        # Reorder columns
        cols = ['Model', 'Net Return %', 'Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Max Drawdown']
        metrics_df = metrics_df[cols]
        
        # Save as CSV
        metrics_df.to_csv('Quantum_Metrics_Report.csv', index=False)
        print("Saved Quantum_Metrics_Report.csv")
        
        # Plot Metrics Table
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        # Fix styling - make cells white with black text or black with white text
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_edgecolor('white')
            cell.set_facecolor('#202020')
            cell.set_text_props(color='white')
            if key[0] == 0:  # Header
                cell.set_facecolor('#404040')
                cell.set_text_props(weight='bold', color='cyan')
        
        plt.title('Quantum Model Performance Metrics', color='white', pad=20)
        plt.savefig('Quantum_Metrics_Report.png', bbox_inches='tight', dpi=150)
        print("Saved Quantum_Metrics_Report.png")

    # 2. Individual Model Plots (Signal vs Price)
    for model in models:
        signal_col = f'{model}_Signal'
        if signal_col not in df.columns:
            continue
            
        # Full View
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        ax1.plot(df.index, df['Close'], color='#cccccc', alpha=0.6, label='Price', linewidth=0.8)
        ax1.set_ylabel('Price', color='white', fontsize=12)
        ax1.tick_params(axis='y', colors='white')
        ax1.tick_params(axis='x', colors='white')
        ax1.grid(True, alpha=0.1, color='gray')
        
        ax2 = ax1.twinx()
        # Area plot for signal to look cleaner than barcode lines
        ax2.fill_between(df.index, df[signal_col], 0, step='pre', alpha=0.3, color='#bf00ff', label=f'{model} Signal')
        ax2.plot(df.index, df[signal_col], drawstyle='steps-post', color='#bf00ff', linewidth=1.0)
        
        ax2.set_ylabel('Signal (0/1)', color='#bf00ff', fontsize=12)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.tick_params(axis='y', colors='#bf00ff')
        
        plt.title(f'{model} - Full Dataset', color='white', fontsize=14)
        plt.savefig(f'{model}_Price_Signal_Full.png', dpi=150)
        print(f"Saved {model}_Price_Signal_Full.png")
        plt.close()
        
        # Zoomed View (Last 1000 bars)
        subset = df.iloc[-1000:]
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        ax1.plot(subset.index, subset['Close'], color='#cccccc', alpha=0.8, label='Price', linewidth=1.5)
        ax1.set_ylabel('Price', color='white', fontsize=12)
        ax1.tick_params(axis='y', colors='white')
        ax1.tick_params(axis='x', colors='white')
        ax1.grid(True, alpha=0.15, color='gray')
        
        ax2 = ax1.twinx()
        ax2.fill_between(subset.index, subset[signal_col], 0, step='pre', alpha=0.4, color='#bf00ff')
        ax2.step(subset.index, subset[signal_col], where='post', color='#bf00ff', linewidth=2.0)
        
        ax2.set_ylabel('Signal (0/1)', color='#bf00ff', fontsize=12)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.tick_params(axis='y', colors='#bf00ff')
        
        plt.title(f'{model} - Last 1000 Bars (Zoomed)', color='white', fontsize=14)
        plt.savefig(f'{model}_Price_Signal_Zoomed.png', dpi=150)
        print(f"Saved {model}_Price_Signal_Zoomed.png")
        plt.close()

    # 3. Draw Circuit Diagrams (Representation)
    # We reconstruct circuits momentarily to draw them
    def draw_circuit(num_features, name):
        if num_features == 1:
            fm = ZFeatureMap(1, reps=2)
        else:
            fm = ZZFeatureMap(num_features, reps=2)
        ansatz = RealAmplitudes(num_features, reps=2)
        qc = QuantumCircuit(num_features)
        qc.append(fm, range(num_features))
        qc.append(ansatz, range(num_features))
        
        qc.draw('mpl', filename=f'{name}_Circuit.png')
        print(f"Saved {name}_Circuit.png")

    # Draw circuits for each model type
    try:
        draw_circuit(2, 'Q_WaveTrend')
        draw_circuit(1, 'Q_CCI')
        draw_circuit(3, 'Q_ADX')
        draw_circuit(1, 'Q_RSA')
        draw_circuit(4, 'Q_4_Ensemble')
        # Q_Master_9 might be too big to draw nicely but try
        draw_circuit(9, 'Q_Master_9')
    except Exception as e:
        print(f"Could not draw circuits (latex/mpl missing?): {e}")

if __name__ == "__main__":
    visualize_results('Quantum_Results_Full.xlsx')
