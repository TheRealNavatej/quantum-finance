import pandas as pd
import numpy as np
import os
from visualize_results import visualize_results

def create_dummy_data():
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    df = pd.DataFrame(index=dates)
    df['Timestamp'] = dates
    df['Close'] = 100 + np.cumsum(np.random.randn(100))
    df['Q_WaveTrend_Signal'] = np.random.randint(0, 2, 100)
    df['Q_WaveTrend_Cum_Ret'] = np.cumprod(1 + np.random.randn(100)*0.01)
    df['Q_WaveTrend_Strategy_Ret'] = np.random.randn(100)*0.01
    
    df.to_excel("Dummy_Results.xlsx")
    print("Created Dummy_Results.xlsx")

if __name__ == "__main__":
    create_dummy_data()
    try:
        visualize_results("Dummy_Results.xlsx")
        print("Visualization test passed!")
    except Exception as e:
        print(f"Visualization test failed: {e}")
    finally:
        if os.path.exists("Dummy_Results.xlsx"):
            os.remove("Dummy_Results.xlsx")
