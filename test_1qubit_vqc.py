import numpy as np
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

def test_1qubit():
    print("Testing 1-qubit VQC...")
    
    # 1 Feature, 1 Qubit
    num_features = 1
    X_train = np.random.rand(10, num_features)
    y_train = np.random.randint(0, 2, 10)
    
    fm = ZFeatureMap(num_features, reps=2)
    ansatz = RealAmplitudes(num_features, reps=2)
    optimizer = COBYLA(maxiter=10)
    sampler = Sampler()
    
    vqc = VQC(
        feature_map=fm,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    try:
        vqc.fit(X_train, y_train)
        print("Fit successful!")
        
        preds = vqc.predict(X_train)
        print(f"Predictions shape: {preds.shape}")
        
    except Exception as e:
        print(f"Fit Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_1qubit()
