import time
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler as AerSampler

def test_speed():
    # Setup
    num_features = 2
    X_train = np.random.rand(10, num_features)
    y_train = np.random.randint(0, 2, 10)
    
    # Model
    fm = ZZFeatureMap(num_features, reps=2)
    ansatz = RealAmplitudes(num_features, reps=2)
    optimizer = COBYLA(maxiter=1)
    sampler = AerSampler(run_options={"shots": 1024})
    
    vqc = VQC(fm, ansatz, optimizer, sampler=sampler)
    
    # Train
    print("Training...")
    vqc.fit(X_train, y_train)
    
    # Predict
    n_predict = 1000
    X_pred = np.random.rand(n_predict, num_features)
    
    print(f"Predicting {n_predict} samples...")
    start = time.time()
    vqc.predict(X_pred)
    duration = time.time() - start
    print(f"Prediction took {duration:.4f}s for {n_predict} samples.")
    print(f"Estimated time for 60,000 samples: {duration * 60:.2f}s")

if __name__ == "__main__":
    test_speed()
