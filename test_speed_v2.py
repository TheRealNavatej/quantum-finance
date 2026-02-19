import time
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.primitives import Sampler as RefSampler

def test_speed():
    # Setup
    num_features = 2
    X_train = np.random.rand(10, num_features)
    y_train = np.random.randint(0, 2, 10)
    
    fm = ZZFeatureMap(num_features, reps=2)
    ansatz = RealAmplitudes(num_features, reps=2)
    optimizer = COBYLA(maxiter=1)
    
    # 1. AerSampler (1024 shots)
    print("Testing AerSampler (1024 shots)...")
    # Modern Qiskit Aer Sampler usually takes run_options in run(), or we use transpile_options.
    # Let's try simple init.
    sampler_aer = AerSampler() 
    # vqc_aer = VQC(fm, ansatz, optimizer, sampler=sampler_aer) 
    # VQC handles the sampler. If we want shots, we might need to set them differently or accept default.
    vqc_aer = VQC(fm, ansatz, optimizer, sampler=sampler_aer)
    vqc_aer.fit(X_train, y_train)
    
    start = time.time()
    vqc_aer.predict(np.random.rand(100, num_features))
    print(f"Aer (100 samples): {time.time() - start:.4f}s")

    # 2. Reference Sampler (Exact)
    print("Testing Reference Sampler (Exact)...")
    sampler_ref = RefSampler()
    vqc_ref = VQC(fm, ansatz, optimizer, sampler=sampler_ref)
    vqc_ref.fit(X_train, y_train)
    
    start = time.time()
    vqc_ref.predict(np.random.rand(100, num_features))
    print(f"Reference (100 samples): {time.time() - start:.4f}s")

if __name__ == "__main__":
    test_speed()
