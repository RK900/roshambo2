import time
import os
import pickle
import numpy as np
from roshambo2.backends.minimal_cuda_backend import PersistentCudaShapeOverlay, SimpleRoshamboData, BitVectorColorGenerator, _quaternion_to_matrix

def main():
    print("Loading Data...")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    m1_path = os.path.join(data_dir, 'anchor_molecule.pkl')
    m2_path = os.path.join(data_dir, 'molecule_0.pkl')
    
    if not os.path.exists(m1_path):
         m1_path = 'data/anchor_molecule.pkl'
         m2_path = 'data/molecule_0.pkl'
         
    m1 = pickle.load(open(m1_path, 'rb'))
    m2 = pickle.load(open(m2_path, 'rb'))


    if isinstance(m1, list): m1 = m1[0]
    if isinstance(m2, list): m2 = m2[0]

    # Create dummy batch of 64 mols
    batch_size = 256
    batch_mols = [m2] * batch_size

    # Prepare Data Wrappers
    print("Preparing Data...")
    q_data = SimpleRoshamboData(m1, name="Query")
    print(f"Query Data Shape: {q_data.f_x.shape}")
    d_data_batch = SimpleRoshamboData(batch_mols, name="DataBatch")
    print(f"Data Batch Shape: {d_data_batch.f_x.shape}")
    
    # Check max atoms needed
    needed_atoms = max(q_data.f_x.shape[1], d_data_batch.f_x.shape[1])
    print(f"Max atoms needed: {needed_atoms}")

    color_gen = BitVectorColorGenerator(n_bits=48)
    
    # Determine max dimensions for Context
    max_mols = 256
    max_atoms = needed_atoms + 10 # ensure capacity
    n_features = color_gen.matrix_size
    
    # Initialize Persistent Context
    print("Initializing Persistent Context...")
    t0 = time.time()
    overlay = PersistentCudaShapeOverlay(max_mols, max_atoms, n_features, n_gpus=1)
    t1 = time.time()
    print(f"Context Initialization took: {t1-t0:.4f}s (Expect ~3-4s)")
    
    # Setup Color Generator
    color_gen = BitVectorColorGenerator(n_bits=48)
    
    # Run loop
    n_batches = 3
    print(f"\nRunning {n_batches} batches of size {batch_size}...")
    
    for i in range(n_batches):
        start = time.time()
        scores = overlay.calculate_overlap_batch(q_data, d_data_batch, mixing=0.5, color_generator=color_gen)
        end = time.time()
        
        # Verify result
        sample_score = scores[0,0,0] # Combo score of first pair
        print(f"Batch {i}: {end-start:.4f}s. Score: {sample_score:.4f}")

    # Non-Persistent Benchmark
    print("\nRunning Non-Persistent Benchmark (Standard Backend)...")
    from roshambo2.backends.minimal_cuda_backend import MinimalCudaShapeOverlay
    
    # We use the same data objects
    # MinimalCudaShapeOverlay will call optimize_overlap_color which creates a FRESH context each time in C++
    
    start = time.time()
    # Note: MinimalCudaShapeOverlay doesn't have calculate_overlap_batch, it has optimize_overlap
    # It takes query_data and data in constructor
    mp_overlay = MinimalCudaShapeOverlay(q_data, d_data_batch, mixing=0.5, start_mode=1,
                                        color_generator=color_gen, n_gpus=1, verbosity=0)
    scores_mp = mp_overlay.optimize_overlap()
    print(f"Non-Persistent Score: {scores_mp[0,0,0]:.4f}")
    end = time.time()
    
    print(f"Non-Persistent Total Time: {end-start:.4f}s")
    print(f"Non-Persistent Time per mol: {(end-start)/batch_size:.4f}s")
    
    print("\nBenchmark Complete.")

if __name__ == '__main__':
    main()
