import time
import os
import numpy as np
from roshambo2.backends.minimal_cuda_backend import PersistentCudaShapeOverlay, SimpleRoshamboData, BitVectorColorGenerator, _quaternion_to_matrix
from roshambo2.classes import Roshambo2DataReaderSDF

def main():
    print("Loading Data from example/...")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example')
    query_path = os.path.join(data_dir, 'query.sdf')
    dataset_path = os.path.join(data_dir, 'dataset.sdf')
    
    # Use Roshambo2DataReaderSDF to properly load SDFs
    # Note: we need to handle the PharmacophoreGenerator. basic_run.py says color=True.
    from roshambo2.pharmacophore import PharmacophoreGenerator
    
    color_gen = PharmacophoreGenerator()
    
    print("Reading Query...")
    q_reader = Roshambo2DataReaderSDF([query_path], color_generator=color_gen, quiet=True, keep_original_coords=True)
    # The reader returns Roshambo2Data objects.
    queries = [q for q in q_reader.get_data()]
    query_data = queries[0]
    
    print("Reading Dataset...")
    d_reader = Roshambo2DataReaderSDF([dataset_path], color_generator=color_gen, quiet=True)
    datasets = [d for d in d_reader.get_data()]
    data_data = datasets[0] # Roshambo2Data object which typically holds many molecules
    
    print(f"Query Name: {query_data.f_names[0]}")
    print(f"Dataset Size: {len(data_data.f_names)}")
    print(f"Query shape: {query_data.f_x.shape}")
    print(f"Data shape: {data_data.f_x.shape}")
    
    # We need to adapt Roshambo2Data to SimpleRoshamboData or simply use PersistentCudaShapeOverlay's expectation
    # PersistentCudaShapeOverlay expects SimpleRoshamboData.
    # SimpleRoshamboData expects a list of dicts with 'graph_nodes' and 'graph_pos'.
    # BUT, Roshambo2Data already has f_x, f_types, f_n_real.
    # We can probably bypass SimpleRoshamboData and pass Roshambo2Data directly if we ensure compatibility
    # or just create a wrapper.
    
    # Let's inspect SimpleRoshamboData vs Roshambo2Data.
    # Roshambo2Data has f_x, f_types, f_n_real just like SimpleRoshamboData creates.
    # So we should be able to pass it directly if we ensure prepare() is called.
    
    print("Init Persistent Context...")
    max_mols = 256
    max_atoms = 128 
    
    # ensure capacity
    needed_atoms = max(query_data.f_x.shape[1], data_data.f_x.shape[1])
    if needed_atoms > max_atoms:
        max_atoms = needed_atoms + 10
        
    overlay = PersistentCudaShapeOverlay(max_mols, max_atoms, n_features=48, n_gpus=1)
    
    # Color gen wrapper
    class SimpleColorGen:
        def __init__(self, pg):
            # The backend expects interaction_matrix_r and p
            # PharmacophoreGenerator has these as constants but not instance vars in same way maybe?
            # actually cuda_backend checks self.color_generator.interaction_matrix_r
            self.interaction_matrix_r = pg.interaction_matrix_r
            self.interaction_matrix_p = pg.interaction_matrix_p
            
    simple_gen = SimpleColorGen(color_gen)
    
    print("Running Batch Calculation...")
    start = time.time()
    # We bypass calculate_overlap_batch's SimpleRoshamboData checks since we have Roshambo2Data
    # but calculate_overlap_batch calls .prepare() on inputs.
    # Roshambo2Data has tofloat32() but maybe not prepare().
    # Let's monkey patch or check.
    
    if not hasattr(query_data, 'prepare'):
        query_data.prepare = query_data.tofloat32
    if not hasattr(data_data, 'prepare'):
        data_data.prepare = data_data.tofloat32
        
    scores = overlay.calculate_overlap_batch(query_data, data_data, mixing=0.5, color_generator=simple_gen)
    end = time.time()
    
    print(f"Total Time: {end-start:.4f}s")
    print(f"Time per mol: {(end-start)/len(data_data.f_names):.4f}s")
    
    # Test again to verify cache/warmup
    print("Running Batch Calculation (Run 2)...")
    start = time.time()
    scores = overlay.calculate_overlap_batch(query_data, data_data, mixing=0.5, color_generator=simple_gen)
    end = time.time()
    print(f"Total Time 2: {end-start:.4f}s")

    # Non-Persistent Benchmark
    print("\nRunning Non-Persistent Benchmark (Standard Backend)...")
    from roshambo2.backends.minimal_cuda_backend import MinimalCudaShapeOverlay
    
    # We use the same data objects
    # MinimalCudaShapeOverlay will call optimize_overlap_color which creates a FRESH context each time in C++
    
    start = time.time()
    # Note: MinimalCudaShapeOverlay doesn't have calculate_overlap_batch, it has optimize_overlap
    # It takes query_data and data in constructor
    mp_overlay = MinimalCudaShapeOverlay(query_data, data_data, start_mode=1, mixing=0.5, 
                                        color_generator=simple_gen, n_gpus=1, verbosity=0)
    scores_mp = mp_overlay.optimize_overlap()
    end = time.time()
    
    print(f"Non-Persistent Total Time: {end-start:.4f}s")
    print(f"Non-Persistent Time per mol: {(end-start)/len(data_data.f_names):.4f}s")


if __name__ == '__main__':
    main()
