import time
import os
import pickle
import numpy as np

print("Time: Import Start")
t0 = time.time()
from roshambo2.backends.minimal_cuda_backend import calculate_overlap, SimpleRoshamboData, BitVectorColorGenerator, MinimalCudaShapeOverlay, optimize_overlap_color, _quaternion_to_matrix
t1 = time.time()
print(f"Time: Import End ({t1-t0:.4f}s)")

# Load Data
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
m1_path = os.path.join(data_dir, 'anchor_molecule.pkl')
m2_path = os.path.join(data_dir, 'molecule_0.pkl')
if not os.path.exists(m1_path):
     m1_path = 'data/anchor_molecule.pkl'
     m2_path = 'data/molecule_0.pkl'

print(f"Time: Data Load Start")
t2 = time.time()
m1 = pickle.load(open(m1_path, 'rb'))
m2 = pickle.load(open(m2_path, 'rb'))
if isinstance(m1, list): m1 = m1[0]
if isinstance(m2, list): m2 = m2[0]
t3 = time.time()
print(f"Time: Data Load End ({t3-t2:.4f}s)")

# Manual Breakdown of calculate_overlap
print(f"Time: Prep Start")
t4 = time.time()
q_data = SimpleRoshamboData(m1, name="Query")
d_data = SimpleRoshamboData(m2, name="Data")

max_type = max(q_data.f_types.max(), d_data.f_types.max())
target_n_bits = max(48, int(max_type))
color_gen = BitVectorColorGenerator(n_bits=target_n_bits)
overlay = MinimalCudaShapeOverlay(q_data, d_data, start_mode=1, mixing=0.5, 
                                  color_generator=color_gen, n_gpus=1, verbosity=1)

overlay.query_data.prepare()
overlay.data.prepare()

im_r = np.ascontiguousarray(overlay.color_generator.interaction_matrix_r)
im_p = np.ascontiguousarray(overlay.color_generator.interaction_matrix_p)
overlay.scores = np.ascontiguousarray(overlay.scores)
t5 = time.time()
print(f"Time: Prep End ({t5-t4:.4f}s)")

print(f"Time: Cuda Call Start")
t6 = time.time()
optimize_overlap_color(
    overlay.query_data.f_x, overlay.query_data.f_types, overlay.query_data.f_n_real, 
    overlay.data.f_x,       overlay.data.f_types,       overlay.data.f_n_real, 
    im_r, 
    im_p, 
    overlay.scores, 
    True, overlay.mixing, overlay.lr_q, overlay.lr_t, overlay.steps, 
    overlay.start_mode, overlay.n_gpus, overlay.verbosity
)
t7 = time.time()
print(f"Time: Cuda Call 1 End ({t7-t6:.4f}s)")

print(f"Time: Batch Benchmark (50 mols) Start")
# Manually create a batch of 50 copies of d_data
n_batch = 50
d_data_batch_x = np.tile(overlay.data.f_x, (n_batch, 1, 1))
d_data_batch_types = np.tile(overlay.data.f_types, (n_batch, 1))
d_data_batch_n_real = np.tile(overlay.data.f_n_real, (n_batch))

# Resize scores for batch
scores_batch = np.zeros((1, n_batch, 20), dtype=np.float32)

t10 = time.time()
optimize_overlap_color(
    overlay.query_data.f_x, overlay.query_data.f_types, overlay.query_data.f_n_real, 
    d_data_batch_x,       d_data_batch_types,       d_data_batch_n_real, 
    im_r, 
    im_p, 
    scores_batch, 
    True, overlay.mixing, overlay.lr_q, overlay.lr_t, overlay.steps, 
    overlay.start_mode, overlay.n_gpus, overlay.verbosity
)
t11 = time.time()
print(f"Time: Batch Benchmark (50 mols) End ({t11-t10:.4f}s)")
print(f"Time per molecule (Batch): {(t11-t10)/n_batch:.4f}s")


print(f"Total Run Time: {t9-t0:.4f}s")
