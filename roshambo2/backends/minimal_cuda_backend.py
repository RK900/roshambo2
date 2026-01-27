import numpy as np
import pickle
import sys
import os
import time

# Try to import the cuda backend
try:
    from _roshambo2_cuda import optimize_overlap_color, CudaOverlapContext
except ImportError:
    print("Warning: _roshambo2_cuda could not be imported. Ensure the compiled extension is available.")
    pass

class BitVectorColorGenerator:
    """
    Color Generator that treats each of the 48 feature bits as a distinct atom type.
    """
    def __init__(self, n_bits=48):
        self.n_bits = n_bits
        self.matrix_size = n_bits + 1
        
        # Initialize r (width) to 1.0 everywhere to avoid division by zero.
        self.interaction_matrix_r = np.ones((self.matrix_size, self.matrix_size), dtype=np.float64)
        self.interaction_matrix_p = np.zeros((self.matrix_size, self.matrix_size), dtype=np.float64)
        
        diag_indices = np.diag_indices(self.matrix_size)
        self.interaction_matrix_p[diag_indices] = 1.0  
        
        # Set p=0 for Real(0)-Real(0) acts so they don't contribute to color score.
        self.interaction_matrix_p[0, 0] = 0.0

class SimpleRoshamboData:
    """
    Minimal wrapper to convert raw coordinate/feature tensors into Roshambo2 formatted arrays.
    Supports single dict or list of dicts.
    """
    def __init__(self, mol_input, name="Mol", allowed_features=None):
        # Normalize input to list
        if isinstance(mol_input, dict):
            self.mol_list = [mol_input]
            self.names = [name]
            self.single_mode = True
        elif isinstance(mol_input, list):
            self.mol_list = mol_input
            self.names = [f"{name}_{i}" for i in range(len(mol_input))]
            self.single_mode = False
        else:
            raise ValueError("Input must be a dict or list of dicts")
            
        self.f_names = self.names
        self.color = True
        
        # 1. First pass: Determine max atoms (N_atoms) across the batch
        max_atoms = 0
        
        # Store processed node/pos data temporarily to avoid re-extraction
        processed_mols = []
        
        # User requested specific features for speed:
        # Default: Formal Charge (18-22), Aromatic (36), HB Acceptor (41), HB Donor (42)
        if allowed_features is None:
            self.allowed_features = {18, 19, 20, 21, 22, 36, 41, 42}
        else:
            self.allowed_features = set(allowed_features)
        
        for mol_dict in self.mol_list:
            nodes = mol_dict['graph_nodes']
            if hasattr(nodes, 'numpy'): nodes = nodes.numpy()
            else: nodes = np.array(nodes)
            
            pos = mol_dict['graph_pos']
            if hasattr(pos, 'numpy'): pos = pos.numpy()
            else: pos = np.array(pos)
            
            # Feature extraction
            # Revert to int (int64) to match roshambo2/utils.py
            n_real = pos.shape[0]
            
            # Feature Atoms
            active_rows, active_cols = np.where(nodes > 0.5)
            
            # Filter features
            if self.allowed_features:
                mask = np.isin(active_cols, list(self.allowed_features))
                active_rows = active_rows[mask]
                active_cols = active_cols[mask]
            
            coords_list = [pos]
            types_list = [np.zeros(n_real, dtype=int)] # Real atoms = Type 0
            
            if len(active_rows) > 0:
                feat_coords = pos[active_rows]
                feat_types = active_cols + 1 # 1-based indexing for features
                coords_list.append(feat_coords)
                types_list.append(feat_types.astype(int))
                
            all_coords = np.concatenate(coords_list, axis=0)
            all_types = np.concatenate(types_list, axis=0)
            
            total_atoms = all_coords.shape[0]
            if total_atoms > max_atoms:
                max_atoms = total_atoms
                
            processed_mols.append({
                'coords': all_coords,
                'types': all_types,
                'n_real': n_real
            })
            
        # 2. Allocate Batched Arrays
        # f_x: (N_mols, max_atoms, 4)
        # f_types: (N_mols, max_atoms)
        # f_n_real: (N_mols)
        
        n_mols = len(self.mol_list)
        self.f_x = np.zeros((n_mols, max_atoms, 4), dtype=np.float32)
        self.f_types = np.zeros((n_mols, max_atoms), dtype=int) # Type 0 is default (real)
        self.f_n_real = np.zeros((n_mols), dtype=int)
        
        # 3. Fill Arrays
        for i, p_mol in enumerate(processed_mols):
            n_current = p_mol['coords'].shape[0]
            
            # Copy coords
            self.f_x[i, :n_current, :3] = p_mol['coords']
            self.f_x[i, :n_current, 3] = 1.0 # weight
            
            # Copy types
            self.f_types[i, :n_current] = p_mol['types']
            
            # N real
            self.f_n_real[i] = p_mol['n_real']
            
        
    def tofloat32(self):
        # We ensure f_x is float32 and CONTIGUOUS
        self.f_x = np.ascontiguousarray(self.f_x.astype(np.float32))
        
    def prepare(self):
        # Ensure strict memory layout
        self.f_n_real = np.ascontiguousarray(self.f_n_real)
        self.f_types = np.ascontiguousarray(self.f_types)
        self.tofloat32()


class MinimalCudaShapeOverlay:
    """Wrapper around _roshambo2_cuda.optimize_overlap_color."""
    def __init__(self, query_data, data, start_mode, mixing=0.5, color_generator=None, n_gpus=1, verbosity=1):
        self.query_data = query_data
        self.data = data
        self.start_mode = start_mode
        self.mixing = mixing
        self.verbosity = verbosity
        
        self.color_generator = color_generator
        self.n_gpus = n_gpus
        
        self.lr_q = 0.1
        self.lr_t = 0.1
        self.steps = 100

        n_q = len(self.query_data.f_names)
        n_d = len(self.data.f_names)
        self.scores = np.zeros((n_q, n_d, 20), dtype=np.float32)

    def optimize_overlap(self):
        self.query_data.prepare()
        self.data.prepare()
        
        im_r = np.ascontiguousarray(self.color_generator.interaction_matrix_r)
        im_p = np.ascontiguousarray(self.color_generator.interaction_matrix_p)
        self.scores = np.ascontiguousarray(self.scores)


        # print("Using CUDA Backend...") # Reduced verbosity


        # print("Using CUDA Backend...") # Reduced verbosity
        optimize_overlap_color(
            self.query_data.f_x, self.query_data.f_types, self.query_data.f_n_real, 
            self.data.f_x,       self.data.f_types,       self.data.f_n_real, 
            im_r, 
            im_p, 
            self.scores, 
            True, self.mixing, self.lr_q, self.lr_t, self.steps, 
            self.start_mode, self.n_gpus, self.verbosity
        )
        return self.scores

class PersistentCudaShapeOverlay:
    """
    Wrapper around _roshambo2_cuda.CudaOverlapContext for training loops.
    Initializes GPU memory once and reuses it.
    """
    def __init__(self, max_mols, max_atoms, n_features=48, n_gpus=1):
        self.n_gpus = n_gpus
        self.max_mols = max_mols
        self.max_atoms = max_atoms
        self.n_features = n_features
        self.scores_dim = 20
        
        # Initialize Persistent Context
        self.ctx = CudaOverlapContext(n_gpus, max_mols, max_atoms, n_features, self.scores_dim)
        
        # Default optimizer settings
        self.lr_q = 0.1
        self.lr_t = 0.1
        self.steps = 100
        self.verbosity = 0

    def calculate_overlap_batch(self, query_data, data_data, start_mode=1, mixing=0.5, color_generator=None):
        """
        Run overlap optimization using the persistent context.
        query_data: SimpleRoshamboData (usually single mol)
        data_data: SimpleRoshamboData (batch of mols)
        """
        # Ensure data is prepared (contiguous float32)
        query_data.prepare()
        data_data.prepare()
        
        # Check sizes
        if data_data.f_x.shape[0] > self.max_mols:
            raise ValueError(f"Batch size {data_data.f_x.shape[0]} exceeds context capacity {self.max_mols}")
        if data_data.f_x.shape[1] > self.max_atoms:
            raise ValueError(f"Molecule size {data_data.f_x.shape[1]} exceeds context capacity {self.max_atoms}")
        if query_data.f_x.shape[1] > self.max_atoms:
            raise ValueError(f"Query Molecule size {query_data.f_x.shape[1]} exceeds context capacity {self.max_atoms}")
        
        # Interaction Matrices
        im_r = np.ascontiguousarray(color_generator.interaction_matrix_r)
        im_p = np.ascontiguousarray(color_generator.interaction_matrix_p)
        
        # Allocate output scores (on host)
        n_q = len(query_data.f_names)
        n_d = len(data_data.f_names)
        scores = np.zeros((n_q, n_d, self.scores_dim), dtype=np.float32)
        
        # Run Optimization
        self.ctx.optimize(
            query_data.f_x, query_data.f_types, query_data.f_n_real, 
            data_data.f_x,       data_data.f_types,       data_data.f_n_real, 
            im_r, 
            im_p, 
            scores, 
            True, mixing, self.lr_q, self.lr_t, self.steps, 
            start_mode, self.verbosity
        )
        return scores

def calculate_overlap(query_dict, data_dict, mixing=0.5, start_mode=1, n_gpus=1, verbosity=0):
    """
    Simpler API to calculate overlap between a query and one or more data molecules.
    
    Args:
        query_dict (dict): Dictionary with 'graph_nodes' and 'graph_pos' for query molecule.
        data_dict (dict or list[dict]): Dictionary (or list of dicts) with 'graph_nodes' and 'graph_pos' for data molecule(s).
        mixing (float): Weight for color score (0.0 to 1.0).
        start_mode (int): Starting mode for optimization (1 is standard).
        n_gpus (int): Number of GPUs to use.
        verbosity (int): Verbosity level.
        
    Returns:
        dict or list[dict]: Dictionary (or list of dicts) containing scores and transformation.
    """
    q_data = SimpleRoshamboData(query_dict, name="Query")
    d_data = SimpleRoshamboData(data_dict, name="Data")
    
    # Dynamically determine n_bits
    max_type = max(q_data.f_types.max(), d_data.f_types.max())
    target_n_bits = max(48, int(max_type))
    
    color_gen = BitVectorColorGenerator(n_bits=target_n_bits)
    
    # Overlay
    overlay = MinimalCudaShapeOverlay(q_data, d_data, start_mode=start_mode, mixing=mixing, 
                                      color_generator=color_gen, n_gpus=n_gpus, verbosity=verbosity)
    
    scores = overlay.optimize_overlap()
    
    # Extract results
    results = []
    
    # scores is (n_q, n_d, 20)
    # n_q is always 1 in this simple API (SimpleRoshamboData could act as batch query too, but API implies single query)
    
    n_data_mols = scores.shape[1]
    
    for i in range(n_data_mols):
        res = scores[0, i]
        results.append({
            "combo_score": float(res[0]),
            "shape_score": float(res[1]),
            "color_score": float(res[2]),
            "rotation_matrix": _quaternion_to_matrix(res[9:13]), 
            "translation_vector": res[13:16],
            "quaternion": res[9:13]
        })
        
    if d_data.single_mode:
        return results[0]
    else:
        return results

def _quaternion_to_matrix(q):
    """Simple helper to convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

if __name__ == '__main__':
    # Example usage
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        m1_path = os.path.join(data_dir, 'anchor_molecule.pkl')
        m2_path = os.path.join(data_dir, 'molecule_0.pkl')
        
        if not os.path.exists(m1_path):
             m1_path = 'data/anchor_molecule.pkl'
             m2_path = 'data/molecule_0.pkl'
             
        print(f"Loading {m1_path} and {m2_path}...")
        m1 = pickle.load(open(m1_path, 'rb'))
        m2 = pickle.load(open(m2_path, 'rb'))

        if isinstance(m1, list): m1 = m1[0]
        if isinstance(m2, list): m2 = m2[0]
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print("Running Single Calculation...")
    start = time.time()
    result = calculate_overlap(m1, m2, verbosity=1)
    end = time.time()
    print(f"Single Calculation took: {end-start:.4f}s")
    
    print("\nSingle Result:")
    print(f"Combo Score: {result['combo_score']:.4f}")
    
    # Batch Test
    print("\nRunning Batch Calculation (50 mols)...")
    batch_mols = [m2] * 50
    start = time.time()
    results = calculate_overlap(m1, batch_mols, verbosity=1)
    end = time.time()
    print(f"Batch Calculation (50 mols) took: {end-start:.4f}s")
    print(f"Time per molecule: {(end-start)/50:.4f}s")
    
    print(f"\nBatch Results (First 3):")
    for i in range(3):
        print(f"Mol {i} Combo Score: {results[i]['combo_score']:.4f}")

