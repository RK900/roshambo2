import pickle
import numpy as np
import os

def inspect_pkl():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    m1_path = os.path.join(data_dir, 'anchor_molecule.pkl')
    
    if not os.path.exists(m1_path):
         m1_path = 'data/anchor_molecule.pkl'
    
    print(f"Loading {m1_path}...")
    m1 = pickle.load(open(m1_path, 'rb'))
    if isinstance(m1, list): m1 = m1[0]
    
    print("Keys:", m1.keys())
    
    nodes = m1['graph_nodes']
    pos = m1['graph_pos']
    
    if hasattr(nodes, 'numpy'): nodes = nodes.numpy()
    if hasattr(pos, 'numpy'): pos = pos.numpy()
    
    print(f"Pos shape: {pos.shape}")
    print(f"Nodes shape: {nodes.shape}")
    
    n_atoms = pos.shape[0]
    n_features_total = np.sum(nodes > 0.5)
    
    print(f"Number of 'Real' Atoms (pos): {n_atoms}")
    print(f"Number of Active Features (nodes > 0.5): {n_features_total}")
    print(f"Total expected size (Real + Features): {n_atoms + n_features_total}")
    
    # Check what features look like
    row, col = np.where(nodes > 0.5)
    print("First 10 features (atom_idx, feat_idx):")
    for r, c in zip(row[:10], col[:10]):
        print(f"  Atom {r}: Feature {c}")

if __name__ == '__main__':
    inspect_pkl()
