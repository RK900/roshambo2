
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from roshambo2.backends.minimal_cuda_backend import PersistentCudaShapeOverlay, SimpleRoshamboData, BitVectorColorGenerator, _quaternion_to_matrix

class PheneOverlayBatchResult:
    """
    Structured result container for PheneShapeOverlay.
    """
    def __init__(self, raw_scores: np.ndarray, num_anchors: int, batch_size: int):
        # raw_scores shape: (num_anchors, batch_size, 20)
        self.raw_scores = raw_scores
        self.num_anchors = num_anchors
        self.batch_size = batch_size

    @property
    def scores(self) -> np.ndarray:
        """Combined Roshambo2 Combo Score (num_anchors, batch_size)"""
        return self.raw_scores[:, :, 0]

    @property
    def shape_scores(self) -> np.ndarray:
        """Shape Only Score (num_anchors, batch_size)"""
        return self.raw_scores[:, :, 1]

    @property
    def color_scores(self) -> np.ndarray:
        """Color Only Score (num_anchors, batch_size)"""
        return self.raw_scores[:, :, 2]

    @property
    def rotation(self) -> np.ndarray:
        """Rotation Matrices (num_anchors, batch_size, 3, 3)"""
        # Quaternion is at indices 9:13 [w, x, y, z]
        # We need to vectorize the conversion
        quats = self.raw_scores[:, :, 9:13] # (NA, B, 4)
        
        # Vectorized conversion would be faster but for simplicity/correctness reuse helper
        # Or implement vectorized numpy version here.
        # Let's do a simple loop for now as this is just result extraction, not inner loop compute.
        # But for large batches loop is slow. Let's vectorize.
        
        w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
        
        # N = num_anchors * batch_size
        # Shape output (NA, B, 3, 3)
        R = np.empty(quats.shape[:-1] + (3, 3), dtype=np.float32)
        
        # Diagonal
        R[..., 0, 0] = 1 - 2*y*y - 2*z*z
        R[..., 1, 1] = 1 - 2*x*x - 2*z*z
        R[..., 2, 2] = 1 - 2*x*x - 2*y*y
        
        # Off-diagonal
        R[..., 0, 1] = 2*x*y - 2*z*w
        R[..., 0, 2] = 2*x*z + 2*y*w
        
        R[..., 1, 0] = 2*x*y + 2*z*w
        R[..., 1, 2] = 2*y*z - 2*x*w
        
        R[..., 2, 0] = 2*x*z - 2*y*w
        R[..., 2, 1] = 2*y*z + 2*x*w
        
        return R

    @property
    def translation(self) -> np.ndarray:
        """Translation Vectors (num_anchors, batch_size, 3)"""
        # Translation is at indices 13:16 [tx, ty, tz]
        return self.raw_scores[:, :, 13:16]


class PheneShapeOverlay:
    """
    Roshambo2 Overlay Wrapper for Phene PyTorch Training Loops.
    
    Handles:
    - Unbatching of PyTorch Geometric style flattened batches.
    - Persistent CUDA Context Management.
    - Anchor vs Batch comparison.
    - Feature filtering for performance.
    """
    
    def __init__(self, max_mols: int = 256, max_atoms: int = 300, n_gpus: int = 1,
                 allowed_features: List[int] = [18, 19, 20, 21, 22, 36, 41, 42]):
        """
        Args:
            max_mols: Maximum batch size expected.
            max_atoms: Maximum atoms (nodes) per molecule expected.
            n_gpus: Number of GPUs to use.
            allowed_features: List of feature indices to use for color alignment.
                              Default is [Charge, Aromatic, HB Donor/Acceptor].
        """
        self.max_mols = max_mols
        self.max_atoms = max_atoms
        self.allowed_features = allowed_features
        
        # Initialize Color Generator
        # Phene generator produces 48 dim features + padding -> 49
        self.color_gen = BitVectorColorGenerator(n_bits=48)
        
        # Initialize Persistent Context
        # Note: minimal_cuda_backend handles the allowed_features filtering in SimpleRoshamboData
        # We just need to ensure we initialize the context with the correct n_features (49)
        self.overlay = PersistentCudaShapeOverlay(
            max_mols=max_mols, 
            max_atoms=max_atoms, 
            n_features=self.color_gen.matrix_size, 
            n_gpus=n_gpus
        )

    def _unbatch(self, batch_data) -> List[Dict[str, np.ndarray]]:
        """
        Splits the flattened PyG batch into a list of individual molecule dictionaries.
        Arguments:
            batch_data: Dictionary containing 'molecular_graph' (PyG Batch) and 'anchor_indices'.
        """
        # Helper to get numpy
        def to_numpy(x):
            if hasattr(x, 'detach'):
                return x.detach().cpu().numpy()
            return x

        # Extract Molecular Graph Object
        # Supports dict access or attribute access
        if isinstance(batch_data, dict):
            mol_graph = batch_data['molecular_graph']
        else:
            mol_graph = batch_data.molecular_graph

        # Extract Tensors from PyG Batch object
        # (.x, .pos, .batch)
        nodes = to_numpy(mol_graph.x)
        pos = to_numpy(mol_graph.pos)
        batch_idx = to_numpy(mol_graph.batch)
        
        # Split into list
        # Since PyG batches are concatenated, batch_idx is monotonic (0,0...1,1...2,2...)
        # We can use np.unique counts to split
        unique_ids, counts = np.unique(batch_idx, return_counts=True)
        
        # Handle case where there might be gaps or empty graphs (unlikely in standard collation but possible)
        # Assuming sequential 0..B-1
        
        split_indices = np.cumsum(counts)[:-1]
        
        split_nodes = np.split(nodes, split_indices)
        split_pos = np.split(pos, split_indices)
        
        mol_list = []
        for n, p in zip(split_nodes, split_pos):
            mol_list.append({
                'graph_nodes': n,
                'graph_pos': p
            })
            
        return mol_list

    def calculate(self, batch_data, mixing: float = 0.5) -> PheneOverlayBatchResult:
        """
        Calculate overlap for all anchors against all candidates in the batch.
        
        Args:
            batch_data: Batch object/dict containing 'molecular_graph' and 'anchor_indices'.
            mixing: Color weight (0.0 to 1.0).
            
        Returns:
            PheneOverlayBatchResult object containing scores, rotation, translation.
        """
        
        # 1. Unbatch everything
        mol_list = self._unbatch(batch_data)
        batch_size = len(mol_list)
        
        if batch_size == 0:
             return PheneOverlayBatchResult(np.zeros((0,0,20)), 0, 0)

        # 2. Identify Anchors
        # Try item access then attribute access
        if isinstance(batch_data, dict):
            anchor_indices = batch_data.get('anchor_indices')
        else:
            anchor_indices = getattr(batch_data, 'anchor_indices', None)

        if anchor_indices is None:
            anchor_indices = []
        else:
            if hasattr(anchor_indices, 'detach'):
                anchor_indices = anchor_indices.detach().cpu().numpy()
            if not isinstance(anchor_indices, list):
                anchor_indices = anchor_indices.tolist()

        num_anchors = len(anchor_indices)
        if num_anchors == 0:
             # Nothing to compare
             return PheneOverlayBatchResult(np.zeros((0, batch_size, 20)), 0, batch_size)

        # 3. Prepare Candidates (All molecules)
        # We construct SimpleRoshamboData for the whole batch
        # Filter is handled by allowed_features in __init__
        
        candidates_data = SimpleRoshamboData(mol_list, name="BatchCandidates", allowed_features=self.allowed_features)
        
        # Allocate result tensor for (NumAnchors, BatchSize, 20)
        all_scores = np.zeros((num_anchors, batch_size, 20), dtype=np.float32)
        
        for i, anchor_idx in enumerate(anchor_indices):
            anchor_idx = int(anchor_idx)
            if anchor_idx >= batch_size: continue
            
            # Extract single anchor molecule
            anchor_mol = mol_list[anchor_idx]
            query_data = SimpleRoshamboData(anchor_mol, name=f"Anchor_{anchor_idx}", allowed_features=self.allowed_features)
            
            # Run Overlay
            # query_data: (1 iter)
            # candidates_data: (N iter)
            # Result: (1, N, 20)
            scores = self.overlay.calculate_overlap_batch(query_data, candidates_data, mixing=mixing, color_generator=self.color_gen)
            
            all_scores[i] = scores[0]
            
        return PheneOverlayBatchResult(all_scores, num_anchors, batch_size)
