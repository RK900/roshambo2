
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch_geometric.utils import to_dense_batch

from roshambo2.backends.minimal_cuda_backend import PersistentCudaShapeOverlay, SimpleRoshamboData, BitVectorColorGenerator, _quaternion_to_matrix


class PheneOverlayBatchResult:
    """
    Structured result container for PheneShapeOverlay.
    Supports both numpy (legacy) and torch tensor (zero-copy) storage.
    """
    def __init__(self, raw_scores, num_anchors: int, batch_size: int):
        # raw_scores shape: (num_anchors, batch_size, 20)
        # Can be np.ndarray or torch.Tensor
        self._raw_scores = raw_scores
        self.num_anchors = num_anchors
        self.batch_size = batch_size
        self._is_torch = isinstance(raw_scores, torch.Tensor)

    @property
    def raw_scores(self):
        """Raw scores as numpy array (for backward compatibility)."""
        if self._is_torch:
            return self._raw_scores.cpu().numpy()
        return self._raw_scores

    @property
    def raw_scores_torch(self) -> torch.Tensor:
        """Raw scores as torch tensor (zero-copy if already on GPU)."""
        if self._is_torch:
            return self._raw_scores
        return torch.from_numpy(self._raw_scores)

    @property
    def scores(self) -> np.ndarray:
        """Combined Roshambo2 Combo Score (num_anchors, batch_size)"""
        return self.raw_scores[:, :, 0]

    @property
    def scores_torch(self) -> torch.Tensor:
        """Combined Roshambo2 Combo Score as torch tensor (num_anchors, batch_size)"""
        return self.raw_scores_torch[:, :, 0]

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
        quats = self.raw_scores[:, :, 9:13]
        w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
        R = np.empty(quats.shape[:-1] + (3, 3), dtype=np.float32)
        R[..., 0, 0] = 1 - 2*y*y - 2*z*z
        R[..., 1, 1] = 1 - 2*x*x - 2*z*z
        R[..., 2, 2] = 1 - 2*x*x - 2*y*y
        R[..., 0, 1] = 2*x*y - 2*z*w
        R[..., 0, 2] = 2*x*z + 2*y*w
        R[..., 1, 0] = 2*x*y + 2*z*w
        R[..., 1, 2] = 2*y*z - 2*x*w
        R[..., 2, 0] = 2*x*z - 2*y*w
        R[..., 2, 1] = 2*y*z + 2*x*w
        return R

    @property
    def rotation_torch(self) -> torch.Tensor:
        """Rotation Matrices as torch tensor (num_anchors, batch_size, 3, 3)"""
        raw = self.raw_scores_torch
        quats = raw[:, :, 9:13]
        w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
        R = torch.empty(quats.shape[:-1] + (3, 3), dtype=torch.float32, device=quats.device)
        R[..., 0, 0] = 1 - 2*y*y - 2*z*z
        R[..., 1, 1] = 1 - 2*x*x - 2*z*z
        R[..., 2, 2] = 1 - 2*x*x - 2*y*y
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
        return self.raw_scores[:, :, 13:16]

    @property
    def translation_torch(self) -> torch.Tensor:
        """Translation Vectors as torch tensor (num_anchors, batch_size, 3)"""
        return self.raw_scores_torch[:, :, 13:16]


class PheneShapeOverlay:
    """
    Roshambo2 Overlay Wrapper for Phene PyTorch Training Loops.

    Handles:
    - Unbatching of PyTorch Geometric style flattened batches.
    - Persistent CUDA Context Management.
    - Anchor vs Batch comparison.
    - Feature filtering for performance.
    - GPU-native zero-copy path (avoids CPU<->GPU round-trips).
    """

    def __init__(self, max_mols: int = 256, max_atoms: int = 300, n_gpus: int = 1,
                 device_id: int = 0,
                 allowed_features: List[int] = [18, 19, 20, 21, 22, 36, 41, 42]):
        """
        Args:
            max_mols: Maximum batch size expected.
            max_atoms: Maximum atoms (nodes) per molecule expected.
            n_gpus: Number of GPUs to use for roshambo computation.
            device_id: CUDA device ID to allocate on (for DDP, pass local GPU index).
            allowed_features: List of feature indices to use for color alignment.
                              Default is [Charge, Aromatic, HB Donor/Acceptor].
        """
        self.max_mols = max_mols
        self.max_atoms = max_atoms
        self.allowed_features = allowed_features
        self._allowed_tensor = None  # Lazily initialized on correct device

        # Initialize Color Generator
        # Phene generator produces 48 dim features + padding -> 49
        self.color_gen = BitVectorColorGenerator(n_bits=48)

        # Initialize Persistent Context on the correct device
        self.overlay = PersistentCudaShapeOverlay(
            max_mols=max_mols,
            max_atoms=max_atoms,
            n_features=self.color_gen.matrix_size,
            n_gpus=n_gpus,
            device_id=device_id,
        )

    def _get_allowed_tensor(self, device):
        """Lazily create the allowed_features tensor on the correct device."""
        if self._allowed_tensor is None or self._allowed_tensor.device != device:
            self._allowed_tensor = torch.tensor(
                sorted(self.allowed_features), device=device, dtype=torch.long
            )
        return self._allowed_tensor

    def _prepare_batch_torch(self, mol_graph):
        """
        GPU-native feature expansion. Replaces _unbatch + SimpleRoshamboData.
        Converts a PyG Batch into padded (B, max_atoms, 4) tensors entirely on GPU.

        Args:
            mol_graph: PyG Batch with .x (N_total, 48), .pos (N_total, 3), .batch (N_total,)

        Returns:
            f_x: (B, max_atoms_dense, 4) float32 CUDA tensor [x, y, z, weight]
            f_types: (B, max_atoms_dense) int32 CUDA tensor
            f_n_real: (B,) int32 CUDA tensor (number of real atoms per molecule)
            num_graphs: int
        """
        x = mol_graph.x           # (N_total, 48)
        pos = mol_graph.pos        # (N_total, 3)
        batch_idx = mol_graph.batch  # (N_total,)
        device = x.device

        num_graphs = int(batch_idx.max().item()) + 1
        allowed = self._get_allowed_tensor(device)

        # Count real atoms per molecule
        n_real = torch.zeros(num_graphs, device=device, dtype=torch.int32)
        n_real.scatter_add_(0, batch_idx.long(), torch.ones(batch_idx.shape[0], device=device, dtype=torch.int32))

        # Feature expansion: find active (> 0.5) features in allowed columns
        feat_vals = x[:, allowed]  # (N_total, n_allowed)
        active_rows, active_cols_local = torch.where(feat_vals > 0.5)

        # Feature types: allowed[col] + 1 (1-based, type 0 = real atom)
        feat_types = (allowed[active_cols_local] + 1).int()
        feat_pos = pos[active_rows]
        feat_batch = batch_idx[active_rows]

        # Concatenate real atoms + feature atoms
        all_pos = torch.cat([pos, feat_pos], dim=0)
        all_types = torch.cat([
            torch.zeros(pos.shape[0], device=device, dtype=torch.int32),
            feat_types
        ], dim=0)
        all_batch = torch.cat([batch_idx, feat_batch], dim=0)

        # Sort by batch index (required by to_dense_batch)
        sort_idx = torch.argsort(all_batch, stable=True)
        all_pos = all_pos[sort_idx]
        all_types = all_types[sort_idx]
        all_batch = all_batch[sort_idx]

        # Pad to dense: (B, max_atoms_dense, 3) and mask
        pos_dense, mask = to_dense_batch(all_pos, all_batch)

        # Types: use float wrapper for to_dense_batch, then cast back
        types_dense, _ = to_dense_batch(
            all_types.float().unsqueeze(-1), all_batch
        )
        types_dense = types_dense.squeeze(-1).int()

        # Build f_x: (B, max_atoms_dense, 4) with weight=1.0 for valid atoms
        weight = mask.float().unsqueeze(-1)
        f_x = torch.cat([pos_dense, weight], dim=-1)

        return f_x.contiguous(), types_dense.contiguous(), n_real, num_graphs

    def _unbatch(self, batch_data) -> List[Dict[str, np.ndarray]]:
        """
        Splits the flattened PyG batch into a list of individual molecule dictionaries.
        Legacy CPU path — kept for backward compatibility.
        """
        def to_numpy(x):
            if hasattr(x, 'detach'):
                return x.detach().cpu().numpy()
            return x

        if isinstance(batch_data, dict):
            mol_graph = batch_data['molecular_graph']
        else:
            mol_graph = batch_data.molecular_graph

        nodes = to_numpy(mol_graph.x)
        pos = to_numpy(mol_graph.pos)
        batch_idx = to_numpy(mol_graph.batch)

        unique_ids, counts = np.unique(batch_idx, return_counts=True)
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
        Automatically uses the zero-copy GPU path when data is on CUDA.

        Args:
            batch_data: Batch object/dict containing 'molecular_graph' and 'anchor_indices'.
            mixing: Color weight (0.0 to 1.0).

        Returns:
            PheneOverlayBatchResult object containing scores, rotation, translation.
        """
        # Extract molecular graph
        if isinstance(batch_data, dict):
            mol_graph = batch_data['molecular_graph']
            anchor_indices = batch_data.get('anchor_indices')
        else:
            mol_graph = batch_data.molecular_graph
            anchor_indices = getattr(batch_data, 'anchor_indices', None)

        # Normalize anchor_indices
        if anchor_indices is None:
            anchor_indices = []
        else:
            if hasattr(anchor_indices, 'detach'):
                anchor_indices = anchor_indices.detach().cpu().tolist()
            elif not isinstance(anchor_indices, list):
                anchor_indices = list(anchor_indices)

        # Check if we can use the zero-copy GPU path
        use_torch_path = (
            hasattr(mol_graph, 'x') and
            mol_graph.x.is_cuda and
            hasattr(self.overlay.ctx, 'optimize_torch')
        )

        if use_torch_path:
            return self._calculate_torch(mol_graph, anchor_indices, mixing)
        else:
            return self._calculate_numpy(batch_data, anchor_indices, mixing)

    def _calculate_torch(self, mol_graph, anchor_indices, mixing: float) -> PheneOverlayBatchResult:
        """
        Zero-copy GPU path. Does feature expansion on GPU and passes tensor
        data_ptrs directly to the CUDA kernel — no CPU<->GPU copies.
        """
        # 1. GPU-native feature expansion for entire batch
        f_x, f_types, f_n_real, batch_size = self._prepare_batch_torch(mol_graph)

        if batch_size == 0:
            device = mol_graph.x.device
            return PheneOverlayBatchResult(
                torch.zeros((0, 0, 20), device=device), 0, 0
            )

        num_anchors = len(anchor_indices)
        if num_anchors == 0:
            device = mol_graph.x.device
            return PheneOverlayBatchResult(
                torch.zeros((0, batch_size, 20), device=device), 0, batch_size
            )

        # 2. Extract anchor data by indexing the dense batch tensors
        valid_anchor_indices = [int(i) for i in anchor_indices if int(i) < batch_size]

        if len(valid_anchor_indices) == 0:
            device = mol_graph.x.device
            return PheneOverlayBatchResult(
                torch.zeros((num_anchors, batch_size, 20), dtype=torch.float32, device=device),
                num_anchors, batch_size
            )

        idx_tensor = torch.tensor(valid_anchor_indices, device=f_x.device, dtype=torch.long)
        query_f_x = f_x[idx_tensor]           # (n_valid_anchors, max_atoms, 4)
        query_f_types = f_types[idx_tensor]    # (n_valid_anchors, max_atoms)
        query_f_n_real = f_n_real[idx_tensor]  # (n_valid_anchors,)

        # 3. Single CUDA call — zero copy
        all_scores = self.overlay.calculate_overlap_batch_torch(
            query_f_x, query_f_types, query_f_n_real,
            f_x, f_types, f_n_real,
            self.color_gen, mixing=mixing
        )

        # 4. Handle invalid anchors by padding
        if len(valid_anchor_indices) < num_anchors:
            device = all_scores.device
            padded = torch.zeros((num_anchors, batch_size, 20), dtype=torch.float32, device=device)
            padded[:len(valid_anchor_indices)] = all_scores
            all_scores = padded

        return PheneOverlayBatchResult(all_scores, num_anchors, batch_size)

    def _calculate_numpy(self, batch_data, anchor_indices, mixing: float) -> PheneOverlayBatchResult:
        """
        Legacy CPU/numpy path. Used as fallback when data is not on CUDA
        or optimize_torch is not available.
        """
        mol_list = self._unbatch(batch_data)
        batch_size = len(mol_list)

        if batch_size == 0:
            return PheneOverlayBatchResult(np.zeros((0, 0, 20)), 0, 0)

        num_anchors = len(anchor_indices)
        if num_anchors == 0:
            return PheneOverlayBatchResult(np.zeros((0, batch_size, 20)), 0, batch_size)

        candidates_data = SimpleRoshamboData(mol_list, name="BatchCandidates", allowed_features=self.allowed_features)

        anchor_mols = []
        valid_anchor_indices = []
        for anchor_idx in anchor_indices:
            anchor_idx = int(anchor_idx)
            if anchor_idx < batch_size:
                anchor_mols.append(mol_list[anchor_idx])
                valid_anchor_indices.append(anchor_idx)

        if len(anchor_mols) == 0:
            return PheneOverlayBatchResult(
                np.zeros((num_anchors, batch_size, 20), dtype=np.float32), num_anchors, batch_size
            )

        query_data = SimpleRoshamboData(anchor_mols, name="Anchors", allowed_features=self.allowed_features)
        all_scores = self.overlay.calculate_overlap_batch(
            query_data, candidates_data, mixing=mixing, color_generator=self.color_gen
        )

        if len(valid_anchor_indices) < num_anchors:
            padded_scores = np.zeros((num_anchors, batch_size, 20), dtype=np.float32)
            for i, _ in enumerate(valid_anchor_indices):
                padded_scores[i] = all_scores[i]
            all_scores = padded_scores

        return PheneOverlayBatchResult(all_scores, num_anchors, batch_size)
