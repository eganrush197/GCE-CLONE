# ABOUTME: Level of Detail (LOD) generation for gaussian splats
# ABOUTME: Implements importance-based, opacity-based, and spatial pruning strategies

import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .mesh_to_gaussian import _SingleGaussian


class LODGenerator:
    """
    Generates multiple levels of detail (LOD) from gaussian splats.

    LOD generation reduces the number of gaussians while preserving visual quality
    by intelligently selecting which gaussians to keep based on different strategies.

    Available Pruning Strategies:

    1. 'importance' (RECOMMENDED - Best Quality)
       - Keeps gaussians with highest visual impact
       - Metric: opacity × volume (product of scales)
       - Best for: General use, balanced quality/performance
       - Example: Large, opaque gaussians are kept; small, transparent ones removed

    2. 'opacity' (Fast, Good Quality)
       - Keeps most opaque gaussians
       - Metric: opacity value only
       - Best for: When opacity is the primary quality indicator
       - Example: Fully opaque gaussians kept; transparent ones removed

    3. 'spatial' (Uniform Coverage)
       - Voxel-based spatial subsampling
       - Metric: spatial distribution in 3D grid
       - Best for: Maintaining uniform coverage across the model
       - Example: One gaussian per voxel, evenly distributed

    Usage:
        # Generate single LOD
        lod_gen = LODGenerator(strategy='importance')
        lod_5k = lod_gen.generate_lod(gaussians, 5000)

        # Generate multiple LODs
        lods = lod_gen.generate_lods(gaussians, [5000, 25000, 100000])
    """

    def __init__(self, strategy: str = 'importance'):
        """
        Initialize LOD generator.

        Args:
            strategy: Pruning strategy - 'importance', 'opacity', or 'spatial'
                     Default: 'importance' (recommended for best quality)
        """
        self.strategy = strategy

        if strategy not in ['importance', 'opacity', 'spatial']:
            raise ValueError(f"Unknown strategy: {strategy}. Must be 'importance', 'opacity', or 'spatial'")
    
    def generate_lod(self, gaussians: 'List[_SingleGaussian]', target_count: int, strategy: str = None) -> 'List[_SingleGaussian]':
        """
        Generate a single LOD level.

        Args:
            gaussians: List of _SingleGaussian objects
            target_count: Target number of gaussians for this LOD
            strategy: Optional override for pruning strategy

        Returns:
            List of _SingleGaussian objects (pruned)
        """
        if strategy:
            old_strategy = self.strategy
            self.strategy = strategy

        if target_count >= len(gaussians):
            result = gaussians
        else:
            result = self._prune_to_count(gaussians, target_count)

        if strategy:
            self.strategy = old_strategy

        return result

    def generate_lods(self, gaussians: 'List[_SingleGaussian]', target_counts: List[int]) -> 'List[List[_SingleGaussian]]':
        """
        Generate multiple LOD levels.

        Args:
            gaussians: List of _SingleGaussian objects
            target_counts: List of target gaussian counts for each LOD

        Returns:
            List of gaussian lists, one per LOD level
        """
        lods = []

        for target_count in sorted(target_counts, reverse=True):
            lod = self.generate_lod(gaussians, target_count)
            lods.append(lod)

        return lods

    def _prune_to_count(self, gaussians: 'List[_SingleGaussian]', target_count: int) -> 'List[_SingleGaussian]':
        """Prune gaussians to target count using selected strategy."""
        if self.strategy == 'importance':
            return self._prune_by_importance(gaussians, target_count)
        elif self.strategy == 'opacity':
            return self._prune_by_opacity(gaussians, target_count)
        elif self.strategy == 'spatial':
            return self._prune_by_spatial(gaussians, target_count)
    
    def _prune_by_importance(self, gaussians: 'List[_SingleGaussian]', target_count: int) -> 'List[_SingleGaussian]':
        """
        Prune by importance score (opacity * volume).

        Gaussians with higher opacity and larger volume are more important.
        """
        # Calculate importance: opacity * volume
        # Volume approximated by product of scales
        importance_scores = []
        for g in gaussians:
            volume = np.prod(g.scales)
            opacity = g.opacity  # Already in linear space (0-1)
            importance = opacity * volume
            importance_scores.append(importance)

        # Keep top N by importance
        importance_scores = np.array(importance_scores)
        indices = np.argsort(importance_scores)[-target_count:]

        return [gaussians[i] for i in sorted(indices)]
    
    def _prune_by_opacity(self, gaussians: 'List[_SingleGaussian]', target_count: int) -> 'List[_SingleGaussian]':
        """
        Prune by opacity threshold.

        Keep the most opaque gaussians.
        """
        opacities = np.array([g.opacity for g in gaussians])

        # Keep top N by opacity
        indices = np.argsort(opacities)[-target_count:]

        return [gaussians[i] for i in sorted(indices)]
    
    def _prune_by_spatial(self, gaussians: 'List[_SingleGaussian]', target_count: int) -> 'List[_SingleGaussian]':
        """
        Prune by spatial subsampling.

        Use voxel grid to ensure even spatial distribution.
        """
        # Calculate grid size based on target count
        grid_size = int(np.cbrt(target_count) * 1.5)

        # Extract positions
        positions = np.array([g.position for g in gaussians])

        # Compute bounding box
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        extent = max_pos - min_pos

        # Assign gaussians to voxels
        voxel_indices = ((positions - min_pos) / (extent + 1e-8) * grid_size).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)

        # Create voxel keys
        voxel_keys = voxel_indices[:, 0] * grid_size**2 + voxel_indices[:, 1] * grid_size + voxel_indices[:, 2]

        # For each voxel, keep the most important gaussian
        unique_voxels = np.unique(voxel_keys)
        selected_indices = []

        # Calculate importance for tie-breaking
        importance = []
        for g in gaussians:
            volume = np.prod(g.scales)
            opacity = g.opacity
            importance.append(opacity * volume)
        importance = np.array(importance)

        for voxel in unique_voxels:
            voxel_mask = voxel_keys == voxel
            voxel_gaussians = np.where(voxel_mask)[0]

            # Select most important gaussian in this voxel
            best_idx = voxel_gaussians[np.argmax(importance[voxel_gaussians])]
            selected_indices.append(best_idx)

            if len(selected_indices) >= target_count:
                break

        indices = selected_indices[:target_count]

        return [gaussians[i] for i in sorted(indices)]

