# ABOUTME: Level of Detail (LOD) generation for gaussian splats
# ABOUTME: Implements importance-based, opacity-based, and spatial pruning strategies

import numpy as np
from typing import List

try:
    from .gaussian_splat import GaussianSplat
except ImportError:
    from gaussian_splat import GaussianSplat


class LODGenerator:
    """Generates multiple levels of detail from a gaussian splat."""
    
    def __init__(self, strategy: str = 'importance'):
        """
        Initialize LOD generator.
        
        Args:
            strategy: Pruning strategy - 'importance', 'opacity', or 'spatial'
        """
        self.strategy = strategy
        
        if strategy not in ['importance', 'opacity', 'spatial']:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def generate_lods(self, gaussians: GaussianSplat, target_counts: List[int]) -> List[GaussianSplat]:
        """
        Generate multiple LOD levels.
        
        Args:
            gaussians: Source gaussian splat
            target_counts: List of target gaussian counts for each LOD
            
        Returns:
            List of GaussianSplat objects, one per LOD level
        """
        lods = []
        
        for target_count in sorted(target_counts, reverse=True):
            if target_count >= gaussians.count:
                lods.append(gaussians)
            else:
                lod = self._prune_to_count(gaussians, target_count)
                lods.append(lod)
        
        return lods
    
    def _prune_to_count(self, gaussians: GaussianSplat, target_count: int) -> GaussianSplat:
        """Prune gaussians to target count using selected strategy."""
        if self.strategy == 'importance':
            return self._prune_by_importance(gaussians, target_count)
        elif self.strategy == 'opacity':
            return self._prune_by_opacity(gaussians, target_count)
        elif self.strategy == 'spatial':
            return self._prune_by_spatial(gaussians, target_count)
    
    def _prune_by_importance(self, gaussians: GaussianSplat, target_count: int) -> GaussianSplat:
        """
        Prune by importance score (opacity * volume).
        
        Gaussians with higher opacity and larger volume are more important.
        """
        # Calculate importance: opacity * volume
        # Volume approximated by product of scales
        volumes = np.prod(np.exp(gaussians.scales), axis=1)
        opacities = 1.0 / (1.0 + np.exp(-gaussians.opacity))  # Sigmoid
        importance = opacities * volumes
        
        # Keep top N by importance
        indices = np.argsort(importance)[-target_count:]
        
        return gaussians.subset(indices)
    
    def _prune_by_opacity(self, gaussians: GaussianSplat, target_count: int) -> GaussianSplat:
        """
        Prune by opacity threshold.
        
        Keep the most opaque gaussians.
        """
        opacities = 1.0 / (1.0 + np.exp(-gaussians.opacity))  # Sigmoid
        
        # Keep top N by opacity
        indices = np.argsort(opacities)[-target_count:]
        
        return gaussians.subset(indices)
    
    def _prune_by_spatial(self, gaussians: GaussianSplat, target_count: int) -> GaussianSplat:
        """
        Prune by spatial subsampling.
        
        Use voxel grid to ensure even spatial distribution.
        """
        # Calculate grid size based on target count
        grid_size = int(np.cbrt(target_count) * 1.5)
        
        # Compute bounding box
        min_pos = gaussians.positions.min(axis=0)
        max_pos = gaussians.positions.max(axis=0)
        extent = max_pos - min_pos
        
        # Assign gaussians to voxels
        voxel_indices = ((gaussians.positions - min_pos) / (extent + 1e-8) * grid_size).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)
        
        # Create voxel keys
        voxel_keys = voxel_indices[:, 0] * grid_size**2 + voxel_indices[:, 1] * grid_size + voxel_indices[:, 2]
        
        # For each voxel, keep the most important gaussian
        unique_voxels = np.unique(voxel_keys)
        selected_indices = []
        
        # Calculate importance for tie-breaking
        volumes = np.prod(np.exp(gaussians.scales), axis=1)
        opacities = 1.0 / (1.0 + np.exp(-gaussians.opacity))
        importance = opacities * volumes
        
        for voxel in unique_voxels:
            voxel_mask = voxel_keys == voxel
            voxel_gaussians = np.where(voxel_mask)[0]
            
            # Select most important gaussian in this voxel
            best_idx = voxel_gaussians[np.argmax(importance[voxel_gaussians])]
            selected_indices.append(best_idx)
            
            if len(selected_indices) >= target_count:
                break
        
        indices = np.array(selected_indices[:target_count])
        
        return gaussians.subset(indices)

