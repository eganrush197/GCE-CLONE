# ABOUTME: Data structure for representing gaussian splat point clouds
# ABOUTME: Stores positions, scales, rotations, colors, and opacity for each gaussian

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GaussianSplat:
    """
    Represents a collection of 3D gaussian splats.
    
    Attributes:
        positions: (N, 3) array of gaussian centers
        scales: (N, 3) array of gaussian scales (log space)
        rotations: (N, 4) array of quaternions (w, x, y, z)
        colors: (N, 3) array of RGB colors [0-1]
        opacity: (N,) array of opacity values [0-1] (log space)
        sh_coefficients: Optional (N, K, 3) spherical harmonics coefficients
    """
    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    colors: np.ndarray
    opacity: np.ndarray
    sh_coefficients: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate gaussian splat data."""
        n = len(self.positions)
        
        assert self.positions.shape == (n, 3), "Positions must be (N, 3)"
        assert self.scales.shape == (n, 3), "Scales must be (N, 3)"
        assert self.rotations.shape == (n, 4), "Rotations must be (N, 4) quaternions"
        assert self.colors.shape == (n, 3), "Colors must be (N, 3)"
        assert self.opacity.shape == (n,), "Opacity must be (N,)"
        
        # Normalize quaternions
        norms = np.linalg.norm(self.rotations, axis=1, keepdims=True)
        self.rotations = self.rotations / (norms + 1e-8)
    
    @property
    def count(self) -> int:
        """Return number of gaussians."""
        return len(self.positions)
    
    def subset(self, indices: np.ndarray) -> 'GaussianSplat':
        """Create a subset of gaussians by indices."""
        return GaussianSplat(
            positions=self.positions[indices],
            scales=self.scales[indices],
            rotations=self.rotations[indices],
            colors=self.colors[indices],
            opacity=self.opacity[indices],
            sh_coefficients=self.sh_coefficients[indices] if self.sh_coefficients is not None else None
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = {
            'positions': self.positions,
            'scales': self.scales,
            'rotations': self.rotations,
            'colors': self.colors,
            'opacity': self.opacity,
        }
        if self.sh_coefficients is not None:
            data['sh_coefficients'] = self.sh_coefficients
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GaussianSplat':
        """Create from dictionary."""
        return cls(
            positions=data['positions'],
            scales=data['scales'],
            rotations=data['rotations'],
            colors=data['colors'],
            opacity=data['opacity'],
            sh_coefficients=data.get('sh_coefficients')
        )

