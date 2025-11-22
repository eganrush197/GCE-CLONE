# ABOUTME: PLY file I/O for gaussian splat format
# ABOUTME: Handles reading and writing .ply files with gaussian splat attributes

import numpy as np
import struct
from pathlib import Path
from typing import Union
from .gaussian_splat import GaussianSplat


def save_ply(gaussians: GaussianSplat, filepath: Union[str, Path]) -> None:
    """
    Save gaussian splats to PLY file format.
    
    Args:
        gaussians: GaussianSplat object to save
        filepath: Output PLY file path
    """
    filepath = Path(filepath)
    
    # Prepare data
    n = gaussians.count
    
    # Convert log-space values to linear for PLY
    scales_linear = np.exp(gaussians.scales)
    opacity_linear = 1.0 / (1.0 + np.exp(-gaussians.opacity))  # Sigmoid
    
    # Convert colors to 0-255 range
    colors_uint8 = (np.clip(gaussians.colors, 0, 1) * 255).astype(np.uint8)
    
    # Write PLY header
    with open(filepath, 'wb') as f:
        # Header
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {n}\n'.encode())
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        f.write(b'property float nx\n')  # Normal (derived from rotation)
        f.write(b'property float ny\n')
        f.write(b'property float nz\n')
        f.write(b'property uchar red\n')
        f.write(b'property uchar green\n')
        f.write(b'property uchar blue\n')
        f.write(b'property float scale_0\n')
        f.write(b'property float scale_1\n')
        f.write(b'property float scale_2\n')
        f.write(b'property float rot_0\n')  # Quaternion w
        f.write(b'property float rot_1\n')  # Quaternion x
        f.write(b'property float rot_2\n')  # Quaternion y
        f.write(b'property float rot_3\n')  # Quaternion z
        f.write(b'property float opacity\n')
        f.write(b'end_header\n')
        
        # Write binary data
        for i in range(n):
            # Position
            f.write(struct.pack('fff', *gaussians.positions[i]))
            
            # Normal (compute from quaternion - simplified to z-axis)
            # For proper implementation, rotate [0,0,1] by quaternion
            normal = _quaternion_rotate(gaussians.rotations[i], np.array([0, 0, 1]))
            f.write(struct.pack('fff', *normal))
            
            # Color
            f.write(struct.pack('BBB', *colors_uint8[i]))
            
            # Scales
            f.write(struct.pack('fff', *scales_linear[i]))
            
            # Rotation (quaternion)
            f.write(struct.pack('ffff', *gaussians.rotations[i]))
            
            # Opacity
            f.write(struct.pack('f', opacity_linear[i]))


def load_ply(filepath: Union[str, Path]) -> GaussianSplat:
    """
    Load gaussian splats from PLY file.
    
    Args:
        filepath: Input PLY file path
        
    Returns:
        GaussianSplat object
    """
    # TODO: Implement PLY loading
    raise NotImplementedError("PLY loading not yet implemented")


def _quaternion_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q.
    
    Args:
        q: Quaternion (w, x, y, z)
        v: Vector (x, y, z)
        
    Returns:
        Rotated vector
    """
    w, x, y, z = q
    vx, vy, vz = v
    
    # Quaternion rotation formula
    t_x = 2 * (y * vz - z * vy)
    t_y = 2 * (z * vx - x * vz)
    t_z = 2 * (x * vy - y * vx)
    
    return np.array([
        vx + w * t_x + (y * t_z - z * t_y),
        vy + w * t_y + (z * t_x - x * t_z),
        vz + w * t_z + (x * t_y - y * t_x)
    ])

