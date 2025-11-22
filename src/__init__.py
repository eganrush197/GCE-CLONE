# ABOUTME: Package initialization for gaussian mesh converter
# ABOUTME: Exports main classes for easy importing

from .mesh_to_gaussian import MeshToGaussianConverter, ConversionConfig
from .gaussian_splat import GaussianSplat
from .lod_generator import LODGenerator
from .ply_io import save_ply, load_ply

__version__ = "0.1.0"

__all__ = [
    "MeshToGaussianConverter",
    "ConversionConfig",
    "GaussianSplat",
    "LODGenerator",
    "save_ply",
    "load_ply",
]

