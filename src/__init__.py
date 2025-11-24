# ABOUTME: Package initialization for gaussian mesh converter
# ABOUTME: Exports main classes for easy importing

try:
    from .mesh_to_gaussian import MeshToGaussianConverter
    from .gaussian_splat import GaussianSplat
    from .lod_generator import LODGenerator
except ImportError:
    # Fallback for when imported as a script
    from mesh_to_gaussian import MeshToGaussianConverter
    from gaussian_splat import GaussianSplat
    from lod_generator import LODGenerator

__version__ = "0.1.0"

__all__ = [
    "MeshToGaussianConverter",
    "GaussianSplat",
    "LODGenerator",
]

