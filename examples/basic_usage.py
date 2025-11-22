#!/usr/bin/env python3
# ABOUTME: Basic usage examples for mesh-to-gaussian converter
# ABOUTME: Demonstrates common conversion workflows

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mesh_to_gaussian import MeshToGaussianConverter, ConversionConfig
from ply_io import save_ply


def example_basic_conversion():
    """Basic mesh to gaussian conversion."""
    print("Example 1: Basic Conversion")
    print("-" * 50)
    
    # Create converter with default settings
    converter = MeshToGaussianConverter()
    
    # Convert mesh
    gaussians = converter.convert('input.obj')
    
    # Save result
    save_ply(gaussians, 'output.ply')
    
    print(f"Converted to {gaussians.count} gaussians")
    print()


def example_custom_config():
    """Conversion with custom configuration."""
    print("Example 2: Custom Configuration")
    print("-" * 50)
    
    # Create custom config
    config = ConversionConfig(
        initialization_strategy='hybrid',
        target_gaussians=50000,
        scale_multiplier=1.2,
        opacity_default=0.95
    )
    
    converter = MeshToGaussianConverter(config)
    gaussians = converter.convert('input.glb')
    save_ply(gaussians, 'output_custom.ply')
    
    print(f"Converted with hybrid strategy: {gaussians.count} gaussians")
    print()


def example_lod_generation():
    """Generate multiple LOD levels."""
    print("Example 3: LOD Generation")
    print("-" * 50)
    
    converter = MeshToGaussianConverter()
    gaussians = converter.convert('input.obj')
    
    # Generate LODs
    lod_counts = [5000, 25000, 100000]
    lods = converter.generate_lods(gaussians, lod_counts)
    
    # Save each LOD
    for count, lod in zip(lod_counts, lods):
        save_ply(lod, f'output_lod{count}.ply')
        print(f"LOD {count}: {lod.count} gaussians")
    
    print()


def example_vertex_strategy():
    """Use vertex-based initialization for low-poly meshes."""
    print("Example 4: Vertex Strategy (Low-Poly)")
    print("-" * 50)
    
    config = ConversionConfig(
        initialization_strategy='vertex',
        scale_multiplier=0.8
    )
    
    converter = MeshToGaussianConverter(config)
    gaussians = converter.convert('lowpoly.obj')
    save_ply(gaussians, 'output_vertex.ply')
    
    print(f"Vertex-based conversion: {gaussians.count} gaussians")
    print()


def example_face_strategy():
    """Use face-based initialization for high-poly meshes."""
    print("Example 5: Face Strategy (High-Poly)")
    print("-" * 50)
    
    config = ConversionConfig(
        initialization_strategy='face',
        samples_per_face=15,
        target_gaussians=100000
    )
    
    converter = MeshToGaussianConverter(config)
    gaussians = converter.convert('highpoly.glb')
    save_ply(gaussians, 'output_face.ply')
    
    print(f"Face-based conversion: {gaussians.count} gaussians")
    print()


def example_with_optimization():
    """Conversion with GPU optimization (requires PyTorch)."""
    print("Example 6: With GPU Optimization")
    print("-" * 50)
    
    config = ConversionConfig(
        initialization_strategy='adaptive',
        optimize=True,
        optimization_iterations=100,
        device='cuda'  # Use 'cpu' if no GPU
    )
    
    converter = MeshToGaussianConverter(config)
    gaussians = converter.convert('input.obj')
    save_ply(gaussians, 'output_optimized.ply')
    
    print(f"Optimized conversion: {gaussians.count} gaussians")
    print()


if __name__ == '__main__':
    print("Mesh to Gaussian Converter - Usage Examples")
    print("=" * 50)
    print()
    
    # Note: These examples assume you have input mesh files
    # Uncomment the examples you want to run
    
    # example_basic_conversion()
    # example_custom_config()
    # example_lod_generation()
    # example_vertex_strategy()
    # example_face_strategy()
    # example_with_optimization()
    
    print("Note: Uncomment the examples you want to run")
    print("Make sure you have input mesh files (input.obj, input.glb, etc.)")

