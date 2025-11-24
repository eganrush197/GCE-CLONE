#!/usr/bin/env python3
# ABOUTME: Basic usage examples for mesh-to-gaussian converter
# ABOUTME: Demonstrates common conversion workflows

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mesh_to_gaussian import MeshToGaussianConverter
from lod_generator import LODGenerator


def example_basic_conversion():
    """Basic mesh to gaussian conversion."""
    print("Example 1: Basic Conversion")
    print("-" * 50)

    # Create converter with default settings
    converter = MeshToGaussianConverter()

    # Load and convert mesh
    mesh = converter.load_mesh('input.obj')
    gaussians = converter.mesh_to_gaussians(mesh, strategy='adaptive')

    # Save result
    converter.save_ply(gaussians, 'output.ply')

    print(f"Converted to {len(gaussians)} gaussians")
    print()


def example_custom_config():
    """Conversion with custom configuration."""
    print("Example 2: Custom Configuration")
    print("-" * 50)

    # Create converter
    converter = MeshToGaussianConverter(device='cpu')

    # Load and convert with hybrid strategy
    mesh = converter.load_mesh('input.glb')
    gaussians = converter.mesh_to_gaussians(mesh, strategy='hybrid', samples_per_face=15)
    converter.save_ply(gaussians, 'output_custom.ply')

    print(f"Converted with hybrid strategy: {len(gaussians)} gaussians")
    print()


def example_lod_generation():
    """Generate multiple LOD levels."""
    print("Example 3: LOD Generation")
    print("-" * 50)

    converter = MeshToGaussianConverter()
    mesh = converter.load_mesh('input.obj')
    gaussians = converter.mesh_to_gaussians(mesh, strategy='hybrid')

    # Generate LODs
    lod_gen = LODGenerator(strategy='importance')
    lod_counts = [5000, 25000, 100000]

    # Save each LOD
    for count in lod_counts:
        lod = lod_gen.generate_lod(gaussians, count)
        converter.save_ply(lod, f'output_lod{count}.ply')
        print(f"LOD {count}: {len(lod)} gaussians")

    print()


def example_vertex_strategy():
    """Use vertex-based initialization for low-poly meshes."""
    print("Example 4: Vertex Strategy (Low-Poly)")
    print("-" * 50)

    converter = MeshToGaussianConverter()
    mesh = converter.load_mesh('lowpoly.obj')
    gaussians = converter.mesh_to_gaussians(mesh, strategy='vertex')
    converter.save_ply(gaussians, 'output_vertex.ply')

    print(f"Vertex-based conversion: {len(gaussians)} gaussians")
    print()


def example_face_strategy():
    """Use face-based initialization for high-poly meshes."""
    print("Example 5: Face Strategy (High-Poly)")
    print("-" * 50)

    converter = MeshToGaussianConverter()
    mesh = converter.load_mesh('highpoly.glb')
    gaussians = converter.mesh_to_gaussians(mesh, strategy='face', samples_per_face=15)
    converter.save_ply(gaussians, 'output_face.ply')

    print(f"Face-based conversion: {len(gaussians)} gaussians")
    print()


def example_with_optimization():
    """Conversion with GPU optimization (requires PyTorch)."""
    print("Example 6: With GPU Optimization")
    print("-" * 50)

    converter = MeshToGaussianConverter(device='cuda')  # Use 'cpu' if no GPU
    mesh = converter.load_mesh('input.obj')
    gaussians = converter.mesh_to_gaussians(mesh, strategy='adaptive')

    # Optimize (if PyTorch available)
    optimized = converter.optimize_gaussians(gaussians, iterations=100)
    converter.save_ply(optimized, 'output_optimized.ply')

    print(f"Optimized conversion: {len(optimized)} gaussians")
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

