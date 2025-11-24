#!/usr/bin/env python3
"""
Quick test script to verify the conversion pipeline works end-to-end.
Creates a simple cube mesh and converts it to gaussian splat.
"""

import sys
from pathlib import Path
import trimesh

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.mesh_to_gaussian import MeshToGaussianConverter
from src.lod_generator import LODGenerator

def main():
    print("=" * 60)
    print("Gaussian Mesh Converter - End-to-End Test")
    print("=" * 60)
    print()
    
    # Create a simple test mesh (cube)
    print("1. Creating test mesh (cube)...")
    cube = trimesh.creation.box(extents=[2, 2, 2])
    print(f"   ✓ Created cube with {len(cube.vertices)} vertices, {len(cube.faces)} faces")
    print()
    
    # Save mesh to file
    mesh_path = Path("test_cube.obj")
    print(f"2. Saving mesh to {mesh_path}...")
    cube.export(mesh_path)
    print(f"   ✓ Saved")
    print()
    
    # Test 1: Vertex strategy
    print("3. Testing VERTEX strategy...")
    converter = MeshToGaussianConverter()
    mesh = converter.load_mesh(str(mesh_path))
    gaussians = converter.mesh_to_gaussians(mesh, strategy='vertex')
    print(f"   ✓ Generated {len(gaussians)} gaussians")

    output_path = Path("test_output_vertex.ply")
    converter.save_ply(gaussians, str(output_path))
    print(f"   ✓ Saved to {output_path} ({output_path.stat().st_size} bytes)")
    print()

    # Test 2: Face strategy
    print("4. Testing FACE strategy...")
    converter = MeshToGaussianConverter()
    mesh = converter.load_mesh(str(mesh_path))
    gaussians = converter.mesh_to_gaussians(mesh, strategy='face', samples_per_face=10)
    print(f"   ✓ Generated {len(gaussians)} gaussians")

    output_path = Path("test_output_face.ply")
    converter.save_ply(gaussians, str(output_path))
    print(f"   ✓ Saved to {output_path} ({output_path.stat().st_size} bytes)")
    print()

    # Test 3: Hybrid strategy
    print("5. Testing HYBRID strategy...")
    converter = MeshToGaussianConverter()
    mesh = converter.load_mesh(str(mesh_path))
    gaussians = converter.mesh_to_gaussians(mesh, strategy='hybrid')
    print(f"   ✓ Generated {len(gaussians)} gaussians")

    output_path = Path("test_output_hybrid.ply")
    converter.save_ply(gaussians, str(output_path))
    print(f"   ✓ Saved to {output_path} ({output_path.stat().st_size} bytes)")
    print()
    
    # Test 4: LOD generation
    print("6. Testing LOD generation...")
    lod_gen = LODGenerator(strategy='importance')
    lod_counts = [10, 25, 50]
    print(f"   ✓ Generating {len(lod_counts)} LOD levels")

    for count in sorted(lod_counts):
        lod = lod_gen.generate_lod(gaussians, count)
        lod_path = Path(f"test_output_lod{count}.ply")
        converter.save_ply(lod, str(lod_path))
        print(f"   ✓ LOD {count}: {len(lod)} gaussians -> {lod_path} ({lod_path.stat().st_size} bytes)")
    print()
    
    # Summary
    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  - test_cube.obj (input mesh)")
    print("  - test_output_vertex.ply")
    print("  - test_output_face.ply")
    print("  - test_output_hybrid.ply")
    print("  - test_output_lod10.ply")
    print("  - test_output_lod25.ply")
    print("  - test_output_lod50.ply")
    print()
    print("You can view these PLY files in:")
    print("  - SuperSplat: https://playcanvas.com/supersplat")
    print("  - Any PLY viewer")
    print()
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

