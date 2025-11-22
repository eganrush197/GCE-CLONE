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

from src.mesh_to_gaussian import MeshToGaussianConverter, ConversionConfig
from src.ply_io import save_ply

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
    config = ConversionConfig(initialization_strategy='vertex')
    converter = MeshToGaussianConverter(config)
    gaussians = converter.convert(mesh_path)
    print(f"   ✓ Generated {gaussians.count} gaussians")
    
    output_path = Path("test_output_vertex.ply")
    save_ply(gaussians, output_path)
    print(f"   ✓ Saved to {output_path} ({output_path.stat().st_size} bytes)")
    print()
    
    # Test 2: Face strategy
    print("4. Testing FACE strategy...")
    config = ConversionConfig(
        initialization_strategy='face',
        samples_per_face=10
    )
    converter = MeshToGaussianConverter(config)
    gaussians = converter.convert(mesh_path)
    print(f"   ✓ Generated {gaussians.count} gaussians")
    
    output_path = Path("test_output_face.ply")
    save_ply(gaussians, output_path)
    print(f"   ✓ Saved to {output_path} ({output_path.stat().st_size} bytes)")
    print()
    
    # Test 3: Hybrid strategy
    print("5. Testing HYBRID strategy...")
    config = ConversionConfig(initialization_strategy='hybrid')
    converter = MeshToGaussianConverter(config)
    gaussians = converter.convert(mesh_path)
    print(f"   ✓ Generated {gaussians.count} gaussians")
    
    output_path = Path("test_output_hybrid.ply")
    save_ply(gaussians, output_path)
    print(f"   ✓ Saved to {output_path} ({output_path.stat().st_size} bytes)")
    print()
    
    # Test 4: LOD generation
    print("6. Testing LOD generation...")
    lod_counts = [10, 25, 50]
    lods = converter.generate_lods(gaussians, lod_counts)
    print(f"   ✓ Generated {len(lods)} LOD levels")
    
    for count, lod in zip(sorted(lod_counts), lods):
        lod_path = Path(f"test_output_lod{count}.ply")
        save_ply(lod, lod_path)
        print(f"   ✓ LOD {count}: {lod.count} gaussians -> {lod_path} ({lod_path.stat().st_size} bytes)")
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

