#!/usr/bin/env python3
"""
Simple wrapper script for mesh2gaussian conversion.
Handles file paths with spaces and provides better error messages.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.mesh_to_gaussian import MeshToGaussianConverter


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert.py <input_mesh> <output_ply> [options]")
        print()
        print("Examples:")
        print('  python convert.py "green tree.glb" output.ply')
        print('  python convert.py model.obj result.ply')
        print()
        print("Options:")
        print("  --strategy [vertex|face|hybrid|adaptive]  (default: adaptive)")
        print("  --samples-per-face N                      (default: 10)")
        print()
        return 1
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    # Parse optional arguments
    strategy = 'adaptive'
    samples_per_face = 10
    
    for i, arg in enumerate(sys.argv[3:], start=3):
        if arg == '--strategy' and i + 1 < len(sys.argv):
            strategy = sys.argv[i + 1]
        elif arg == '--samples-per-face' and i + 1 < len(sys.argv):
            samples_per_face = int(sys.argv[i + 1])
    
    # Validate input file
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1
    
    print("=" * 60)
    print("Gaussian Mesh Converter")
    print("=" * 60)
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")
    print(f"Strategy: {strategy}")
    print("=" * 60)
    print()
    
    try:
        # Create converter
        print("Loading mesh...")
        converter = MeshToGaussianConverter()

        # Load mesh
        mesh = converter.load_mesh(str(input_path))

        # Convert to gaussians
        print("Converting to gaussian splats...")
        gaussians = converter.mesh_to_gaussians(mesh, strategy=strategy, samples_per_face=samples_per_face)
        print(f"✓ Generated {len(gaussians)} gaussians")
        print()

        # Save
        print(f"Saving to {output_path}...")
        converter.save_ply(gaussians, str(output_path))
        file_size = output_path.stat().st_size
        print(f"✓ Saved {file_size:,} bytes")
        print()
        
        print("=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print()
        print(f"Output file: {output_path.absolute()}")
        print()
        print("View your gaussian splat at:")
        print("  https://playcanvas.com/supersplat")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

