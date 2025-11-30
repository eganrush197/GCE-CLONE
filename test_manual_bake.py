"""
Manual test script for Blender baker.
This will help us debug the baking process.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stage1_baker import BlenderBaker

# Blender path
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 3.1\blender.exe"

# Create baker
print("Creating BlenderBaker...")
baker = BlenderBaker(blender_executable=BLENDER_EXE)

# Test asset
test_asset = Path("test_assets/simple_cube.blend")

if not test_asset.exists():
    print(f"ERROR: Test asset not found: {test_asset}")
    sys.exit(1)

print(f"\nTest asset: {test_asset}")
print(f"Test asset exists: {test_asset.exists()}")
print(f"Test asset size: {test_asset.stat().st_size} bytes")

# Create output directory
output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

print(f"\nOutput directory: {output_dir}")

# Bake
print("\n" + "="*60)
print("STARTING BAKE")
print("="*60)

try:
    obj_path, texture_path = baker.bake(
        str(test_asset),
        output_dir=str(output_dir),
        texture_resolution=512,  # Small for fast testing
        timeout=120
    )
    
    print("\n" + "="*60)
    print("BAKE SUCCESSFUL!")
    print("="*60)
    print(f"OBJ: {obj_path}")
    print(f"Texture: {texture_path}")
    
    # Check outputs
    print(f"\nOBJ exists: {obj_path.exists()}")
    print(f"OBJ size: {obj_path.stat().st_size} bytes")
    
    print(f"\nTexture exists: {texture_path.exists()}")
    print(f"Texture size: {texture_path.stat().st_size} bytes")
    
    # Check MTL
    mtl_path = obj_path.with_suffix('.mtl')
    print(f"\nMTL exists: {mtl_path.exists()}")
    if mtl_path.exists():
        print(f"MTL size: {mtl_path.stat().st_size} bytes")
        print("\nMTL content:")
        print(mtl_path.read_text())
    
except Exception as e:
    print(f"\n❌ BAKE FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

