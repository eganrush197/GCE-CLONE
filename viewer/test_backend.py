#!/usr/bin/env python3
"""Quick test script for backend components."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from viewer.backend.ply_parser import PLYParser

def test_ply_parser():
    """Test PLY parser with existing files."""
    parser = PLYParser()
    
    # Find PLY files
    output_clouds = Path(__file__).parent.parent / "output_clouds"
    ply_files = list(output_clouds.glob("*.ply"))
    
    if not ply_files:
        print("No PLY files found in output_clouds/")
        return
    
    print(f"Found {len(ply_files)} PLY files:")
    for ply_file in ply_files:
        print(f"  - {ply_file.name}")
    
    # Test get_file_info
    print("\n" + "="*70)
    print("Testing get_file_info()...")
    print("="*70)
    
    for ply_file in ply_files[:2]:  # Test first 2 files
        print(f"\nFile: {ply_file.name}")
        try:
            info = parser.get_file_info(str(ply_file))
            print(f"  Vertex count: {info['vertex_count']:,}")
            print(f"  File size: {info['size']:,} bytes ({info['size'] / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Test parse_ply (only on smaller file)
    print("\n" + "="*70)
    print("Testing parse_ply()...")
    print("="*70)
    
    # Find smallest file
    smallest_file = min(ply_files, key=lambda f: f.stat().st_size)
    print(f"\nParsing smallest file: {smallest_file.name}")
    
    try:
        data = parser.parse_ply(str(smallest_file))
        print(f"  Vertex count: {data['vertex_count']:,}")
        print(f"  Gaussians keys: {list(data['gaussians'].keys())}")
        print(f"  First position: {data['gaussians']['positions'][0]}")
        print(f"  First color: {data['gaussians']['colors'][0]}")
        print(f"  First scale: {data['gaussians']['scales'][0]}")
        print(f"  First opacity: {data['gaussians']['opacities'][0]}")
        print("\n✅ PLY parser working correctly!")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ply_parser()

