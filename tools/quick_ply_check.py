#!/usr/bin/env python3
"""Quick PLY color diagnostic - samples only first N vertices."""

import struct
import numpy as np
import sys
from pathlib import Path


def quick_check_ply(ply_path: str, sample_count: int = 1000):
    """Quick check of PLY file colors."""
    with open(ply_path, 'rb') as f:
        # Read header
        vertex_count = 0
        
        while True:
            line = f.readline().decode().strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line == 'end_header':
                break
        
        print('=' * 60)
        print(f'PLY FILE: {Path(ply_path).name}')
        print('=' * 60)
        print(f'Total vertices: {vertex_count:,}')
        print(f'Sampling first {sample_count:,} vertices...\n')
        
        # Read sample vertices (68 bytes each)
        bytes_per_vertex = 68
        sample_count = min(sample_count, vertex_count)
        
        sh_dc = np.zeros((sample_count, 3), dtype=np.float32)
        opacities = np.zeros(sample_count, dtype=np.float32)
        positions = np.zeros((sample_count, 3), dtype=np.float32)
        
        for i in range(sample_count):
            data = f.read(bytes_per_vertex)
            if len(data) < bytes_per_vertex:
                break
            
            # Unpack: x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, ...
            values = struct.unpack('<17f', data)
            positions[i] = values[0:3]
            sh_dc[i] = values[6:9]
            opacities[i] = values[9]
        
        # Convert SH DC to RGB
        rgb = sh_dc + 0.5
        
        # Display first few
        print('FIRST 10 SAMPLES:')
        print('-' * 60)
        for i in range(min(10, sample_count)):
            print(f'V{i}: pos=({positions[i, 0]:.3f}, {positions[i, 1]:.3f}, {positions[i, 2]:.3f})')
            print(f'     SH DC=({sh_dc[i, 0]:.4f}, {sh_dc[i, 1]:.4f}, {sh_dc[i, 2]:.4f})')
            print(f'     RGB=({rgb[i, 0]:.3f}, {rgb[i, 1]:.3f}, {rgb[i, 2]:.3f}), opacity={opacities[i]:.3f}')
        
        # Statistics
        print(f'\n{"="*60}')
        print(f'STATISTICS (from {sample_count:,} samples):')
        print('=' * 60)
        
        print(f'\nSH DC:')
        print(f'  Min: ({sh_dc[:, 0].min():.4f}, {sh_dc[:, 1].min():.4f}, {sh_dc[:, 2].min():.4f})')
        print(f'  Max: ({sh_dc[:, 0].max():.4f}, {sh_dc[:, 1].max():.4f}, {sh_dc[:, 2].max():.4f})')
        print(f'  Mean: ({sh_dc[:, 0].mean():.4f}, {sh_dc[:, 1].mean():.4f}, {sh_dc[:, 2].mean():.4f})')
        print(f'  Variance: {sh_dc.var():.6f}')
        
        print(f'\nRGB (from SH DC):')
        print(f'  Min: ({rgb[:, 0].min():.3f}, {rgb[:, 1].min():.3f}, {rgb[:, 2].min():.3f})')
        print(f'  Max: ({rgb[:, 0].max():.3f}, {rgb[:, 1].max():.3f}, {rgb[:, 2].max():.3f})')
        print(f'  Mean: ({rgb[:, 0].mean():.3f}, {rgb[:, 1].mean():.3f}, {rgb[:, 2].mean():.3f})')
        
        # Check for black
        black_count = np.sum((rgb < 0.1).all(axis=1))
        gray_count = np.sum(np.abs(rgb - 0.5).max(axis=1) < 0.01)
        
        print(f'\nColor Analysis:')
        print(f'  Black vertices (RGB < 0.1): {black_count:,} ({100*black_count/sample_count:.1f}%)')
        print(f'  Gray vertices (RGB ≈ 0.5): {gray_count:,} ({100*gray_count/sample_count:.1f}%)')
        
        print(f'\nOpacity:')
        print(f'  Min: {opacities.min():.3f}')
        print(f'  Max: {opacities.max():.3f}')
        print(f'  Mean: {opacities.mean():.3f}')
        
        # Diagnosis
        print(f'\n{"="*60}')
        print('DIAGNOSIS:')
        print('=' * 60)
        
        if sh_dc.var() < 0.0001:
            if np.abs(sh_dc.mean()) < 0.01:
                print('❌ ALL GRAY - Colors defaulted to (0.5, 0.5, 0.5)')
                print('   → Color sampling completely failed')
            elif (rgb < 0.1).all():
                print('❌ ALL BLACK - Colors are (0.0, 0.0, 0.0)')
                print('   → Texture sampling returned black')
            else:
                print('❌ ALL SAME COLOR - No variance')
        elif black_count > sample_count * 0.8:
            print('⚠️  MOSTLY BLACK - 80%+ vertices are black')
            print('   → Partial texture sampling failure')
        elif gray_count > sample_count * 0.8:
            print('⚠️  MOSTLY GRAY - 80%+ vertices are default gray')
            print('   → Partial color sampling failure')
        else:
            print('✅ COLORS PRESENT - Variance detected')
            print(f'   → Color system appears to be working')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python quick_ply_check.py <ply_file> [sample_count]')
        sys.exit(1)
    
    ply_path = sys.argv[1]
    sample_count = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    quick_check_ply(ply_path, sample_count)

