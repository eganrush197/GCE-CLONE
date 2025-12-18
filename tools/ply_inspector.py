#!/usr/bin/env python3
"""PLY file inspector for verifying gaussian splat output."""

import struct
import numpy as np
import sys
from pathlib import Path


def parse_ply(ply_path: str, sample_count: int = 10):
    """Parse PLY file and display contents."""
    with open(ply_path, 'rb') as f:
        # Read header
        properties = []
        vertex_count = 0
        
        while True:
            line = f.readline().decode().strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))
            elif line == 'end_header':
                break
        
        print('=' * 60)
        print(f'PLY FILE: {ply_path}')
        print('=' * 60)
        print(f'Vertex count: {vertex_count:,}')
        print(f'Properties ({len(properties)}):')
        for name, ptype in properties:
            print(f'  - {name} ({ptype})')
        
        # Calculate bytes per vertex
        type_sizes = {'float': 4, 'uchar': 1, 'int': 4, 'double': 8}
        bytes_per_vertex = sum(type_sizes.get(t, 4) for _, t in properties)
        print(f'\nBytes per vertex: {bytes_per_vertex}')
        
        # Read all vertices for statistics
        print(f'\nReading {vertex_count:,} vertices...')
        all_data = f.read(vertex_count * bytes_per_vertex)
        
        # Parse into arrays
        sh_dc = np.zeros((vertex_count, 3), dtype=np.float32)
        rgb = np.zeros((vertex_count, 3), dtype=np.uint8)
        positions = np.zeros((vertex_count, 3), dtype=np.float32)
        opacities = np.zeros(vertex_count, dtype=np.float32)
        
        for i in range(vertex_count):
            offset = i * bytes_per_vertex
            data = all_data[offset:offset + bytes_per_vertex]
            
            prop_offset = 0
            for name, ptype in properties:
                if ptype == 'float':
                    val = struct.unpack('<f', data[prop_offset:prop_offset+4])[0]
                    prop_offset += 4
                    if name == 'x': positions[i, 0] = val
                    elif name == 'y': positions[i, 1] = val
                    elif name == 'z': positions[i, 2] = val
                    elif name == 'f_dc_0': sh_dc[i, 0] = val
                    elif name == 'f_dc_1': sh_dc[i, 1] = val
                    elif name == 'f_dc_2': sh_dc[i, 2] = val
                    elif name == 'opacity': opacities[i] = val
                elif ptype == 'uchar':
                    val = data[prop_offset]
                    prop_offset += 1
                    if name == 'red': rgb[i, 0] = val
                    elif name == 'green': rgb[i, 1] = val
                    elif name == 'blue': rgb[i, 2] = val
        
        # Display sample vertices
        print(f'\n{"="*60}')
        print(f'SAMPLE VERTICES (first {sample_count}):')
        print('=' * 60)
        
        for i in range(min(sample_count, vertex_count)):
            rgb_from_sh = sh_dc[i] + 0.5
            print(f'V{i}: pos=({positions[i, 0]:.3f}, {positions[i, 1]:.3f}, {positions[i, 2]:.3f})')
            print(f'     SH DC=({sh_dc[i, 0]:.4f}, {sh_dc[i, 1]:.4f}, {sh_dc[i, 2]:.4f})')
            print(f'     RGB from SH=({rgb_from_sh[0]:.3f}, {rgb_from_sh[1]:.3f}, {rgb_from_sh[2]:.3f})')
            print(f'     Direct RGB=({rgb[i, 0]}, {rgb[i, 1]}, {rgb[i, 2]}), opacity={opacities[i]:.3f}')
        
        # Statistics
        print(f'\n{"="*60}')
        print('COLOR STATISTICS:')
        print('=' * 60)
        
        rgb_from_sh_all = sh_dc + 0.5
        print(f'SH DC -> RGB:')
        print(f'  Min: ({rgb_from_sh_all[:, 0].min():.3f}, {rgb_from_sh_all[:, 1].min():.3f}, {rgb_from_sh_all[:, 2].min():.3f})')
        print(f'  Max: ({rgb_from_sh_all[:, 0].max():.3f}, {rgb_from_sh_all[:, 1].max():.3f}, {rgb_from_sh_all[:, 2].max():.3f})')
        print(f'  Mean: ({rgb_from_sh_all[:, 0].mean():.3f}, {rgb_from_sh_all[:, 1].mean():.3f}, {rgb_from_sh_all[:, 2].mean():.3f})')
        
        print(f'\nDirect RGB (0-255):')
        print(f'  Min: ({rgb[:, 0].min()}, {rgb[:, 1].min()}, {rgb[:, 2].min()})')
        print(f'  Max: ({rgb[:, 0].max()}, {rgb[:, 1].max()}, {rgb[:, 2].max()})')
        print(f'  Mean: ({rgb[:, 0].mean():.1f}, {rgb[:, 1].mean():.1f}, {rgb[:, 2].mean():.1f})')
        
        # Check for black vertices
        black_count = np.sum((rgb_from_sh_all < 0.1).all(axis=1))
        print(f'\nBlack vertices (RGB < 0.1): {black_count:,} ({100*black_count/vertex_count:.1f}%)')
        
        print(f'\nOpacity stats: min={opacities.min():.3f}, max={opacities.max():.3f}, mean={opacities.mean():.3f}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ply_inspector.py <ply_file> [sample_count]')
        sys.exit(1)
    
    ply_path = sys.argv[1]
    sample_count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    parse_ply(ply_path, sample_count)

