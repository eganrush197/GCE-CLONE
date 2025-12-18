"""
Check if UV clamping is causing dark colors by sampling edge pixels.

Investigate:
1. UV distribution (how many are outside [0, 1]?)
2. What happens when we clamp vs wrap UVs
3. Are edge pixels dark?
"""

import numpy as np
from pathlib import Path
from PIL import Image
import os
import json

def find_latest_packed_dir():
    """Find the most recent packed directory"""
    temp_dir = Path(os.environ['TEMP'])
    packed_dirs = list(temp_dir.glob('gaussian_packed_*'))
    if not packed_dirs:
        print('No packed directories found!')
        return None
    return max(packed_dirs, key=lambda p: p.stat().st_mtime)

def main():
    packed_dir = find_latest_packed_dir()
    if not packed_dir:
        return
    
    print(f'Using packed directory: {packed_dir}')
    print('=' * 80)
    
    # Load UVs
    uv_file = packed_dir / 'uv0.npy'
    if not uv_file.exists():
        print('UV file not found!')
        return
    
    uvs = np.load(uv_file)
    print(f'\nLoaded UVs: {uvs.shape}')
    print('-' * 80)
    
    # UV Statistics
    print(f'\nUV Statistics:')
    print(f'  U: min={uvs[:, 0].min():.3f}, max={uvs[:, 0].max():.3f}, mean={uvs[:, 0].mean():.3f}')
    print(f'  V: min={uvs[:, 1].min():.3f}, max={uvs[:, 1].max():.3f}, mean={uvs[:, 1].mean():.3f}')
    
    # Check how many UVs are outside [0, 1]
    outside_u = np.sum((uvs[:, 0] < 0) | (uvs[:, 0] > 1))
    outside_v = np.sum((uvs[:, 1] < 0) | (uvs[:, 1] > 1))
    outside_both = np.sum(((uvs[:, 0] < 0) | (uvs[:, 0] > 1)) | ((uvs[:, 1] < 0) | (uvs[:, 1] > 1)))
    
    print(f'\nUVs outside [0, 1]:')
    print(f'  U outside: {outside_u:,} / {len(uvs):,} ({100*outside_u/len(uvs):.1f}%)')
    print(f'  V outside: {outside_v:,} / {len(uvs):,} ({100*outside_v/len(uvs):.1f}%)')
    print(f'  Either outside: {outside_both:,} / {len(uvs):,} ({100*outside_both/len(uvs):.1f}%)')
    
    # Load a texture to test edge pixel colors
    textures_dir = packed_dir / 'textures'
    texture_files = list(textures_dir.glob('*'))
    
    if texture_files:
        # Use first texture
        tex_file = texture_files[0]
        print(f'\n\nTesting with texture: {tex_file.name}')
        print('-' * 80)
        
        img = Image.open(tex_file)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Handle different formats
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        h, w = img_array.shape[:2]
        
        # Sample edge pixels
        print(f'\nEdge Pixel Colors:')
        print(f'  Top-left (0, 0): RGB({img_array[0, 0, 0]:.3f}, {img_array[0, 0, 1]:.3f}, {img_array[0, 0, 2]:.3f})')
        print(f'  Top-right (0, {w-1}): RGB({img_array[0, w-1, 0]:.3f}, {img_array[0, w-1, 1]:.3f}, {img_array[0, w-1, 2]:.3f})')
        print(f'  Bottom-left ({h-1}, 0): RGB({img_array[h-1, 0, 0]:.3f}, {img_array[h-1, 0, 1]:.3f}, {img_array[h-1, 0, 2]:.3f})')
        print(f'  Bottom-right ({h-1}, {w-1}): RGB({img_array[h-1, w-1, 0]:.3f}, {img_array[h-1, w-1, 1]:.3f}, {img_array[h-1, w-1, 2]:.3f})')
        
        # Average edge color
        edge_pixels = np.concatenate([
            img_array[0, :, :],  # Top edge
            img_array[-1, :, :],  # Bottom edge
            img_array[:, 0, :],  # Left edge
            img_array[:, -1, :],  # Right edge
        ])
        edge_mean = edge_pixels.mean(axis=0)
        center_mean = img_array[h//4:3*h//4, w//4:3*w//4, :].mean(axis=(0, 1))
        
        print(f'\nAverage Colors:')
        print(f'  Edge pixels: RGB({edge_mean[0]:.3f}, {edge_mean[1]:.3f}, {edge_mean[2]:.3f})')
        print(f'  Center pixels: RGB({center_mean[0]:.3f}, {center_mean[1]:.3f}, {center_mean[2]:.3f})')
        
        edge_brightness = edge_mean.mean()
        center_brightness = center_mean.mean()
        
        if edge_brightness < center_brightness * 0.7:
            print(f'\n⚠️  EDGE PIXELS ARE DARKER than center!')
            print(f'   Edge brightness: {edge_brightness:.3f}')
            print(f'   Center brightness: {center_brightness:.3f}')
            print(f'   Ratio: {edge_brightness/center_brightness:.2f}x')
        else:
            print(f'\n✓ Edge pixels are similar brightness to center')
    
    # Verdict
    print(f'\n\n🎯 VERDICT:')
    print('=' * 80)
    
    if outside_both > len(uvs) * 0.3:
        print(f'⚠️  UV CLAMPING COULD BE AN ISSUE!')
        print(f'   {100*outside_both/len(uvs):.1f}% of UVs are outside [0, 1]')
        print(f'   These UVs are being CLAMPED to edge pixels')
        print(f'   If edge pixels are dark, this causes darkening')
        print(f'\n💡 POSSIBLE FIX:')
        print(f'   Use UV wrapping (modulo) instead of clamping:')
        print(f'   u = u % 1.0')
        print(f'   v = v % 1.0')
    else:
        print(f'✓ UV clamping is NOT a major issue')
        print(f'   Only {100*outside_both/len(uvs):.1f}% of UVs are outside [0, 1]')

if __name__ == '__main__':
    main()

