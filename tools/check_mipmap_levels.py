"""
Check if mipmap level selection is causing dark colors.

Analyze:
1. Gaussian scales in the output PLY
2. What mipmap levels would be selected
3. If those levels are too low (too blurry/averaged)
"""

import struct
import numpy as np

def main():
    ply_path = 'output_clouds/packed-tree_full.ply'
    
    print(f'Analyzing: {ply_path}')
    print('=' * 80)
    
    with open(ply_path, 'rb') as f:
        # Skip header
        while True:
            line = f.readline().decode('ascii').strip()
            if line == 'end_header':
                break
        
        # Read first 10000 gaussians
        n_samples = 10000
        scales = []
        
        for i in range(n_samples):
            data = f.read(68)  # 17 floats * 4 bytes
            if len(data) < 68:
                break
            
            # Unpack: 3 pos, 3 normal, 3 sh_dc, 4 rot, 3 scale, 1 opacity
            values = struct.unpack('<17f', data)
            
            # Scales are at indices 13, 14, 15 (in log space)
            scale_x = np.exp(values[13])
            scale_y = np.exp(values[14])
            scale_z = np.exp(values[15])
            
            scales.append([scale_x, scale_y, scale_z])
        
        scales = np.array(scales)
    
    print(f'\nAnalyzed {len(scales)} gaussians')
    print('-' * 80)
    
    # Statistics
    print(f'\nGaussian Scale Statistics:')
    print(f'  X: min={scales[:, 0].min():.4f}, max={scales[:, 0].max():.4f}, mean={scales[:, 0].mean():.4f}')
    print(f'  Y: min={scales[:, 1].min():.4f}, max={scales[:, 1].max():.4f}, mean={scales[:, 1].mean():.4f}')
    print(f'  Z: min={scales[:, 2].min():.4f}, max={scales[:, 2].max():.4f}, mean={scales[:, 2].mean():.4f}')
    
    # Calculate mipmap levels using the same formula as the code
    max_scale = np.maximum(scales[:, 0], scales[:, 1])
    mipmap_levels = np.log2(np.maximum(max_scale, 0.1))
    mipmap_levels = np.clip(mipmap_levels, 0.0, 7.0)
    
    print(f'\nMipmap Level Statistics:')
    print(f'  Min: {mipmap_levels.min():.2f}')
    print(f'  Max: {mipmap_levels.max():.2f}')
    print(f'  Mean: {mipmap_levels.mean():.2f}')
    print(f'  Median: {np.median(mipmap_levels):.2f}')
    
    # Distribution
    print(f'\nMipmap Level Distribution:')
    for level in range(8):
        count = np.sum((mipmap_levels >= level) & (mipmap_levels < level + 1))
        pct = 100 * count / len(mipmap_levels)
        print(f'  Level {level}: {count:,} ({pct:.1f}%)')
    
    # Verdict
    print(f'\n🎯 VERDICT:')
    print('=' * 80)
    
    mean_level = mipmap_levels.mean()
    if mean_level > 3.0:
        print(f'⚠️  MIPMAPPING IS LIKELY THE ISSUE!')
        print(f'   Mean mipmap level is {mean_level:.2f} (very low resolution)')
        print(f'   Level 3 = 1/8 resolution, Level 4 = 1/16 resolution')
        print(f'   Sampling from such low-res mipmaps causes blurry, averaged colors')
        print(f'\n💡 POSSIBLE FIXES:')
        print(f'   1. Reduce gaussian scales (make them smaller)')
        print(f'   2. Adjust mipmap level calculation formula')
        print(f'   3. Disable mipmapping entirely for testing')
    elif mean_level > 1.5:
        print(f'⚠️  Mipmapping may be contributing to the issue')
        print(f'   Mean mipmap level is {mean_level:.2f}')
        print(f'   Some gaussians are sampling from low-res mipmaps')
    else:
        print(f'✓ Mipmapping is NOT the issue')
        print(f'   Mean mipmap level is {mean_level:.2f} (mostly full resolution)')
    
    # Show some examples
    print(f'\n📍 SAMPLE GAUSSIANS:')
    print('-' * 80)
    for i in [0, 100, 500, 1000, 5000]:
        if i < len(scales):
            s = scales[i]
            level = mipmap_levels[i]
            resolution_factor = 2 ** level
            print(f'  [{i}] Scale=({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}) → Mipmap level {level:.2f} (1/{resolution_factor:.1f} resolution)')

if __name__ == '__main__':
    main()

