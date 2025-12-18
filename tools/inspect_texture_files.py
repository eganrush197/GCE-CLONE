"""
Inspect texture files to check if they're being loaded correctly.
Check for:
1. Image format and mode
2. Color profile / ICC profile
3. Actual pixel values
4. Comparison with what we expect
"""

import numpy as np
from pathlib import Path
from PIL import Image
import os

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
    
    textures_dir = packed_dir / 'textures'
    if not textures_dir.exists():
        print('Textures directory not found!')
        return
    
    # Get all texture files
    texture_files = sorted(list(textures_dir.glob('*')))
    print(f'\nFound {len(texture_files)} texture files')
    print('=' * 80)
    
    # Inspect each texture
    for i, tex_file in enumerate(texture_files[:10]):  # First 10 only
        print(f'\n[{i+1}] {tex_file.name}')
        print('-' * 80)
        
        try:
            img = Image.open(tex_file)
            
            # Basic info
            print(f'Size: {img.size}')
            print(f'Mode: {img.mode}')
            print(f'Format: {img.format}')
            
            # Check for ICC profile
            if 'icc_profile' in img.info:
                print(f'ICC Profile: Present ({len(img.info["icc_profile"])} bytes)')
            else:
                print(f'ICC Profile: None')
            
            # Check for other metadata
            if 'gamma' in img.info:
                print(f'Gamma: {img.info["gamma"]}')
            
            # Load as numpy array (same as our code)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Handle different formats
            if len(img_array.shape) == 2:
                # Grayscale
                mean_val = img_array.mean()
                min_val = img_array.min()
                max_val = img_array.max()
                print(f'Grayscale: mean={mean_val:.3f}, min={min_val:.3f}, max={max_val:.3f}')
                
                # Sample center pixel
                h, w = img_array.shape
                center_val = img_array[h//2, w//2]
                print(f'Center pixel: {center_val:.3f}')
                
            elif img_array.shape[2] >= 3:
                # RGB or RGBA
                rgb = img_array[:, :, :3]
                mean_rgb = rgb.mean(axis=(0, 1))
                min_rgb = rgb.min(axis=(0, 1))
                max_rgb = rgb.max(axis=(0, 1))
                
                print(f'RGB Mean: ({mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f})')
                print(f'RGB Min:  ({min_rgb[0]:.3f}, {min_rgb[1]:.3f}, {min_rgb[2]:.3f})')
                print(f'RGB Max:  ({max_rgb[0]:.3f}, {max_rgb[1]:.3f}, {max_rgb[2]:.3f})')
                
                # Sample center pixel
                h, w = rgb.shape[:2]
                center_rgb = rgb[h//2, w//2]
                print(f'Center pixel: RGB({center_rgb[0]:.3f}, {center_rgb[1]:.3f}, {center_rgb[2]:.3f})')
                
                # Check if it looks like a green leaf texture
                is_green = mean_rgb[1] > mean_rgb[0] and mean_rgb[1] > mean_rgb[2]
                is_bright = mean_rgb.mean() > 0.4
                
                if is_green and is_bright:
                    print('✓ Looks like a BRIGHT GREEN texture (leaf?)')
                elif is_green:
                    print('⚠️  Looks like a DARK GREEN texture')
                elif mean_rgb.mean() < 0.3:
                    print('⚠️  DARK texture')
                
                # Check if alpha channel exists
                if img_array.shape[2] == 4:
                    alpha = img_array[:, :, 3]
                    print(f'Alpha: mean={alpha.mean():.3f}, min={alpha.min():.3f}, max={alpha.max():.3f}')
            
        except Exception as e:
            print(f'ERROR: {e}')
    
    print('\n\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    
    # Count texture types
    green_count = 0
    dark_count = 0
    bright_count = 0
    
    for tex_file in texture_files:
        try:
            img = Image.open(tex_file)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                rgb = img_array[:, :, :3]
                mean_rgb = rgb.mean(axis=(0, 1))
                
                is_green = mean_rgb[1] > mean_rgb[0] and mean_rgb[1] > mean_rgb[2]
                is_bright = mean_rgb.mean() > 0.4
                is_dark = mean_rgb.mean() < 0.3
                
                if is_green:
                    green_count += 1
                if is_bright:
                    bright_count += 1
                if is_dark:
                    dark_count += 1
        except:
            pass
    
    print(f'\nTotal textures: {len(texture_files)}')
    print(f'Green textures: {green_count} ({100*green_count/len(texture_files):.0f}%)')
    print(f'Bright textures (>0.4): {bright_count} ({100*bright_count/len(texture_files):.0f}%)')
    print(f'Dark textures (<0.3): {dark_count} ({100*dark_count/len(texture_files):.0f}%)')
    
    print(f'\n🎯 VERDICT:')
    if dark_count > len(texture_files) * 0.5:
        print(f'⚠️  MOST TEXTURES ARE DARK!')
        print(f'   This suggests the source textures in the Blender file are dark,')
        print(f'   OR Blender is exporting them incorrectly.')
    elif bright_count > len(texture_files) * 0.5:
        print(f'✓ Most textures are bright - texture loading seems OK')
    else:
        print(f'⚠️  Mixed texture brightness - need more investigation')

if __name__ == '__main__':
    main()

