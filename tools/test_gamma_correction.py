"""
Test if gamma correction fixes the dark colors issue.

Compare:
1. Current method: Load texture as-is (sRGB treated as linear)
2. Gamma corrected: Load texture and apply sRGB→linear conversion
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

def srgb_to_linear(srgb):
    """
    Convert sRGB color space to linear color space.
    
    sRGB uses a gamma curve of approximately 2.2 for perceptual uniformity.
    For physically-based rendering, we need linear color space.
    
    Accurate formula:
    - if sRGB <= 0.04045: linear = sRGB / 12.92
    - else: linear = ((sRGB + 0.055) / 1.055) ^ 2.4
    
    Approximate formula (faster):
    - linear = sRGB ^ 2.2
    """
    # Use accurate formula
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4)
    )
    return linear

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
    
    # Find a texture file (preferably a diffuse/color texture)
    texture_files = list(textures_dir.glob('*'))
    if not texture_files:
        print('No texture files found!')
        return
    
    # Use the first texture
    tex_file = texture_files[0]
    print(f'\nTesting with texture: {tex_file.name}')
    print('-' * 80)
    
    # Load texture
    img = Image.open(tex_file)
    print(f'Image size: {img.size}')
    print(f'Image mode: {img.mode}')
    
    # Method 1: Current (no gamma correction)
    img_array_srgb = np.array(img, dtype=np.float32) / 255.0
    
    # Handle different formats
    if len(img_array_srgb.shape) == 2:
        # Grayscale
        rgb_srgb = np.stack([img_array_srgb] * 3, axis=-1)
    elif img_array_srgb.shape[2] >= 3:
        # RGB or RGBA
        rgb_srgb = img_array_srgb[:, :, :3]
    else:
        print('Unsupported image format!')
        return
    
    # Method 2: With gamma correction
    rgb_linear = srgb_to_linear(rgb_srgb)
    
    # Compare statistics
    print(f'\n📊 COMPARISON:')
    print('-' * 80)
    
    mean_srgb = rgb_srgb.mean(axis=(0, 1))
    mean_linear = rgb_linear.mean(axis=(0, 1))
    
    print(f'Current method (sRGB as linear):')
    print(f'  Mean RGB: ({mean_srgb[0]:.3f}, {mean_srgb[1]:.3f}, {mean_srgb[2]:.3f})')
    print(f'  Overall brightness: {mean_srgb.mean():.3f}')
    
    print(f'\nWith gamma correction (sRGB→linear):')
    print(f'  Mean RGB: ({mean_linear[0]:.3f}, {mean_linear[1]:.3f}, {mean_linear[2]:.3f})')
    print(f'  Overall brightness: {mean_linear.mean():.3f}')
    
    brightness_increase = mean_linear.mean() / mean_srgb.mean()
    print(f'\n💡 Brightness increase: {brightness_increase:.2f}x ({100*(brightness_increase-1):.0f}% brighter)')
    
    # Sample a few pixels
    print(f'\n📍 SAMPLE PIXELS (center of image):')
    print('-' * 80)
    h, w = rgb_srgb.shape[:2]
    sample_points = [
        (h//2, w//2),
        (h//4, w//4),
        (3*h//4, 3*w//4),
    ]
    
    for i, (y, x) in enumerate(sample_points):
        srgb_pixel = rgb_srgb[y, x]
        linear_pixel = rgb_linear[y, x]
        
        print(f'\nPixel {i+1} at ({y}, {x}):')
        print(f'  sRGB:   RGB({srgb_pixel[0]:.3f}, {srgb_pixel[1]:.3f}, {srgb_pixel[2]:.3f})')
        print(f'  Linear: RGB({linear_pixel[0]:.3f}, {linear_pixel[1]:.3f}, {linear_pixel[2]:.3f})')
        
        pixel_increase = linear_pixel.mean() / srgb_pixel.mean() if srgb_pixel.mean() > 0 else 0
        print(f'  Increase: {pixel_increase:.2f}x')
    
    # Verdict
    print(f'\n\n🎯 VERDICT:')
    print('=' * 80)
    if brightness_increase > 1.5:
        print(f'✅ GAMMA CORRECTION IS THE ISSUE!')
        print(f'   Textures are {100*(brightness_increase-1):.0f}% brighter with gamma correction.')
        print(f'   This explains why the output is too dark!')
        print(f'\n💡 FIX: Apply sRGB→linear conversion when loading textures:')
        print(f'   base_texture = srgb_to_linear(base_texture)')
    elif brightness_increase > 1.2:
        print(f'⚠️  Gamma correction helps but may not be the only issue.')
        print(f'   Textures are {100*(brightness_increase-1):.0f}% brighter with gamma correction.')
    else:
        print(f'❌ Gamma correction is NOT the main issue.')
        print(f'   Brightness increase is only {100*(brightness_increase-1):.0f}%.')

if __name__ == '__main__':
    main()

