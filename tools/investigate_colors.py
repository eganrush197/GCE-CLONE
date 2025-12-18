"""
Investigate color darkening issue by checking:
1. Vertex colors (are they dark?)
2. Texture brightness (are textures dark?)
3. Vertex color blending (is multiply mode causing darkening?)
"""

import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys

def find_latest_packed_dir():
    """Find the most recent packed directory"""
    import os
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
    
    # Load manifest
    manifest_path = packed_dir / 'material_manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # ========== INVESTIGATION 1: VERTEX COLORS ==========
    print('\n[1] VERTEX COLOR INVESTIGATION')
    print('-' * 80)
    
    vc_path = manifest.get('vertex_colors')
    if vc_path:
        vc_file = packed_dir / Path(vc_path).name
        if vc_file.exists():
            vertex_colors = np.load(vc_file)
            print(f'✓ Vertex colors loaded: {vertex_colors.shape}')
            
            # Statistics
            print(f'\nVertex Color Statistics (RGB only):')
            print(f'  Min:    ({vertex_colors[:, 0].min():.3f}, {vertex_colors[:, 1].min():.3f}, {vertex_colors[:, 2].min():.3f})')
            print(f'  Max:    ({vertex_colors[:, 0].max():.3f}, {vertex_colors[:, 1].max():.3f}, {vertex_colors[:, 2].max():.3f})')
            print(f'  Mean:   ({vertex_colors[:, 0].mean():.3f}, {vertex_colors[:, 1].mean():.3f}, {vertex_colors[:, 2].mean():.3f})')
            print(f'  Median: ({np.median(vertex_colors[:, 0]):.3f}, {np.median(vertex_colors[:, 1]):.3f}, {np.median(vertex_colors[:, 2]):.3f})')
            
            # Check distribution
            white_count = np.sum(np.all(vertex_colors[:, :3] > 0.99, axis=1))
            dark_count = np.sum(np.all(vertex_colors[:, :3] < 0.5, axis=1))
            print(f'\nDistribution:')
            print(f'  White (>0.99): {white_count:,} / {len(vertex_colors):,} ({100*white_count/len(vertex_colors):.1f}%)')
            print(f'  Dark (<0.5):   {dark_count:,} / {len(vertex_colors):,} ({100*dark_count/len(vertex_colors):.1f}%)')
            
            # Verdict
            mean_brightness = vertex_colors[:, :3].mean()
            print(f'\n📊 VERDICT:')
            if mean_brightness > 0.95:
                print(f'   ✅ Vertex colors are WHITE (mean={mean_brightness:.3f}) - NOT causing darkening')
            elif mean_brightness < 0.7:
                print(f'   ⚠️  Vertex colors are DARK (mean={mean_brightness:.3f}) - LIKELY CAUSING DARKENING!')
            else:
                print(f'   ⚠️  Vertex colors are MEDIUM (mean={mean_brightness:.3f}) - MAY be causing some darkening')
        else:
            print(f'✗ Vertex color file not found: {vc_file}')
    else:
        print('✓ No vertex colors in manifest - vertex colors NOT causing darkening')
    
    # Check blend mode
    blend_mode = manifest.get('vertex_color_blend_mode', 'multiply')
    print(f'\nVertex color blend mode: {blend_mode}')
    if blend_mode == 'multiply' and vc_path:
        print('  ⚠️  Multiply mode will darken textures if vertex colors are < 1.0')
    
    # ========== INVESTIGATION 2: TEXTURE BRIGHTNESS ==========
    print('\n\n[2] TEXTURE BRIGHTNESS INVESTIGATION')
    print('-' * 80)
    
    textures_dir = packed_dir / 'textures'
    if textures_dir.exists():
        texture_files = list(textures_dir.glob('*'))
        print(f'Found {len(texture_files)} texture files')
        
        # Sample a few textures
        sample_count = min(5, len(texture_files))
        print(f'\nSampling {sample_count} textures:')
        
        for i, tex_file in enumerate(texture_files[:sample_count]):
            try:
                img = Image.open(tex_file)
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Handle different formats
                if len(img_array.shape) == 2:
                    # Grayscale
                    mean_brightness = img_array.mean()
                    print(f'  [{i+1}] {tex_file.name}: {img.size} grayscale, mean={mean_brightness:.3f}')
                elif img_array.shape[2] >= 3:
                    # RGB or RGBA
                    rgb = img_array[:, :, :3]
                    mean_rgb = rgb.mean(axis=(0, 1))
                    mean_brightness = mean_rgb.mean()
                    print(f'  [{i+1}] {tex_file.name}: {img.size} RGB({mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f}), mean={mean_brightness:.3f}')
                    
                    # Check if it's dark
                    if mean_brightness < 0.3:
                        print(f'       ⚠️  DARK texture!')
                    elif mean_brightness > 0.7:
                        print(f'       ✓ Bright texture')
            except Exception as e:
                print(f'  [{i+1}] {tex_file.name}: Error loading - {e}')
    else:
        print('✗ Textures directory not found')
    
    # ========== INVESTIGATION 3: COMBINED EFFECT ==========
    print('\n\n[3] COMBINED EFFECT SIMULATION')
    print('-' * 80)
    
    if vc_path and vc_file.exists():
        # Simulate multiply blending
        sample_texture_color = np.array([0.6, 0.8, 0.4])  # Bright green (expected)
        sample_vc = vertex_colors[len(vertex_colors)//2, :3]  # Middle vertex color
        
        if blend_mode == 'multiply':
            result = sample_texture_color * sample_vc
        else:
            result = sample_texture_color
        
        print(f'Sample texture color: RGB({sample_texture_color[0]:.3f}, {sample_texture_color[1]:.3f}, {sample_texture_color[2]:.3f})')
        print(f'Sample vertex color:  RGB({sample_vc[0]:.3f}, {sample_vc[1]:.3f}, {sample_vc[2]:.3f})')
        print(f'Blend mode: {blend_mode}')
        print(f'Result color:         RGB({result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f})')
        
        darkening_factor = result.mean() / sample_texture_color.mean()
        print(f'\nDarkening factor: {darkening_factor:.2f}x ({100*(1-darkening_factor):.0f}% darker)')
        
        if darkening_factor < 0.7:
            print('⚠️  SIGNIFICANT DARKENING from vertex color multiply!')

if __name__ == '__main__':
    main()

