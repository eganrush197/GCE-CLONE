"""
Check texture file color channels to diagnose BGR vs RGB issue.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

def check_texture(texture_path):
    """Check a texture file's color channels."""
    print(f"\n{'='*60}")
    print(f"TEXTURE: {texture_path.name}")
    print(f"{'='*60}")
    
    img = Image.open(texture_path)
    print(f"Mode: {img.mode}")
    print(f"Size: {img.size}")
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    if len(img_array.shape) == 3:
        print(f"Shape: {img_array.shape} (H, W, C)")
        print(f"Channels: {img_array.shape[2]}")
        
        # Sample center pixel
        h, w, c = img_array.shape
        center_y, center_x = h // 2, w // 2
        center_pixel = img_array[center_y, center_x]
        
        print(f"\nCenter pixel (as loaded):")
        if c >= 3:
            print(f"  Channel 0: {center_pixel[0]:.3f}")
            print(f"  Channel 1: {center_pixel[1]:.3f}")
            print(f"  Channel 2: {center_pixel[2]:.3f}")
            if c == 4:
                print(f"  Channel 3 (alpha): {center_pixel[3]:.3f}")
        
        # Calculate channel statistics
        print(f"\nChannel statistics:")
        for i in range(min(c, 3)):
            channel_mean = img_array[:, :, i].mean()
            channel_max = img_array[:, :, i].max()
            channel_min = img_array[:, :, i].min()
            print(f"  Channel {i}: mean={channel_mean:.3f}, min={channel_min:.3f}, max={channel_max:.3f}")
        
        # Determine dominant channel
        channel_means = [img_array[:, :, i].mean() for i in range(min(c, 3))]
        dominant_idx = np.argmax(channel_means)
        channel_names = ['Red', 'Green', 'Blue']
        print(f"\nDominant channel: {dominant_idx} ({channel_names[dominant_idx]})")
        
        # Check if this looks like a green texture (leaves) or brown texture (bark)
        if c >= 3:
            r_mean, g_mean, b_mean = channel_means[0], channel_means[1], channel_means[2]
            if g_mean > r_mean and g_mean > b_mean:
                print(f"  → Appears to be GREEN-dominant (likely leaves)")
            elif r_mean > g_mean and r_mean > b_mean:
                print(f"  → Appears to be RED-dominant (likely bark/brown)")
            elif b_mean > r_mean and b_mean > g_mean:
                print(f"  → Appears to be BLUE-dominant (unusual)")
            else:
                print(f"  → Appears to be NEUTRAL/GREY")
    else:
        print(f"Shape: {img_array.shape} (H, W) - Grayscale")
        print(f"Mean: {img_array.mean():.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_texture_channels.py <texture_directory>")
        print("Example: python check_texture_channels.py temp_packed-tree/textures")
        sys.exit(1)
    
    texture_dir = Path(sys.argv[1])
    
    if not texture_dir.exists():
        print(f"Error: Directory not found: {texture_dir}")
        sys.exit(1)
    
    # Find all texture files
    texture_files = list(texture_dir.glob("*.png")) + list(texture_dir.glob("*.jpg")) + list(texture_dir.glob("*.jpeg"))
    
    if not texture_files:
        print(f"No texture files found in {texture_dir}")
        sys.exit(1)
    
    print(f"Found {len(texture_files)} texture files")
    
    for texture_file in sorted(texture_files):
        try:
            check_texture(texture_file)
        except Exception as e:
            print(f"\nError checking {texture_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

