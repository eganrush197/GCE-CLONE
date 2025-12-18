#!/usr/bin/env python3
"""Diagnostic script to check Stage 1 extraction output."""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import sys
from collections import Counter


def diagnose_stage1_output(temp_dir: str):
    """Check Stage 1 extraction output."""
    temp_path = Path(temp_dir)
    
    print("=" * 70)
    print("STAGE 1 DIAGNOSTIC")
    print("=" * 70)
    print(f"Directory: {temp_path}\n")
    
    # Check manifest
    manifest_path = temp_path / "material_manifest.json"
    if not manifest_path.exists():
        print("[FAIL] material_manifest.json NOT FOUND")
        return False
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"[OK] Manifest loaded")
    print(f"  Materials: {len(manifest.get('materials', {}))}")
    print(f"  Face materials: {len(manifest.get('face_materials', []))}")
    print(f"  UV layers: {list(manifest.get('uv_layers', {}).keys())}")
    print(f"  Default UV layer: {manifest.get('uv_layer')}")
    
    # Check textures
    print("\n" + "=" * 70)
    print("TEXTURES:")
    print("=" * 70)
    for mat_name, mat_data in manifest.get('materials', {}).items():
        diffuse = mat_data.get('diffuse')
        if diffuse:
            diffuse_path = diffuse.get('path') if isinstance(diffuse, dict) else diffuse
            full_path = temp_path / Path(diffuse_path).name if not Path(diffuse_path).is_absolute() else Path(diffuse_path)
            
            if full_path.exists():
                img = Image.open(full_path)
                arr = np.array(img)
                mean_color = arr.mean(axis=(0,1))
                is_black = arr.max() < 10
                status = "[FAIL] ALL BLACK" if is_black else "[OK]"
                print(f"  {mat_name}: {status}")
                print(f"    Size: {img.size}, Mode: {img.mode}")
                print(f"    Mean color: {mean_color[:3]}")
                print(f"    Min: {arr.min()}, Max: {arr.max()}")
            else:
                print(f"  {mat_name}: [FAIL] File not found: {diffuse_path}")
        else:
            print(f"  {mat_name}: [WARN] No diffuse texture")
    
    # Check UVs
    print("\n" + "=" * 70)
    print("UV LAYERS:")
    print("=" * 70)
    for uv_name, uv_path in manifest.get('uv_layers', {}).items():
        uv_file = temp_path / Path(uv_path).name if not Path(uv_path).is_absolute() else Path(uv_path)
        if uv_file.exists():
            uvs = np.load(uv_file)
            in_range = (uvs >= -0.1).all() and (uvs <= 1.1).all()
            status = "[OK]" if in_range else "[WARN] Out of range"
            print(f"  {uv_name}: {status}")
            print(f"    Shape: {uvs.shape}")
            print(f"    Range: [{uvs.min():.3f}, {uvs.max():.3f}]")
            print(f"    Mean: {uvs.mean():.3f}")
        else:
            print(f"  {uv_name}: [FAIL] File not found: {uv_path}")
    
    # Check face_materials
    print("\n" + "=" * 70)
    print("FACE MATERIALS DISTRIBUTION:")
    print("=" * 70)
    face_mats = manifest.get('face_materials', [])
    counts = Counter(face_mats)
    total = len(face_mats)
    print(f"Total faces: {total:,}")
    for mat, count in counts.most_common():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {mat}: {count:,} faces ({pct:.1f}%)")
    
    # Check for issues
    print("\n" + "=" * 70)
    print("ISSUES DETECTED:")
    print("=" * 70)
    
    issues = []
    
    # Check for None materials
    none_count = counts.get(None, 0)
    if none_count > 0:
        pct = 100 * none_count / total
        if pct > 50:
            issues.append(f"ERROR: {pct:.0f}% of faces have no material")
        else:
            issues.append(f"WARNING: {pct:.0f}% of faces have no material")
    
    # Check material name overlap
    mat_names_in_manifest = set(manifest.get('materials', {}).keys())
    mat_names_in_faces = set(face_mats) - {None}
    
    if mat_names_in_faces and mat_names_in_manifest:
        overlap = mat_names_in_faces.intersection(mat_names_in_manifest)
        if not overlap:
            issues.append("CRITICAL: NO OVERLAP between face materials and texture materials!")
            issues.append(f"  Face materials: {mat_names_in_faces}")
            issues.append(f"  Texture materials: {mat_names_in_manifest}")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  No critical issues detected")
    
    return len([i for i in issues if 'CRITICAL' in i or 'ERROR' in i]) == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_stage1.py <temp_directory>")
        sys.exit(1)
    diagnose_stage1_output(sys.argv[1])

