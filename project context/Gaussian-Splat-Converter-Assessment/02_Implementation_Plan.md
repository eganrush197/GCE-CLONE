# Gaussian Splat Pipeline: Implementation Plan

**Document 2 of 3**  
**Date:** December 2024  
**Status:** Ready for Implementation  
**Test Asset:** `packed-tree.blend`

---

## Executive Summary

This document provides a step-by-step implementation plan to fix the color system. The approach is:

1. **Diagnose First** — Run diagnostics to identify exact failure point
2. **Fix Incrementally** — Address issues one at a time, testing after each
3. **Validate Thoroughly** — Confirm colors work end-to-end before moving on

**Estimated Time:** 4-8 hours depending on root cause

---

## Table of Contents

1. [Phase 0: Diagnostic Investigation](#phase-0-diagnostic-investigation)
2. [Phase 1: Fix Identified Issues](#phase-1-fix-identified-issues)
3. [Phase 2: Add Validation Layer](#phase-2-add-validation-layer)
4. [Phase 3: End-to-End Verification](#phase-3-end-to-end-verification)
5. [Code Changes Reference](#code-changes-reference)

---

## Phase 0: Diagnostic Investigation

**Goal:** Identify exactly where color data is lost.

### Step 0.1: Run PLY Inspector

First, determine what's actually in the output file.

```bash
python tools/ply_inspector.py output_clouds/packed-tree_full.ply 20
```

**What to look for:**

```
SH DC -> RGB:
  Min: (X.XXX, X.XXX, X.XXX)
  Max: (X.XXX, X.XXX, X.XXX)
  Mean: (X.XXX, X.XXX, X.XXX)

Black vertices (RGB < 0.1): XXXXXX (XX.X%)

Opacity stats: min=X.XXX, max=X.XXX, mean=X.XXX
```

**Interpretation:**

| Mean RGB | Black % | Opacity Mean | Diagnosis |
|----------|---------|--------------|-----------|
| ≈ 0.0 | ~100% | > 0.5 | Colors sampled as black |
| ≈ 0.5 | ~0% | > 0.5 | Bug in viewer, not pipeline |
| Any | Any | ≈ 0.0 | Opacity culling issue |
| ≈ 0.0 | ~100% | ≈ 0.0 | Both color AND opacity broken |

**Record the results before proceeding.**

### Step 0.2: Examine Stage 1 Output

Check what the packed extraction produced.

```bash
# Find the temp directory (or use --keep-temp flag)
python cli.py packed-tree.blend ./output --packed --keep-temp --verbose

# Then examine:
ls -la ./output/_temp*/textures/
cat ./output/_temp*/material_manifest.json | python -m json.tool
```

**Checklist:**

- [ ] `textures/` directory exists
- [ ] PNG files are present
- [ ] PNG files are NOT all black (open one in image viewer)
- [ ] `material_manifest.json` exists
- [ ] Manifest has `materials` with `diffuse` entries
- [ ] Manifest has `face_materials` array
- [ ] Manifest has `uv_layers` entries

### Step 0.3: Check Verbose Pipeline Output

Run with full verbosity and examine logs:

```bash
python cli.py packed-tree.blend ./output --packed --verbose 2>&1 | tee pipeline_log.txt
```

**Key log messages to find:**

```
# Good signs:
[INFO] Loaded textures for X materials
[INFO] Loaded UV layer 'uv0': (XXXXX, 2)
[DEBUG] Material groups for sampling:
[DEBUG]   MaterialName: XXXXX samples

# Bad signs:
[WARNING] Default UV layer 'X' not found
[WARNING] Failed to load diffuse for X
[DEBUG]   None: XXXXX samples          # ← All samples have no material!
```

### Step 0.4: Create Minimal Test Script

Create a focused diagnostic script:

```python
#!/usr/bin/env python3
"""Diagnostic script to trace color sampling."""

import json
import numpy as np
from pathlib import Path
from PIL import Image

def diagnose_stage1_output(temp_dir: str):
    """Check Stage 1 extraction output."""
    temp_path = Path(temp_dir)
    
    print("=" * 60)
    print("STAGE 1 DIAGNOSTIC")
    print("=" * 60)
    
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
    print("\nTextures:")
    for mat_name, mat_data in manifest.get('materials', {}).items():
        diffuse = mat_data.get('diffuse')
        if diffuse:
            diffuse_path = diffuse.get('path') if isinstance(diffuse, dict) else diffuse
            if Path(diffuse_path).exists():
                img = Image.open(diffuse_path)
                arr = np.array(img)
                mean_color = arr.mean(axis=(0,1))
                is_black = arr.max() < 10
                status = "[FAIL] ALL BLACK" if is_black else "[OK]"
                print(f"  {mat_name}: {status} (mean: {mean_color[:3]})")
            else:
                print(f"  {mat_name}: [FAIL] File not found: {diffuse_path}")
        else:
            print(f"  {mat_name}: [WARN] No diffuse texture")
    
    # Check UVs
    print("\nUV Layers:")
    for uv_name, uv_path in manifest.get('uv_layers', {}).items():
        if Path(uv_path).exists():
            uvs = np.load(uv_path)
            in_range = (uvs >= 0).all() and (uvs <= 1).all()
            status = "[OK]" if in_range else "[WARN] Out of range"
            print(f"  {uv_name}: {status} shape={uvs.shape} range=[{uvs.min():.3f}, {uvs.max():.3f}]")
        else:
            print(f"  {uv_name}: [FAIL] File not found: {uv_path}")
    
    # Check face_materials
    print("\nFace Materials Distribution:")
    face_mats = manifest.get('face_materials', [])
    from collections import Counter
    counts = Counter(face_mats)
    for mat, count in counts.most_common():
        print(f"  {mat}: {count} faces ({100*count/len(face_mats):.1f}%)")
    
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diagnose_color.py <temp_directory>")
        sys.exit(1)
    diagnose_stage1_output(sys.argv[1])
```

Save as `tools/diagnose_color.py` and run:

```bash
python tools/diagnose_color.py ./output/_temp_whatever/
```

---

## Phase 1: Fix Identified Issues

Based on diagnostic results, apply the appropriate fix.

### Fix 1A: If Textures Are Black (FP-1)

**Problem:** Blender extraction produces black textures.

**Solution:** Fix `extract_packed.py` texture handling.

```python
# In extract_packed_textures(), add validation:

def extract_packed_textures(output_path):
    # ... existing extraction code ...
    
    # Add validation after extraction
    for img_name, img_path in extracted.items():
        img = Image.open(img_path)
        arr = np.array(img)
        if arr.max() < 10:
            print(f"  [WARN] Texture {img_name} appears to be all black!")
            print(f"         This may indicate the texture wasn't properly packed.")
    
    return extracted
```

**Root cause check:** Are textures actually packed in the .blend file?

```python
# Add to extract_packed.py, before extraction:
print("Checking for packed images...")
for img in bpy.data.images:
    packed = "PACKED" if img.packed_file else "EXTERNAL"
    has_data = "HAS_DATA" if img.has_data else "NO_DATA"
    print(f"  {img.name}: {packed}, {has_data}")
```

### Fix 1B: If UV Layer Not Found (FP-6)

**Problem:** UV layer name mismatch between manifest and loaded data.

**Solution:** Use the first available UV layer as fallback.

**File:** `src/mesh_to_gaussian.py`

**In `_mesh_to_gaussians_multi_material()`, around line 1340:**

```python
# BEFORE:
default_uv_layer = manifest.get('uv_layer')
if default_uv_layer and default_uv_layer in uv_layers:
    loop_uvs = uv_layers[default_uv_layer]
else:
    self.logger.warning("Default UV layer '%s' not found in loaded UV layers", default_uv_layer)

# AFTER:
default_uv_layer = manifest.get('uv_layer')
if default_uv_layer and default_uv_layer in uv_layers:
    loop_uvs = uv_layers[default_uv_layer]
    self.logger.info("Using UV layer: %s", default_uv_layer)
elif uv_layers:
    # Fallback: use first available UV layer
    fallback_name = next(iter(uv_layers.keys()))
    loop_uvs = uv_layers[fallback_name]
    self.logger.warning(
        "Default UV layer '%s' not found. Using fallback: '%s'",
        default_uv_layer, fallback_name
    )
else:
    self.logger.error("NO UV LAYERS AVAILABLE - colors will be wrong!")
    loop_uvs = None
```

### Fix 1C: If Material Groups Are Empty (FP-7)

**Problem:** `face_materials` doesn't match texture names.

**Solution:** Add case-insensitive matching and debug logging.

**File:** `src/mesh_to_gaussian.py`

**In `_sample_multi_material_colors()`, around line 1150:**

```python
# Add at the start of the function:
self.logger.info("=== COLOR SAMPLING DEBUG ===")
self.logger.info("face_materials sample: %s", face_materials[:5] if face_materials else "EMPTY")
self.logger.info("material_textures keys: %s", list(material_textures.keys()))

# Check for case mismatch
face_mat_set = set(face_materials) if face_materials else set()
tex_mat_set = set(material_textures.keys())
self.logger.info("Unique face materials: %s", face_mat_set)
self.logger.info("Available texture materials: %s", tex_mat_set)

if face_mat_set and tex_mat_set and not face_mat_set.intersection(tex_mat_set):
    self.logger.error("NO OVERLAP between face materials and texture materials!")
    self.logger.error("This is likely a naming mismatch bug.")
```

### Fix 1D: If Texture Paths Are Wrong (FP-5)

**Problem:** Manifest contains relative paths that don't resolve.

**Solution:** Make paths absolute in manifest or resolve them during loading.

**File:** `src/mesh_to_gaussian.py`

**In `_load_material_textures()`, around line 780:**

```python
# BEFORE:
diffuse_path = diffuse_entry['path'] if isinstance(diffuse_entry, dict) else diffuse_entry
img = Image.open(diffuse_path)

# AFTER:
diffuse_path = diffuse_entry['path'] if isinstance(diffuse_entry, dict) else diffuse_entry

# Try to resolve path
diffuse_path = Path(diffuse_path)
if not diffuse_path.exists():
    # Try relative to manifest location
    manifest_dir = Path(manifest.get('_manifest_path', '.')).parent
    alt_path = manifest_dir / diffuse_path.name
    if alt_path.exists():
        diffuse_path = alt_path
        self.logger.debug("Resolved texture path: %s", diffuse_path)
    else:
        self.logger.error("Texture not found: %s (also tried: %s)", diffuse_path, alt_path)
        continue

img = Image.open(diffuse_path)
```

**Also add manifest path tracking in orchestrator:**

**File:** `src/pipeline/orchestrator.py`

**In `_run_packed_extraction()`, after loading manifest:**

```python
# Add manifest location for path resolution
manifest['_manifest_path'] = str(self.temp_dir / "material_manifest.json")
```

### Fix 1E: If Opacity Culls Everything (FP-9)

**Problem:** Transparency sampling returns all zeros, culling all gaussians.

**Solution:** Add diagnostic logging and optional bypass.

**File:** `src/mesh_to_gaussian.py`

**In `_mesh_to_gaussians_multi_material()`, around line 1430:**

```python
# After opacity sampling, add diagnostic:
if 'transparency' in textures:
    # ... existing sampling code ...
    
    # Add diagnostic
    non_zero_opacity = (opacities > 0.1).sum()
    self.logger.info("Opacity diagnostic: %d/%d samples have opacity > 0.1 (%.1f%%)",
                    non_zero_opacity, n_samples, 100 * non_zero_opacity / n_samples)
    
    if non_zero_opacity == 0:
        self.logger.warning("ALL samples have low opacity! Disabling opacity culling.")
        opacities[:] = 1.0  # Force all opaque for debugging
```

---

## Phase 2: Add Validation Layer

After fixing identified issues, add validation to catch future problems.

### Step 2.1: Create PipelineValidator Class

**File:** `src/pipeline/validator.py` (new file)

```python
#!/usr/bin/env python3
# ABOUTME: Validation layer for pipeline outputs
# ABOUTME: Catches color and data integrity issues early

import json
import numpy as np
from pathlib import Path
from typing import List, Optional
from PIL import Image
import logging


class PipelineValidator:
    """Validates pipeline stage outputs to catch issues early."""
    
    def __init__(self):
        self.logger = logging.getLogger('gaussian_pipeline.validator')
        self.issues: List[str] = []
    
    def validate_stage1_output(self, output_dir: Path) -> bool:
        """
        Validate packed extraction output.
        
        Returns:
            True if valid, False if critical issues found
        """
        self.issues = []
        output_dir = Path(output_dir)
        
        self.logger.info("Validating Stage 1 output: %s", output_dir)
        
        # Check manifest exists
        manifest_path = output_dir / "material_manifest.json"
        if not manifest_path.exists():
            self.issues.append("CRITICAL: material_manifest.json not found")
            return False
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Check materials
        materials = manifest.get('materials', {})
        if not materials:
            self.issues.append("CRITICAL: No materials in manifest")
            return False
        
        # Check textures exist and aren't black
        for mat_name, mat_data in materials.items():
            diffuse = mat_data.get('diffuse')
            if diffuse:
                path = diffuse.get('path') if isinstance(diffuse, dict) else diffuse
                if not Path(path).exists():
                    self.issues.append(f"ERROR: Texture not found: {path}")
                else:
                    img = Image.open(path)
                    arr = np.array(img)
                    if arr.max() < 10:
                        self.issues.append(f"WARNING: Texture appears black: {path}")
                    mean = arr.mean()
                    self.logger.debug("Texture %s: mean=%.1f, max=%d", 
                                     Path(path).name, mean, arr.max())
        
        # Check face_materials
        face_materials = manifest.get('face_materials', [])
        if not face_materials:
            self.issues.append("CRITICAL: No face_materials in manifest")
            return False
        
        # Check for None entries
        none_count = sum(1 for m in face_materials if m is None)
        if none_count > 0:
            pct = 100 * none_count / len(face_materials)
            if pct > 50:
                self.issues.append(f"ERROR: {pct:.0f}% of faces have no material")
            else:
                self.issues.append(f"WARNING: {pct:.0f}% of faces have no material")
        
        # Check UV layers
        uv_layers = manifest.get('uv_layers', {})
        if not uv_layers:
            self.issues.append("WARNING: No UV layers exported")
        else:
            for uv_name, uv_path in uv_layers.items():
                if not Path(uv_path).exists():
                    self.issues.append(f"ERROR: UV layer not found: {uv_path}")
                else:
                    uvs = np.load(uv_path)
                    if uvs.min() < -1 or uvs.max() > 2:
                        self.issues.append(f"WARNING: UV coords out of range: {uv_name}")
        
        # Report
        critical = any("CRITICAL" in i for i in self.issues)
        errors = sum(1 for i in self.issues if "ERROR" in i)
        warnings = sum(1 for i in self.issues if "WARNING" in i)
        
        self.logger.info("Validation: %d critical, %d errors, %d warnings",
                        1 if critical else 0, errors, warnings)
        
        for issue in self.issues:
            if "CRITICAL" in issue:
                self.logger.error(issue)
            elif "ERROR" in issue:
                self.logger.error(issue)
            else:
                self.logger.warning(issue)
        
        return not critical
    
    def validate_gaussians(self, gaussians: list) -> bool:
        """
        Validate gaussian output before PLY writing.
        
        Returns:
            True if valid, False if critical issues found
        """
        self.issues = []
        
        if not gaussians:
            self.issues.append("CRITICAL: No gaussians generated")
            return False
        
        # Extract data
        sh_dcs = np.array([g.sh_dc for g in gaussians])
        opacities = np.array([g.opacity for g in gaussians])
        positions = np.array([g.position for g in gaussians])
        
        # Check for NaN/Inf
        if not np.isfinite(positions).all():
            self.issues.append("ERROR: Positions contain NaN/Inf")
        if not np.isfinite(sh_dcs).all():
            self.issues.append("ERROR: SH DC values contain NaN/Inf")
        
        # Check color variance
        color_var = sh_dcs.var()
        if color_var < 0.0001:
            rgb = sh_dcs + 0.5
            mean_rgb = rgb.mean(axis=0)
            self.issues.append(
                f"WARNING: All colors are identical! Mean RGB: ({mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f})"
            )
            if (mean_rgb < 0.1).all():
                self.issues.append("ERROR: All colors are BLACK - color sampling failed!")
            elif (np.abs(mean_rgb - 0.5) < 0.01).all():
                self.issues.append("ERROR: All colors are GRAY - fallback to default!")
        
        # Check opacity
        if opacities.mean() < 0.1:
            self.issues.append("WARNING: Very low average opacity - model may be invisible")
        
        # Check SH DC range
        if sh_dcs.min() < -0.6 or sh_dcs.max() > 0.6:
            self.issues.append("WARNING: SH DC values outside expected range [-0.5, 0.5]")
        
        # Report
        for issue in self.issues:
            if "CRITICAL" in issue or "ERROR" in issue:
                self.logger.error(issue)
            else:
                self.logger.warning(issue)
        
        critical = any("CRITICAL" in i or "ERROR" in i for i in self.issues)
        return not critical
    
    def get_issues(self) -> List[str]:
        """Return list of issues found during validation."""
        return self.issues.copy()
```

### Step 2.2: Integrate Validator into Pipeline

**File:** `src/pipeline/orchestrator.py`

**Add import at top:**
```python
from .validator import PipelineValidator
```

**Add validation after Stage 1:**
```python
def _run_packed_extraction(self) -> Tuple[Path, dict]:
    # ... existing extraction code ...
    
    # Validate Stage 1 output
    validator = PipelineValidator()
    if not validator.validate_stage1_output(self.temp_dir):
        issues = validator.get_issues()
        raise RuntimeError(
            f"Stage 1 validation failed:\n" + "\n".join(issues)
        )
    
    return obj_path, manifest
```

**Add validation before PLY save:**
```python
def _run_stage2(self, obj_path: Path, manifest: dict = None) -> List:
    # ... existing conversion code ...
    
    # Validate gaussians before returning
    validator = PipelineValidator()
    if not validator.validate_gaussians(gaussians):
        issues = validator.get_issues()
        self.logger.error("Gaussian validation issues:\n%s", "\n".join(issues))
        # Don't fail, but log prominently
    
    return gaussians
```

---

## Phase 3: End-to-End Verification

### Step 3.1: Create Test RGB Cube

Create a simple test asset with known colors:

1. In Blender, create a cube
2. Create 6 materials: Red, Green, Blue, Cyan, Magenta, Yellow
3. Assign one material per face
4. Save as `test_rgb_cube.blend`

**Expected colors (RGB):**
- Face 1: Red (1.0, 0.0, 0.0) → SH DC (0.5, -0.5, -0.5)
- Face 2: Green (0.0, 1.0, 0.0) → SH DC (-0.5, 0.5, -0.5)
- Face 3: Blue (0.0, 0.0, 1.0) → SH DC (-0.5, -0.5, 0.5)
- Face 4: Cyan (0.0, 1.0, 1.0) → SH DC (-0.5, 0.5, 0.5)
- Face 5: Magenta (1.0, 0.0, 1.0) → SH DC (0.5, -0.5, 0.5)
- Face 6: Yellow (1.0, 1.0, 0.0) → SH DC (0.5, 0.5, -0.5)

### Step 3.2: Run Full Pipeline Test

```bash
# Convert test cube
python cli.py test_rgb_cube.blend ./output --packed --verbose

# Inspect output
python tools/ply_inspector.py output/test_rgb_cube_full.ply 100

# Expected output should show:
# - SH DC values spanning from -0.5 to 0.5
# - Color variance > 0
# - Black vertices < 20%
```

### Step 3.3: Visual Verification

1. Start viewer: `python viewer/server.py`
2. Open http://localhost:8000
3. Load `test_rgb_cube_full.ply`
4. Verify 6 distinct colors are visible

### Step 3.4: Test with packed-tree.blend

```bash
python cli.py packed-tree.blend ./output --packed --verbose

python tools/ply_inspector.py output/packed-tree_full.ply 50

# Expected: 
# - Non-zero color variance
# - Mean RGB showing green-ish (tree colors)
# - Less than 50% black vertices
```

---

## Code Changes Reference

### Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `src/mesh_to_gaussian.py` | Add debug logging, fix UV fallback | P0 |
| `src/stage1_baker/blender_scripts/extract_packed.py` | Add texture validation | P0 |
| `src/pipeline/orchestrator.py` | Add manifest path tracking, validation | P1 |
| `src/pipeline/validator.py` | New file - validation layer | P1 |

### Files to Create

| File | Purpose |
|------|---------|
| `src/pipeline/validator.py` | Validation layer |
| `tools/diagnose_color.py` | Diagnostic script |
| `test_assets/test_rgb_cube.blend` | Test asset |

### Verification Checklist

After all fixes:

- [ ] `test_rgb_cube.blend` produces 6 distinct colors
- [ ] PLY inspector shows color variance > 0
- [ ] PLY inspector shows black vertices < 20%
- [ ] Viewer displays colors correctly
- [ ] `packed-tree.blend` produces visible colors
- [ ] No validation errors in verbose output

---

## Rollback Plan

If fixes break something else:

1. All changes should be in a feature branch
2. Each fix should be a separate commit
3. Test after each commit
4. If a fix causes new problems, revert that commit only

```bash
# Create feature branch
git checkout -b fix/color-system

# Commit each fix separately
git add src/mesh_to_gaussian.py
git commit -m "Add debug logging for color sampling"

# Test, then continue with next fix
```

---

## Success Criteria

The color system is fixed when:

1. **PLY Inspector Test:** `test_rgb_cube` output shows 6 distinct SH DC clusters
2. **Visual Test:** Viewer displays distinct colors for test cube
3. **Real Asset Test:** `packed-tree.blend` shows green/brown tree colors
4. **Validation Test:** No critical/error issues from validator
5. **Regression Test:** Existing geometry/positioning still works

---

## Next Steps

After implementation:

1. See **Document 3: Validation & Testing Specification** for detailed test procedures
2. Consider adding automated tests for color preservation
3. Consider refactoring `mesh_to_gaussian.py` into smaller modules (optional, lower priority)
