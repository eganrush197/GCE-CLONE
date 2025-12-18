# Gaussian Splat Pipeline: Validation & Testing Specification

**Document 3 of 3**  
**Date:** December 2024  
**Status:** Ready for Use  
**Test Asset:** `packed-tree.blend`

---

## Executive Summary

This document defines the test procedures, acceptance criteria, and validation checkpoints for verifying the color system fix. Use this as a checklist during implementation.

---

## Table of Contents

1. [Test Assets](#1-test-assets)
2. [Diagnostic Procedures](#2-diagnostic-procedures)
3. [Acceptance Criteria](#3-acceptance-criteria)
4. [Test Procedures](#4-test-procedures)
5. [Validation Checkpoints](#5-validation-checkpoints)
6. [Troubleshooting Guide](#6-troubleshooting-guide)

---

## 1. Test Assets

### 1.1 Required Test Assets

| Asset | Purpose | Expected Output |
|-------|---------|-----------------|
| `test_rgb_cube.blend` | Known-color validation | 6 distinct colors |
| `packed-tree.blend` | Real-world test | Green/brown colors |

### 1.2 Creating test_rgb_cube.blend

**Step-by-step in Blender:**

1. Open Blender, delete default cube
2. Add > Mesh > Cube
3. Tab into Edit Mode
4. Create 6 materials:
   - `Mat_Red`: Base Color = (1, 0, 0)
   - `Mat_Green`: Base Color = (0, 1, 0)
   - `Mat_Blue`: Base Color = (0, 0, 1)
   - `Mat_Cyan`: Base Color = (0, 1, 1)
   - `Mat_Magenta`: Base Color = (1, 0, 1)
   - `Mat_Yellow`: Base Color = (1, 1, 0)
5. Select each face, assign corresponding material
6. Tab out of Edit Mode
7. File > Save As > `test_rgb_cube.blend`

**Alternative: Textured version**

1. Create a 64x64 PNG with 6 solid color squares
2. UV unwrap cube to the texture
3. This tests texture sampling path

### 1.3 Test Asset Validation

Before using a test asset, verify it in Blender:

```
1. Open asset in Blender
2. Check Materials panel shows expected materials
3. Check each face has correct material assigned
4. Check UV Map exists (for textured assets)
5. Render preview looks correct
```

---

## 2. Diagnostic Procedures

### 2.1 PLY Inspector Analysis

**Command:**
```bash
python tools/ply_inspector.py <ply_file> [sample_count]
```

**Key Metrics to Record:**

| Metric | How to Find | Healthy Value |
|--------|-------------|---------------|
| SH DC Mean | "Mean:" line | Not (0,0,0) or (0.5,0.5,0.5) |
| SH DC Variance | Max - Min range | > 0.1 |
| Black Vertex % | "Black vertices" line | < 30% |
| Opacity Mean | "Opacity stats" line | > 0.5 |

**Recording Template:**

```
File: ___________________
Date: ___________________

SH DC Stats:
  Min: (_____, _____, _____)
  Max: (_____, _____, _____)
  Mean: (_____, _____, _____)

Color Health:
  Black vertices: _____% 
  Color variance: _____ (Good if > 0.1)

Opacity Stats:
  Min: _____
  Max: _____
  Mean: _____

Assessment: [ ] PASS  [ ] FAIL
Notes: _______________________
```

### 2.2 Stage 1 Output Inspection

**Procedure:**

```bash
# 1. Run with keep-temp flag
python cli.py <input>.blend ./output --packed --keep-temp --verbose

# 2. Find temp directory
ls -la ./output/

# 3. Check texture extraction
ls -la ./output/*temp*/textures/

# 4. Verify textures aren't black
# Open any PNG in image viewer

# 5. Check manifest
cat ./output/*temp*/material_manifest.json | python -m json.tool

# 6. Check UV layers
ls -la ./output/*temp*/*.npy
```

**Checklist:**

- [ ] Temp directory created
- [ ] `textures/` directory contains PNG files
- [ ] PNG files have visible colors (not black)
- [ ] `material_manifest.json` exists
- [ ] Manifest has `materials` section with entries
- [ ] Manifest has `face_materials` array (non-empty)
- [ ] Manifest has `uv_layers` section
- [ ] UV `.npy` files exist

### 2.3 Verbose Log Analysis

**Key Log Patterns:**

```bash
# Run with verbose
python cli.py <input>.blend ./output --packed --verbose 2>&1 | tee log.txt

# Search for key patterns
grep -E "(Loaded|WARNING|ERROR|Material groups|avg color)" log.txt
```

**Good Signs:**
```
[INFO] Loaded textures for 3 materials
[INFO] Loaded UV layer 'uv0': (12000, 2)
[DEBUG] Material groups for sampling:
[DEBUG]   Trunk: 5000 samples
[DEBUG]   Leaves: 15000 samples
[DEBUG]   Trunk avg color: RGB(0.400, 0.300, 0.200)
[DEBUG]   Leaves avg color: RGB(0.200, 0.500, 0.100)
```

**Bad Signs:**
```
[WARNING] Default UV layer 'uv0' not found
[WARNING] Failed to load diffuse for MaterialName
[DEBUG]   None: 20000 samples    # ← All samples without material!
[DEBUG] avg color: RGB(0.000, 0.000, 0.000)  # ← Black!
[INFO] Culled 20000 samples (100.0%)  # ← Everything culled!
```

---

## 3. Acceptance Criteria

### 3.1 Minimum Viable Fix

The color system is fixed when ALL of the following pass:

| Criterion | Test | Pass Condition |
|-----------|------|----------------|
| AC-1 | PLY Inspector on test_rgb_cube | 6 distinct SH DC clusters |
| AC-2 | Visual test_rgb_cube in viewer | 6 visible colors |
| AC-3 | PLY Inspector on packed-tree | Mean RGB ≠ (0,0,0) |
| AC-4 | Visual packed-tree in viewer | Visible green/brown |
| AC-5 | Validation layer | No CRITICAL errors |

### 3.2 Full Fix

In addition to minimum viable:

| Criterion | Test | Pass Condition |
|-----------|------|----------------|
| AC-6 | Black vertex percentage | < 20% |
| AC-7 | Color variance | > 0.05 |
| AC-8 | Opacity preservation | Mean > 0.5 |
| AC-9 | Geometry preserved | Bounding box unchanged |
| AC-10 | Performance | < 2x slowdown from baseline |

### 3.3 Regression Criteria

Ensure existing functionality isn't broken:

| Criterion | Test | Pass Condition |
|-----------|------|----------------|
| RC-1 | Vertex positions | Match before/after |
| RC-2 | Gaussian count | Similar to before |
| RC-3 | LOD generation | Still works |
| RC-4 | Viewer loading | Still works |
| RC-5 | Binary streaming | Still works |

---

## 4. Test Procedures

### 4.1 Test Procedure: RGB Cube

**Purpose:** Verify known colors are preserved

**Setup:**
- `test_rgb_cube.blend` created per Section 1.2
- Pipeline running with fixes applied

**Steps:**

1. **Convert:**
   ```bash
   python cli.py test_rgb_cube.blend ./output --packed --verbose
   ```

2. **Inspect PLY:**
   ```bash
   python tools/ply_inspector.py output/test_rgb_cube_full.ply 100
   ```

3. **Verify SH DC values:**
   - Should see values near:
     - (0.5, -0.5, -0.5) for red
     - (-0.5, 0.5, -0.5) for green
     - (-0.5, -0.5, 0.5) for blue
     - etc.

4. **Visual verification:**
   ```bash
   python viewer/server.py
   # Open http://localhost:8000
   # Load test_rgb_cube_full.ply
   ```

5. **Verify 6 colors visible**

**Pass Criteria:**
- [ ] PLY shows 6 distinct SH DC value clusters
- [ ] Viewer shows 6 distinct colors
- [ ] No validation errors in log

### 4.2 Test Procedure: Packed Tree

**Purpose:** Verify real-world asset works

**Setup:**
- `packed-tree.blend` available
- Pipeline running with fixes applied

**Steps:**

1. **Convert:**
   ```bash
   python cli.py packed-tree.blend ./output --packed --verbose
   ```

2. **Check logs for errors:**
   ```bash
   grep -E "(ERROR|WARNING|FAIL)" output.log
   ```

3. **Inspect PLY:**
   ```bash
   python tools/ply_inspector.py output/packed-tree_full.ply 50
   ```

4. **Verify color stats:**
   - Mean RGB should NOT be (0,0,0) or (0.5,0.5,0.5)
   - Should show green-ish values for leaves
   - Should show brown-ish values for trunk

5. **Visual verification:**
   ```bash
   python viewer/server.py
   # Open http://localhost:8000
   # Load packed-tree_full.ply
   ```

6. **Verify tree has visible colors**

**Pass Criteria:**
- [ ] No CRITICAL validation errors
- [ ] PLY mean RGB ≠ (0,0,0)
- [ ] Black vertex % < 50%
- [ ] Viewer shows colored tree

### 4.3 Test Procedure: Regression Check

**Purpose:** Ensure existing functionality works

**Steps:**

1. **Run LOD generation:**
   ```bash
   python cli.py packed-tree.blend ./output --packed --lod 5000 25000 100000
   ```

2. **Verify LOD files created:**
   ```bash
   ls -la output/*lod*.ply
   ```

3. **Load each LOD in viewer:**
   - Each should display with colors
   - Each should have progressively fewer points

4. **Test baking pipeline (if supported):**
   ```bash
   python cli.py packed-tree.blend ./output --verbose
   # (without --packed flag)
   ```

**Pass Criteria:**
- [ ] LOD files created
- [ ] LODs display correctly
- [ ] Both pipelines produce output

---

## 5. Validation Checkpoints

Use these checkpoints during development to catch issues early.

### 5.1 After Stage 1 (Extraction)

**Check before proceeding to Stage 2:**

```python
# Quick validation script
import json
from pathlib import Path
from PIL import Image
import numpy as np

def validate_stage1(temp_dir):
    temp_dir = Path(temp_dir)
    
    # Check manifest
    manifest = json.load(open(temp_dir / "material_manifest.json"))
    print(f"Materials: {len(manifest['materials'])}")
    print(f"Face materials: {len(manifest['face_materials'])}")
    
    # Check textures
    for mat_name, mat_data in manifest['materials'].items():
        if mat_data.get('diffuse'):
            path = mat_data['diffuse'].get('path', mat_data['diffuse'])
            if Path(path).exists():
                img = np.array(Image.open(path))
                print(f"{mat_name}: max={img.max()}, mean={img.mean():.1f}")
            else:
                print(f"{mat_name}: FILE NOT FOUND")
    
    return True

# Usage: validate_stage1("./output/_temp_xxx/")
```

### 5.2 After Color Sampling

**Check before building gaussians:**

```python
# Add to _sample_multi_material_colors() temporarily:

# After sampling, before return:
self.logger.info("COLOR SAMPLING CHECKPOINT:")
self.logger.info("  Samples: %d", len(colors))
self.logger.info("  Color mean: (%.3f, %.3f, %.3f)", *colors.mean(axis=0))
self.logger.info("  Color variance: %.4f", colors.var())
self.logger.info("  Black (< 0.1): %d (%.1f%%)", 
                (colors.max(axis=1) < 0.1).sum(),
                100 * (colors.max(axis=1) < 0.1).mean())

if colors.var() < 0.001:
    self.logger.error("CHECKPOINT FAIL: No color variance!")
```

### 5.3 After Gaussian Creation

**Check before PLY save:**

```python
# Add to orchestrator._run_stage2() temporarily:

# After gaussians created:
sh_dcs = np.array([g.sh_dc for g in gaussians])
rgb = sh_dcs + 0.5

print("GAUSSIAN CHECKPOINT:")
print(f"  Count: {len(gaussians)}")
print(f"  RGB mean: ({rgb.mean(axis=0)[0]:.3f}, {rgb.mean(axis=0)[1]:.3f}, {rgb.mean(axis=0)[2]:.3f})")
print(f"  RGB variance: {rgb.var():.4f}")

if rgb.var() < 0.001:
    print("CHECKPOINT FAIL: All gaussians same color!")
```

### 5.4 After PLY Save

**Check file before viewer:**

```bash
# Quick check
python tools/ply_inspector.py output/file.ply 10

# Look for:
# - SH DC values NOT all zero
# - Color variance > 0
# - Opacity mean > 0.5
```

---

## 6. Troubleshooting Guide

### 6.1 Symptom: All Black Output

**Possible Causes & Fixes:**

| Cause | Diagnostic | Fix |
|-------|------------|-----|
| Textures not extracted | Check `textures/` dir | Fix Blender script |
| Textures are black | Open PNG manually | Check Blender material |
| UV mismatch | Check UV stats | Use fallback UV |
| Material name mismatch | Compare manifest vs code | Case-insensitive match |
| All samples culled | Check "Culled X%" log | Disable opacity culling |

### 6.2 Symptom: All Gray Output

**Diagnosis:** Fallback to default color

**Possible Causes:**
- `_has_texture_visual()` returns False
- Material groups are empty
- Texture loading fails silently

**Debug Steps:**
1. Add logging to `_has_texture_visual()`
2. Print material_groups contents
3. Check texture loading logs

### 6.3 Symptom: Partial Colors

**Diagnosis:** Some faces colored, some black

**Possible Causes:**
- Some materials missing textures
- Face-material mapping incomplete
- UV coverage issues

**Debug Steps:**
1. Check which materials are in manifest
2. Check face_materials distribution
3. Verify UV coverage in Blender

### 6.4 Symptom: Wrong Colors

**Diagnosis:** Colors present but incorrect

**Possible Causes:**
- UV coordinates wrong
- Texture channel confusion (RGB vs BGR)
- SH DC conversion error

**Debug Steps:**
1. Sample known coordinates manually
2. Check texture loading in isolation
3. Verify SH DC math: `sh_dc = rgb - 0.5`

### 6.5 Symptom: Viewer Shows Black but PLY is Correct

**Diagnosis:** Bug in viewer, not pipeline

**Debug Steps:**
1. Check PLY inspector shows correct values
2. Check browser console for shader errors
3. Check binary transfer is sending correct data
4. Verify `vColor = instanceSHDC + 0.5` in shader

---

## Appendix A: Quick Reference Commands

```bash
# Full verbose run with temp files kept
python cli.py packed-tree.blend ./output --packed --verbose --keep-temp

# Inspect PLY output
python tools/ply_inspector.py output/packed-tree_full.ply 50

# Check Stage 1 output
cat output/_temp*/material_manifest.json | python -m json.tool
ls -la output/_temp*/textures/

# Start viewer
python viewer/server.py

# Search logs for issues
grep -E "(ERROR|WARNING|BLACK|variance)" log.txt
```

## Appendix B: Expected Values Reference

**SH DC for common colors:**

| Color | RGB | SH DC |
|-------|-----|-------|
| Black | (0, 0, 0) | (-0.5, -0.5, -0.5) |
| White | (1, 1, 1) | (0.5, 0.5, 0.5) |
| Gray | (0.5, 0.5, 0.5) | (0, 0, 0) |
| Red | (1, 0, 0) | (0.5, -0.5, -0.5) |
| Green | (0, 1, 0) | (-0.5, 0.5, -0.5) |
| Blue | (0, 0, 1) | (-0.5, -0.5, 0.5) |
| Cyan | (0, 1, 1) | (-0.5, 0.5, 0.5) |
| Magenta | (1, 0, 1) | (0.5, -0.5, 0.5) |
| Yellow | (1, 1, 0) | (0.5, 0.5, -0.5) |

**Conversion formulas:**
```
Encoding: sh_dc = rgb - 0.5
Decoding: rgb = sh_dc + 0.5
```

## Appendix C: Checklist Template

```
Date: ____________
Tester: ____________
Asset: ____________

PRE-TEST:
[ ] Test asset validated in Blender
[ ] Pipeline code updated
[ ] Clean output directory

STAGE 1:
[ ] Textures extracted (count: ___)
[ ] Textures not black
[ ] Manifest valid
[ ] Face materials populated
[ ] UV layers exported

STAGE 2:
[ ] Materials loaded (count: ___)
[ ] UV layers loaded
[ ] Color sampling checkpoint passed
[ ] Gaussian count: ___

OUTPUT:
[ ] PLY created
[ ] PLY inspector: mean RGB = (___,___,___)
[ ] PLY inspector: black % = ___%
[ ] PLY inspector: opacity mean = ___

VIEWER:
[ ] File loads
[ ] Colors visible
[ ] Colors correct

RESULT: [ ] PASS  [ ] FAIL

Notes:
_________________________________
_________________________________
```
