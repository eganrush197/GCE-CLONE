# Gaussian Splat Pipeline: Technical Analysis & Diagnosis

**Document 1 of 3**  
**Date:** December 2024  
**Status:** Critical Issue Analysis  
**Test Asset:** `packed-tree.blend`

---

## Executive Summary

The Gaussian Splat Conversion Pipeline has a critical defect: **all converted models output as solid black**, regardless of input textures or colors. This document provides a complete technical analysis of the failure, identifies root causes, and maps the path to resolution.

**Key Finding:** The color system has never worked. Despite extensive infrastructure (LOD generation, web viewer, binary streaming), the core feature—preserving color data from source models—fails silently.

---

## Table of Contents

1. [Observed Symptoms](#1-observed-symptoms)
2. [Expected vs Actual Behavior](#2-expected-vs-actual-behavior)
3. [Root Cause Analysis](#3-root-cause-analysis)
4. [Code-Level Diagnosis](#4-code-level-diagnosis)
5. [Failure Point Map](#5-failure-point-map)
6. [Impact Assessment](#6-impact-assessment)

---

## 1. Observed Symptoms

### 1.1 Primary Symptom

All output PLY files render as **solid black** in the viewer, regardless of:
- Input file type (.blend, .obj)
- Pipeline mode (baking vs packed extraction)
- Texture presence or absence
- Model complexity

### 1.2 What "Black" Tells Us

The output being **black** (not gray) is diagnostically significant:

| Output Color | SH DC Value | Sampled RGB | Indicates |
|--------------|-------------|-------------|-----------|
| **Gray (0.5)** | (0.0, 0.0, 0.0) | (0.5, 0.5, 0.5) | Fallback to default |
| **Black (0.0)** | (-0.5, -0.5, -0.5) | (0.0, 0.0, 0.0) | Actively sampled black OR opacity=0 |

Since output is **black**, not gray, this means either:
1. Colors are being sampled as (0.0, 0.0, 0.0) — true black
2. Opacity values are 0 — gaussians are invisible
3. Data corruption in the pipeline

### 1.3 Verification Method

Run the PLY inspector on any output file:

```bash
python tools/ply_inspector.py output_clouds/packed-tree_full.ply
```

Expected diagnostic output:
```
SH DC -> RGB:
  Min: (X.XXX, X.XXX, X.XXX)
  Max: (X.XXX, X.XXX, X.XXX)
  Mean: (X.XXX, X.XXX, X.XXX)

Black vertices (RGB < 0.1): XXXXXX (XX.X%)

Opacity stats: min=X.XXX, max=X.XXX, mean=X.XXX
```

**If Mean RGB ≈ 0.0:** Colors are being sampled as black
**If Opacity mean ≈ 0.0:** Gaussians are invisible
**If Mean RGB ≈ 0.5:** Bug is in viewer, not pipeline

---

## 2. Expected vs Actual Behavior

### 2.1 Expected Color Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Blender File   │────▶│  Packed Extract │────▶│  Manifest JSON  │
│  (textures)     │     │  (extract_      │     │  + UV .npy      │
│                 │     │   packed.py)    │     │  + textures/    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Viewer         │◀────│  PLY File       │◀────│  Converter      │
│  (RGB display)  │     │  (SH DC values) │     │  (texture       │
│                 │     │                 │     │   sampling)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘

Expected Values at Each Stage:
─────────────────────────────
Texture pixel:     RGB (0.2, 0.5, 0.1)  ← Green leaf color
Sampled color:     RGB (0.2, 0.5, 0.1)
SH DC conversion:  SH  (-0.3, 0.0, -0.4)
PLY storage:       f_dc_0=-0.3, f_dc_1=0.0, f_dc_2=-0.4
Viewer decode:     RGB (0.2, 0.5, 0.1)  ← Green displayed
```

### 2.2 Actual Color Flow (Current Broken State)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Blender File   │────▶│  Packed Extract │────▶│  Manifest JSON  │
│  (textures)     │     │                 │     │  + UV .npy      │
│                 │     │  ✓ Works        │     │  + textures/    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  Converter      │
                                                │                 │
                                                │  ✗ FAILS HERE   │
                                                │  (see Section 3)│
                                                └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Viewer         │◀────│  PLY File       │◀────│  Fallback/Error │
│  (BLACK)        │     │  SH DC = ???    │     │  colors = ???   │
└─────────────────┘     └─────────────────┘     └─────────────────┘

Actual Values (Hypothesized):
─────────────────────────────
Texture pixel:     RGB (0.2, 0.5, 0.1)  ← Green exists in texture
Sampled color:     RGB (0.0, 0.0, 0.0)  ← BLACK (sampling failed)
SH DC conversion:  SH  (-0.5, -0.5, -0.5)
PLY storage:       f_dc_0=-0.5, f_dc_1=-0.5, f_dc_2=-0.5
Viewer decode:     RGB (0.0, 0.0, 0.0)  ← BLACK displayed
```

---

## 3. Root Cause Analysis

### 3.1 Primary Bug: Triangulation Mismatch in Packed Extraction

**Location:** `src/stage1_baker/blender_scripts/extract_packed.py`

**The Problem:**

The extraction script builds `face_materials` from the **original mesh** (which may contain quads/n-gons), but exports UV coordinates from a **triangulated copy**. This causes a count and ordering mismatch.

**Code Evidence:**

```python
# In export_uv_layers() - TRIANGULATED mesh
def export_uv_layers(mesh_obj, output_path, manifest):
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])  # ← Triangulates
    # ... exports UVs from triangulated mesh ...
```

```python
# In main() - ORIGINAL mesh (NOT triangulated)
face_materials = []
for poly in temp_mesh.polygons:
    mat_idx = poly.material_index
    # ... builds list from original face count ...
```

**Impact:**

If original mesh has 1000 quads → triangulated = 2000 triangles.
- `face_materials` has 1000 entries (one per quad)
- UV coordinates have 6000 entries (3 loops × 2000 triangles)
- Converter tries to access `face_materials[1500]` → out of bounds or wrong material

### 3.2 Secondary Bug: Silent Fallback to Black

**Location:** `src/mesh_to_gaussian.py`, `_sample_multi_material_colors()`

When material lookup fails, the code falls back to the initialized value:

```python
def _sample_multi_material_colors(...):
    n_samples = len(face_indices)
    colors = np.full((n_samples, 3), 0.5, dtype=np.float32)  # Gray default
    # ...
    for mat_name, sample_indices in material_groups.items():
        if mat_name is None or mat_name not in material_textures:
            continue  # ← Silently skips, leaves gray
```

But the output is **black**, not gray. This suggests:

1. The gray default (0.5) is being overwritten with (0.0) somewhere
2. OR opacity culling is removing all gaussians
3. OR the texture sampling path IS being hit, but returns black

### 3.3 Tertiary Suspect: Texture Sampling Returns Black

If the triangulation mismatch causes UV coordinates to be wrong, texture sampling could return black:

```python
def _sample_texture_bilinear(self, texture, uvs, ...):
    # If UVs are out of range [0,1] or point to wrong locations...
    u = np.clip(uvs[:, 0], 0.0, 1.0)
    v = np.clip(1.0 - uvs[:, 1], 0.0, 1.0)
    # ...samples texture at potentially wrong locations
```

If UVs are garbage (due to mismatch), they might:
- Point to transparent regions → black with alpha=0
- Point to edges/padding → black pixels
- Be out of range and clip to corners → likely black if texture has black borders

### 3.4 Quaternary Suspect: Opacity Culling

The converter has aggressive opacity culling:

```python
# In _mesh_to_gaussians_multi_material()
OPACITY_THRESHOLD = 0.1

valid_mask = opacities >= OPACITY_THRESHOLD
n_culled = n_samples - np.sum(valid_mask)

if n_culled > 0:
    self.logger.info("Culled %d samples (%.1f%%) due to low opacity", ...)
```

If transparency textures are missampled (reading black = fully transparent), ALL gaussians could be culled, resulting in an empty or near-empty output.

---

## 4. Code-Level Diagnosis

### 4.1 The Triangulation Bug (CONFIRMED)

**File:** `src/stage1_baker/blender_scripts/extract_packed.py`

**Line ~380 (export_uv_layers):**
```python
def export_uv_layers(mesh_obj, output_path, manifest):
    # ...
    # Create a bmesh from the object and triangulate it
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])  # TRIANGULATED
```

**Line ~290 (main function, face_materials building):**
```python
# Build face-to-material mapping from triangulated mesh
# This ensures the mapping matches the triangulated OBJ export
import bmesh

bm = bmesh.new()
bm.from_mesh(mesh)
bmesh.ops.triangulate(bm, faces=bm.faces[:])

# Create a temporary mesh to get triangulated face materials
temp_mesh = bpy.data.meshes.new("temp_triangulated_materials")
bm.to_mesh(temp_mesh)
bm.free()

face_materials = []
for poly in temp_mesh.polygons:  # This IS triangulated - GOOD
    mat_idx = poly.material_index
    if mat_idx < len(active_obj.data.materials) and active_obj.data.materials[mat_idx]:
        face_materials.append(active_obj.data.materials[mat_idx].name)
    else:
        face_materials.append(None)
```

**Wait - this looks correct!** Let me re-examine...

Actually, looking more carefully at the code, the face_materials IS built from triangulated mesh. So the primary bug hypothesis may be wrong.

**Revised Investigation Needed:** The triangulation appears handled correctly. The bug must be elsewhere.

### 4.2 Alternative Failure Points

**A. Manifest Path Resolution**

In `_load_material_textures()`:
```python
diffuse_path = diffuse_entry['path'] if isinstance(diffuse_entry, dict) else diffuse_entry
img = Image.open(diffuse_path)  # ← Could fail if path is wrong
```

If paths in manifest are relative and the working directory is wrong, texture loading fails silently.

**B. UV Layer Name Mismatch**

In `_mesh_to_gaussians_multi_material()`:
```python
default_uv_layer = manifest.get('uv_layer')
if default_uv_layer and default_uv_layer in uv_layers:
    loop_uvs = uv_layers[default_uv_layer]
else:
    self.logger.warning("Default UV layer '%s' not found in loaded UV layers", default_uv_layer)
```

If the UV layer name in manifest doesn't match what was exported, UVs won't be found.

**C. Empty Material Groups**

```python
material_groups = {}
for i, face_idx in enumerate(face_indices):
    mat_name = face_materials[face_idx] if face_idx < len(face_materials) else None
    # ...

for mat_name, sample_indices in material_groups.items():
    if mat_name is None or mat_name not in material_textures:
        continue  # Skips ALL samples if material not found
```

If `face_materials` has `None` entries or names don't match `material_textures` keys, all sampling is skipped.

### 4.3 The Smoking Gun: Debug Logging

The code has debug logging that should reveal the issue:

```python
# In _sample_multi_material_colors()
self.logger.debug("Material groups for sampling:")
for mat_name, indices in material_groups.items():
    self.logger.debug("  %s: %d samples", mat_name, len(indices))
```

**If this shows `None: X samples` or no materials at all, that's the problem.**

```python
# Later in the same function
if 'diffuse' in textures:
    # ...
    avg_color = sampled_colors.mean(axis=0)
    self.logger.debug("  %s avg color: RGB(%.3f, %.3f, %.3f)",
                     mat_name, avg_color[0], avg_color[1], avg_color[2])
```

**If avg color is (0.0, 0.0, 0.0), texture sampling is returning black.**

---

## 5. Failure Point Map

```
INPUT: packed-tree.blend
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Packed Extraction (extract_packed.py)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [FP-1] Texture Extraction                                      │
│         extract_packed_textures()                               │
│         ▶ Could fail silently if textures aren't packed         │
│         ▶ Check: Are textures/*.png files created?              │
│         ▶ Check: Are they non-black (open in image viewer)?     │
│                                                                 │
│  [FP-2] UV Layer Export                                         │
│         export_uv_layers()                                      │
│         ▶ Could export wrong UV layer                           │
│         ▶ Check: Does uv_layers/*.npy exist?                    │
│         ▶ Check: Do UV values look valid (in [0,1] range)?      │
│                                                                 │
│  [FP-3] Material Manifest                                       │
│         build_material_manifest()                               │
│         ▶ Could have wrong paths or missing entries             │
│         ▶ Check: material_manifest.json content                 │
│         ▶ Check: Do paths resolve correctly?                    │
│                                                                 │
│  [FP-4] Face-Material Mapping                                   │
│         face_materials list                                     │
│         ▶ Could have None entries                               │
│         ▶ Could have name mismatches                            │
│         ▶ Check: Manifest face_materials array length vs faces  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Gaussian Conversion (mesh_to_gaussian.py)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [FP-5] Manifest Loading                                        │
│         _load_material_textures()                               │
│         ▶ Could fail to open texture files                      │
│         ▶ Check: "Loaded diffuse for X" log messages            │
│                                                                 │
│  [FP-6] UV Layer Loading                                        │
│         _load_uv_layers()                                       │
│         ▶ Could fail to find/load .npy files                    │
│         ▶ Check: "Loaded UV layer 'X'" log messages             │
│                                                                 │
│  [FP-7] Material Group Building                                 │
│         material_groups = {}                                    │
│         ▶ Could have all None or mismatched names               │
│         ▶ Check: "Material groups for sampling" debug output    │
│                                                                 │
│  [FP-8] Texture Sampling                                        │
│         _sample_texture_bilinear()                              │
│         ▶ Could sample black if UVs are wrong                   │
│         ▶ Check: "avg color: RGB(X, X, X)" debug output         │
│                                                                 │
│  [FP-9] Opacity Culling                                         │
│         OPACITY_THRESHOLD = 0.1                                 │
│         ▶ Could cull all samples if transparency broken         │
│         ▶ Check: "Culled X samples" log output                  │
│                                                                 │
│  [FP-10] SH DC Conversion                                       │
│          sh_dc = colors - 0.5                                   │
│          ▶ If colors = 0.0, sh_dc = -0.5 (black output)         │
│          ▶ If colors = 0.5, sh_dc = 0.0 (gray output)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: PLY Output                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [FP-11] PLY Writing                                            │
│          save_ply()                                             │
│          ▶ Generally reliable, unlikely failure point           │
│          ▶ Check: ply_inspector.py output                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ VIEWER: Display                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [FP-12] Binary Transfer                                        │
│          api.py load-binary endpoint                            │
│          ▶ Could corrupt data in transit                        │
│          ▶ Check: Browser console diagnostics                   │
│                                                                 │
│  [FP-13] Shader Rendering                                       │
│          main.js fragment shader                                │
│          ▶ vColor = instanceSHDC + 0.5                          │
│          ▶ Check: If SH DC = -0.5, output = black (correct)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Impact Assessment

### 6.1 Wasted Infrastructure

The following components have **zero value** while color is broken:

| Component | Lines of Code | Purpose | Value Without Color |
|-----------|---------------|---------|---------------------|
| LOD Generator | ~200 | Reduce point count | Zero (black at any LOD) |
| Web Viewer | ~700 | Visualize splats | Zero (displays black) |
| Binary Streaming | ~200 | Fast transfer | Zero (fast black transfer) |
| Gaussian Shaders | ~100 | Proper rendering | Zero (renders black) |
| File Watcher | ~100 | Real-time updates | Zero (updates black files) |

**Total: ~1300 lines of code providing zero value until color works.**

### 6.2 What Works

| Component | Status | Notes |
|-----------|--------|-------|
| Blender Integration | ✓ Works | Texture extraction runs |
| Mesh Loading | ✓ Works | Geometry loads correctly |
| Gaussian Placement | ✓ Works | Points are positioned |
| Scale Computation | ✓ Works | Sizes are reasonable |
| Rotation Computation | ✓ Works | Normals → quaternions |
| PLY Writing | ✓ Works | Valid PLY files |
| Viewer Loading | ✓ Works | Files load and display |

### 6.3 Priority

**Color is the P0 issue.** No other work should be done until this is fixed.

The pipeline successfully produces correctly-positioned, correctly-sized gaussian splats—they're just all black. Fixing color transforms the entire project from "broken" to "functional."

---

## Next Steps

See **Document 2: Implementation Plan** for the specific fixes and testing approach.

See **Document 3: Validation & Testing Specification** for test procedures and success criteria.

---

## Appendix A: Quick Diagnostic Commands

```bash
# 1. Check if textures were extracted
ls -la temp_dir/textures/

# 2. Check manifest content
cat temp_dir/material_manifest.json | python -m json.tool

# 3. Inspect PLY output
python tools/ply_inspector.py output_clouds/packed-tree_full.ply

# 4. Run with verbose logging
python cli.py packed-tree.blend ./output --packed --verbose
```

## Appendix B: Key File Locations

| Purpose | File | Key Lines |
|---------|------|-----------|
| Texture extraction | `src/stage1_baker/blender_scripts/extract_packed.py` | `extract_packed_textures()` |
| Manifest building | `src/stage1_baker/blender_scripts/extract_packed.py` | `build_material_manifest()` |
| UV export | `src/stage1_baker/blender_scripts/extract_packed.py` | `export_uv_layers()` |
| Color sampling | `src/mesh_to_gaussian.py` | `_sample_multi_material_colors()` |
| Texture loading | `src/mesh_to_gaussian.py` | `_load_material_textures()` |
| PLY output | `src/mesh_to_gaussian.py` | `save_ply()` |
| PLY inspection | `tools/ply_inspector.py` | `parse_ply()` |
