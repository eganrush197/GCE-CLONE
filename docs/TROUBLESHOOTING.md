# Unified Gaussian Pipeline - Troubleshooting Guide

**Last Updated:** December 19, 2025

This guide covers common issues, diagnostic procedures, and troubleshooting steps for the Gaussian Splat conversion pipeline.

---

## Quick Diagnostic Commands

```bash
# Inspect PLY file colors and statistics
python tools/ply_inspector.py output/model_full.ply 20

# View PLY header structure
Get-Content output/model_full.ply -Head 30       # Windows PowerShell
head -30 output/model_full.ply                   # Linux/Mac

# Run conversion with temp files kept for debugging
python -m src.pipeline.orchestrator model.blend --output output --packed --keep-temp

# Start the viewer
cd viewer
python server.py
```

---

## Common Issues

### 1. All Black Output

**Symptom:** The converted model appears completely black in the viewer.

**Possible Causes & Solutions:**

| Cause | Diagnostic | Fix |
|-------|------------|-----|
| Textures not extracted | Check `temp_*/textures/` directory | Verify Blender script ran correctly |
| Textures are black | Open PNG files manually | Check Blender material setup |
| UV mismatch | Check UV layer names | Use `--uv-layer` flag to specify correct layer |
| All samples culled | Check log for "Culled X%" | Increase opacity threshold |

**Debug Steps:**
1. Run with `--keep-temp` flag and inspect the temp folder
2. Check if texture files exist and contain valid image data
3. Use `ply_inspector.py` to verify SH DC values aren't all (-0.5, -0.5, -0.5)

---

### 2. All Gray Output

**Symptom:** The model appears uniformly gray.

**Diagnosis:** Fallback to default color (0.5, 0.5, 0.5) which converts to SH DC (0.0, 0.0, 0.0).

**Possible Causes:**
- Texture not loaded by the mesh loader
- Material groups are empty
- `_has_texture_visual()` returns False

**Debug Steps:**
1. Check if MTL file exists and references texture with `map_Kd`
2. Verify texture file path is correct (relative vs absolute)
3. Use `--keep-temp` and check intermediate files

---

### 3. Wrong Colors (Inverted/Swapped)

**Symptom:** Colors are present but incorrect (e.g., leaves are brown, bark is green).

**Status:** This was fixed in December 2025 with an R/G channel swap at lines 413-416 of `mesh_to_gaussian.py`.

**If issue persists:**
1. Verify you have the latest code with the channel swap fix
2. Check if the issue is in the viewer vs the PLY file
3. Use `ply_inspector.py` to verify the PLY contains expected color values

---

### 4. Partial Colors

**Symptom:** Some parts of the model are colored, some are black/gray.

**Possible Causes:**
- Some materials missing textures
- Face-material mapping incomplete
- UV coverage issues on some faces

**Debug Steps:**
1. Check `material_manifest.json` in temp folder
2. Verify all materials have texture assignments
3. Check UV coverage in Blender (some faces may have no UVs)

---

### 5. "Blender not found" Error

**Solution:**
```bash
# Specify full path to Blender executable
python cli.py model.blend ./output --blender "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"
```

Or add Blender to your system PATH.

---

### 6. Out of Memory Error

**Solution:**
```bash
# Use CPU instead of GPU
python cli.py model.obj ./output --device cpu

# Or reduce LOD levels
python cli.py model.obj ./output --lod 1000 5000 10000
```

---

## PLY Inspector Reference

The `ply_inspector.py` tool provides detailed color statistics:

```bash
python tools/ply_inspector.py output/model_full.ply 20
```

**Key Metrics:**

| Metric | Healthy Value | Issue Indicator |
|--------|---------------|-----------------|
| SH DC Mean | NOT (0,0,0) | All gray = texture not loaded |
| SH DC Variance | > 0.1 | Low variance = single color |
| Black Vertex % | < 30% | High % = color system issue |
| Opacity Mean | > 0.5 | Low opacity = visibility issues |

---

## SH DC Color Reference

| Color | RGB | SH DC |
|-------|-----|-------|
| Black | (0, 0, 0) | (-0.5, -0.5, -0.5) |
| White | (1, 1, 1) | (0.5, 0.5, 0.5) |
| Gray | (0.5, 0.5, 0.5) | (0, 0, 0) |
| Red | (1, 0, 0) | (0.5, -0.5, -0.5) |
| Green | (0, 1, 0) | (-0.5, 0.5, -0.5) |
| Blue | (0, 0, 1) | (-0.5, -0.5, 0.5) |

**Conversion formulas:**
- Encoding: `sh_dc = rgb - 0.5`
- Decoding: `rgb = sh_dc + 0.5`

---

## Viewer Issues

### Viewer Shows Black but PLY is Correct

**Diagnosis:** The PLY file has correct colors but the viewer doesn't display them.

**Debug Steps:**
1. Check browser console for JavaScript errors
2. Verify the shader is using `vColor = instanceSHDC + 0.5`
3. Check that binary transfer is sending correct data

---

## Getting Help

For issues not covered here:
1. Check `project context/Color-System-Technical-Documentation.md` for detailed color system info
2. Check `project context/Gaussian_Splat_Viewer_Technical_Specification.md` for architecture details
3. Open an issue on the repository

---

**End of Troubleshooting Guide**

