# Unified Gaussian Pipeline - User Guide

**Version:** 1.0  
**Last Updated:** December 1, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Command-Line Interface](#command-line-interface)
5. [Supported File Formats](#supported-file-formats)
6. [Configuration Options](#configuration-options)
7. [Understanding LOD Levels](#understanding-lod-levels)
8. [Workflow Examples](#workflow-examples)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Overview

The Unified Gaussian Pipeline converts 3D mesh files into Gaussian Splat representations (PLY format) for ultra-fast web rendering. It supports:

- **Blender files** (.blend) with procedural shaders → automatic texture baking
- **Standard mesh formats** (.obj, .glb, .fbx) with textures → direct conversion
- **Multiple LOD levels** for adaptive quality rendering
- **Four sampling strategies** for optimal gaussian placement
- **Texture-aware color sampling** for accurate appearance

### What Are Gaussian Splats?

Gaussian Splats are a 3D representation using gaussian distributions instead of traditional polygons. They enable:
- ✅ Ultra-fast rendering (60+ FPS on web browsers)
- ✅ Photorealistic quality
- ✅ Small file sizes with LOD support
- ✅ No complex shader compilation

---

## Installation

### Prerequisites

1. **Python 3.8+** (tested with Python 3.13)
2. **Blender 3.0+** (only required for .blend files)
3. **CUDA** (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd gaussian-mesh-converter

# Install Python dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test the CLI
python cli.py --help

# Test with a simple conversion (if you have an OBJ file)
python cli.py path/to/model.obj ./output
```

---

## Quick Start

### Convert an OBJ File

```bash
python cli.py model.obj ./output
```

This creates:
- `output/model_full.ply` - Full resolution gaussian splat
- `output/model_lod100000.ply` - 100,000 gaussians
- `output/model_lod25000.ply` - 25,000 gaussians
- `output/model_lod5000.ply` - 5,000 gaussians

### Convert a Blender File (Procedural Shaders)

```bash
python cli.py tree.blend ./output --blender "C:\Program Files\Blender Foundation\Blender 3.1\blender.exe"
```

This will:
1. Bake procedural shaders to textures (4096x4096 default)
2. Export as OBJ with texture
3. Convert to gaussian splats
4. Generate LOD levels

### Convert a Blender File (Packed Textures)

```bash
python cli.py packed-tree.blend ./output --packed --uv-layer uv0
```

This will:
1. Extract embedded textures from .blend file
2. Export mesh with materials
3. Convert to gaussian splats with texture sampling
4. Generate LOD levels

**Note:** This is faster than baking and preserves original texture quality. See [PACKED_TEXTURE_PIPELINE.md](PACKED_TEXTURE_PIPELINE.md) for details.

---

## Command-Line Interface

### Basic Syntax

```bash
python cli.py <input_file> <output_dir> [options]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `input_file` | Path to input file (.blend, .obj, .glb, .fbx) | `model.obj` |
| `output_dir` | Directory for output PLY files | `./output` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lod` | `5000 25000 100000` | LOD levels (gaussian counts) |
| `--strategy` | `hybrid` | Sampling strategy: `vertex`, `face`, `hybrid`, `adaptive` |
| `--lod-strategy` | `importance` | LOD strategy: `importance`, `opacity`, `spatial` |
| `--texture-resolution` | `4096` | Texture baking resolution (512-8192) |
| `--blender` | `blender` | Path to Blender executable |
| `--keep-temp` | `false` | Keep temporary files for debugging |
| `--device` | `cpu` | Computation device: `cpu` or `cuda` |
| `--packed` | `false` | Use packed texture extraction mode |
| `--uv-layer` | `uv0` | UV layer name for packed mode |
| `--vertex-color-blend` | `multiply` | Vertex color blending: `multiply`, `add`, `overlay`, `replace`, `none` |
| `--texture-filter` | `bilinear` | Texture filtering: `nearest`, `bilinear` |
| `--no-mipmaps` | `false` | Disable mipmap generation |

### Examples

```bash
# Custom LOD levels
python cli.py model.obj ./output --lod 1000 10000 50000

# High-quality texture baking
python cli.py tree.blend ./output --texture-resolution 8192

# GPU acceleration
python cli.py large_model.glb ./output --device cuda

# Keep temporary files for debugging
python cli.py model.blend ./output --keep-temp

# Adaptive sampling strategy
python cli.py character.obj ./output --strategy adaptive

# Packed texture mode with vertex colors
python cli.py model.blend ./output --packed --vertex-color-blend multiply

# High-quality texture filtering
python cli.py model.blend ./output --packed --texture-filter bilinear
```

---

## Supported File Formats

### .blend (Blender Files)

**Processing:** Stage 1 (Baking) → Stage 2 (Conversion)

**Requirements:**
- Blender 3.0+ installed
- Specify Blender path with `--blender` if not in PATH

**What Happens:**
1. Blender opens in headless mode
2. Procedural shaders are baked to a texture
3. Original UVs are preserved (overlapping UVs supported)
4. Exports as OBJ + MTL + texture PNG
5. Converts to gaussian splats

**Best For:**
- Models with procedural materials (noise, gradients, etc.)
- Complex shader networks
- Non-texture-mapped models

### .obj (Wavefront OBJ)

**Processing:** Stage 2 (Conversion) only

**Requirements:**
- MTL file in same directory (for materials)
- Texture images referenced in MTL

**What Happens:**
1. Loads mesh geometry
2. Reads MTL file for materials
3. Loads textures (PNG, JPG, etc.)
4. Samples colors from textures using UVs
5. Converts to gaussian splats

**Best For:**
- Standard textured models
- Models exported from 3D software
- Simple workflows without Blender

### .glb (GL Transmission Format Binary)

**Processing:** Stage 2 (Conversion) only

**Requirements:**
- Embedded or external textures

**What Happens:**
- Same as OBJ, but uses GLB's embedded materials

**Best For:**
- Web-ready 3D models
- Models from Sketchfab, etc.
- Self-contained files

### .fbx (Autodesk FBX)

**Status:** ⏳ Planned for Phase 4

---

## Configuration Options

### Sampling Strategies

Controls how gaussians are placed on the mesh:

| Strategy | Description | Best For | Speed |
|----------|-------------|----------|-------|
| `vertex` | One gaussian per vertex | Low-poly models | ⚡ Fastest |
| `face` | Gaussians distributed across faces | Smooth surfaces | 🐢 Slower |
| `hybrid` | Combines vertex + face sampling | General use | ⚖️ Balanced |
| `adaptive` | Density based on curvature | Complex geometry | 🐢 Slowest |

**Recommendation:** Use `hybrid` (default) for most cases.

### LOD Strategies

Controls how gaussians are reduced for LOD levels:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `importance` | Keeps most important gaussians | General use (default) |
| `opacity` | Prioritizes visible gaussians | Transparent objects |
| `spatial` | Uniform spatial distribution | Even coverage |

**Recommendation:** Use `importance` (default) for most cases.

### Texture Resolution

For .blend files, controls the baking resolution:

| Resolution | Quality | File Size | Bake Time |
|------------|---------|-----------|-----------|
| 512 | Low | Small | Fast |
| 1024 | Medium | Medium | Medium |
| 2048 | High | Large | Slow |
| 4096 | Very High | Very Large | Very Slow |
| 8192 | Ultra | Huge | Extremely Slow |

**Recommendation:**
- Use `2048` for most models
- Use `4096` (default) for high-quality assets
- Use `512` for quick tests

---

## Understanding LOD Levels

LOD (Level of Detail) allows adaptive quality based on viewing distance.

### Default LOD Levels

```bash
--lod 5000 25000 100000
```

This creates 4 files:
1. **Full resolution** - All gaussians (e.g., 500,000)
2. **LOD 100000** - Reduced to 100,000 gaussians
3. **LOD 25000** - Reduced to 25,000 gaussians
4. **LOD 5000** - Reduced to 5,000 gaussians

### Choosing LOD Levels

| Use Case | Recommended LODs | Rationale |
|----------|------------------|-----------|
| **Mobile web** | `1000 5000 10000` | Small files, fast loading |
| **Desktop web** | `5000 25000 100000` | Balanced quality/performance |
| **High-end rendering** | `50000 100000 250000` | Maximum quality |
| **Quick preview** | `100 500` | Fastest processing |

### Example: Mobile-Optimized

```bash
python cli.py model.obj ./output --lod 1000 5000 10000
```

### Example: High-Quality

```bash
python cli.py model.obj ./output --lod 50000 100000 250000
```

---

## Workflow Examples

### Example 1: Simple OBJ Conversion

**Scenario:** You have a textured OBJ file from Blender export.

```bash
# Basic conversion
python cli.py character.obj ./output

# Output:
# - character_full.ply (all gaussians)
# - character_lod100000.ply
# - character_lod25000.ply
# - character_lod5000.ply
```

**Processing time:** ~10 seconds for typical model

### Example 2: Blender Procedural Model

**Scenario:** You have a .blend file with procedural materials (noise textures, gradients, etc.)

```bash
# Convert with high-quality baking
python cli.py tree.blend ./output \
  --blender "C:\Program Files\Blender Foundation\Blender 3.1\blender.exe" \
  --texture-resolution 4096 \
  --strategy hybrid

# Output:
# - tree_full.ply
# - tree_lod100000.ply
# - tree_lod25000.ply
# - tree_lod5000.ply
# - temp/ (if --keep-temp is used)
```

**Processing time:** ~30-60 seconds (depends on baking complexity)

### Example 3: Batch Processing

**Scenario:** Convert multiple models with consistent settings.

```bash
# Create a batch script (batch_convert.sh)
#!/bin/bash

for file in models/*.obj; do
  basename=$(basename "$file" .obj)
  python cli.py "$file" "output/$basename" --lod 5000 25000 100000
done
```

### Example 4: Mobile Game Assets

**Scenario:** Create lightweight assets for mobile games.

```bash
# Low LOD levels, fast processing
python cli.py prop.obj ./mobile_assets \
  --lod 500 2000 5000 \
  --strategy vertex \
  --device cpu

# Output optimized for mobile:
# - prop_full.ply (~10,000 gaussians)
# - prop_lod5000.ply
# - prop_lod2000.ply
# - prop_lod500.ply (ultra-low for distant objects)
```

### Example 5: High-Quality Archival

**Scenario:** Maximum quality for archival or high-end rendering.

```bash
# High resolution, many gaussians
python cli.py statue.obj ./archive \
  --lod 100000 250000 500000 \
  --strategy adaptive \
  --device cuda

# Output:
# - statue_full.ply (potentially millions of gaussians)
# - statue_lod500000.ply
# - statue_lod250000.ply
# - statue_lod100000.ply
```

**Processing time:** Several minutes (GPU recommended)

### Example 6: Debugging Workflow

**Scenario:** Something isn't working, need to inspect intermediate files.

```bash
# Keep temporary files for inspection
python cli.py problematic.blend ./debug_output \
  --keep-temp \
  --texture-resolution 1024

# Inspect temp files:
# - temp/baked_model.obj (baked mesh)
# - temp/baked_texture.png (baked texture)
# - temp/baked_model.mtl (material file)
```

---

## Troubleshooting

### Issue: "Blender not found"

**Error:**
```
❌ File not found: Blender not found: blender
Please install Blender or specify correct path.
```

**Solution:**
```bash
# Specify full path to Blender
python cli.py model.blend ./output --blender "C:\Program Files\Blender Foundation\Blender 3.1\blender.exe"

# Or add Blender to PATH (Windows)
setx PATH "%PATH%;C:\Program Files\Blender Foundation\Blender 3.1"
```

### Issue: "Input file not found"

**Error:**
```
❌ File not found: [Errno 2] No such file or directory: 'model.obj'
```

**Solution:**
- Check file path is correct
- Use absolute paths if relative paths don't work
- Verify file extension matches actual file

### Issue: "No texture found"

**Symptom:** Gaussians are all gray/white

**Solution:**
1. Check MTL file exists in same directory as OBJ
2. Verify MTL references texture with `map_Kd` line
3. Ensure texture file exists at referenced path
4. Use `--keep-temp` to inspect intermediate files

### Issue: "Out of memory"

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use CPU instead of GPU
python cli.py model.obj ./output --device cpu

# Or reduce LOD levels
python cli.py model.obj ./output --lod 1000 5000 10000
```

### Issue: "Baking takes too long"

**Solution:**
```bash
# Reduce texture resolution
python cli.py model.blend ./output --texture-resolution 1024

# Or use smaller test first
python cli.py model.blend ./output --texture-resolution 512 --lod 100 500
```

### Issue: "Colors look wrong"

**Possible causes:**
1. **Missing textures** - Check MTL file references
2. **Wrong UV mapping** - Verify UVs in original model
3. **Gamma correction** - Textures may need sRGB conversion

**Debug:**
```bash
# Keep temp files and inspect baked texture
python cli.py model.blend ./output --keep-temp

# Check temp/baked_texture.png visually
```

---

## Advanced Usage

### Python API

You can use the pipeline programmatically:

```python
from pathlib import Path
from pipeline import Pipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    input_file=Path("model.obj"),
    output_dir=Path("./output"),
    lod_levels=[5000, 25000, 100000],
    strategy='hybrid',
    lod_strategy='importance',
    device='cpu'
)

# Run pipeline
pipeline = Pipeline(config)
output_files = pipeline.run()

print(f"Generated {len(output_files)} files:")
for f in output_files:
    print(f"  - {f}")
```

### Custom Processing

For advanced users who want to customize the pipeline:

```python
from mesh_to_gaussian import MeshToGaussianConverter
from lod_generator import LODGenerator

# Stage 2: Manual conversion
converter = MeshToGaussianConverter(device='cpu')
converter.load_mesh("model.obj")
gaussians = converter.mesh_to_gaussians(strategy='hybrid')

# Custom LOD generation
lod_gen = LODGenerator(strategy='importance')
lod_5k = lod_gen.generate_lod(gaussians, target_count=5000)

# Save
converter.save_ply(gaussians, "output/full.ply")
converter.save_ply(lod_5k, "output/lod5k.ply")
```

### Integration with Web Viewers

The generated PLY files can be used with gaussian splat web viewers:

```html
<!-- Example with three.js gaussian splat loader -->
<script type="module">
  import { GaussianSplatLoader } from 'gaussian-splat-loader';

  const loader = new GaussianSplatLoader();

  // Load appropriate LOD based on distance
  const distance = camera.position.distanceTo(object.position);

  let plyFile;
  if (distance < 10) {
    plyFile = 'model_full.ply';
  } else if (distance < 50) {
    plyFile = 'model_lod100000.ply';
  } else if (distance < 100) {
    plyFile = 'model_lod25000.ply';
  } else {
    plyFile = 'model_lod5000.ply';
  }

  loader.load(plyFile, (splat) => {
    scene.add(splat);
  });
</script>
```

---

## Performance Tips

1. **Use GPU for large models** - `--device cuda` can be 10-100x faster
2. **Start with low resolution** - Test with `--texture-resolution 512` first
3. **Use vertex strategy for speed** - `--strategy vertex` is fastest
4. **Reduce LOD levels for testing** - `--lod 100 500` for quick iterations
5. **Keep temp files for debugging** - `--keep-temp` helps diagnose issues

---

## File Size Reference

Approximate output file sizes:

| Gaussian Count | File Size | Use Case |
|----------------|-----------|----------|
| 1,000 | ~50 KB | Ultra-low LOD |
| 5,000 | ~250 KB | Mobile distant view |
| 25,000 | ~1.2 MB | Desktop distant view |
| 100,000 | ~5 MB | Desktop close view |
| 500,000 | ~25 MB | High-quality close view |
| 1,000,000+ | ~50+ MB | Archival/maximum quality |

---

## Support

For issues, questions, or contributions:
- **GitHub Issues:** [repository-url]/issues
- **Documentation:** See `project context/` directory
- **Examples:** See `examples/` directory

---

## License

[Your license here]

---

**Happy Gaussian Splatting! 🎉**


