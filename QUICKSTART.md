# Quick Start Guide

Get up and running with the Gaussian Mesh Converter in 5 minutes.

## Installation

### 1. Install Dependencies

```bash
# Basic installation (CPU only)
pip install -r requirements.txt

# Or with GPU support (optional)
pip install -r requirements.txt
pip install torch torchvision
```

### 2. Verify Installation

```bash
# Test imports
python -c "import trimesh, numpy, scipy; print('✅ Dependencies OK')"
```

## Basic Usage

### Command Line (Simplest)

```bash
# Convert a mesh file
python mesh2gaussian input.obj output.ply

# That's it! You now have a gaussian splat PLY file
```

### With Options

```bash
# Generate multiple LOD levels
python mesh2gaussian input.glb output.ply --lod 5000,25000,100000

# Use specific strategy
python mesh2gaussian input.obj output.ply --strategy hybrid

# With GPU optimization (requires PyTorch)
python mesh2gaussian input.obj output.ply --optimize --device cuda
```

### Python API

```python
from src.mesh_to_gaussian import MeshToGaussianConverter
from src.lod_generator import LODGenerator

# Create converter
converter = MeshToGaussianConverter(device='cpu')  # or 'cuda' for GPU

# Load mesh (auto-loads MTL colors)
mesh = converter.load_mesh('input.obj')

# Convert to gaussians
gaussians = converter.mesh_to_gaussians(mesh, strategy='hybrid', samples_per_face=10)

# Save result
converter.save_ply(gaussians, 'output.ply')

print(f"✅ Created {len(gaussians)} gaussians")

# Optional: Generate LODs
lod_gen = LODGenerator(strategy='importance')
lod_5k = lod_gen.generate_lod(gaussians, 5000)
converter.save_ply(lod_5k, 'output_lod5k.ply')
```

## Common Workflows

### Low-Poly Game Asset

```bash
# Use vertex strategy for clean low-poly meshes
python mesh2gaussian lowpoly_character.obj character.ply --strategy vertex
```

**Why vertex strategy?** Low-poly models have well-defined vertices that represent
the shape perfectly. One gaussian per vertex is efficient and accurate.

### High-Poly Scanned Model

```bash
# Use face strategy with LODs for detailed meshes
python mesh2gaussian scanned_statue.glb statue.ply \
  --strategy face \
  --lod 10000,50000,200000
```

**Why face strategy?** High-poly scanned models have dense triangles. Sampling on
faces captures surface detail better than just vertices.

**LOD Strategy:** Uses 'importance' by default (opacity × volume) for best quality.

### Textured Architectural Model

```bash
# Hybrid strategy works best for textured models
python mesh2gaussian building.glb building.ply \
  --strategy hybrid \
  --samples-per-face 15
```

**Why hybrid strategy?** Combines vertex precision with face sampling to capture
both geometry and texture details.

## Understanding LOD Strategies

When generating LODs, you can choose different pruning strategies:

```bash
# Importance (RECOMMENDED) - Best quality
python mesh2gaussian model.obj output.ply --lod 5000,25000,100000
# Uses opacity × volume metric

# Opacity - Fast, good quality
# (Currently set via LODGenerator in Python API)

# Spatial - Uniform coverage
# (Currently set via LODGenerator in Python API)
```

**Python API for LOD strategies:**
```python
from src.lod_generator import LODGenerator

# Importance (recommended)
lod_gen = LODGenerator(strategy='importance')
lod_5k = lod_gen.generate_lod(gaussians, 5000)

# Opacity (fast)
lod_gen = LODGenerator(strategy='opacity')
lod_5k = lod_gen.generate_lod(gaussians, 5000)

# Spatial (uniform)
lod_gen = LODGenerator(strategy='spatial')
lod_5k = lod_gen.generate_lod(gaussians, 5000)
```

## Testing Your Installation

```bash
# Run the test suite
pytest tests/ -v

# Expected output: 8 passed in ~3-6 seconds
# ✅ TestGaussianSplat: 2 tests
# ✅ TestMeshToGaussianConverter: 3 tests
# ✅ TestLODGenerator: 3 tests
```

## Viewing Results

The output PLY files can be viewed in:
- **SuperSplat** (https://playcanvas.com/supersplat) - Web-based viewer
- **Blender** with gaussian splat plugins
- **Unity/Unreal** with gaussian splat renderers
- Any PLY viewer (will show as point cloud)

## Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the project directory
cd "New Gaussian Converter DMTG"

# Install dependencies
pip install -r requirements.txt
```

### "Mesh file not found"
```bash
# Use absolute paths or ensure file is in current directory
python mesh2gaussian /full/path/to/input.obj output.ply
```

### Slow conversion
```bash
# Reduce samples for faster conversion
python mesh2gaussian input.obj output.ply --samples-per-face 5

# Or use vertex strategy (fastest)
python mesh2gaussian input.obj output.ply --strategy vertex
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/basic_usage.py](examples/basic_usage.py) for more examples
- See [project context/](project context/) for technical details
- Experiment with different strategies and parameters

## Performance Reference

| Mesh Size | Strategy | Time (CPU) | Gaussians |
|-----------|----------|------------|-----------|
| 1K verts  | vertex   | ~1 sec     | ~1K       |
| 10K verts | hybrid   | ~5 sec     | ~15K      |
| 100K verts| face     | ~30 sec    | ~200K     |

**Happy converting! 🎉**

