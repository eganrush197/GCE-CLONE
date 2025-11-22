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
from src.ply_io import save_ply

# Create converter
converter = MeshToGaussianConverter()

# Convert mesh
gaussians = converter.convert('input.obj')

# Save result
save_ply(gaussians, 'output.ply')

print(f"✅ Created {gaussians.count} gaussians")
```

## Common Workflows

### Low-Poly Game Asset

```bash
# Use vertex strategy for clean low-poly meshes
python mesh2gaussian lowpoly_character.obj character.ply --strategy vertex
```

### High-Poly Scanned Model

```bash
# Use face strategy with LODs for detailed meshes
python mesh2gaussian scanned_statue.glb statue.ply \
  --strategy face \
  --lod 10000,50000,200000
```

### Textured Architectural Model

```bash
# Hybrid strategy works best for textured models
python mesh2gaussian building.glb building.ply \
  --strategy hybrid \
  --samples-per-face 15
```

## Testing Your Installation

```bash
# Run the test suite
pytest tests/ -v

# Should see all tests passing
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

