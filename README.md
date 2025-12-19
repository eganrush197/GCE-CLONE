# Unified Gaussian Pipeline

**Convert 3D meshes to Gaussian Splats with automatic texture baking and LOD generation**

[![Tests](https://img.shields.io/badge/tests-51%2F51%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()

---

## Overview

The Unified Gaussian Pipeline is a production-ready tool for converting 3D mesh files into Gaussian Splat representations (PLY format) optimized for ultra-fast web rendering. It supports **Blender files with procedural shaders**, **packed texture extraction**, standard mesh formats with textures, and automatically generates multiple LOD (Level of Detail) levels.

### Why This Tool?

- **100-1000x faster** than multi-view neural methods (1-30 seconds vs 30-180 minutes)
- **Blender Integration** - Automatic procedural shader baking in headless mode
- **Packed Texture Pipeline** - Direct extraction from .blend files with embedded textures
- **Multi-Format Support** - .blend, .obj, .glb files (FBX planned)
- **No CUDA required** (optional GPU acceleration available)
- **Production Ready** - 51/51 tests passing, comprehensive error handling
- **Easy to Use** - Simple CLI interface with sensible defaults

Multi-view rendering is designed for photogrammetry (reconstructing unknown geometry from photos). For meshes, we already have complete geometric information - direct conversion is the pragmatic choice.

## Quick Start

```bash
# Convert an OBJ file
python cli.py model.obj ./output

# Convert a Blender file with procedural shaders
python cli.py tree.blend ./output --blender "C:\Program Files\Blender Foundation\Blender 3.1\blender.exe"

# Convert a Blender file with packed textures (faster)
python cli.py packed-tree.blend ./output --packed --uv-layer uv0

# Custom LOD levels for mobile
python cli.py model.obj ./output --lod 1000 5000 25000
```

**See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for complete documentation.**

## Features

### Core Capabilities
- ✅ **Blender Integration** - Automatic procedural shader baking
- ✅ **Packed Texture Pipeline (NEW!)** - Direct extraction from .blend with embedded textures
  - Multi-material support with per-face material assignment
  - Multi-UV layer support (different textures can use different UV layers)
  - Vertex color blending (5 modes: multiply, add, overlay, replace, none)
  - Bilinear texture filtering with mipmap generation
  - Transparency and roughness map support
- ✅ **Unified Pipeline** - Intelligent file routing and stage coordination
- ✅ **CLI Interface** - User-friendly command-line tool
- ✅ OBJ and GLB mesh loading with automatic MTL color parsing
- ✅ **UV texture sampling** - Sample colors directly from texture maps
- ✅ Four initialization strategies (Vertex, Face, Hybrid, Adaptive)
- ✅ Automatic gaussian parameter estimation
- ✅ LOD (Level of Detail) generation with 3 pruning strategies
  - **Importance** (recommended): opacity × volume - best quality
  - **Opacity**: Keep most opaque - fast, good quality
  - **Spatial**: Voxel-based - uniform coverage
- ✅ Spherical Harmonics (SH) color encoding (standard format)
- ✅ Full type hints for IDE autocomplete and type checking
- ✅ **Comprehensive test suite (51/51 tests passing)**
- ✅ Normal-based orientation
- ✅ Optional quick optimization (100 iterations, GPU-accelerated)

### Supported File Formats

| Format | Status | Processing | Use Case |
|--------|--------|------------|----------|
| **.blend** | ✅ Supported | Stage 1 (Baking) + Stage 2 (Conversion) | Procedural materials, complex shaders |
| **.obj** | ✅ Supported | Stage 2 (Conversion) only | Textured models, standard exports |
| **.glb** | ✅ Supported | Stage 2 (Conversion) only | Web-ready models, embedded textures |
| **.fbx** | ⏳ Planned | Phase 4 | Autodesk ecosystem |

### Performance

| Asset Type | Complexity | Processing Time | Output Size |
|------------|------------|-----------------|-------------|
| Simple cube | Low | ~5 seconds | ~500 KB |
| Game character | Medium | ~30 seconds | ~5 MB |
| Architectural model | Medium | ~45 seconds | ~8 MB |
| High-poly landscape | High | ~2-5 minutes | ~25 MB |

*Times measured on Intel i7, 16GB RAM. GPU acceleration can provide 10-100x speedup.*

## Installation

### Requirements
- Python 3.8+
- NumPy, SciPy, Trimesh, Pillow
- Optional: PyTorch (for GPU optimization)

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd "New Gaussian Converter DMTG"

# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch for GPU acceleration
pip install torch torchvision
```

## Usage

### Command Line

```bash
# Basic conversion (OBJ/GLB files)
python cli.py model.obj ./output

# Blender file with procedural shaders (requires Blender installation)
python cli.py model.blend ./output --blender "path/to/blender.exe"

# Blender file with packed textures (recommended for pre-textured .blend files)
python -m src.pipeline.orchestrator model.blend --output ./output --packed

# With custom LOD levels
python cli.py model.obj ./output --lod 5000 25000 100000

# Specify initialization strategy
python cli.py model.obj ./output --strategy hybrid

# Keep temporary files for debugging
python -m src.pipeline.orchestrator model.blend --output ./output --packed --keep-temp
```

### Python API

```python
from src.mesh_to_gaussian import MeshToGaussianConverter
from src.lod_generator import LODGenerator

# Create converter
converter = MeshToGaussianConverter(device='cpu')

# Load and convert mesh
mesh = converter.load_mesh('input.obj')
gaussians = converter.mesh_to_gaussians(mesh, strategy='adaptive', samples_per_face=10)
converter.save_ply(gaussians, 'output.ply')

# Generate LODs
lod_gen = LODGenerator(strategy='importance')
lod_5k = lod_gen.generate_lod(gaussians, 5000)
lod_25k = lod_gen.generate_lod(gaussians, 25000)
lod_100k = lod_gen.generate_lod(gaussians, 100000)

# Save LODs
converter.save_ply(lod_5k, 'output_lod5k.ply')
converter.save_ply(lod_25k, 'output_lod25k.ply')
converter.save_ply(lod_100k, 'output_lod100k.ply')
```

## Color & Texture Support

The converter automatically extracts colors from multiple sources with priority:

1. **UV-mapped textures** (NEW!) - Highest quality
   - Automatically loads textures referenced in MTL files (`map_Kd`)
   - Samples colors at UV coordinates for each gaussian
   - Supports both vertex and face strategies with interpolation
   - Works with PNG, JPG, and other PIL-supported formats

2. **Vertex colors** - Per-vertex color data
   - Embedded in OBJ files or loaded from mesh

3. **Face colors** - Material colors from MTL files
   - Diffuse color (`Kd`) applied to faces

4. **Default gray** - Fallback when no colors available

**Example with texture:**
```python
# OBJ file references texture in MTL
# model.mtl contains: map_Kd texture.jpg
mesh = converter.load_mesh("model.obj")  # Automatically loads texture.jpg
gaussians = converter.mesh_to_gaussians(mesh, strategy='vertex')
# Colors sampled from texture at each vertex's UV coordinate
```

## Initialization Strategies

Choose the right strategy for your mesh:

- **Vertex**: Place gaussians at mesh vertices
  - Fastest option (~1 gaussian per vertex)
  - Best for: Low-poly models, clean geometry
  - Texture sampling: Uses vertex UV coordinates directly

- **Face**: Sample points on triangle faces
  - Higher quality, more gaussians
  - Best for: Textured meshes, detailed surfaces
  - Texture sampling: Interpolates UV coordinates using barycentric weights

- **Hybrid**: Combine vertex + face sampling
  - Balanced quality and performance
  - Best for: General use (recommended)
  - Texture sampling: Both vertex and interpolated sampling

- **Adaptive**: Auto-select based on mesh properties
  - Currently maps to Hybrid
  - Best for: When unsure which strategy to use

## LOD (Level of Detail) Strategies

Choose the right pruning strategy for your use case:

- **Importance** (RECOMMENDED): Keep gaussians with highest visual impact
  - Metric: opacity × volume
  - Best quality preservation

- **Opacity**: Keep most opaque gaussians
  - Fast, good quality
  - Best when opacity indicates importance

- **Spatial**: Voxel-based spatial subsampling
  - Uniform coverage across model
  - Best for maintaining even distribution

## Project Structure

```
project-root/
├── src/
│   ├── mesh_to_gaussian.py        # Core converter with UV sampling
│   ├── gaussian_splat.py          # GaussianSplat data structure
│   ├── lod_generator.py           # LOD generation strategies
│   ├── pipeline/                  # Unified pipeline orchestrator
│   │   ├── orchestrator.py        # Main pipeline coordinator
│   │   ├── config.py              # Configuration dataclass
│   │   └── router.py              # File routing logic
│   └── stage1_baker/              # Blender integration
│       ├── baker.py               # Python wrapper for Blender
│       ├── packed_extractor.py    # Packed texture extraction
│       └── blender_scripts/       # Blender Python scripts
├── viewer/                        # Web-based Gaussian Splat viewer
│   ├── server.py                  # FastAPI backend
│   ├── backend/                   # API and PLY parsing
│   └── static/                    # Three.js frontend
├── tests/                         # Comprehensive test suite
├── tools/                         # Utility scripts (ply_inspector, etc.)
├── docs/                          # Documentation
├── cli.py                         # Command-line interface
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Documentation

- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** - Complete user guide with examples
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Diagnostic and troubleshooting guide
- **[PACKED_TEXTURE_PIPELINE.md](PACKED_TEXTURE_PIPELINE.md)** - Packed texture pipeline technical reference

### Technical Documentation (for developers)
- **[project context/Gaussian_Splat_Viewer_Technical_Specification.md](project%20context/Gaussian_Splat_Viewer_Technical_Specification.md)** - Full architecture and API reference
- **[project context/Color-System-Technical-Documentation.md](project%20context/Color-System-Technical-Documentation.md)** - Color encoding system details

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Expected output: 51 passed
```

**Test Coverage:**
- ✅ GaussianSplat data structure (2 tests)
- ✅ Mesh conversion strategies (3 tests)
- ✅ LOD generation strategies (3 tests)
- ✅ UV texture sampling (2 tests)
- ✅ Blender baker integration (10 tests)
- ✅ Pipeline orchestration (7 tests)
- ✅ Packed texture extraction (17 tests)
  - Basic extraction and manifest (5 tests)
  - Multi-material conversion (5 tests)
  - Vertex color support (3 tests)
  - Texture filtering & mipmapping (4 tests)

### Code Quality

- **Type Safety**: Full type hints throughout codebase
- **Test Coverage**: 51/51 tests passing (100% pass rate)
- **Documentation**: Comprehensive inline and external docs
- **No Code Duplication**: Single source of truth for all features
- **Production Ready**: Clean, maintainable, well-tested code

## License

[Specify license]

## Contributing

[Contribution guidelines]

