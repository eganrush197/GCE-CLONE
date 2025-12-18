# Gaussian Splat Conversion Pipeline - Technical Specification
## Current State Documentation for Architecture Review

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Status:** Current Implementation State  
**Purpose:** Architecture review, bottleneck diagnosis, and future planning

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Components](#3-pipeline-components)
4. [API Reference](#4-api-reference)
5. [Data Structures](#5-data-structures)
6. [Color System Analysis](#6-color-system-analysis)
7. [Technical Debt & Known Issues](#7-technical-debt--known-issues)
8. [Integration Points](#8-integration-points)
9. [Recommendations](#9-recommendations)

---

## 1. Executive Summary

### 1.1 Project Purpose

This project converts 3D mesh models (OBJ, FBX, BLEND) into Gaussian Splat point clouds (PLY format) for real-time rendering. The system consists of:

1. **Conversion Pipeline**: Transforms meshes to gaussian splats with texture/color sampling
2. **Web Viewer**: Three.js-based viewer for visualizing output PLY files

### 1.2 Current State Overview

| Component | Status | Notes |
|-----------|--------|-------|
| Baking Pipeline | Functional | Blender-based texture baking |
| Packed Pipeline | Functional | Packed texture extraction |
| Mesh-to-Gaussian Converter | Functional | Core conversion works |
| LOD Generator | Functional | Three strategies implemented |
| Web Viewer | Functional | Custom shader rendering |
| **Color System** | **BROKEN** | **Models output as black** |

### 1.3 Critical Issue: Color System Failure

**The #1 priority issue**: Models have **never successfully retained color data** through the conversion pipeline. All output models appear black in the viewer.

**Impact**: 
- Both pipelines (baking and packed) are affected
- The issue is in the conversion stage, not the viewer
- PLY files contain `sh_dc = [0.0, 0.0, 0.0]` (gray fallback converted to black)

### 1.4 Project History Context

The project has undergone significant pivots:
- Originally attempted neural network training with virtual camera arrays
- Started with INRIA gaussian splatting approach
- Experimented with PyTorch3D library
- Tried both headless and headed Blender execution
- Current approach: Direct mesh-to-gaussian conversion with texture sampling

---

## 2. System Architecture

### 2.1 Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Pipeline Orchestration | Python | 3.8+ | Main pipeline coordination |
| 3D Processing | Trimesh | >=3.23.0 | Mesh loading and manipulation |
| Texture Processing | Pillow | >=10.0.0 | Image loading and sampling |
| Numerical Computing | NumPy | >=1.24.0 | Vectorized operations |
| Spatial Algorithms | SciPy | >=1.11.0 | KD-tree for scale computation |
| Blender Integration | Blender | 3.x/4.x | Texture baking and extraction |
| Web Backend | FastAPI | >=0.104.1 | REST API and WebSocket |
| Web Frontend | Three.js | r160 | WebGL rendering |
| File Watching | watchdog | >=3.0.0 | Directory monitoring |

### 2.2 Directory Structure

```
project-root/
├── cli.py                          # Command-line interface
├── requirements.txt                # Python dependencies
├── src/
│   ├── pipeline/
│   │   ├── orchestrator.py         # Pipeline class - main coordinator
│   │   └── config.py               # PipelineConfig dataclass
│   ├── stage1_baker/
│   │   ├── baker.py                # BlenderBaker class
│   │   ├── packed_extractor.py     # PackedExtractor class
│   │   └── blender_scripts/
│   │       ├── bake_and_export.py  # Blender baking script
│   │       └── extract_packed.py   # Blender extraction script
│   ├── mesh_to_gaussian.py         # MeshToGaussianConverter (2085 lines)
│   ├── gaussian_splat.py           # GaussianSplat dataclass
│   └── lod_generator.py            # LODGenerator class
├── viewer/
│   ├── server.py                   # FastAPI entry point
│   ├── backend/
│   │   ├── api.py                  # REST endpoints (399 lines)
│   │   ├── ply_parser.py           # Binary PLY parser (293 lines)
│   │   └── file_watcher.py         # Directory watcher (116 lines)
│   └── static/
│       ├── index.html              # Main HTML page
│       ├── css/style.css           # Styling
│       └── js/main.js              # Three.js application (715 lines)
├── tools/
│   └── ply_inspector.py            # PLY file diagnostic tool
├── tests/
│   └── test_converter.py           # Unit tests
└── output_clouds/                  # Default output directory
    └── LOD_output/                 # LOD files subdirectory
```

### 2.3 High-Level Data Flow

```
INPUT FILES                    STAGE 1                      STAGE 2                    OUTPUT
─────────────────────────────────────────────────────────────────────────────────────────────────
                         ┌─────────────────┐
  .blend ───────────────►│  BlenderBaker   │
  .fbx                   │  (bake_and_     │
  .obj                   │   export.py)    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ OBJ + Baked     │
                         │ Texture         │─────┐
                         └─────────────────┘     │
                                                 │     ┌────────────────────┐
                                                 ├────►│ MeshToGaussian     │────► .ply
                                                 │     │ Converter          │
                         ┌─────────────────┐     │     │                    │
  .blend ───────────────►│ PackedExtractor │     │     │ - load_mesh()      │
  (packed textures)      │ (extract_       │─────┘     │ - mesh_to_gauss()  │
                         │  packed.py)     │           │ - save_ply()       │
                         └────────┬────────┘           └─────────┬──────────┘
                                  │                              │
                                  ▼                              ▼
                         ┌─────────────────┐           ┌─────────────────┐
                         │ OBJ + Manifest  │           │  LODGenerator   │────► _lod1.ply
                         │ + Textures      │           │                 │────► _lod2.ply
                         └─────────────────┘           └─────────────────┘
```

---

## 3. Pipeline Components

### 3.1 Entry Points

#### CLI (cli.py)

Primary command-line interface for the conversion pipeline.

```bash
# Basic usage
python cli.py input.blend --output output_clouds/

# Packed pipeline (for pre-packed textures)
python cli.py input.blend --packed --output output_clouds/

# With options
python cli.py input.blend \
    --output output_clouds/ \
    --samples-per-face 3 \
    --strategy hybrid \
    --lod-levels 3 \
    --uv-layer "UVMap" \
    --vertex-color-blend multiply
```

**Key CLI Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | `output_clouds/` | Output directory |
| `--packed` | False | Use packed texture extraction instead of baking |
| `--samples-per-face` | 3 | Gaussian samples per mesh face |
| `--strategy` | `hybrid` | Sampling strategy: vertex, face, hybrid |
| `--lod-levels` | 3 | Number of LOD levels to generate |
| `--uv-layer` | None | Specific UV layer to use |
| `--vertex-color-blend` | `multiply` | Blend mode: multiply, add, overlay, replace, none |

### 3.2 Pipeline Orchestrator

**File:** `src/pipeline/orchestrator.py`

The `Pipeline` class coordinates the full conversion workflow.

```python
class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.blender_baker = BlenderBaker(...)
        self.packed_extractor = PackedExtractor(...)
        self.converter = MeshToGaussianConverter(...)
        self.lod_generator = LODGenerator(...)

    def run(self, input_path: Path) -> Path:
        # Stage 1: Extract/Bake textures
        if self.config.use_packed:
            stage1_output = self._run_packed_extraction(input_path)
        else:
            stage1_output = self._run_stage1(input_path)

        # Stage 2: Convert to gaussians
        gaussians = self._run_stage2(stage1_output)

        # Stage 3: Generate LODs
        if self.config.lod_levels > 0:
            self._generate_lods(gaussians)

        return output_path
```

**Configuration (`PipelineConfig`):**

```python
@dataclass
class PipelineConfig:
    output_dir: Path
    blender_path: Optional[str] = None
    use_packed: bool = False
    samples_per_face: int = 3
    strategy: str = 'hybrid'
    lod_levels: int = 3
    lod_ratios: List[float] = field(default_factory=lambda: [0.5, 0.25, 0.1])
    uv_layer: Optional[str] = None
    vertex_color_blend: str = 'multiply'
```

### 3.3 Stage 1: Blender Integration

#### BlenderBaker (`src/stage1_baker/baker.py`)

Executes Blender in headless mode to bake textures onto geometry.

```python
class BlenderBaker:
    def __init__(self, blender_path: Optional[str] = None):
        self.blender_path = blender_path or self._find_blender()

    def bake(self, input_path: Path, output_dir: Path) -> Path:
        """
        Run Blender headless to bake textures.

        Returns:
            Path to exported OBJ file with baked texture
        """
        # Executes: blender --background input.blend --python bake_and_export.py
```

**Blender Script (`bake_and_export.py`):**

Key functions:
- `prepare_for_bake()` - Sets up UV maps and materials
- `bake_combined_texture()` - Bakes diffuse/emission to single texture
- `export_obj_with_mtl()` - Exports triangulated OBJ
- `dump_bake_diagnostic()` - Diagnostic function for debugging black bakes

#### PackedExtractor (`src/stage1_baker/packed_extractor.py`)

Extracts pre-packed textures from Blender files without baking.

```python
class PackedExtractor:
    def extract(self, input_path: Path, output_dir: Path) -> Path:
        """
        Extract packed textures and create material manifest.

        Returns:
            Path to output directory containing:
            - model.obj (triangulated mesh)
            - textures/ (extracted texture files)
            - material_manifest.json
            - uv_layers/*.npy (UV coordinates per layer)
            - vertex_colors.npy (if present)
        """
```

**Blender Script (`extract_packed.py`):**

Key functions:
- `extract_packed_textures()` - Unpacks embedded textures
- `analyze_material_textures()` - Maps shader nodes to texture files
- `export_uv_layers()` - Saves UV coordinates as numpy arrays
- `export_vertex_colors()` - Saves per-loop vertex colors
- `create_material_manifest()` - Creates JSON mapping materials to textures

**Material Manifest Structure:**

```json
{
  "obj_file": "model.obj",
  "materials": {
    "Material_Name": {
      "diffuse": {"path": "textures/diffuse.png", "uv_layer": "UVMap"},
      "transparency": {"path": "textures/alpha.png", "uv_layer": "UVMap"},
      "roughness": {"path": "textures/roughness.png", "uv_layer": "UVMap"},
      "normal": {"path": "textures/normal.png", "uv_layer": "UVMap"}
    }
  },
  "face_materials": [0, 0, 1, 1, 2, ...],
  "uv_layers": {
    "UVMap": "uv_layers/UVMap.npy",
    "UVMap.001": "uv_layers/UVMap.001.npy"
  },
  "vertex_colors": "vertex_colors.npy",
  "uv_layer": "UVMap"
}
```

### 3.4 Stage 2: Mesh to Gaussian Conversion

**File:** `src/mesh_to_gaussian.py` (2085 lines)

The core conversion engine that transforms meshes into gaussian splats.

#### MeshToGaussianConverter Class

```python
class MeshToGaussianConverter:
    def __init__(self, device: str = 'cpu',
                 use_mipmaps: bool = False,
                 max_texture_cache_size: int = 10):
        self.device = device
        self.use_mipmaps = use_mipmaps
        self._texture_cache = OrderedDict()  # LRU cache

    def load_mesh(self, path: str) -> trimesh.Trimesh:
        """Load mesh with MTL and texture support."""

    def mesh_to_gaussians(self, mesh: trimesh.Trimesh,
                          strategy: str = 'hybrid',
                          samples_per_face: int = 3,
                          manifest: Optional[dict] = None) -> List[_SingleGaussian]:
        """
        Convert mesh to list of gaussians.

        Strategies:
        - 'vertex': One gaussian per vertex
        - 'face': N samples per face (barycentric)
        - 'hybrid': Both vertex and face sampling
        """

    def save_ply(self, gaussians: List[_SingleGaussian], output_path: str):
        """Save gaussians to binary PLY file."""
```

#### Key Internal Methods

| Method | Purpose | Lines |
|--------|---------|-------|
| `_load_obj_with_mtl()` | OBJ loading with MTL color parsing | 150-280 |
| `_has_texture_visual()` | Validates texture availability | 291-326 |
| `_sample_vertex_colors_vectorized()` | Main color sampling (CRITICAL) | 588-634 |
| `_sample_texture_batch()` | Batch texture sampling with cache | 382-450 |
| `_sample_colors_vectorized()` | Face-based color sampling | 478-546 |
| `_compute_vertex_scales_fast()` | KD-tree based scale computation | 548-586 |
| `_mesh_to_gaussians_multi_material()` | Multi-material texture handling | 1253-1550 |
| `_sample_multi_material_colors()` | Per-material color sampling | 1150-1251 |
| `_normals_to_quaternions_vectorized()` | Convert normals to rotations | ~1550-1600 |

### 3.5 Stage 3: LOD Generation

**File:** `src/lod_generator.py`

```python
class LODGenerator:
    def generate_lod(self, gaussians: List[_SingleGaussian],
                     target_count: int,
                     strategy: str = 'importance') -> List[_SingleGaussian]:
        """
        Reduce gaussian count using specified strategy.

        Strategies:
        - 'importance': Sort by opacity * volume, keep top N
        - 'opacity': Sort by opacity only
        - 'spatial': Voxel-based decimation (preserves distribution)
        """
```

**LOD Strategies:**

| Strategy | Algorithm | Best For |
|----------|-----------|----------|
| `importance` | `score = opacity * (sx * sy * sz)` | General use |
| `opacity` | `score = opacity` | Dense point clouds |
| `spatial` | Voxel grid → keep one per cell | Large scenes |

---

## 4. API Reference

### 4.1 Viewer Backend API

**File:** `viewer/backend/api.py`

#### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/files` | GET | List all PLY files in output directory |
| `/api/load/{filename}` | GET | Load PLY file as JSON (legacy) |
| `/api/load-binary/{filename}` | GET | Load PLY file as binary stream |
| `/api/lod/{filename}` | POST | Generate LOD for file |
| `/api/watch` | WebSocket | Real-time file change notifications |

#### GET /api/files

Lists available PLY files with metadata.

**Response:**
```json
{
  "files": [
    {
      "name": "model.ply",
      "path": "output_clouds/model.ply",
      "size": 1234567,
      "modified": "2024-12-15T10:30:00",
      "vertex_count": 50000
    }
  ],
  "lod_files": [
    {
      "name": "model_lod1.ply",
      "parent": "model.ply",
      "reduction": 0.5
    }
  ]
}
```

#### GET /api/load-binary/{filename}

Streams gaussian data in custom binary format for efficiency.

**Binary Protocol:**
```
Header (8 bytes):
  - Magic number: 0x47535053 ('GSPS') - 4 bytes
  - Gaussian count: uint32 - 4 bytes

Per Gaussian (56 bytes):
  - position: float32[3] (x, y, z)
  - scale: float32[3] (sx, sy, sz)
  - rotation: float32[4] (qw, qx, qy, qz)
  - sh_dc: float32[3] (r-0.5, g-0.5, b-0.5)
  - opacity: float32

Total: 8 + (N * 56) bytes
```

#### WebSocket /api/watch

Real-time notifications when PLY files are added, modified, or deleted.

**Message Format:**
```json
{
  "type": "file_added|file_modified|file_deleted",
  "file": {
    "name": "model.ply",
    "path": "output_clouds/model.ply"
  }
}
```

### 4.2 PLY Parser

**File:** `viewer/backend/ply_parser.py`

```python
class PLYParser:
    def parse(self, file_path: str) -> Dict:
        """
        Parse binary PLY file.

        Supports two formats:
        - Legacy (20 properties): includes RGB uchar
        - Phase 1 (17 properties): SH DC only

        Returns:
            {
                'positions': np.ndarray[N, 3],
                'scales': np.ndarray[N, 3],
                'rotations': np.ndarray[N, 4],
                'colors': np.ndarray[N, 3],  # Converted from SH DC
                'opacities': np.ndarray[N]
            }
        """
```

**Color Conversion:**
```python
# SH DC to RGB (in parser)
colors = sh_dc + 0.5
colors = np.clip(colors, 0.0, 1.0)
```

### 4.3 File Watcher

**File:** `viewer/backend/file_watcher.py`

```python
class FileWatcher:
    def __init__(self, watch_dir: str, callback: Callable):
        self.observer = Observer()
        self.handler = PLYEventHandler(callback)

    def start(self):
        self.observer.schedule(self.handler, self.watch_dir, recursive=True)
        self.observer.start()
```

Uses `watchdog` library to monitor `output_clouds/` directory for `.ply` file changes.

### 4.4 Frontend Application

**File:** `viewer/static/js/main.js` (715 lines)

#### Key Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `initScene()` | 50-100 | Three.js scene setup |
| `loadGaussiansBinary()` | 154-250 | Binary stream loading |
| `parseBinaryGaussians()` | 260-344 | Parse binary protocol |
| `createGaussianSplats()` | 455-494 | Create instanced mesh |
| `updateStats()` | 500-520 | FPS and point count display |

#### Binary Stream Parser

```javascript
function parseBinaryGaussians(buffer) {
    const view = new DataView(buffer);

    // Read header
    const magic = view.getUint32(0, true);  // 0x47535053
    const count = view.getUint32(4, true);

    // Parse gaussians (56 bytes each)
    const gaussians = [];
    let offset = 8;

    for (let i = 0; i < count; i++) {
        gaussians.push({
            position: [view.getFloat32(offset, true), ...],
            scale: [...],
            rotation: [...],
            sh_dc: [...],
            opacity: view.getFloat32(offset + 52, true)
        });
        offset += 56;
    }
    return gaussians;
}
```

#### Custom Gaussian Shader

**Vertex Shader:**
```glsl
attribute vec3 instancePosition;
attribute vec3 instanceScale;
attribute vec4 instanceRotation;
attribute vec3 instanceSHDC;
attribute float instanceOpacity;

varying vec3 vColor;
varying float vOpacity;
varying vec2 vUv;

void main() {
    // Convert SH DC to RGB
    vColor = instanceSHDC + 0.5;
    vOpacity = instanceOpacity;
    vUv = uv;

    // Apply instance transform
    vec3 transformed = position * instanceScale;
    transformed = rotateByQuaternion(transformed, instanceRotation);
    transformed += instancePosition;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
}
```

**Fragment Shader:**
```glsl
varying vec3 vColor;
varying float vOpacity;
varying vec2 vUv;

void main() {
    // Gaussian falloff from center
    float dist = length(vUv - 0.5) * 2.0;
    float alpha = exp(-dist * dist * 4.0) * vOpacity;

    gl_FragColor = vec4(vColor, alpha);
}
```

---

## 5. Data Structures

### 5.1 Internal Gaussian Representation

**File:** `src/mesh_to_gaussian.py`

```python
@dataclass
class _SingleGaussian:
    position: np.ndarray      # [x, y, z] - world space position
    scales: np.ndarray        # [sx, sy, sz] - log-space scales
    rotation: np.ndarray      # [w, x, y, z] - quaternion (normalized)
    opacity: float            # [0, 1] - visibility
    sh_dc: np.ndarray         # [r, g, b] - spherical harmonics DC term
    sh_rest: Optional[np.ndarray] = None  # Higher-order SH (unused)
```

**SH DC Color Encoding:**
```
Storage:   sh_dc = RGB - 0.5    (range: [-0.5, 0.5])
Display:   RGB = sh_dc + 0.5    (range: [0.0, 1.0])

Examples:
  White (1,1,1) → sh_dc (0.5, 0.5, 0.5)
  Black (0,0,0) → sh_dc (-0.5, -0.5, -0.5)
  Gray (0.5,0.5,0.5) → sh_dc (0.0, 0.0, 0.0)
  Red (1,0,0) → sh_dc (0.5, -0.5, -0.5)
```

### 5.2 PLY File Format

**Binary Little-Endian Format (17 properties):**

```
ply
format binary_little_endian 1.0
element vertex N
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0      # SH DC Red
property float f_dc_1      # SH DC Green
property float f_dc_2      # SH DC Blue
property float opacity
property float scale_0     # Log-space scale X
property float scale_1     # Log-space scale Y
property float scale_2     # Log-space scale Z
property float rot_0       # Quaternion W
property float rot_1       # Quaternion X
property float rot_2       # Quaternion Y
property float rot_3       # Quaternion Z
end_header
[Binary: N × 68 bytes]
```

### 5.3 Trimesh Visual Types

Understanding these is critical for debugging color issues:

| Visual Type | Description | Color Source |
|-------------|-------------|--------------|
| `ColorVisuals` | Per-vertex RGBA | `mesh.visual.vertex_colors` |
| `TextureVisuals` | UV-mapped texture | `mesh.visual.material.image` |
| `None/Empty` | No visual data | Falls back to gray |

**Visual Type Detection:**
```python
# In _has_texture_visual() - lines 291-326
def _has_texture_visual(self, mesh) -> bool:
    if not hasattr(mesh, 'visual'):
        return False
    if not isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        return False
    if mesh.visual.uv is None:
        return False
    if mesh.visual.material is None:
        return False
    if not hasattr(mesh.visual.material, 'image'):
        return False
    if mesh.visual.material.image is None:
        return False
    return True
```

---

## 6. Color System Analysis

### 6.1 The Problem

**Symptom:** All converted models appear black in the viewer.

**Root Cause Hypothesis:** Texture data is not being properly associated with the mesh during Trimesh loading, causing the color sampling to fall back to gray (0.5, 0.5, 0.5), which when converted to SH DC becomes (0.0, 0.0, 0.0).

### 6.2 Color Flow Trace

```
STAGE 1: BLENDER EXPORT
─────────────────────────────────────────────────────────────────
Baking Pipeline:
  bake_and_export.py
    → bake_combined_texture()     # Creates baked_texture.png
    → export_obj_with_mtl()       # Exports model.obj + model.mtl

  Expected Output:
    model.obj (with UV coords)
    model.mtl (with map_Kd baked_texture.png)
    baked_texture.png

  ⚠️ POTENTIAL ISSUE: Bake may produce all-black texture
     (dump_bake_diagnostic() exists for debugging)

Packed Pipeline:
  extract_packed.py
    → extract_packed_textures()   # Unpacks to textures/
    → create_material_manifest()  # Creates JSON mapping
    → export_uv_layers()          # Saves UV as .npy
    → export_vertex_colors()      # Saves vertex colors as .npy

  ⚠️ POTENTIAL ISSUE: Manifest may not be consumed properly

─────────────────────────────────────────────────────────────────
STAGE 2: MESH LOADING (mesh_to_gaussian.py)
─────────────────────────────────────────────────────────────────
load_mesh(path)
  → trimesh.load(path, force='mesh')

  ⚠️ CRITICAL ISSUE: Trimesh may not associate texture correctly

  _load_obj_with_mtl() attempts:
    1. Check if MTL file exists
    2. Parse MTL for map_Kd directive
    3. Manually attach texture to mesh.visual

  But this may fail if:
    - MTL path parsing is wrong
    - Texture file path is relative/absolute mismatch
    - Trimesh converts TextureVisuals to ColorVisuals

─────────────────────────────────────────────────────────────────
STAGE 2: COLOR SAMPLING
─────────────────────────────────────────────────────────────────
_sample_vertex_colors_vectorized(mesh, vertex_indices)  # Lines 588-634
  │
  ├─► Check: _has_texture_visual(mesh)?
  │   │
  │   ├─► YES: _sample_texture_batch(mesh, uvs)
  │   │        → Returns sampled RGB from texture
  │   │
  │   └─► NO:  Check for vertex colors?
  │            │
  │            ├─► YES: Use mesh.visual.vertex_colors
  │            │
  │            └─► NO:  ⚠️ FALLBACK TO GRAY (0.5, 0.5, 0.5)
  │
  └─► colors array (N, 3) RGB values

─────────────────────────────────────────────────────────────────
STAGE 2: SH DC CONVERSION
─────────────────────────────────────────────────────────────────
mesh_to_gaussians()
  │
  ├─► colors = self._sample_vertex_colors_vectorized(...)
  │
  └─► sh_dc = colors - 0.5

      If colors = (0.5, 0.5, 0.5)  # Gray fallback
      Then sh_dc = (0.0, 0.0, 0.0)

─────────────────────────────────────────────────────────────────
STAGE 3: PLY OUTPUT
─────────────────────────────────────────────────────────────────
save_ply(gaussians, output_path)
  → Writes sh_dc as f_dc_0, f_dc_1, f_dc_2

  If sh_dc = (0.0, 0.0, 0.0), output appears gray (not black)

  ⚠️ If output is BLACK, sh_dc may be NEGATIVE or something
     else is wrong in the shader.

─────────────────────────────────────────────────────────────────
VIEWER: COLOR DISPLAY
─────────────────────────────────────────────────────────────────
ply_parser.py:
  colors = sh_dc + 0.5

main.js (vertex shader):
  vColor = instanceSHDC + 0.5;

  If sh_dc = (0.0, 0.0, 0.0):
    Display color = (0.5, 0.5, 0.5) = GRAY

  If output is BLACK:
    Either sh_dc = (-0.5, -0.5, -0.5)
    Or shader is not receiving color data
    Or opacity = 0
```

### 6.3 Likely Failure Points (Priority Order)

| Priority | Location | Issue | Evidence |
|----------|----------|-------|----------|
| **1** | `_has_texture_visual()` | Returns `False` - texture not attached | Fallback to gray |
| **2** | `trimesh.load()` | TextureVisuals not created from MTL | Common Trimesh issue |
| **3** | `_load_obj_with_mtl()` | Manual texture attachment fails | Path resolution issues |
| **4** | Blender bake | Black texture output | `dump_bake_diagnostic()` exists |
| **5** | Manifest consumption | Multi-material path not used | Packed pipeline specific |
| **6** | UV coordinates | Missing or invalid UVs | Would cause sampling failure |

### 6.4 Diagnostic Tools

**PLY Inspector (`tools/ply_inspector.py`):**
```bash
python tools/ply_inspector.py output_clouds/model.ply

# Output shows:
# - SH DC values (f_dc_0, f_dc_1, f_dc_2)
# - Min/max/mean of each property
# - Helps verify if color data is present
```

**Bake Diagnostic (`bake_and_export.py`):**
```python
def dump_bake_diagnostic():
    # Saves intermediate textures for debugging
    # Check if baked texture is black
```

---

## 7. Technical Debt & Known Issues

### 7.1 Critical Issues

| ID | Issue | Impact | Location | Priority |
|----|-------|--------|----------|----------|
| **C-01** | Color data not preserved | All models black | `mesh_to_gaussian.py` | **P0** |
| **C-02** | Texture→Trimesh association failure | Silent fallback to gray | `load_mesh()` | **P0** |
| **C-03** | MTL parsing unreliable | Textures not loaded | `_load_obj_with_mtl()` | **P0** |

### 7.2 High Priority Issues

| ID | Issue | Impact | Location | Priority |
|----|-------|--------|----------|----------|
| **H-01** | No error/warning when color sampling fails | Hard to debug | `_sample_vertex_colors_vectorized()` | P1 |
| **H-02** | Manifest not used in baking pipeline | Color data lost | `orchestrator.py` | P1 |
| **H-03** | Blender path detection fragile | Pipeline fails silently | `baker.py` | P1 |
| **H-04** | No validation of intermediate outputs | Issues propagate | Throughout pipeline | P1 |

### 7.3 Medium Priority Issues

| ID | Issue | Impact | Location | Priority |
|----|-------|--------|----------|----------|
| **M-01** | Large mesh_to_gaussian.py (2085 lines) | Hard to maintain | `mesh_to_gaussian.py` | P2 |
| **M-02** | Multiple color sampling paths | Logic complexity | `mesh_to_gaussian.py` | P2 |
| **M-03** | LRU texture cache not bounded well | Memory issues | `_sample_texture_batch()` | P2 |
| **M-04** | Viewer lacks error boundaries | Crashes on bad data | `main.js` | P2 |
| **M-05** | No unit tests for color sampling | Regression risk | `tests/` | P2 |

### 7.4 Low Priority Issues

| ID | Issue | Impact | Location | Priority |
|----|-------|--------|----------|----------|
| **L-01** | Hardcoded paths in some places | Portability | Various | P3 |
| **L-02** | Missing type hints | IDE support | Python files | P3 |
| **L-03** | Inconsistent logging levels | Debug difficulty | Throughout | P3 |
| **L-04** | Viewer UI not responsive | Mobile unusable | `style.css` | P3 |

### 7.5 Code Quality Observations

**mesh_to_gaussian.py Complexity:**
- 2085 lines in single file
- Multiple sampling strategies with overlapping logic
- `_sample_vertex_colors_vectorized()` has deep nesting
- Multi-material path (`_mesh_to_gaussians_multi_material`) partially duplicates single-material path

**Recommended Refactoring:**
1. Extract `ColorSampler` class with clear interface
2. Extract `TextureLoader` class to handle Trimesh quirks
3. Split strategies into separate strategy classes
4. Add validation layer between pipeline stages

---

## 8. Integration Points

### 8.1 External Dependencies

| Dependency | Version | Purpose | Risk Level |
|------------|---------|---------|------------|
| **Blender** | 3.x/4.x | Texture baking/extraction | HIGH - Version-specific API changes |
| **Trimesh** | ≥3.23.0 | Mesh loading | HIGH - TextureVisuals handling varies |
| **Pillow** | ≥10.0.0 | Texture sampling | LOW - Stable API |
| **NumPy** | ≥1.24.0 | Numerical operations | LOW - Stable API |
| **FastAPI** | ≥0.104.1 | Web server | LOW - Stable API |
| **Three.js** | r160 | WebGL rendering | MEDIUM - Breaking changes between versions |

### 8.2 Blender Integration Details

**Execution Model:**
```python
# Headless execution
subprocess.run([
    blender_path,
    '--background',
    str(input_path),
    '--python', str(script_path),
    '--', json.dumps(args)
])
```

**Version Compatibility:**
- Blender 3.x: Cycles baking API
- Blender 4.x: Updated material node API

**Common Issues:**
1. Blender not in PATH
2. Python environment mismatch (Blender uses bundled Python)
3. Relative vs absolute paths in script arguments

### 8.3 Trimesh Integration Details

**Loading Behavior:**

```python
# Current approach
mesh = trimesh.load(path, force='mesh')

# Issue: force='mesh' may convert TextureVisuals to ColorVisuals
# or drop visual data entirely

# Alternative approach (not implemented)
scene = trimesh.load(path, force='scene')
# Then iterate scene.geometry to preserve materials
```

**Known Trimesh Quirks:**
1. OBJ+MTL loading requires files in same directory
2. `force='mesh'` concatenates multi-object scenes, losing per-object materials
3. TextureVisuals requires UV + material + image, any missing = fallback
4. Converting between visual types can lose data

### 8.4 File Format Integration

**Input Formats:**

| Format | Support | Notes |
|--------|---------|-------|
| `.blend` | Full | Native Blender, best support |
| `.fbx` | Partial | Blender import required |
| `.obj` | Full | Direct Trimesh loading |
| `.gltf/.glb` | Untested | Should work via Trimesh |

**Output Format:**

| Format | Support | Notes |
|--------|---------|-------|
| `.ply` | Full | Binary little-endian, 17 properties |

### 8.5 Inter-Component Data Flow

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   CLI.py    │────────►│ Orchestrator│────────►│   Output    │
│             │         │             │         │   Files     │
└─────────────┘         └──────┬──────┘         └─────────────┘
                               │                       │
              ┌────────────────┼────────────────┐      │
              ▼                ▼                ▼      ▼
       ┌───────────┐    ┌───────────┐    ┌───────────────┐
       │  Baker/   │    │ Converter │    │  LOD          │
       │ Extractor │    │           │    │  Generator    │
       └─────┬─────┘    └─────┬─────┘    └───────────────┘
             │                │
             ▼                ▼
      ┌────────────┐   ┌────────────┐
      │ Intermediate│  │  Trimesh   │
      │ Files      │   │  (in-mem)  │
      └────────────┘   └────────────┘

Intermediate Files:
  - model.obj
  - model.mtl
  - textures/*.png
  - material_manifest.json
  - uv_layers/*.npy
  - vertex_colors.npy
```

---

## 9. Recommendations

### 9.1 Immediate Actions (Fix Color System)

#### Option A: Bypass Trimesh for Color Sampling

Create a dedicated texture sampling path that doesn't rely on Trimesh's visual system:

```python
class DirectTextureSampler:
    def __init__(self, manifest_path: str):
        self.manifest = json.load(open(manifest_path))
        self.textures = self._load_textures()
        self.uv_layers = self._load_uv_layers()

    def sample(self, face_indices: np.ndarray,
               barycentric: np.ndarray) -> np.ndarray:
        """
        Sample colors directly from texture files using
        UV coordinates from manifest, bypassing Trimesh.
        """
        # Get material per face
        face_materials = self.manifest['face_materials']

        # For each face, get UV coords and sample texture
        colors = np.zeros((len(face_indices), 3))
        for mat_name, mat_info in self.manifest['materials'].items():
            mask = face_materials == mat_idx
            texture = self.textures[mat_info['diffuse']['path']]
            uv = self.uv_layers[mat_info['diffuse']['uv_layer']]
            # Sample texture at UV coordinates
            colors[mask] = self._sample_texture(texture, uv, ...)

        return colors
```

**Pros:** Complete control over color pipeline
**Cons:** Requires packed pipeline with manifest

#### Option B: Fix Trimesh Integration

Debug and fix the existing Trimesh-based color sampling:

1. Add extensive logging to `_has_texture_visual()`
2. Print visual type at each stage
3. Test with known-good OBJ+MTL+texture files
4. Consider using `trimesh.load(path, force='scene')` instead of `force='mesh'`

```python
def load_mesh(self, path: str) -> trimesh.Trimesh:
    logging.info(f"Loading mesh from {path}")

    # Try scene mode first to preserve materials
    scene = trimesh.load(path, force='scene')
    logging.info(f"Loaded scene with {len(scene.geometry)} geometries")

    for name, geom in scene.geometry.items():
        logging.info(f"  {name}: visual type = {type(geom.visual)}")
        if hasattr(geom.visual, 'material'):
            logging.info(f"    material = {geom.visual.material}")
            if hasattr(geom.visual.material, 'image'):
                logging.info(f"    image = {geom.visual.material.image}")

    # Then concatenate with visual preservation
    mesh = trimesh.util.concatenate(list(scene.geometry.values()))
    return mesh
```

**Pros:** Fixes root cause
**Cons:** Trimesh behavior may vary by version

#### Option C: Use Packed Pipeline Exclusively

Leverage the manifest system which already exports UV and texture data:

1. Make packed pipeline the default
2. Load UV coordinates from `.npy` files directly
3. Load textures directly from manifest paths
4. Sample using numpy/Pillow, not Trimesh

**Pros:** Most reliable data source
**Cons:** Requires Blender preprocessing for all inputs

### 9.2 Short-Term Improvements

1. **Add Validation Layer**
   ```python
   class PipelineValidator:
       def validate_stage1_output(self, output_dir: Path) -> List[str]:
           """Return list of issues with Stage 1 output."""
           issues = []
           if not (output_dir / 'model.obj').exists():
               issues.append("Missing model.obj")
           # Check texture file exists and is not black
           # Check UV coordinates are valid
           return issues
   ```

2. **Add Color Debug Mode**
   ```bash
   python cli.py input.blend --debug-color
   # Outputs diagnostic images showing:
   # - Texture as loaded
   # - UV visualization
   # - Per-face color samples
   ```

3. **Add Integration Tests**
   ```python
   def test_color_preservation():
       """End-to-end test with known-color model."""
       # Use a model with solid red, green, blue faces
       result = pipeline.run("test_rgb_cube.blend")
       gaussians = load_ply(result)

       # Verify colors match expected
       assert any(is_red(g.sh_dc) for g in gaussians)
       assert any(is_green(g.sh_dc) for g in gaussians)
       assert any(is_blue(g.sh_dc) for g in gaussians)
   ```

### 9.3 Long-Term Architecture Recommendations

1. **Modular Pipeline Architecture**
   ```
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │  Importer   │──►│  Processor  │──►│  Exporter   │
   │  (pluggable)│   │  (core)     │   │  (pluggable)│
   └─────────────┘   └─────────────┘   └─────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
   BlenderImporter   GaussianProcessor   PLYExporter
   OBJImporter       LODProcessor        SplatExporter
   GLTFImporter
   ```

2. **Standardized Intermediate Format**
   ```python
   @dataclass
   class IntermediateMesh:
       vertices: np.ndarray        # [N, 3]
       faces: np.ndarray           # [M, 3]
       uvs: np.ndarray             # [N, 2] or [M*3, 2]
       normals: np.ndarray         # [N, 3]
       materials: List[Material]   # Per-material textures
       face_materials: np.ndarray  # [M] material index per face
   ```

   All importers output this format, all processors consume it.

3. **Separate Color Sampling Module**
   ```python
   # src/color/sampler.py
   class ColorSampler(ABC):
       @abstractmethod
       def sample(self, mesh: IntermediateMesh,
                  points: np.ndarray) -> np.ndarray:
           """Return RGB colors for given points."""

   class TextureSampler(ColorSampler):
       """Sample from UV-mapped textures."""

   class VertexColorSampler(ColorSampler):
       """Interpolate vertex colors."""

   class ConstantColorSampler(ColorSampler):
       """Return constant color (fallback)."""
   ```

4. **Configuration-Driven Pipeline**
   ```yaml
   # pipeline.yaml
   stages:
     - name: import
       type: blender_import
       config:
         extract_textures: true

     - name: sample_colors
       type: texture_sampler
       config:
         fallback: vertex_colors

     - name: convert
       type: mesh_to_gaussian
       config:
         strategy: hybrid
         samples_per_face: 3

     - name: export
       type: ply_exporter
   ```

### 9.4 Testing Strategy

1. **Unit Tests:** Individual functions with mocked dependencies
2. **Integration Tests:** Full pipeline with known test assets
3. **Visual Regression Tests:** Compare rendered output images
4. **Performance Benchmarks:** Track conversion time and memory usage

**Test Assets Needed:**
- `test_solid_colors.blend` - Cube with red/green/blue faces
- `test_textured.blend` - Simple textured model
- `test_multi_material.blend` - Multiple materials
- `test_vertex_colors.blend` - Vertex color only model

---

## Appendix A: Running the System

### Start Conversion Pipeline

```bash
# Basic conversion
python cli.py model.blend --output output_clouds/

# Packed pipeline (recommended until color issue fixed)
python cli.py model.blend --packed --output output_clouds/

# With LOD generation
python cli.py model.blend --lod-levels 3 --output output_clouds/
```

### Start Viewer

```bash
cd viewer
python server.py
# Open http://localhost:8000
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Inspect PLY Output

```bash
python tools/ply_inspector.py output_clouds/model.ply
```

---

## Appendix B: Quick Reference

### Color Conversion Formulas

```
Encoding (mesh_to_gaussian.py):
  sh_dc = rgb - 0.5

Decoding (viewer):
  rgb = sh_dc + 0.5
  rgb = clamp(rgb, 0.0, 1.0)
```

### Binary Protocol (Viewer Streaming)

```
Magic: 0x47535053 (4 bytes)
Count: uint32 (4 bytes)
Gaussians: [
  position: float32[3]   (12 bytes)
  scale: float32[3]      (12 bytes)
  rotation: float32[4]   (16 bytes)
  sh_dc: float32[3]      (12 bytes)
  opacity: float32       (4 bytes)
] × Count                (56 bytes each)
```

### Key File Locations

| Purpose | File |
|---------|------|
| Entry point | `cli.py` |
| Color sampling | `src/mesh_to_gaussian.py:588-634` |
| Texture validation | `src/mesh_to_gaussian.py:291-326` |
| Baking script | `src/stage1_baker/blender_scripts/bake_and_export.py` |
| Extraction script | `src/stage1_baker/blender_scripts/extract_packed.py` |
| PLY parsing | `viewer/backend/ply_parser.py` |
| Gaussian shader | `viewer/static/js/main.js:499-550` |

---

*End of Technical Specification*


