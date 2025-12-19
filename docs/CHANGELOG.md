# Changelog

All notable changes to the Unified Gaussian Pipeline project.

## [Unreleased]

### Fixed
- R/G channel swap in color encoding (December 18, 2025)
  - Applied fix at `mesh_to_gaussian.py` lines 413-416
  - Resolves issue where leaves appeared brown and bark appeared green
  - Swaps R and G channels to match expected color space conventions

---

## [1.3.0] - December 1, 2025 - Phase 3: Pipeline Orchestrator

**Status:** ✅ COMPLETE  
**Tests:** 17/17 passing

### Added
- Unified pipeline orchestrator (`src/pipeline/orchestrator.py`)
- Pipeline configuration with validation (`src/pipeline/config.py`)
- Intelligent file routing (`src/pipeline/router.py`)
- Command-line interface (`cli.py`)
- Lazy initialization for components

### Features
- Automatic detection of file types (.blend, .obj, .glb)
- Single command converts any supported format
- Configurable LOD levels, strategies, and options
- Progress reporting with execution time

### Example Usage
```bash
python cli.py model.obj ./output --lod 5000 25000 100000
```

---

## [1.2.0] - November 30, 2025 - Phase 2: Blender Baker Integration

**Status:** ✅ COMPLETE  
**Tests:** 7/7 passing

### Added
- Blender baker module (`src/stage1_baker/baker.py`)
- Blender export script (`src/stage1_baker/blender_scripts/bake_and_export.py`)
- Packed texture extractor (`src/stage1_baker/packed_extractor.py`)
- Headless Blender subprocess execution
- UV preservation with secondary "BakeUV" layer

### Fixed
- Windows subprocess hanging issue (Blender Ctrl+C signal)
- Implemented file polling for reliable completion detection

### Performance
- Simple cube bake: ~0.5 seconds (60x faster than target)

---

## [1.1.0] - November 28, 2025 - Phase 1: UV Texture Sampling

**Status:** ✅ COMPLETE  
**Tests:** 2/2 passing

### Added
- UV texture sampling in `mesh_to_gaussian.py`
- MTL `map_Kd` directive parsing
- PIL-based texture loading
- Bilinear texture interpolation
- Texture caching with LRU eviction

### Features
- Automatic texture loading from MTL references
- Color sampling at UV coordinates for each gaussian
- Support for PNG, JPG, and other PIL formats
- Fallback to vertex/face colors when textures unavailable

---

## [1.0.0] - November 2025 - Initial Release

### Core Features
- Mesh-to-Gaussian converter (`src/mesh_to_gaussian.py`)
- LOD generator with 3 strategies (`src/lod_generator.py`)
- Web-based viewer (`viewer/`)
- Binary PLY export with Spherical Harmonics

### Supported Formats
- OBJ with MTL materials
- GLB with embedded textures
- Blender files (via baking pipeline)

### Initialization Strategies
- Vertex: One gaussian per vertex
- Face: Distributed across faces
- Hybrid: Combined vertex + face
- Adaptive: Curvature-based density

### LOD Strategies
- Importance: opacity × volume (recommended)
- Opacity: Keep most visible
- Spatial: Uniform distribution

---

## Project Milestones

| Date | Milestone | Details |
|------|-----------|---------|
| Dec 18, 2025 | Color fix | R/G channel swap applied |
| Dec 1, 2025 | Phase 3 complete | Pipeline orchestrator |
| Nov 30, 2025 | Phase 2 complete | Blender baker |
| Nov 28, 2025 | Phase 1 complete | UV texture sampling |
| Nov 2025 | Initial release | Core converter + viewer |

---

## Total Test Coverage

- Converter tests: 10 tests
- Baker tests: 7 tests  
- Pipeline tests: 17 tests
- Packed texture tests: 17 tests
- **Total: 51/51 tests passing**

