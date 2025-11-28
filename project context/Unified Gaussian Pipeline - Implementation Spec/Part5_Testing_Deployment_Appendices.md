# Unified Gaussian Pipeline - Implementation Specification
## Part 5: Testing, Deployment & Appendices

**Version:** 1.0  
**Date:** November 27, 2025

---

## 7. Testing Strategy

### 7.1 Test Coverage Requirements

**Minimum Coverage:** 90% across all modules

| Module | Target Coverage | Critical Paths |
|--------|----------------|----------------|
| `mesh_to_gaussian.py` | 95% | Texture loading, UV sampling |
| `baker.py` | 90% | Subprocess management, validation |
| `orchestrator.py` | 90% | Stage coordination, error handling |
| `router.py` | 100% | File routing logic |
| `config.py` | 100% | Validation logic |

### 7.2 Running Tests

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific phase:**
```bash
pytest tests/test_texture_sampling.py -v
pytest tests/test_baker.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_fbx_support.py -v
```

**Coverage report:**
```bash
pytest tests/ --cov=src --cov-report=html
```

### 7.3 Pre-commit Checklist

1. Run full test suite
2. Check code coverage (must be ≥ 90%)
3. Run linter: `pylint src/`
4. Verify all ABOUTME comments present
5. Check that no tests are skipped unintentionally

---

## 8. Deployment & Configuration

### 8.1 System Requirements

**Software:**
- Python 3.8+
- Blender 3.0+ (for Stage 1)
- CUDA 11.8+ (optional, for GPU acceleration)

**Hardware:**
- RAM: 16 GB minimum, 32 GB recommended
- Storage: 10 GB free space for temp files
- GPU: Optional (NVIDIA with CUDA support)

### 8.2 Installation

**Step 1: Install Python dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Install Blender**
- Download from https://www.blender.org/download/
- Add to PATH or note executable location

**Step 3: Configure Blender path**
```bash
# Test Blender installation
blender --version

# If not in PATH, specify location:
python cli.py model.blend ./output --blender "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe"
```

**Step 4: Verify installation**
```bash
# Run test suite
pytest tests/ -v

# Try example conversion
python cli.py test_assets/procedural_cube.blend ./test_output
```

### 8.3 Performance Tuning

**For faster iteration:**
```bash
# Reduce texture resolution
python cli.py model.blend ./output --texture-resolution 2048

# Use fewer LOD levels
python cli.py model.blend ./output --lod 5000 25000

# Use vertex strategy (faster)
python cli.py model.blend ./output --strategy vertex
```

**For best quality:**
```bash
# High-res textures
python cli.py model.blend ./output --texture-resolution 8192

# Fine-grained LODs
python cli.py model.blend ./output --lod 5000 10000 25000 50000 100000
```

---

## Appendix A: Data Flow Diagrams

### Complete System Data Flow

```
INPUT FILE
    │
    ├─ .blend (procedural shaders)
    │     │
    │     ▼
    │  ┌──────────────────────────────────┐
    │  │  STAGE 1: BLENDER BAKER          │
    │  │  - Load in Blender (headless)    │
    │  │  - Detect existing UVs           │
    │  │  - Create secondary UV layer     │
    │  │  - Bake shaders → 4K texture     │
    │  │  - Export OBJ + MTL              │
    │  └──────────────┬───────────────────┘
    │                 │
    │                 ▼
    │           OBJ + Texture
    │                 │
    ├─────────────────┘
    │
    ├─ .obj/.glb/.fbx
    │
    ▼
┌──────────────────────────────────┐
│  STAGE 2: GAUSSIAN CONVERTER     │
│  - Load mesh with trimesh        │
│  - Parse MTL for texture ref     │
│  - Load texture with PIL         │
│  - Generate gaussians            │
│  - Sample colors from texture    │
└──────────────┬───────────────────┘
               │
               ▼
        List[Gaussian]
               │
               ▼
┌──────────────────────────────────┐
│  LOD GENERATOR                   │
│  - Compute importance            │
│  - Select top N                  │
│  - Create subsets                │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  PLY EXPORT                      │
│  - Pack position, scales, etc    │
│  - Write binary PLY              │
│  Output: Multiple .ply files     │
└──────────────────────────────────┘
```

### UV Texture Sampling Mathematics

```
MESH TRIANGLE
    vertices: v0, v1, v2
    UVs:      uv0, uv1, uv2

SAMPLE POINT P
    │
    ▼
1. BARYCENTRIC COORDINATES
   p = α·v0 + β·v1 + γ·v2
   where α + β + γ = 1
    │
    ▼
2. UV INTERPOLATION
   uv_p = α·uv0 + β·uv1 + γ·uv2
    │
    ▼
3. PIXEL COORDINATE
   u_pixel = u × (width - 1)
   v_pixel = (1-v) × (height - 1)  # V-flip
    │
    ▼
4. TEXTURE SAMPLING
   color = texture.getpixel((u_px, v_px))
    │
    ▼
GAUSSIAN COLOR
```

---

## Appendix B: File Format Specifications

### MTL File Format

**Supported Directives:**

```mtl
newmtl <material_name>
Kd <r> <g> <b>          # Diffuse color [0-1]
map_Kd <filename>       # Diffuse texture path (NEW in Phase 1)
```

**Example:**
```mtl
newmtl TreeBark
Kd 0.4 0.3 0.2
map_Kd bark_texture.png
```

### PLY File Format (Output)

**Header:**
```
ply
format binary_little_endian 1.0
element vertex <N>
property float x
property float y
property float z
property float nx           # Normal (unused)
property float ny
property float nz
property float f_dc_0       # Spherical Harmonics DC (R)
property float f_dc_1       # SH DC (G)
property float f_dc_2       # SH DC (B)
property float opacity      # Opacity [0-1]
property float scale_0      # Log scale X
property float scale_1      # Log scale Y
property float scale_2      # Log scale Z
property float rot_0        # Quaternion W
property float rot_1        # Quaternion X
property float rot_2        # Quaternion Y
property float rot_3        # Quaternion Z
end_header
<binary data>
```

**Data per gaussian:** 17 floats × 4 bytes = 68 bytes

**File size formula:**
```
Size (MB) ≈ 0.068 × N

Examples:
  5,000 gaussians   ≈ 0.34 MB
  25,000 gaussians  ≈ 1.7 MB
  100,000 gaussians ≈ 6.8 MB
  1M gaussians      ≈ 68 MB
```

---

## Appendix C: Troubleshooting Guide

### Common Issues

**Issue 1: Blender not found**
```
Error: Blender executable failed: blender
```
**Solution:**
```bash
python cli.py model.blend ./output --blender "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe"
```

**Issue 2: Texture not loaded**
```
⚠️ Texture not found: Skull.jpg
```
**Solution:**
- Verify texture file exists in same directory as OBJ
- Check MTL file references correct filename
- Ensure texture path is relative, not absolute

**Issue 3: All gray gaussians**
```
Generated 100000 gaussians (all gray)
```
**Solution:**
- Check mesh has UV coordinates: `hasattr(mesh.visual, 'uv')`
- Verify texture loaded: `hasattr(mesh.visual.material, 'image')`
- Test with simple textured cube first

**Issue 4: Out of memory**
```
MemoryError: Unable to allocate array
```
**Solution:**
```bash
# Reduce samples per face
python cli.py model.blend ./output --strategy vertex

# Or generate fewer LOD levels
python cli.py model.blend ./output --lod 5000 25000
```

**Issue 5: Baking timeout**
```
subprocess.TimeoutExpired: Blender baking exceeded 600s
```
**Solution:**
- Reduce texture resolution: `--texture-resolution 2048`
- Simplify mesh before baking
- Increase timeout in `baker.py` if needed

### Debug Mode

**Enable verbose logging:**
```python
# In cli.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Keep temp files for inspection:**
```bash
python cli.py model.blend ./output --keep-temp
```

### Performance Optimization

**Slow Stage 1 (Blender baking):**
- Reduce texture resolution
- Use GPU if available (modify `bake_and_export.py`)
- Simplify mesh topology

**Slow Stage 2 (Gaussian generation):**
- Use `vertex` strategy instead of `hybrid`
- Reduce `samples_per_face`
- Use GPU device: `--device cuda`

**Large output files:**
- Reduce LOD levels
- Use lower gaussian counts

---

## Summary Checklist

### Phase 1: UV Texture Sampling ✅ COMPLETE (Nov 28, 2025)
- [x] ✅ Implement `_load_obj_with_mtl()` enhancement
- [x] ✅ Implement `_sample_texture_color()` method
- [x] ✅ Update face sampling to use texture
- [x] ✅ Write and pass all tests (2/2 passing)
- [x] ✅ Verify skull model with texture works
- [x] ✅ Commit code

### Phase 2: Blender Baker
- [ ] Create `stage1_baker/` directory structure
- [ ] Implement `baker.py` Python wrapper
- [ ] Implement `blender_scripts/bake_and_export.py`
- [ ] Write and pass all tests
- [ ] Create test assets
- [ ] Verify real .blend file bakes successfully
- [ ] Commit code

### Phase 3: Pipeline Orchestrator
- [ ] Create `pipeline/` directory structure
- [ ] Implement `config.py` dataclass
- [ ] Implement `router.py` routing logic
- [ ] Implement `orchestrator.py` main pipeline
- [ ] Create `cli.py` unified interface
- [ ] Write and pass all tests
- [ ] Test end-to-end workflows
- [ ] Commit code

### Phase 4: FBX Support (Optional)
- [ ] Add FBX routing
- [ ] Write and pass tests
- [ ] Update documentation
- [ ] Commit code

### Final Integration
- [ ] Run complete test suite (all phases)
- [ ] Verify >90% code coverage
- [ ] Update README with usage examples
- [ ] Create user documentation
- [ ] Performance benchmark on real assets
- [ ] Final commit and tag release

---

## Quick Reference

### File Structure
```
GCE CLONE/
├── src/
│   ├── stage1_baker/          # NEW
│   ├── stage2_converter/      # ENHANCED
│   └── pipeline/              # NEW
├── tests/                     # EXPANDED
├── test_assets/               # NEW
└── cli.py                     # NEW
```

### Key Commands
```bash
# Development
pytest tests/ -v --cov=src

# Usage
python cli.py input.blend ./output
python cli.py input.obj ./output --lod 5000 25000 100000

# With options
python cli.py input.blend ./output \
  --strategy hybrid \
  --lod-strategy importance \
  --texture-resolution 4096 \
  --keep-temp
```

### Contact & Support

For implementation questions:
1. Review relevant section in this documentation
2. Check `CURRENT_PROJECT_STATE.md` for known issues
3. Consult appendices for troubleshooting

---

**END OF SPECIFICATION**

**Document Version:** 1.0  
**Total Estimated Implementation Time:** 4-6 weeks  
**Team Size:** 5-7 developers

Each phase is designed to be completed independently with full test coverage before moving to the next phase. Follow TDD methodology throughout implementation.
