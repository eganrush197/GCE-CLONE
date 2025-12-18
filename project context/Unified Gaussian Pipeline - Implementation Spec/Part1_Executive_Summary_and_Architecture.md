# Unified Gaussian Pipeline - Implementation Specification
## Part 1: Executive Summary & System Architecture

**Version:** 1.3
**Date:** November 27, 2025
**Last Updated:** December 1, 2025
**Project:** Gaussian Mesh Converter + Blender Point Cloud Pipeline Integration
**Target Team:** 5-7 Python developers with 3D graphics knowledge

---

## 1. Executive Summary

### 1.1 Implementation Status

| Phase | Status | Completion Date | Tests | Notes |
|-------|--------|----------------|-------|-------|
| **Phase 1: UV Texture Sampling** | ✅ **COMPLETE** | Nov 28, 2025 | 2/2 ✅ | Texture loading & UV sampling working |
| **Phase 2: Blender Baker** | ✅ **COMPLETE** | Nov 30, 2025 | 7/7 ✅ | Headless baking, UV preservation working |
| **Phase 3: Pipeline Orchestrator** | ✅ **COMPLETE** | Dec 1, 2025 | 17/17 ✅ | Unified pipeline with CLI working |
| **Phase 4: FBX Support** | ⏳ PENDING | - | 0/? | Optional |

**Total Test Coverage:** 34/34 tests passing ✅ (10 converter + 7 baker + 17 pipeline)
**Remaining Timeline:** 1 week (Phase 4 - optional)

### 1.2 Project Goal

Build a unified pipeline that converts Blender files with procedural shaders into gaussian splat PLY files with multiple LOD levels, while also supporting direct conversion of standard mesh formats.

### 1.3 Current State

**Existing Capabilities:**
- ✅ OBJ/GLB mesh loading with normalization
- ✅ UV texture sampling - MTL `map_Kd` support with PIL image loading
- ✅ **Blender Baker** - Headless shader baking with UV preservation
- ✅ **Unified Pipeline (NEW!)** - Automatic file routing and stage coordination
- ✅ **CLI Interface (NEW!)** - User-friendly command-line tool
- ✅ Gaussian generation (4 strategies: vertex, face, hybrid, adaptive)
- ✅ LOD generation (3 strategies: importance, opacity, spatial)
- ✅ Binary PLY export with Spherical Harmonics
- ✅ Comprehensive test suite (34/34 tests passing)

**Remaining Gaps:**
- ⏳ FBX format support (Phase 4 - optional)

### 1.4 What We're Building

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED PIPELINE FLOW                        │
└─────────────────────────────────────────────────────────────────┘

Input: .blend, .obj, .glb, .fbx
           │
           ▼
    ┌──────────────┐
    │   ROUTER     │  Determines which stage(s) needed
    └──────┬───────┘
           │
    ┌──────┴───────────────────────────────┐
    │                                      │
    ▼                                      ▼
┌─────────────────┐                  ┌────────────────┐
│  STAGE 1:       │                  │  STAGE 2:      │
│  BLENDER BAKER  │──────────────────▶│  GAUSSIAN      │
│  (New)          │  OBJ + Texture   │  CONVERTER     │
│                 │                  │  (Enhanced)    │
│ - Bake shaders  │                  │ - Load texture │
│ - Preserve UVs  │                  │ - Sample colors│
│ - Export OBJ    │                  │ - Gen gaussians│
└─────────────────┘                  │ - Create LODs  │
                                     └────────┬───────┘
                                              │
                                              ▼
                                     ┌────────────────┐
                                     │  OUTPUT:       │
                                     │  Multiple PLYs │
                                     │  - Full res    │
                                     │  - LOD 100k    │
                                     │  - LOD 25k     │
                                     │  - LOD 5k      │
                                     └────────────────┘
```

### 1.4 Success Criteria

1. ✅ .blend files with procedural shaders → colored gaussian splats
2. ✅ Texture-mapped OBJ files → accurate colored gaussians
3. ✅ User-selectable LOD levels
4. ✅ Processing time: < 5 minutes for typical assets
5. ✅ All tests passing with >90% coverage

---

## 2. System Architecture

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROJECT STRUCTURE                           │
└─────────────────────────────────────────────────────────────────┘

GCE CLONE/
├── src/
│   ├── stage1_baker/                    # NEW
│   │   ├── __init__.py
│   │   ├── baker.py                     # Python wrapper
│   │   └── blender_scripts/
│   │       └── bake_and_export.py       # Blender Python script
│   │
│   ├── stage2_converter/                # EXISTING (moved & enhanced)
│   │   ├── __init__.py
│   │   ├── mesh_to_gaussian.py         # ENHANCED with texture sampling
│   │   ├── gaussian_splat.py           # UNCHANGED
│   │   └── lod_generator.py            # UNCHANGED
│   │
│   ├── pipeline/                        # NEW
│   │   ├── __init__.py
│   │   ├── orchestrator.py             # Main pipeline logic
│   │   ├── router.py                   # File routing logic
│   │   └── config.py                   # Configuration dataclass
│   │
│   └── utils/                           # NEW
│       ├── __init__.py
│       ├── file_utils.py               # Temp file management
│       └── validation.py               # Input validation
│
├── tests/
│   ├── test_texture_sampling.py        # NEW - Phase 1
│   ├── test_baker.py                   # NEW - Phase 2
│   ├── test_pipeline.py                # NEW - Phase 3
│   ├── test_fbx_support.py             # NEW - Phase 4
│   └── test_converter.py               # EXISTING
│
├── test_assets/                        # NEW - Test data
│   ├── textured_cube/
│   │   ├── cube.obj
│   │   ├── cube.mtl
│   │   └── cube_texture.png
│   ├── procedural_tree/
│   │   └── tree.blend
│   └── fbx_model/
│       └── model.fbx
│
├── cli.py                              # NEW - Unified CLI
├── convert.py                          # EXISTING - Keep for backwards compat
└── mesh2gaussian                       # EXISTING - Keep for backwards compat
```

### 2.2 Module Responsibilities

| Module | Responsibility | Key Classes/Functions |
|--------|---------------|----------------------|
| `stage1_baker/baker.py` | Blender subprocess management | `BlenderBaker` |
| `stage1_baker/blender_scripts/bake_and_export.py` | UV preservation, shader baking, OBJ export | `bake_and_export()` |
| `stage2_converter/mesh_to_gaussian.py` | Texture loading, UV sampling, gaussian generation | `MeshToGaussianConverter` |
| `pipeline/orchestrator.py` | Pipeline coordination, progress tracking | `Pipeline` |
| `pipeline/router.py` | File type detection, stage routing | `FileRouter` |
| `pipeline/config.py` | User settings, validation | `PipelineConfig` |

### 2.3 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA FLOW DIAGRAM                           │
└─────────────────────────────────────────────────────────────────┘

USER INPUT
    │
    ├─ File Path (str)
    ├─ LOD Levels (List[int])
    ├─ Strategy (str)
    └─ Output Directory (Path)
    │
    ▼
┌──────────────────┐
│   FileRouter     │
│  - Detect type   │
│  - Validate      │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
.blend    .obj/.glb/.fbx
    │         │
    │         └─────────────────┐
    │                           │
    ▼                           │
┌──────────────────┐            │
│  BlenderBaker    │            │
│  Input: .blend   │            │
│  Output: OBJ +   │            │
│         Texture  │            │
└────────┬─────────┘            │
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ MeshToGaussian      │
         │ Input: OBJ/GLB/FBX  │
         │        + Texture    │
         │ Process:            │
         │   1. Load mesh      │
         │   2. Load texture   │
         │   3. Sample UVs     │
         │   4. Gen gaussians  │
         │ Output: List[Gauss] │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   LODGenerator      │
         │ Input: List[Gauss]  │
         │ Output: Multiple    │
         │         List[Gauss] │
         │         (LOD levels)│
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │    PLY Export       │
         │ Output: Multiple    │
         │         .ply files  │
         └─────────────────────┘
```

### 2.4 Implementation Timeline

| Phase | Duration | Team Size | Status | Dependencies |
|-------|----------|-----------|--------|--------------|
| Phase 1: UV Texture Sampling | 1 week | 2 devs | ✅ **COMPLETE** | None |
| Phase 2: Blender Baker | 1-2 weeks | 2-3 devs | ⏳ PENDING | Phase 1 complete ✅ |
| Phase 3: Pipeline Orchestrator | 1-2 weeks | 2 devs | ⏳ PENDING | Phases 1 & 2 complete |
| Phase 4: FBX Support (Optional) | 3-5 days | 1 dev | ⏳ PENDING | Phase 3 complete |

**Original Estimated Time:** 4-6 weeks
**Remaining Time:** 3-5 weeks (Phases 2-4)
**Phase 1 Completed:** November 28, 2025 ✅

### 2.5 Key Technical Decisions

**Decision 1: Two-Stage Pipeline**
- **Rationale:** Separates Blender-specific operations from general mesh processing
- **Benefit:** Stage 2 can work standalone for non-.blend inputs
- **Trade-off:** Added complexity in orchestration

**Decision 2: UV Preservation Strategy**
- **Rationale:** Tree assets use overlapping UVs for memory efficiency
- **Implementation:** Create secondary UV layer for baking, preserve original
- **Benefit:** Maintains artist-intended UV layouts

**Decision 3: List[_SingleGaussian] vs GaussianSplat**
- **Rationale:** List provides flexibility during conversion
- **Implementation:** Use dataclass for individual gaussians
- **Benefit:** Easier debugging and testing

**Decision 4: TDD Mandatory**
- **Rationale:** Complex 3D math requires verification at each step
- **Implementation:** Write failing tests first, minimal code to pass
- **Benefit:** High confidence, fewer regressions

---

## Next Steps

1. Read **Part 2: Phase 1 Implementation** for UV texture sampling details
2. Review current codebase to understand existing implementations
3. Set up development environment with Blender and dependencies
4. Begin Phase 1 implementation with TDD approach

---

**Continue to Part 2 for Phase 1: UV Texture Sampling Implementation**
