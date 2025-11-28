# Unified Gaussian Pipeline - Implementation Specification
## Part 3: Phase 2 - Blender Baker Integration

**Version:** 1.0  
**Date:** November 27, 2025

---

## Phase 2: Blender Baker Integration

**Estimated Duration:** 1-2 weeks  
**Priority:** HIGH - Enables procedural shader support  
**Assigned Team Size:** 2-3 developers  
**Dependencies:** Phase 1 must be complete

### Overview

This phase implements the Blender subprocess integration that bakes procedural shaders to textures. This is the "Stage 1" of the unified pipeline.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    BLENDER BAKER ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────────┘

USER PROCESS (Python)
    │
    ├─ BlenderBaker.bake()
    │   │
    │   ├─ Validate .blend file
    │   ├─ Create temp directory
    │   ├─ Build subprocess command
    │   │
    │   ▼
    │  ┌────────────────────────────────┐
    │  │  subprocess.run([              │
    │  │    blender.exe,                │
    │  │    -b, file.blend,             │
    │  │    -P, bake_and_export.py,     │
    │  │    --, temp_dir                │
    │  │  ])                            │
    │  └────────────────┬───────────────┘
    │                   │
    │  ┌────────────────▼───────────────┐
    │  │   BLENDER PROCESS              │
    │  │   (Headless)                   │
    │  │                                │
    │  │   bake_and_export.py:          │
    │  │   1. Select all mesh objects   │
    │  │   2. Preserve original UVs     │
    │  │   3. Create secondary UV layer │
    │  │   4. Bake shaders → 4K texture │
    │  │   5. Save texture as PNG       │
    │  │   6. Export OBJ + MTL          │
    │  └────────────────┬───────────────┘
    │                   │
    │                   ▼
    │   TEMP DIRECTORY
    │   ├─ baked_texture.png  (4096x4096)
    │   ├─ model.obj
    │   └─ model.mtl  (references texture)
    │
    ├─ BlenderBaker.validate_output()
    │   ├─ Check OBJ exists
    │   ├─ Check texture exists
    │   └─ Check file sizes
    │
    └─ Return: (obj_path, texture_path)
```

### UV Preservation Strategy

**Problem:** Tree assets often use overlapping UVs where multiple leaves share the same UV space for memory efficiency.

**Naive Approach (WRONG):**
```python
# This destroys overlapping UVs
bpy.ops.uv.smart_project()
```

**Correct Approach:**
```python
# 1. Detect existing UVs
if mesh.uv_layers:
    original_uv = mesh.uv_layers.active.name
    
    # 2. Create secondary UV layer
    bake_uv = mesh.uv_layers.new(name="BakeUV")
    mesh.uv_layers.active = bake_uv
    
    # 3. Unwrap ONLY the new layer
    bpy.ops.uv.smart_project(island_margin=0.02)

# During baking:
# - Shader reads from original UV layer
# - Bake writes to active UV layer (BakeUV)
```

**Why This Works:**
- Blender's shader evaluator reads `uv_layers[0]` (original UVs)
- Blender's baker writes to `uv_layers.active` (BakeUV)
- Original topology preserved for shader evaluation
- New layout optimized for bake coverage

### Implementation Files

#### File 1: `src/stage1_baker/__init__.py`

```python
# ABOUTME: Stage 1 - Blender baker module
# ABOUTME: Exports BlenderBaker class for procedural shader baking

from .baker import BlenderBaker

__all__ = ['BlenderBaker']
```

#### File 2: `src/stage1_baker/baker.py`

```python
# ABOUTME: Python wrapper for Blender subprocess operations
# ABOUTME: Manages headless Blender execution for shader baking

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional
import time


class BlenderBaker:
    """
    Manages Blender subprocess for baking procedural shaders to textures.
    
    This class provides a Python interface to invoke Blender headlessly,
    execute baking scripts, and return the resulting OBJ + texture files.
    
    Usage:
        baker = BlenderBaker(blender_executable="/path/to/blender")
        obj_path, texture_path = baker.bake("model.blend", output_dir="./temp")
    """
    
    def __init__(self, blender_executable: str = "blender"):
        """
        Initialize Blender baker.
        
        Args:
            blender_executable: Path to Blender executable.
                               Default: "blender" (assumes in PATH)
        """
        self.blender_exe = blender_executable
        self._validate_blender()
        
        # Path to the Blender Python script
        self.script_path = Path(__file__).parent / "blender_scripts" / "bake_and_export.py"
        
        if not self.script_path.exists():
            raise FileNotFoundError(f"Bake script not found: {self.script_path}")
    
    def _validate_blender(self):
        """Verify Blender executable exists and is callable."""
        try:
            result = subprocess.run(
                [self.blender_exe, "--version"],
                capture_output=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"Blender executable failed: {self.blender_exe}")
            
            version_info = result.stdout.decode('utf-8')
            print(f"✓ Found Blender: {version_info.split()[1]}")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Blender not found: {self.blender_exe}\n"
                f"Please install Blender or specify correct path."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Blender --version timed out")
    
    def bake(self, 
             blend_file: str, 
             output_dir: Optional[str] = None,
             texture_resolution: int = 4096,
             timeout: int = 600) -> Tuple[Path, Path]:
        """
        Bake procedural shaders from .blend file to texture + OBJ.
        
        Args:
            blend_file: Path to input .blend file
            output_dir: Output directory (default: temp directory)
            texture_resolution: Texture size in pixels (default: 4096)
            timeout: Max baking time in seconds (default: 600)
            
        Returns:
            Tuple of (obj_path, texture_path)
            
        Raises:
            FileNotFoundError: If blend file doesn't exist
            RuntimeError: If baking fails
            subprocess.TimeoutExpired: If baking exceeds timeout
        """
        blend_path = Path(blend_file)
        
        if not blend_path.exists():
            raise FileNotFoundError(f"Blend file not found: {blend_file}")
        
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(tempfile.mkdtemp(prefix="blender_bake_"))
        
        print(f"Baking {blend_path.name} to {output_path}")
        print(f"Texture resolution: {texture_resolution}x{texture_resolution}")
        
        # Expected output files
        obj_path = output_path / f"{blend_path.stem}.obj"
        texture_path = output_path / "baked_texture.png"
        
        # Build Blender command
        cmd = [
            str(self.blender_exe),
            "-b",                           # Background mode
            str(blend_path),                # Input file
            "-P", str(self.script_path),    # Python script
            "--",                           # Separator for script args
            str(output_path),               # Output directory
            str(texture_resolution)         # Texture size
        ]
        
        # Execute Blender
        start_time = time.time()
        
        try:
            print("Starting Blender baker (this may take 1-5 minutes)...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                text=True
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                raise RuntimeError(
                    f"Blender baking failed with code {result.returncode}\n"
                    f"Check logs above for details"
                )
            
            print(f"✓ Baking complete in {elapsed:.1f}s")
            
            # Validate output
            self._validate_output(obj_path, texture_path)
            
            return obj_path, texture_path
            
        except subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired(
                cmd, timeout,
                f"Blender baking exceeded {timeout}s timeout"
            )
    
    def _validate_output(self, obj_path: Path, texture_path: Path):
        """
        Verify that baking produced valid output files.
        
        Raises:
            RuntimeError: If output is invalid
        """
        # Check OBJ exists
        if not obj_path.exists():
            raise RuntimeError(f"OBJ not created: {obj_path}")
        
        # Check texture exists
        if not texture_path.exists():
            raise RuntimeError(f"Texture not created: {texture_path}")
        
        # Check file sizes
        obj_size = obj_path.stat().st_size
        texture_size = texture_path.stat().st_size
        
        if obj_size < 100:
            raise RuntimeError(f"OBJ too small ({obj_size} bytes), likely empty")
        
        if texture_size < 1000:
            raise RuntimeError(f"Texture too small ({texture_size} bytes), likely failed")
        
        print(f"✓ OBJ: {obj_size / 1e6:.1f} MB")
        print(f"✓ Texture: {texture_size / 1e6:.1f} MB")
    
    def cleanup_temp(self, path: Path):
        """Delete temporary baking directory."""
        if path.exists() and "blender_bake_" in str(path):
            shutil.rmtree(path)
            print(f"✓ Cleaned up temp dir: {path}")
```

#### File 3: `src/stage1_baker/blender_scripts/bake_and_export.py`

```python
# ABOUTME: Blender Python script for procedural shader baking
# ABOUTME: Preserves UV topology, bakes to texture, exports OBJ

import bpy
import sys
from pathlib import Path

# Get arguments passed after "--"
argv = sys.argv
argv = argv[argv.index("--") + 1:]

if len(argv) < 1:
    print("ERROR: No output directory provided")
    sys.exit(1)

output_dir = Path(argv[0])
texture_resolution = int(argv[1]) if len(argv) > 1 else 4096

print(f"Blender Baker Script")
print(f"Output directory: {output_dir}")
print(f"Texture resolution: {texture_resolution}")


def bake_and_export():
    """Main baking function."""
    
    # 1. SELECT AND PREPARE MESHES
    bpy.ops.object.select_all(action='DESELECT')
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not mesh_objects:
        print("ERROR: No mesh objects found in scene")
        sys.exit(1)
    
    print(f"Found {len(mesh_objects)} mesh object(s)")
    
    # Select all meshes
    for obj in mesh_objects:
        obj.select_set(True)
    
    bpy.context.view_layer.objects.active = mesh_objects[0]
    
    # Join into single mesh (optional - depends on asset)
    if len(mesh_objects) > 1:
        print("Joining multiple meshes...")
        bpy.ops.object.join()
    
    active_obj = bpy.context.active_object
    mesh = active_obj.data
    
    # 2. UV LAYER MANAGEMENT - PRESERVE EXISTING UVS
    print("Managing UV layers...")
    
    original_uv = None
    
    if mesh.uv_layers:
        print(f"  Found {len(mesh.uv_layers)} existing UV layer(s)")
        original_uv = mesh.uv_layers.active.name
        
        # Create secondary UV layer for baking
        bake_uv = mesh.uv_layers.new(name="BakeUV")
        mesh.uv_layers.active = bake_uv
        
        print(f"  Created secondary UV layer: {bake_uv.name}")
        print(f"  Original UV preserved: {original_uv}")
        
        # Unwrap the new layer only
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        print("  Unwrapped BakeUV layer")
    else:
        print("  No existing UVs - creating new UV map")
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # 3. SETUP MATERIAL FOR BAKING
    print("Setting up material for baking...")
    
    # Create or use existing material
    if not active_obj.data.materials:
        mat = bpy.data.materials.new(name="BakeMaterial")
        active_obj.data.materials.append(mat)
        mat.use_nodes = True
    else:
        mat = active_obj.data.materials[0]
        mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    
    # 4. CREATE IMAGE FOR BAKING
    print(f"Creating {texture_resolution}x{texture_resolution} bake target...")
    
    img_name = "BakedTexture"
    
    # Remove existing image if present
    if img_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[img_name])
    
    img = bpy.data.images.new(
        img_name, 
        width=texture_resolution, 
        height=texture_resolution,
        alpha=True
    )
    
    # Add Image Texture node for baking target
    bake_node = nodes.new(type='ShaderNodeTexImage')
    bake_node.image = img
    bake_node.select = True
    nodes.active = bake_node  # Critical: marks bake target
    
    print("  Bake target node created and activated")
    
    # 5. EXECUTE BAKE
    print("Starting bake operation (this may take several minutes)...")
    
    # Set render engine to Cycles for better quality
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'  # Use GPU if available
    bpy.context.scene.cycles.samples = 128   # Reduce for speed, increase for quality
    
    try:
        bpy.ops.object.bake(
            type='DIFFUSE',
            pass_filter={'COLOR'},
            margin=16,              # Padding around UV islands
            use_clear=True,
            save_mode='INTERNAL'
        )
        print("✓ Bake complete!")
    except Exception as e:
        print(f"ERROR during baking: {e}")
        sys.exit(1)
    
    # 6. SAVE TEXTURE
    texture_path = output_dir / "baked_texture.png"
    img.filepath_raw = str(texture_path)
    img.file_format = 'PNG'
    
    try:
        img.save()
        print(f"✓ Saved texture: {texture_path}")
    except Exception as e:
        print(f"ERROR saving texture: {e}")
        sys.exit(1)
    
    # 7. EXPORT OBJ WITH MTL
    obj_path = output_dir / f"{Path(bpy.data.filepath).stem}.obj"
    
    print(f"Exporting OBJ: {obj_path}")
    
    try:
        bpy.ops.export_scene.obj(
            filepath=str(obj_path),
            use_selection=False,
            use_materials=True,
            use_uvs=True,
            path_mode='RELATIVE'  # MTL references texture relatively
        )
        print(f"✓ Exported OBJ: {obj_path}")
    except Exception as e:
        print(f"ERROR exporting OBJ: {e}")
        sys.exit(1)
    
    print("="*60)
    print("BAKING COMPLETE")
    print(f"Output directory: {output_dir}")
    print(f"  - {obj_path.name}")
    print(f"  - {texture_path.name}")
    print("="*60)


# Execute
try:
    bake_and_export()
    sys.exit(0)
except Exception as e:
    print(f"FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
```

### Acceptance Criteria

**Phase 2 is complete when:**

- [ ] All tests in `test_baker.py` pass
- [ ] Real .blend file with procedural shaders bakes successfully
- [ ] Baked output loads correctly in Phase 1's texture sampler
- [ ] UV topology preserved (test with overlapping UV asset)
- [ ] Code committed with message: "Phase 2: Add Blender baker integration"
- [ ] Documentation updated with Blender installation requirements

### Performance Benchmarks

| Asset Type | Vertices | Bake Time | Texture Size | OBJ Size |
|-----------|----------|-----------|--------------|----------|
| Simple cube | 8 | 15-30s | 4 MB | 1 KB |
| Tree (medium) | 5K | 60-120s | 15 MB | 500 KB |
| Tree (complex) | 20K | 180-300s | 20 MB | 2 MB |

---

**Continue to Part 4 for Phase 3: Pipeline Orchestrator**
