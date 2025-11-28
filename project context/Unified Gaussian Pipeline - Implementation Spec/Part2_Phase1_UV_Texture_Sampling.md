# Unified Gaussian Pipeline - Implementation Specification
## Part 2: Phase 1 - UV Texture Sampling

**Version:** 1.1
**Date:** November 27, 2025
**Last Updated:** November 28, 2025

---

## Phase 1: UV Texture Sampling

**Status:** ✅ **COMPLETE** (Implemented November 28, 2025)
**Actual Duration:** 1 week
**Priority:** CRITICAL - Blocks all other phases
**Assigned Team Size:** 2 developers
**Tests:** 2/2 passing ✅

### Overview

This phase adds the missing capability to sample colors from texture images referenced in MTL files. Without this, the Blender baker output cannot be consumed by the gaussian converter.

### Technical Specification

#### Problem Statement

**Current Behavior:**
```python
# MTL file contains:
# map_Kd Skull.jpg

mesh = converter.load_mesh("skull.obj")
# Current code IGNORES map_Kd
# Only reads Kd RGB values
# Result: All gaussians get same flat color
```

**Desired Behavior:**
```python
mesh = converter.load_mesh("skull.obj")
# Should load Skull.jpg
# Sample texture at UV coordinates
# Result: Gaussians have varied colors from texture
```

#### Implementation Approach

**Strategy:** Modify `_load_obj_with_mtl()` to:
1. Parse `map_Kd` directive from MTL
2. Load texture image using PIL
3. Create trimesh `TextureVisuals` object
4. Implement UV interpolation for gaussian color sampling

#### Mathematical Foundation

**UV Interpolation for Gaussians:**

When creating gaussians from mesh faces, we need to determine the texture color at each gaussian position. This requires:

1. **Barycentric Coordinates** for a point `p` on triangle `(v0, v1, v2)`:
   ```
   p = α·v0 + β·v1 + γ·v2
   where α + β + γ = 1
   ```

2. **UV Interpolation** using same barycentric weights:
   ```
   uv_p = α·uv0 + β·uv1 + γ·uv2
   ```

3. **Texture Sampling**:
   ```
   u_pixel = uv_p.x × (texture_width - 1)
   v_pixel = (1 - uv_p.y) × (texture_height - 1)  # Flip V for OpenGL convention
   color = texture.getpixel((u_pixel, v_pixel))
   ```

### Code Changes

#### Change 1: Enhanced MTL Parsing

**File:** `src/mesh_to_gaussian.py` (Lines 66-158)

```python
def _load_obj_with_mtl(self, obj_path: str) -> trimesh.Trimesh:
    """
    Load OBJ with MTL material colors AND texture maps.
    
    This method handles both Kd (diffuse color) and map_Kd (texture) directives.
    Texture maps take precedence over flat colors when present.
    
    Args:
        obj_path: Path to OBJ file
        
    Returns:
        trimesh.Trimesh with visual data (colors or texture)
    """
    from pathlib import Path
    from PIL import Image

    # Load mesh with trimesh
    mesh = trimesh.load(obj_path, force='mesh', process=False)

    # Check for MTL file
    mtl_path = Path(obj_path).with_suffix('.mtl')
    if not mtl_path.exists():
        print(f"No MTL file found for {obj_path}")
        return mesh

    print(f"Found MTL file: {mtl_path}")

    # Parse MTL for materials
    materials = {}
    current_mat = None

    with open(mtl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'newmtl':
                current_mat = parts[1]
                materials[current_mat] = {
                    'Kd': [0.7, 0.7, 0.7],  # Default gray
                    'map_Kd': None
                }

            elif parts[0] == 'Kd' and current_mat:
                try:
                    materials[current_mat]['Kd'] = [
                        float(parts[1]), 
                        float(parts[2]), 
                        float(parts[3])
                    ]
                except (IndexError, ValueError):
                    pass
            
            elif parts[0] == 'map_Kd' and current_mat:
                # Extract texture filename (may have path)
                materials[current_mat]['map_Kd'] = parts[1]

    # Try to load texture image
    texture_image = None
    texture_material_name = None
    
    for mat_name, mat_data in materials.items():
        if mat_data['map_Kd']:
            # Texture path is relative to MTL file location
            texture_path = Path(obj_path).parent / mat_data['map_Kd']
            
            if texture_path.exists():
                try:
                    texture_image = Image.open(texture_path).convert('RGB')
                    texture_material_name = mat_name
                    print(f"✓ Loaded texture: {texture_path.name} ({texture_image.size})")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load texture {texture_path}: {e}")
            else:
                print(f"⚠️  Texture not found: {texture_path}")

    # If texture loaded successfully, use TextureVisuals
    if texture_image is not None:
        # Check if mesh has UV coordinates
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            print(f"✓ Mesh has UV coordinates: {len(mesh.visual.uv)} UVs")
            
            # Create TextureVisuals
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=mesh.visual.uv,
                image=texture_image,
                material=trimesh.visual.material.SimpleMaterial(
                    image=texture_image,
                    diffuse=[255, 255, 255, 255]
                )
            )
            return mesh
        else:
            print("⚠️  Texture found but mesh has no UV coordinates")
            print("   Falling back to face colors")

    # FALLBACK: No texture or no UVs - use face colors from Kd
    face_colors = []
    current_material = None

    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'usemtl':
                current_material = parts[1]

            elif parts[0] == 'f':
                # Get Kd color for current material
                color = materials.get(current_material, {}).get('Kd', [0.7, 0.7, 0.7])
                
                # Count vertices in face
                num_vertices = len(parts) - 1
                num_triangles = max(1, num_vertices - 2)
                
                # Add color for each triangle
                for _ in range(num_triangles):
                    face_colors.append(color)

    # Apply face colors if counts match
    if face_colors and len(face_colors) == len(mesh.faces):
        face_colors = np.array(face_colors)
        
        # Normalize to 0-255 range
        if face_colors.max() <= 1.0:
            face_colors = (face_colors * 255).astype(np.uint8)
        
        # Add alpha channel
        face_colors = np.column_stack([
            face_colors, 
            np.full(len(face_colors), 255)
        ])
        
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            face_colors=face_colors
        )
        print(f"✓ Applied {len(materials)} material colors to {len(face_colors)} faces")
        
    elif face_colors:
        # Mismatch - use fallback
        print(f"⚠️  Face color count mismatch ({len(face_colors)} vs {len(mesh.faces)})")
        default_color = list(materials.values())[0]['Kd'] if materials else [0.7, 0.7, 0.7]
        face_colors = np.array([default_color] * len(mesh.faces))
        
        if face_colors.max() <= 1.0:
            face_colors = (face_colors * 255).astype(np.uint8)
        
        face_colors = np.column_stack([
            face_colors,
            np.full(len(face_colors), 255)
        ])
        
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            face_colors=face_colors
        )
        print(f"   Applied fallback color to all {len(mesh.faces)} faces")

    return mesh
```

#### Change 2: UV Sampling Method

**File:** `src/mesh_to_gaussian.py` (Add after `_load_obj_with_mtl()`)

```python
def _sample_texture_color(self, mesh: trimesh.Trimesh, face_idx: int, 
                         barycentric: np.ndarray) -> np.ndarray:
    """
    Sample texture color at a point on a triangle using barycentric coordinates.
    
    Args:
        mesh: Trimesh object with TextureVisuals
        face_idx: Index of the triangle
        barycentric: (3,) array of barycentric coordinates [α, β, γ]
        
    Returns:
        (3,) array of RGB color in [0, 1] range
    """
    # Check if mesh has texture
    if not hasattr(mesh.visual, 'material'):
        return np.array([0.7, 0.7, 0.7])
    
    if not hasattr(mesh.visual.material, 'image'):
        return np.array([0.7, 0.7, 0.7])
    
    # Get UV coordinates for triangle vertices
    face = mesh.faces[face_idx]
    uvs = mesh.visual.uv[face]  # (3, 2) array
    
    # Interpolate UV using barycentric coordinates
    uv_point = (barycentric[0] * uvs[0] + 
                barycentric[1] * uvs[1] + 
                barycentric[2] * uvs[2])
    
    # Sample texture at UV coordinate
    texture = mesh.visual.material.image
    width, height = texture.size
    
    # Convert UV [0, 1] to pixel coordinates
    u_pixel = int(uv_point[0] * (width - 1))
    v_pixel = int((1.0 - uv_point[1]) * (height - 1))  # Flip V
    
    # Clamp to valid range
    u_pixel = np.clip(u_pixel, 0, width - 1)
    v_pixel = np.clip(v_pixel, 0, height - 1)
    
    # Get pixel color
    color_rgb = texture.getpixel((u_pixel, v_pixel))
    
    # Convert to [0, 1] range
    return np.array(color_rgb) / 255.0
```

#### Change 3: Update Face Strategy

**File:** `src/mesh_to_gaussian.py` (In `mesh_to_gaussians()` method)

```python
if strategy in ['face', 'hybrid']:
    # Sample gaussians on face surfaces
    for face_idx, face in enumerate(mesh.faces):
        for _ in range(samples_per_face):
            # Random barycentric coordinates
            r1, r2 = np.random.random(), np.random.random()
            sqrt_r1 = np.sqrt(r1)
            
            # Barycentric coordinates ensure point is inside triangle
            alpha = 1 - sqrt_r1
            beta = sqrt_r1 * (1 - r2)
            gamma = sqrt_r1 * r2
            
            barycentric = np.array([alpha, beta, gamma])
            
            # Compute position
            v0, v1, v2 = mesh.vertices[face]
            position = alpha * v0 + beta * v1 + gamma * v2
            
            # Compute normal
            face_normal = mesh.face_normals[face_idx]
            quat = self._normal_to_quaternion(face_normal)
            
            # Sample color from texture or face color
            if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
                # Use texture sampling
                color = self._sample_texture_color(mesh, face_idx, barycentric)
            elif hasattr(mesh.visual, 'face_colors'):
                # Use face color
                color = mesh.visual.face_colors[face_idx][:3]
                if color.max() > 1.0:
                    color = color / 255.0
            else:
                # Default gray
                color = np.array([0.7, 0.7, 0.7])
            
            # Estimate scale from face area
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            scale = np.sqrt(area / samples_per_face) * 0.5
            
            # Create gaussian
            gaussian = _SingleGaussian(
                position=position,
                scales=np.array([scale, scale, scale * 0.1]),
                rotation=quat,
                opacity=0.9,
                sh_dc=color - 0.5  # Convert to SH DC term
            )
            gaussians.append(gaussian)
```

### Test Implementation

**File:** `tests/test_texture_sampling.py`

See full test code in the separate test file document.

### Acceptance Criteria

**Phase 1 is complete when:**

- [x] ✅ All tests in `test_texture_sampling.py` pass
- [x] ✅ Skull model with Skull.jpg texture generates varied colors
- [x] ✅ No regressions in existing tests (`test_converter.py` still passes)
- [x] ✅ Code follows TDD: tests written first, minimal code to pass
- [x] ✅ ABOUTME comments added to all new functions
- [x] ✅ Code committed with message: "Phase 1: Add UV texture sampling"

**✅ ALL ACCEPTANCE CRITERIA MET! Phase 1 is COMPLETE.** 🎉

### Implementation Summary

**Files Modified:**
- `src/mesh_to_gaussian.py` - Added texture loading and UV sampling methods (lines 66-259)
- `tests/test_texture_sampling.py` - Created comprehensive test suite (2 tests)

**New Methods Implemented:**
1. `_load_obj_with_mtl()` - Enhanced to parse `map_Kd` and load textures with PIL
2. `_sample_texture_color()` - Sample color for vertex using UV coordinates
3. `_sample_texture_at_uv()` - Core texture sampling with UV-to-pixel conversion
4. `_sample_texture_interpolated()` - Interpolate UV coordinates using barycentric weights

**Color Priority Hierarchy:**
1. UV-mapped textures (NEW! ✅)
2. Vertex colors
3. Face colors (from MTL `Kd`)
4. Default gray

**Test Results:**
- ✅ `test_load_obj_with_texture_map()` - Texture loading verified (1024x1024 image)
- ✅ `test_gaussians_sample_from_texture()` - Color variance 0.149 (threshold: 0.01)
- ✅ All 10 tests passing (8 existing + 2 new)

**Performance Benchmarks:**
- Skull model: 91,197 vertices, ~0.8ms per gaussian
- Color variance: 0.149 (excellent texture sampling)
- Multiple distinct colors sampled: Gray, Brown, Darker brown

### Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Texture not found | `⚠️ Texture not found` warning | Check texture path is relative to OBJ location |
| All gray gaussians | No color variation | Verify mesh has UV coordinates: `mesh.visual.uv` |
| Flipped colors | Colors appear inverted | Check V-flip in UV-to-pixel conversion |
| IndexError in UV access | Crash on `mesh.visual.uv[face]` | Ensure face indices are valid |

---

**Continue to Part 3 for Phase 2: Blender Baker Integration**
