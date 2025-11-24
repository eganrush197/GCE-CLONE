# Color & Texture Enhancement Implementation Plan

## Overview
This document outlines the planned enhancements to the color/texture support system in the Gaussian Mesh Converter. These improvements will add texture sampling, better color validation, user feedback, and advanced material property support.

---

## Enhancement 1: Texture Coordinate Support

### Goal
Enable sampling of actual texture images at UV coordinates instead of just using material diffuse colors.

### Current State
- ✅ Material diffuse colors (Kd) are extracted from MTL files
- ❌ Texture images referenced in MTL files are ignored
- ❌ UV coordinates are not utilized

### Implementation Plan

#### Step 1: Add Texture Sampling Helper Method
**Location**: `src/mesh_to_gaussian.py` - Add new method to `MeshToGaussianConverter` class

```python
def _sample_texture_color(self, mesh, face_idx, barycentric_coords):
    """
    Sample texture color at barycentric coordinates on a face.
    
    Args:
        mesh: Trimesh object with UV coordinates and material
        face_idx: Index of the face to sample
        barycentric_coords: (w1, w2, w3) barycentric weights
        
    Returns:
        np.ndarray: RGB color in 0-1 range, or None if no texture available
    """
    # Check if mesh has UV coordinates
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        return None
    
    # Check if mesh has material with texture
    if not hasattr(mesh.visual, 'material') or mesh.visual.material is None:
        return None
    
    # Get UV coordinates for the face vertices
    uv_indices = mesh.faces[face_idx]
    uv_coords = mesh.visual.uv[uv_indices]
    
    # Interpolate UV using barycentric coordinates
    uv = (barycentric_coords[0] * uv_coords[0] +
          barycentric_coords[1] * uv_coords[1] +
          barycentric_coords[2] * uv_coords[2])
    
    # Sample texture image
    if hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
        from PIL import Image
        img = mesh.visual.material.image
        
        # Convert UV to pixel coordinates (handle wrapping)
        x = int(uv[0] * img.width) % img.width
        y = int((1 - uv[1]) * img.height) % img.height  # Flip Y for image coordinates
        
        # Get pixel color
        pixel = img.getpixel((x, y))
        
        # Handle different image modes (RGB, RGBA, L, etc.)
        if isinstance(pixel, int):  # Grayscale
            return np.array([pixel, pixel, pixel]) / 255.0
        else:
            return np.array(pixel[:3]) / 255.0
    
    return None
```

#### Step 2: Integrate into Face Strategy
**Location**: `mesh_to_gaussians()` method, face sampling section (around line 222)

**Current color extraction:**
```python
# Interpolate vertex colors if available
if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
    # ... existing code ...
elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
    # ... existing code ...
else:
    color = np.array([0.5, 0.5, 0.5])
```

**Enhanced version:**
```python
# Try texture sampling first (highest quality)
color = self._sample_texture_color(mesh, face_idx, (w1, w2, w3))

# Fall back to vertex color interpolation
if color is None and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
    v_colors = mesh.visual.vertex_colors[face][:, :3]
    if v_colors.max() > 1.0:
        v_colors = v_colors / 255.0
    color = (w1 * v_colors[0] + w2 * v_colors[1] + w3 * v_colors[2])

# Fall back to face colors
elif color is None and hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
    face_color = mesh.visual.face_colors[face_idx][:3]
    color = face_color / 255.0 if face_color.max() > 1.0 else face_color

# Final fallback to default gray
if color is None:
    color = np.array([0.5, 0.5, 0.5])
```

#### Step 3: Update MTL Parser for Texture References
**Location**: `_load_obj_with_mtl()` method (around line 80)

**Add texture map parsing:**
```python
elif parts[0] == 'map_Kd' and current_mat:
    # Diffuse texture map
    texture_path = ' '.join(parts[1:])  # Handle paths with spaces
    materials[current_mat]['texture'] = texture_path
```

**Then load textures and attach to mesh:**
```python
# After parsing MTL, load textures
for mat_name, mat_data in materials.items():
    if 'texture' in mat_data:
        texture_path = Path(mtl_path).parent / mat_data['texture']
        if texture_path.exists():
            from PIL import Image
            mat_data['image'] = Image.open(texture_path)
```

#### Dependencies
- PIL/Pillow (already in requirements.txt ✅)

#### Testing Plan
1. Create test OBJ with UV coordinates and texture
2. Verify texture colors are sampled correctly
3. Test with various image formats (PNG, JPG, BMP)
4. Test fallback when texture is missing

---

## Enhancement 2: Color Validation

### Goal
Ensure all colors are properly normalized and clamped to valid ranges, preventing rendering artifacts.

### Current State
- ⚠️ Color normalization is done inline in multiple places
- ⚠️ No validation that colors are in valid range after processing
- ⚠️ Potential for values outside 0-1 range

### Implementation Plan

#### Step 1: Add Color Normalization Helper
**Location**: `src/mesh_to_gaussian.py` - Add new method to `MeshToGaussianConverter` class

```python
def _normalize_color(self, color):
    """
    Ensure color is in valid 0-1 range.
    
    Args:
        color: Color array (can be 0-1 or 0-255 range)
        
    Returns:
        np.ndarray: Color normalized to 0-1 range, clipped to valid values
    """
    color = np.array(color, dtype=np.float32)
    
    # Detect if color is in 0-255 range
    if color.max() > 1.0:
        color = color / 255.0
    
    # Clamp to valid range
    color = np.clip(color, 0.0, 1.0)
    
    return color
```

#### Step 2: Replace Inline Normalization
**Locations to update:**
1. Line 169-171 (vertex strategy, vertex colors)
2. Line 176-178 (vertex strategy, face colors)
3. Line 225-227 (face strategy, vertex colors)
4. Line 234-237 (face strategy, face colors)

**Replace patterns like:**
```python
if color.max() > 1.0:
    color = color / 255.0
```

**With:**
```python
color = self._normalize_color(color)
```

#### Step 3: Add Validation Before SH Conversion
**Location**: Before creating GaussianSplat objects (lines 185, 241)

```python
# Validate color before converting to SH
color = self._normalize_color(color)
assert 0.0 <= color.min() and color.max() <= 1.0, f"Invalid color range: {color}"

gaussian = GaussianSplat(
    # ... other params ...
    sh_dc=color - 0.5  # SH DC term centered at 0
)
```

#### Testing Plan
1. Test with colors in 0-1 range
2. Test with colors in 0-255 range
3. Test with out-of-range colors (negative, >255)
4. Verify no crashes or artifacts

---

## Enhancement 3: Handle Missing Colors More Gracefully

### Goal
Provide clear feedback to users when color data is missing or incomplete, helping them understand why their output might be gray.

### Current State
- ❌ Silent fallback to gray when no colors found
- ❌ No indication of color source used
- ❌ Users don't know if colors were successfully extracted

### Implementation Plan

#### Step 1: Add Color Source Tracking
**Location**: `mesh_to_gaussians()` method

```python
# Track color sources for statistics
color_sources = {
    'texture': 0,
    'vertex_color': 0,
    'face_color': 0,
    'default_gray': 0
}
```

#### Step 2: Update Color Extraction to Track Source
**In vertex strategy:**
```python
if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
    color = mesh.visual.vertex_colors[i][:3]
    color = self._normalize_color(color)
    color_sources['vertex_color'] += 1
elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
    # ... find face color ...
    color_sources['face_color'] += 1
else:
    color = np.array([0.5, 0.5, 0.5])
    color_sources['default_gray'] += 1
```

#### Step 3: Print Color Statistics
**Location**: End of `mesh_to_gaussians()` method

```python
# Print color source statistics
print(f"Color sources:")
if color_sources['texture'] > 0:
    print(f"  ✅ Texture sampled: {color_sources['texture']} gaussians")
if color_sources['vertex_color'] > 0:
    print(f"  ✅ Vertex colors: {color_sources['vertex_color']} gaussians")
if color_sources['face_color'] > 0:
    print(f"  ✅ Face colors: {color_sources['face_color']} gaussians")
if color_sources['default_gray'] > 0:
    print(f"  ⚠️  Default gray: {color_sources['default_gray']} gaussians")

# Warning if all colors are default
if color_sources['default_gray'] == len(gaussians):
    print()
    print("⚠️  WARNING: No color data found in mesh!")
    print("   Possible solutions:")
    print("   1. Check if MTL file exists alongside OBJ file")
    print("   2. Verify MTL contains 'Kd' (diffuse color) values")
    print("   3. Check if mesh has vertex colors or textures")
    print()
```

#### Step 4: Add Mesh Inspection on Load
**Location**: `load_mesh()` method, after loading

```python
# Inspect color capabilities
print(f"Mesh color capabilities:")
has_vertex_colors = hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None
has_face_colors = hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None
has_uv = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
has_material = hasattr(mesh.visual, 'material') and mesh.visual.material is not None

print(f"  Vertex colors: {'✅' if has_vertex_colors else '❌'}")
print(f"  Face colors: {'✅' if has_face_colors else '❌'}")
print(f"  UV coordinates: {'✅' if has_uv else '❌'}")
print(f"  Material: {'✅' if has_material else '❌'}")

if not any([has_vertex_colors, has_face_colors, has_material]):
    print(f"  ⚠️  No color data detected - output will be gray")
```

---

## Enhancement 4: Support for Ambient/Specular Colors

### Goal
Utilize additional material properties (ambient, specular, emissive) to create more realistic gaussian colors.

### Current State
- ✅ Diffuse color (Kd) is extracted
- ❌ Ambient color (Ka) is ignored
- ❌ Specular color (Ks) is ignored  
- ❌ Emissive color (Ke) is ignored

### Implementation Plan

#### Step 1: Extend MTL Parser
**Location**: `_load_obj_with_mtl()` method

```python
# Current material structure
materials[current_mat] = {
    'diffuse': [0.7, 0.7, 0.7],
    'ambient': None,
    'specular': None,
    'emissive': None
}

# Parse additional properties
elif parts[0] == 'Ka' and current_mat:
    materials[current_mat]['ambient'] = [float(parts[1]), float(parts[2]), float(parts[3])]
elif parts[0] == 'Ks' and current_mat:
    materials[current_mat]['specular'] = [float(parts[1]), float(parts[2]), float(parts[3])]
elif parts[0] == 'Ke' and current_mat:
    materials[current_mat]['emissive'] = [float(parts[1]), float(parts[2]), float(parts[3])]
```

#### Step 2: Add Material Blending Function
**Location**: New method in `MeshToGaussianConverter` class

```python
def _blend_material_colors(self, material_data, blend_mode='diffuse_only'):
    """
    Blend material color components into final color.
    
    Args:
        material_data: Dict with 'diffuse', 'ambient', 'specular', 'emissive'
        blend_mode: How to combine colors
            - 'diffuse_only': Use only diffuse (current behavior)
            - 'diffuse_ambient': Blend diffuse + ambient
            - 'full': Blend all components
            
    Returns:
        np.ndarray: Final RGB color
    """
    diffuse = np.array(material_data.get('diffuse', [0.7, 0.7, 0.7]))
    
    if blend_mode == 'diffuse_only':
        return diffuse
    
    ambient = np.array(material_data.get('ambient', diffuse * 0.2))
    
    if blend_mode == 'diffuse_ambient':
        # Blend diffuse with ambient (ambient adds to shadows)
        return diffuse * 0.8 + ambient * 0.2
    
    elif blend_mode == 'full':
        specular = np.array(material_data.get('specular', [0, 0, 0]))
        emissive = np.array(material_data.get('emissive', [0, 0, 0]))
        
        # Weighted blend
        color = (diffuse * 0.7 +      # Main color
                 ambient * 0.2 +       # Shadow tint
                 specular * 0.05 +     # Highlight tint
                 emissive * 0.05)      # Glow
        
        return np.clip(color, 0, 1)
    
    return diffuse
```

#### Step 3: Add Configuration Option
**Location**: Add to `ConversionConfig` or as parameter

```python
def __init__(self, device='cuda', material_blend_mode='diffuse_only'):
    self.device = device
    self.material_blend_mode = material_blend_mode
```

#### Step 4: Use in Color Extraction
**Location**: When extracting material colors

```python
# Instead of:
color = materials[current_material]

# Use:
color = self._blend_material_colors(
    materials[current_material],
    blend_mode=self.material_blend_mode
)
```

---

## Implementation Priority

1. **HIGH**: Enhancement 2 (Color Validation) - Prevents bugs
2. **HIGH**: Enhancement 3 (Missing Colors Feedback) - Improves UX
3. **MEDIUM**: Enhancement 1 (Texture Support) - Major feature
4. **LOW**: Enhancement 4 (Ambient/Specular) - Nice to have

## Testing Strategy

### Unit Tests
- Test color normalization with various inputs
- Test texture sampling with mock images
- Test material blending calculations

### Integration Tests
- Test with real OBJ/MTL files
- Test with textured GLB files
- Test with vertex-colored meshes

### Visual Tests
- Compare output in gaussian splat viewer
- Verify colors match original mesh
- Check for artifacts or incorrect colors

## Documentation Updates Needed

1. Update `COLOR & TEXTURE SUPPORT.md` with new features
2. Add examples for texture usage
3. Document material blending modes
4. Add troubleshooting for texture issues

---

## Estimated Effort

- Enhancement 1: 4-6 hours (texture sampling is complex)
- Enhancement 2: 1-2 hours (straightforward refactoring)
- Enhancement 3: 2-3 hours (tracking and reporting)
- Enhancement 4: 2-3 hours (parsing and blending)

**Total**: 9-14 hours of development + testing

