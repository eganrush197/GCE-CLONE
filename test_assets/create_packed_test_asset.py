"""
Script to create test asset for packed texture extraction tests.

This creates a cube with:
- 2 materials (Material_Red and Material_Blue)
- Embedded textures (diffuse, roughness, transparency)
- Vertex colors (gradient)
- Proper UV mapping

Run with: blender --background --python create_packed_test_asset.py
"""

import bpy
import numpy as np
from pathlib import Path


def clear_scene():
    """Clear all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear all materials
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)
    
    # Clear all images
    for img in bpy.data.images:
        bpy.data.images.remove(img)


def create_texture_image(name, width, height, color_func):
    """
    Create a texture image procedurally.
    
    Args:
        name: Image name
        width, height: Image dimensions
        color_func: Function(x, y) -> (r, g, b, a) that generates pixel colors
    
    Returns:
        bpy.types.Image
    """
    img = bpy.data.images.new(name, width=width, height=height, alpha=True)
    
    # Generate pixels
    pixels = []
    for y in range(height):
        for x in range(width):
            r, g, b, a = color_func(x, y, width, height)
            pixels.extend([r, g, b, a])
    
    # Set pixels
    img.pixels = pixels
    
    # Pack the image (embed in .blend file)
    img.pack()
    
    return img


def create_red_material():
    """Create Material_Red with diffuse and roughness textures."""
    mat = bpy.data.materials.new(name="Material_Red")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    # Create red diffuse texture (solid red)
    red_diffuse = create_texture_image(
        "red_diffuse.png", 64, 64,
        lambda x, y, w, h: (1.0, 0.0, 0.0, 1.0)  # Solid red
    )
    
    diffuse_tex = nodes.new(type='ShaderNodeTexImage')
    diffuse_tex.image = red_diffuse
    diffuse_tex.location = (-400, 100)
    
    # Create roughness texture (0.5 gray)
    red_roughness = create_texture_image(
        "red_roughness.png", 64, 64,
        lambda x, y, w, h: (0.5, 0.5, 0.5, 1.0)  # 0.5 roughness
    )
    
    roughness_tex = nodes.new(type='ShaderNodeTexImage')
    roughness_tex.image = red_roughness
    roughness_tex.location = (-400, -200)
    
    # Connect nodes
    links.new(diffuse_tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(roughness_tex.outputs['Color'], bsdf.inputs['Roughness'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat


def create_blue_material():
    """Create Material_Blue with diffuse and transparency textures."""
    mat = bpy.data.materials.new(name="Material_Blue")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    # Create blue diffuse texture with alpha channel (checkerboard transparency)
    blue_diffuse = create_texture_image(
        "blue_diffuse.png", 64, 64,
        lambda x, y, w, h: (
            0.0, 0.0, 1.0,  # Blue color
            1.0 if ((x // 8) + (y // 8)) % 2 == 0 else 0.3  # Checkerboard alpha
        )
    )
    
    diffuse_tex = nodes.new(type='ShaderNodeTexImage')
    diffuse_tex.image = blue_diffuse
    diffuse_tex.location = (-400, 100)
    
    # Connect nodes
    links.new(diffuse_tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(diffuse_tex.outputs['Alpha'], bsdf.inputs['Alpha'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Enable transparency
    mat.blend_method = 'BLEND'

    return mat


def create_cube_with_materials():
    """Create cube and assign materials to different faces."""
    # Create cube
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    cube = bpy.context.active_object
    cube.name = "TestCube"

    # UV unwrap
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create materials
    mat_red = create_red_material()
    mat_blue = create_blue_material()

    # Assign materials to cube
    cube.data.materials.append(mat_red)
    cube.data.materials.append(mat_blue)

    # Assign materials to faces
    # Cube has 6 faces, each split into 2 triangles = 12 polygons after triangulation
    # But before triangulation, we have 6 quads
    # First 3 faces (0-2) = Material_Red (index 0)
    # Last 3 faces (3-5) = Material_Blue (index 1)
    for i, poly in enumerate(cube.data.polygons):
        if i < 3:
            poly.material_index = 0  # Material_Red
        else:
            poly.material_index = 1  # Material_Blue

    return cube


def add_vertex_colors(cube):
    """Add vertex colors to cube (gradient from white to black)."""
    mesh = cube.data

    # Create vertex color layer
    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name="Col")

    color_layer = mesh.vertex_colors.active

    # Get vertex positions for gradient calculation
    vertices = mesh.vertices

    # Find min/max Z for gradient
    z_coords = [v.co.z for v in vertices]
    z_min = min(z_coords)
    z_max = max(z_coords)
    z_range = z_max - z_min if z_max != z_min else 1.0

    # Assign colors per loop (face corner)
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            loop = mesh.loops[loop_idx]
            vertex = vertices[loop.vertex_index]

            # Gradient based on Z coordinate (white at top, black at bottom)
            t = (vertex.co.z - z_min) / z_range
            color = (t, t, t, 1.0)  # Grayscale gradient

            color_layer.data[loop_idx].color = color


def main():
    """Main function to create test asset."""
    print("\n" + "="*60)
    print("CREATING PACKED TEXTURE TEST ASSET")
    print("="*60)

    # Clear scene
    print("\n[1/4] Clearing scene...")
    clear_scene()

    # Create cube with materials
    print("[2/4] Creating cube with materials...")
    cube = create_cube_with_materials()
    print(f"  ✓ Created cube with {len(cube.data.materials)} materials")
    print(f"  ✓ Material 0: {cube.data.materials[0].name}")
    print(f"  ✓ Material 1: {cube.data.materials[1].name}")

    # Add vertex colors
    print("[3/4] Adding vertex colors...")
    add_vertex_colors(cube)
    print(f"  ✓ Added vertex color layer: {cube.data.vertex_colors.active.name}")

    # Pack all images
    print("[4/4] Packing textures...")
    bpy.ops.file.pack_all()
    packed_count = sum(1 for img in bpy.data.images if img.packed_file)
    print(f"  ✓ Packed {packed_count} textures")

    # Save file
    output_path = Path(__file__).parent / "test_packed_multi_material.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

    print("\n" + "="*60)
    print("✓ TEST ASSET CREATED SUCCESSFULLY")
    print("="*60)
    print(f"Output: {output_path}")
    print(f"\nAsset Details:")
    print(f"  - Cube with 6 faces (12 triangles when triangulated)")
    print(f"  - Material_Red: Faces 0-2 (solid red + roughness)")
    print(f"  - Material_Blue: Faces 3-5 (blue + checkerboard alpha)")
    print(f"  - Vertex colors: White-to-black gradient (top to bottom)")
    print(f"  - Textures: {packed_count} embedded images")
    print(f"  - UV Layer: UVMap (default)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

