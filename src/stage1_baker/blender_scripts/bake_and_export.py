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
    bpy.context.scene.cycles.samples = 32    # Reduced for faster testing (was 128)
    
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

    # Convert to absolute path for Blender
    texture_path_abs = texture_path.resolve()

    img.filepath_raw = str(texture_path_abs)
    img.file_format = 'PNG'

    try:
        img.save()
        print(f"✓ Saved texture: {texture_path_abs}")
    except Exception as e:
        print(f"ERROR saving texture: {e}")
        sys.exit(1)

    # 7. UPDATE MATERIAL TO USE BAKED TEXTURE
    print("Updating material to use baked texture...")

    # Clear existing nodes
    nodes.clear()

    # Create simple material with baked texture
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (400, 0)

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    tex_node = nodes.new(type='ShaderNodeTexImage')
    tex_node.image = img
    tex_node.location = (-400, 0)

    # Connect nodes
    links = mat.node_tree.links
    links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    print("  Material updated to use baked texture")

    # 8. EXPORT OBJ WITH MTL
    obj_path = output_dir / f"{Path(bpy.data.filepath).stem}.obj"

    # Convert to absolute path for Blender
    obj_path_abs = obj_path.resolve()

    print(f"Exporting OBJ: {obj_path_abs}")

    try:
        bpy.ops.export_scene.obj(
            filepath=str(obj_path_abs),
            use_selection=False,
            use_materials=True,
            use_uvs=True,
            path_mode='RELATIVE'  # MTL references texture relatively
        )
        print(f"✓ Exported OBJ: {obj_path_abs}")
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
    # Force Blender to quit
    bpy.ops.wm.quit_blender()
except Exception as e:
    print(f"FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

