# ABOUTME: Blender Python script for extracting packed textures from .blend files
# ABOUTME: Creates material manifest JSON and exports mesh with UVs preserved

import bpy  # type: ignore  # Only available inside Blender
import sys
import os
import json
import time
from pathlib import Path


class Timer:
    """Simple timer for Blender script."""

    def __init__(self, name):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        print(f"[TIMER] {self.name} started...")
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"[OK] {self.name} complete in {self.elapsed:.1f}s")


# Get arguments passed after "--"
argv = sys.argv
argv = argv[argv.index("--") + 1:]

if len(argv) < 1:
    print("ERROR: No output directory provided")
    sys.exit(1)

output_dir = Path(argv[0])
uv_layer_name = argv[1] if len(argv) > 1 else 'uv0'

print(f"Packed Texture Extractor Script")
print(f"Output directory: {output_dir}")
print(f"UV layer to use: {uv_layer_name}")
print("=" * 60)


def find_texture_node_for_input(node_tree, input_socket):
    """
    Trace back from an input socket to find the connected Image Texture node.
    Handles intermediate nodes like Mix RGB, Color Ramp, etc.

    Returns: (image, image_node, uv_layer_name) or (None, None, None)
    """
    if not input_socket.is_linked:
        return None, None, None

    source_node = input_socket.links[0].from_node

    # Direct image texture connection
    if source_node.type == 'TEX_IMAGE' and source_node.image:
        # Find the UV layer used by this texture node
        uv_layer_name = None

        # Check if the Vector input is connected to a UV Map node
        vector_input = source_node.inputs.get('Vector')
        if vector_input and vector_input.is_linked:
            uv_node = vector_input.links[0].from_node
            if uv_node.type == 'UVMAP':
                uv_layer_name = uv_node.uv_map

        return source_node.image, source_node, uv_layer_name

    # Handle intermediate nodes - trace further back
    # Common patterns: MixRGB, ColorRamp, Math, etc.
    traceable_types = {'MIX_RGB', 'MIX', 'VALTORGB', 'MATH', 'SEPARATE_COLOR',
                       'COMBINE_COLOR', 'INVERT', 'HUE_SAT', 'BRIGHT_CONTRAST',
                       'GAMMA', 'CURVE_RGB', 'SEPRGB', 'COMBRGB'}

    if source_node.type in traceable_types:
        # Try to find image texture in any color/value input
        for inp in source_node.inputs:
            if inp.type in {'RGBA', 'VALUE', 'VECTOR'}:
                img, node, uv_layer = find_texture_node_for_input(node_tree, inp)
                if img:
                    return img, node, uv_layer

    return None, None, None


def analyze_material(mat):
    """
    Analyze a material to find which textures are used for each PBR channel.

    Returns dict with:
        - diffuse: image name or None
        - diffuse_uv_layer: UV layer name for diffuse texture or None
        - transparency: image name or None (separate texture)
        - transparency_uv_layer: UV layer name for transparency texture or None
        - diffuse_has_alpha: bool (if diffuse texture has alpha channel for transparency)
        - roughness: image name or None
        - roughness_uv_layer: UV layer name for roughness texture or None
        - is_glossy: bool (True if roughness texture should be inverted)
        - normal: image name or None
        - normal_uv_layer: UV layer name for normal texture or None
    """
    result = {
        'diffuse': None,
        'diffuse_uv_layer': None,
        'transparency': None,
        'transparency_uv_layer': None,
        'diffuse_has_alpha': False,
        'roughness': None,
        'roughness_uv_layer': None,
        'is_glossy': False,
        'normal': None,
        'normal_uv_layer': None
    }

    if not mat or not mat.use_nodes:
        return result

    nodes = mat.node_tree.nodes

    # Find the Principled BSDF node
    principled = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled = node
            break

    if not principled:
        # Try to find any shader and trace back
        return result

    # DIFFUSE: Check Base Color input
    base_color_input = principled.inputs.get('Base Color')
    if base_color_input:
        img, tex_node, uv_layer = find_texture_node_for_input(mat.node_tree, base_color_input)
        if img:
            result['diffuse'] = img.name
            result['diffuse_uv_layer'] = uv_layer
            # Check if this image has alpha channel
            if img.channels == 4:
                result['diffuse_has_alpha'] = True

    # TRANSPARENCY: Check Alpha input, or Mix Shader with Transparent BSDF
    alpha_input = principled.inputs.get('Alpha')
    if alpha_input:
        img, tex_node, uv_layer = find_texture_node_for_input(mat.node_tree, alpha_input)
        if img:
            result['transparency'] = img.name
            result['transparency_uv_layer'] = uv_layer

    # Also check for Mix Shader with Transparent BSDF pattern
    # (Common in SpeedTree materials)
    for node in nodes:
        if node.type == 'MIX_SHADER':
            fac_input = node.inputs.get('Fac')
            if fac_input:
                img, tex_node, uv_layer = find_texture_node_for_input(mat.node_tree, fac_input)
                if img and not result['transparency']:
                    result['transparency'] = img.name
                    result['transparency_uv_layer'] = uv_layer

    # ROUGHNESS: Check Roughness input
    roughness_input = principled.inputs.get('Roughness')
    if roughness_input:
        img, tex_node, uv_layer = find_texture_node_for_input(mat.node_tree, roughness_input)
        if img:
            result['roughness'] = img.name
            result['roughness_uv_layer'] = uv_layer
            # Check if it's a glossiness map (should be inverted)
            img_name_lower = img.name.lower()
            if 'gloss' in img_name_lower or 'glossy' in img_name_lower or 'glossiness' in img_name_lower:
                result['is_glossy'] = True

    # NORMAL: Check Normal input (usually through Normal Map node)
    normal_input = principled.inputs.get('Normal')
    if normal_input and normal_input.is_linked:
        normal_source = normal_input.links[0].from_node
        if normal_source.type == 'NORMAL_MAP':
            color_input = normal_source.inputs.get('Color')
            if color_input:
                img, tex_node, uv_layer = find_texture_node_for_input(mat.node_tree, color_input)
                if img:
                    result['normal'] = img.name
                    result['normal_uv_layer'] = uv_layer

    return result


def extract_packed_textures(output_path):
    """
    Extract all packed textures to the output directory.

    Returns dict mapping image name -> extracted file path
    """
    textures_dir = output_path / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)

    extracted = {}

    for img in bpy.data.images:
        # Skip non-file images (generated, viewer, etc.)
        if img.source not in {'FILE', 'GENERATED'}:
            continue

        # Skip images without data
        if not img.has_data and not img.packed_file:
            print(f"  [WARN] Skipping {img.name} - no data")
            continue

        # Determine output filename - prefer image name for reliability
        original_name = img.name if img.name else Path(img.filepath).name

        # Step 1: Remove Blender's duplicate suffix if present (.001, .002, etc.)
        # These appear AFTER the extension: "texture.png.001"
        base_name = original_name
        if '.' in original_name:
            name_without_suffix, potential_suffix = original_name.rsplit('.', 1)
            if potential_suffix.isdigit() and len(potential_suffix) == 3:
                base_name = name_without_suffix

        # Step 2: Ensure we have a valid image extension
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tga', '.exr', '.bmp', '.tiff')
        if not base_name.lower().endswith(valid_extensions):
            base_name += '.png'

        output_file = textures_dir / base_name

        # Handle duplicates by adding number
        counter = 1
        while output_file.exists():
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix
            output_file = textures_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        try:
            if img.packed_file:
                # Unpack to specific location
                print(f"  [UNPACK] {img.name} -> {output_file.name}")
                img.unpack(method='WRITE_LOCAL')

                # Find where Blender unpacked it and move to our location
                unpacked_path = bpy.path.abspath(img.filepath)
                if Path(unpacked_path).exists() and str(unpacked_path) != str(output_file):
                    import shutil
                    shutil.move(unpacked_path, output_file)
            else:
                # Image is external - save a copy
                print(f"  [SAVE] {img.name} -> {output_file.name}")
                # Set filepath and save
                original_filepath = img.filepath
                img.filepath_raw = str(output_file)
                img.save()
                img.filepath_raw = original_filepath

            extracted[img.name] = str(output_file)

        except Exception as e:
            print(f"  [ERROR] Failed to extract {img.name}: {e}")
            # Try alternative method - save directly
            try:
                img.filepath_raw = str(output_file)
                img.file_format = 'PNG'
                img.save()
                extracted[img.name] = str(output_file)
                print(f"  [OK] Saved via alternative method: {output_file.name}")
            except Exception as e2:
                print(f"  [ERROR] Alternative method also failed: {e2}")

    return extracted


def export_vertex_colors(mesh_obj, output_path):
    """
    Export vertex colors from mesh as numpy array.

    Vertex colors in Blender are stored per-loop (face corner), not per-vertex.
    This is important for proper interpolation.
    The mesh is triangulated before export to match the OBJ export.

    Args:
        mesh_obj: Blender mesh object
        output_path: Directory to save vertex_colors.npy

    Returns:
        Path to saved file, or None if no vertex colors found
    """
    import numpy as np
    import bmesh

    mesh = mesh_obj.data

    # Check if mesh has vertex color layers
    if not mesh.vertex_colors or len(mesh.vertex_colors) == 0:
        print("  [INFO] No vertex color layers found")
        return None

    # Create a bmesh from the object and triangulate it
    # This ensures vertex color export matches the triangulated OBJ export
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    # Create a temporary mesh to hold the triangulated data
    temp_mesh = bpy.data.meshes.new("temp_triangulated_vc")
    bm.to_mesh(temp_mesh)
    bm.free()

    # Check if temp mesh has vertex color layers
    if not temp_mesh.vertex_colors or len(temp_mesh.vertex_colors) == 0:
        print("  [INFO] No vertex color layers found after triangulation")
        bpy.data.meshes.remove(temp_mesh)
        return None

    # Use the active vertex color layer
    color_layer = temp_mesh.vertex_colors.active
    if not color_layer:
        color_layer = temp_mesh.vertex_colors[0]

    print(f"  [EXTRACT] Vertex color layer: '{color_layer.name}'")

    # Vertex colors are stored per-loop (face corner)
    # Each polygon has N loops (3 for triangles after triangulation)
    num_loops = len(temp_mesh.loops)
    print(f"  [INFO] Triangulated mesh has {num_loops} loops (face corners)")

    # Extract RGBA colors for each loop
    loop_colors = np.zeros((num_loops, 4), dtype=np.float32)

    for i, loop in enumerate(temp_mesh.loops):
        # Get color for this loop
        color = color_layer.data[i].color  # RGBA tuple
        loop_colors[i] = [color[0], color[1], color[2], color[3]]

    # Save as numpy array
    output_file = output_path / "vertex_colors.npy"
    np.save(output_file, loop_colors)

    print(f"  [OK] Saved vertex colors: {output_file.name} ({loop_colors.shape})")

    # Print color statistics
    avg_color = loop_colors.mean(axis=0)
    print(f"  [INFO] Average color: RGBA({avg_color[0]:.3f}, {avg_color[1]:.3f}, "
          f"{avg_color[2]:.3f}, {avg_color[3]:.3f})")

    # Clean up temporary mesh
    bpy.data.meshes.remove(temp_mesh)

    return output_file


def export_uv_layers(mesh_obj, output_path, manifest):
    """
    Export all UV layers referenced in the manifest as numpy arrays.

    UV coordinates are stored per-loop (face corner), matching vertex colors.
    The mesh is triangulated before export to match the OBJ export.

    Args:
        mesh_obj: Blender mesh object
        output_path: Directory to save UV layer files
        manifest: Material manifest to determine which UV layers are needed

    Returns:
        Dict mapping UV layer name -> file path
    """
    import numpy as np
    import bmesh

    # Collect all UV layers referenced in the manifest
    required_uv_layers = set()
    for mat_name, textures in manifest['materials'].items():
        for tex_type in ['diffuse', 'transparency', 'roughness', 'normal']:
            if textures.get(tex_type) and isinstance(textures[tex_type], dict):
                uv_layer = textures[tex_type].get('uv_layer')
                if uv_layer:
                    required_uv_layers.add(uv_layer)

    # Also add the default UV layer
    if manifest.get('uv_layer'):
        required_uv_layers.add(manifest['uv_layer'])

    if not required_uv_layers:
        print("  [INFO] No UV layers to export")
        return {}

    print(f"\n[EXPORT] Exporting {len(required_uv_layers)} UV layer(s)...")

    # Create a bmesh from the object and triangulate it
    # This ensures UV export matches the triangulated OBJ export
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    # Create a temporary mesh to hold the triangulated data
    temp_mesh = bpy.data.meshes.new("temp_triangulated")
    bm.to_mesh(temp_mesh)
    bm.free()

    # Check available UV layers in mesh
    available_uv_layers = {uv.name for uv in temp_mesh.uv_layers}
    print(f"  [INFO] Available UV layers: {sorted(available_uv_layers)}")

    exported_uv_files = {}
    num_loops = len(temp_mesh.loops)

    for uv_layer_name in sorted(required_uv_layers):
        if uv_layer_name not in available_uv_layers:
            print(f"  [WARN] UV layer '{uv_layer_name}' not found in mesh, skipping")
            continue

        uv_layer = temp_mesh.uv_layers[uv_layer_name]

        # Extract UV coordinates for each loop
        loop_uvs = np.zeros((num_loops, 2), dtype=np.float32)

        for i, loop in enumerate(temp_mesh.loops):
            uv = uv_layer.data[i].uv
            loop_uvs[i] = [uv[0], uv[1]]

        # Save as numpy array
        output_file = output_path / f"{uv_layer_name}.npy"
        np.save(output_file, loop_uvs)

        exported_uv_files[uv_layer_name] = str(output_file)
        print(f"  [OK] Saved UV layer '{uv_layer_name}': {output_file.name} ({loop_uvs.shape})")

    # Clean up temporary mesh
    bpy.data.meshes.remove(temp_mesh)

    return exported_uv_files


def get_face_material_indices(mesh_obj):
    """
    Get a list of which material index each face uses.

    Returns: list of material indices, one per face
    """
    mesh = mesh_obj.data
    return [poly.material_index for poly in mesh.polygons]


def build_material_manifest(mesh_objects, extracted_textures, uv_layer):
    """
    Build the material manifest mapping materials to their textures.

    Returns dict structure for JSON export
    """
    manifest = {
        'uv_layer': uv_layer,
        'materials': {},
        'face_materials': []  # List of material names per face (for merged mesh)
    }

    # Collect all materials from all mesh objects
    all_materials = set()
    for obj in mesh_objects:
        for mat in obj.data.materials:
            if mat:
                all_materials.add(mat)

    print(f"\n[ANALYZE] Analyzing {len(all_materials)} materials...")

    for mat in all_materials:
        print(f"\n  Material: {mat.name}")

        analysis = analyze_material(mat)

        # Map image names to extracted file paths
        # New structure: each texture is a dict with 'path' and 'uv_layer'
        textures = {
            'diffuse': None,
            'transparency': None,
            'diffuse_has_alpha': analysis['diffuse_has_alpha'],
            'roughness': None,
            'is_glossy': analysis['is_glossy'],
            'normal': None
        }

        if analysis['diffuse'] and analysis['diffuse'] in extracted_textures:
            textures['diffuse'] = {
                'path': extracted_textures[analysis['diffuse']],
                'uv_layer': analysis['diffuse_uv_layer'] or uv_layer  # Fallback to default
            }
            uv_note = f" (UV: {textures['diffuse']['uv_layer']})" if textures['diffuse']['uv_layer'] else ""
            print(f"    [OK] Diffuse: {Path(textures['diffuse']['path']).name}{uv_note}")

        if analysis['transparency'] and analysis['transparency'] in extracted_textures:
            textures['transparency'] = {
                'path': extracted_textures[analysis['transparency']],
                'uv_layer': analysis['transparency_uv_layer'] or uv_layer
            }
            uv_note = f" (UV: {textures['transparency']['uv_layer']})" if textures['transparency']['uv_layer'] else ""
            print(f"    [OK] Transparency: {Path(textures['transparency']['path']).name}{uv_note}")
        elif analysis['diffuse_has_alpha']:
            print(f"    [OK] Transparency: Using diffuse alpha channel")

        if analysis['roughness'] and analysis['roughness'] in extracted_textures:
            textures['roughness'] = {
                'path': extracted_textures[analysis['roughness']],
                'uv_layer': analysis['roughness_uv_layer'] or uv_layer
            }
            gloss_note = " (glossy, will invert)" if analysis['is_glossy'] else ""
            uv_note = f" (UV: {textures['roughness']['uv_layer']})" if textures['roughness']['uv_layer'] else ""
            print(f"    [OK] Roughness: {Path(textures['roughness']['path']).name}{gloss_note}{uv_note}")

        if analysis['normal'] and analysis['normal'] in extracted_textures:
            textures['normal'] = {
                'path': extracted_textures[analysis['normal']],
                'uv_layer': analysis['normal_uv_layer'] or uv_layer
            }
            uv_note = f" (UV: {textures['normal']['uv_layer']})" if textures['normal']['uv_layer'] else ""
            print(f"    [OK] Normal: {Path(textures['normal']['path']).name}{uv_note}")

        manifest['materials'][mat.name] = textures

    return manifest


def export_mesh_with_materials(output_path, uv_layer):
    """
    Export the mesh as OBJ with materials and UV coordinates preserved.
    """
    # Get all mesh objects
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    if not mesh_objects:
        print("ERROR: No mesh objects found in scene")
        sys.exit(1)

    print(f"\n[EXPORT] Found {len(mesh_objects)} mesh object(s)")

    # Select all mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    # Join if multiple meshes
    if len(mesh_objects) > 1:
        print("  Joining meshes...")
        bpy.ops.object.join()

    active_obj = bpy.context.active_object
    mesh = active_obj.data

    # Ensure the specified UV layer is active
    actual_uv_layer = uv_layer
    if uv_layer in mesh.uv_layers:
        mesh.uv_layers.active = mesh.uv_layers[uv_layer]
        print(f"  [OK] Using UV layer: {uv_layer}")
    else:
        # List available UV layers
        available = [uv.name for uv in mesh.uv_layers]
        if available:
            mesh.uv_layers.active = mesh.uv_layers[0]
            actual_uv_layer = available[0]  # Update to actual layer name
            print(f"  [WARN] UV layer '{uv_layer}' not found. Using '{actual_uv_layer}' instead.")
            print(f"    Available: {available}")
        else:
            print("  [WARN] No UV layers found!")
            actual_uv_layer = None

    # Triangulate the mesh ONCE before export
    # This ensures face materials mapping matches the exported OBJ exactly
    # (Previously we triangulated separately for mapping and export, causing mismatches)
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    # Apply the triangulated bmesh back to the actual mesh
    # This modifies the joined mesh (not the original source meshes)
    bm.to_mesh(mesh)
    bm.free()

    # Update mesh to reflect changes
    mesh.update()

    print(f"  [INFO] Triangulated mesh: {len(mesh.polygons)} faces")

    # Build face-to-material mapping from the now-triangulated mesh
    face_materials = []
    for poly in mesh.polygons:
        mat_idx = poly.material_index
        if mat_idx < len(active_obj.data.materials) and active_obj.data.materials[mat_idx]:
            face_materials.append(active_obj.data.materials[mat_idx].name)
        else:
            face_materials.append(None)

    print(f"  [INFO] Built face materials mapping for {len(face_materials)} triangulated faces")

    # Export OBJ
    obj_path = output_path / f"{Path(bpy.data.filepath).stem}.obj"

    print(f"  Exporting to: {obj_path}")

    # Use appropriate export operator based on Blender version
    # Blender 3.2+ uses wm.obj_export, older versions use export_scene.obj
    blender_version = bpy.app.version
    use_new_exporter = blender_version >= (3, 2, 0)

    if use_new_exporter:
        # Blender 3.2+ (new exporter)
        # Note: export_triangulated_mesh=False because mesh is already triangulated above
        bpy.ops.wm.obj_export(
            filepath=str(obj_path),
            export_selected_objects=True,
            export_uv=True,
            export_normals=True,
            export_materials=True,
            export_triangulated_mesh=False,
            path_mode='RELATIVE'
        )
    else:
        # Blender 3.1 and older (legacy exporter)
        # Note: use_triangles=False because mesh is already triangulated above
        bpy.ops.export_scene.obj(
            filepath=str(obj_path),
            use_selection=True,
            use_uvs=True,
            use_normals=True,
            use_materials=True,
            use_triangles=False,
            path_mode='RELATIVE'
        )

    print(f"  [OK] Mesh exported: {obj_path.name}")

    # Return the active (joined) object, not the original list, and the actual UV layer used
    return obj_path, [active_obj], face_materials, actual_uv_layer


def main():
    """Main extraction function."""
    print("\n" + "=" * 60)
    print("PACKED TEXTURE EXTRACTION")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract all packed textures
    with Timer("Texture extraction"):
        extracted_textures = extract_packed_textures(output_dir)
        print(f"\n  [OK] Extracted {len(extracted_textures)} textures")

    # 2. Export mesh with UVs
    with Timer("Mesh export"):
        obj_path, mesh_objects, face_materials, actual_uv_layer = export_mesh_with_materials(
            output_dir, uv_layer_name
        )

    # 2.5. Export vertex colors (if present)
    vertex_color_file = None
    with Timer("Vertex color extraction"):
        if mesh_objects:
            vertex_color_file = export_vertex_colors(mesh_objects[0], output_dir)
            if vertex_color_file:
                print(f"\n  [OK] Vertex colors exported")
            else:
                print(f"\n  [INFO] No vertex colors to export")

    # 3. Build material manifest (use actual UV layer, not requested one)
    with Timer("Manifest generation"):
        manifest = build_material_manifest(
            mesh_objects, extracted_textures, actual_uv_layer
        )
        manifest['face_materials'] = face_materials
        manifest['obj_file'] = str(obj_path)

        # Add vertex color file if present
        if vertex_color_file:
            manifest['vertex_colors'] = str(vertex_color_file)
        else:
            manifest['vertex_colors'] = None

    # 3.5. Export UV layers (after manifest is built so we know which layers are needed)
    uv_layer_files = {}
    with Timer("UV layer export"):
        if mesh_objects:
            uv_layer_files = export_uv_layers(mesh_objects[0], output_dir, manifest)
            if uv_layer_files:
                print(f"\n  [OK] Exported {len(uv_layer_files)} UV layer(s)")
                # Add UV layer files to manifest
                manifest['uv_layers'] = uv_layer_files
            else:
                print(f"\n  [INFO] No UV layers to export")
                manifest['uv_layers'] = {}

    # Write manifest JSON (after UV layers are added)
    with Timer("Manifest save"):
        manifest_path = output_dir / "material_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n  [OK] Manifest saved: {manifest_path.name}")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print(f"   Output directory: {output_dir}")
    print(f"   Mesh file: {obj_path.name}")
    print(f"   Textures: {len(extracted_textures)} files")
    print(f"   Materials: {len(manifest['materials'])} analyzed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
