# ABOUTME: Blender Python script for procedural shader baking
# ABOUTME: Preserves UV topology, bakes to texture, exports OBJ

import bpy  # type: ignore  # Only available inside Blender
import sys
import os
import time
from pathlib import Path


class Timer:
    """Simple timer for Blender script (can't import from utils)."""

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

def dump_bake_diagnostic(obj, bake_image, bake_type):
    """
    Dump comprehensive diagnostic info about the bake setup.
    This helps debug black bake issues by showing the exact state
    of objects, materials, UV layers, and node connections.
    """
    print("\n" + "=" * 80)
    print("[DIAG] BAKE DIAGNOSTIC DUMP")
    print("=" * 80)

    # 1. OBJECT STATE
    print("\n[OBJ] OBJECT STATE:")
    print(f"  Name: {obj.name}")
    print(f"  Type: {obj.type}")
    print(f"  Selected: {obj.select_get()}")
    print(f"  Active: {bpy.context.view_layer.objects.active == obj}")
    print(f"  Visible: {obj.visible_get()}")
    print(f"  Hide render: {obj.hide_render}")
    if obj.type == 'MESH':
        print(f"  Vertex count: {len(obj.data.vertices)}")
        print(f"  Polygon count: {len(obj.data.polygons)}")
        print(f"  Material slots: {len(obj.material_slots)}")

    # 2. UV LAYERS
    print("\n[UV] UV LAYERS:")
    if obj.type == 'MESH':
        mesh = obj.data
        if mesh.uv_layers:
            print(f"  Total UV layers: {len(mesh.uv_layers)}")
            for i, uv_layer in enumerate(mesh.uv_layers):
                active_marker = " <- ACTIVE" if uv_layer.active else ""
                render_marker = " <- RENDER" if uv_layer.active_render else ""
                print(f"    [{i}] '{uv_layer.name}'{active_marker}{render_marker}")

            # Check if BakeUV exists and has data
            if "BakeUV" in mesh.uv_layers:
                bake_uv = mesh.uv_layers["BakeUV"]
                # Sample a few UV coordinates to verify they're not all zero
                sample_count = min(10, len(bake_uv.data))
                all_zero = True
                for i in range(sample_count):
                    uv = bake_uv.data[i].uv
                    if uv[0] != 0.0 or uv[1] != 0.0:
                        all_zero = False
                        break
                if all_zero:
                    print(f"  [WARN] WARNING: BakeUV appears to have all zero coordinates!")
                else:
                    print(f"  [OK] BakeUV has non-zero UV coordinates")
        else:
            print("  [WARN] NO UV LAYERS FOUND!")

    # 3. BAKE TARGET IMAGE
    print("\n[IMG] BAKE TARGET IMAGE:")
    print(f"  Name: {bake_image.name}")
    print(f"  Size: {bake_image.size[0]}x{bake_image.size[1]}")
    print(f"  Generated: {bake_image.generated_type if hasattr(bake_image, 'generated_type') else 'N/A'}")
    print(f"  Is dirty: {bake_image.is_dirty}")
    print(f"  Filepath: {bake_image.filepath or '(internal)'}")

    # 4. MATERIALS
    print("\n[MAT] MATERIALS:")
    if obj.type == 'MESH':
        mesh = obj.data
        print(f"  Total materials on mesh: {len(mesh.materials)}")

        for mat_idx, mat in enumerate(mesh.materials):
            if mat is None:
                print(f"\n  [{mat_idx}] [WARN] NONE (empty slot)")
                continue

            print(f"\n  [{mat_idx}] '{mat.name}':")
            print(f"       Uses nodes: {mat.use_nodes}")

            if not mat.use_nodes:
                print(f"       [WARN] Material does not use nodes - cannot bake!")
                continue

            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            print(f"       Node count: {len(nodes)}")
            print(f"       Link count: {len(links)}")

            # Find active node
            active_node = nodes.active
            if active_node:
                print(f"       Active node: '{active_node.name}' (type: {active_node.type})")
                if active_node.type == 'TEX_IMAGE':
                    if active_node.image:
                        print(f"         -> Image: '{active_node.image.name}'")
                        print(f"         -> Is bake target: {active_node.image == bake_image}")
                    else:
                        print(f"         -> [WARN] No image assigned!")
            else:
                print(f"       [WARN] NO ACTIVE NODE!")

            # Find all image texture nodes
            tex_nodes = [n for n in nodes if n.type == 'TEX_IMAGE']
            print(f"       Image texture nodes: {len(tex_nodes)}")
            for tex_node in tex_nodes:
                selected_marker = " [SELECTED]" if tex_node.select else ""
                active_marker = " [ACTIVE]" if tex_node == active_node else ""
                if tex_node.image:
                    is_bake = " <- BAKE TARGET" if tex_node.image == bake_image else ""
                    print(f"         - '{tex_node.name}': '{tex_node.image.name}'{is_bake}{selected_marker}{active_marker}")
                    # Check if image is loaded
                    if tex_node.image.size[0] == 0 or tex_node.image.size[1] == 0:
                        print(f"           [WARN] Image has zero size - NOT LOADED!")
                    if tex_node.image != bake_image and not tex_node.image.has_data:
                        print(f"           [WARN] Image has no data - may be missing!")
                    # Check UV input - CRITICAL for debugging
                    vector_input = tex_node.inputs.get('Vector')
                    if vector_input and vector_input.is_linked:
                        uv_source = vector_input.links[0].from_node
                        if uv_source.type == 'UVMAP':
                            print(f"           UV: '{uv_source.uv_map}' (via UV Map node)")
                        else:
                            print(f"           UV: from '{uv_source.name}' ({uv_source.type})")
                    else:
                        print(f"           UV: IMPLICIT (uses active UV layer)")
                else:
                    print(f"         - '{tex_node.name}': [WARN] NO IMAGE{selected_marker}{active_marker}")

            # Find output node and trace connections
            output_node = None
            for node in nodes:
                if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
                    output_node = node
                    break

            if output_node:
                surface_input = output_node.inputs.get('Surface')
                if surface_input and surface_input.is_linked:
                    connected_node = surface_input.links[0].from_node
                    print(f"       Output connected to: '{connected_node.name}' (type: {connected_node.type})")

                    # If it's a shader, show what's connected to Base Color
                    if connected_node.type == 'BSDF_PRINCIPLED':
                        base_color = connected_node.inputs.get('Base Color')
                        if base_color:
                            if base_color.is_linked:
                                color_source = base_color.links[0].from_node
                                print(f"         Base Color <- '{color_source.name}' (type: {color_source.type})")
                            else:
                                color_val = base_color.default_value[:3]
                                print(f"         Base Color = ({color_val[0]:.2f}, {color_val[1]:.2f}, {color_val[2]:.2f})")
                    elif connected_node.type == 'MIX_SHADER':
                        # Show both shader inputs
                        for i, inp_name in enumerate(['Shader', 'Shader_001']):
                            inp = connected_node.inputs[i + 1] if i + 1 < len(connected_node.inputs) else None
                            if inp and inp.is_linked:
                                shader = inp.links[0].from_node
                                print(f"         Mix input {i}: '{shader.name}' (type: {shader.type})")
                else:
                    print(f"       [WARN] Output node has no surface connection!")
            else:
                print(f"       [WARN] No active output node found!")

        # Check material assignment on faces
        mat_face_counts = {}
        for poly in mesh.polygons:
            idx = poly.material_index
            mat_face_counts[idx] = mat_face_counts.get(idx, 0) + 1

        print(f"\n  [STATS] Face material distribution:")
        for mat_idx, count in sorted(mat_face_counts.items()):
            mat_name = mesh.materials[mat_idx].name if mat_idx < len(mesh.materials) and mesh.materials[mat_idx] else "NONE"
            pct = (count / len(mesh.polygons)) * 100
            print(f"       Material {mat_idx} ('{mat_name}'): {count} faces ({pct:.1f}%)")

    # 5. BAKE SETTINGS
    print("\n[CFG] BAKE SETTINGS:")
    print(f"  Bake type: {bake_type}")
    print(f"  Render engine: {bpy.context.scene.render.engine}")
    if bpy.context.scene.render.engine == 'CYCLES':
        print(f"  Cycles device: {bpy.context.scene.cycles.device}")
        print(f"  Samples: {bpy.context.scene.cycles.samples}")
    print(f"  Bake margin: {bpy.context.scene.render.bake.margin}")
    print(f"  Use clear: {bpy.context.scene.render.bake.use_clear}")
    print(f"  Use selected to active: {bpy.context.scene.render.bake.use_selected_to_active}")

    # 6. SCENE LIGHTS
    print("\n[LIGHT] SCENE LIGHTS:")
    lights = [obj for obj in bpy.data.objects if obj.type == 'LIGHT']
    if lights:
        for light in lights:
            print(f"  - '{light.name}': {light.data.type}, energy={light.data.energy}")
    else:
        print("  [WARN] NO LIGHTS IN SCENE!")

    print("\n" + "=" * 80)
    print("[DIAG] END DIAGNOSTIC DUMP")
    print("=" * 80 + "\n")


# Get arguments passed after "--"
argv = sys.argv
argv = argv[argv.index("--") + 1:]

if len(argv) < 1:
    print("ERROR: No output directory provided")
    sys.exit(1)

output_dir = Path(argv[0])
texture_resolution = int(argv[1]) if len(argv) > 1 else 4096
use_gpu = argv[2].lower() == 'true' if len(argv) > 2 else False
samples = int(argv[3]) if len(argv) > 3 else 32
renderer_mode = argv[4] if len(argv) > 4 else 'auto'  # 'auto', 'cycles', 'eevee'
use_denoise = argv[5].lower() == 'true' if len(argv) > 5 else False
forced_bake_type = argv[6] if len(argv) > 6 else 'auto'  # 'auto', 'DIFFUSE', 'COMBINED', 'EMIT'

print(f"Blender Baker Script")
print(f"Output directory: {output_dir}")
print(f"Texture resolution: {texture_resolution}")
print(f"Use GPU: {use_gpu}")
print(f"Samples: {samples}")
print(f"Renderer mode: {renderer_mode}")
print(f"Denoise: {use_denoise}")
print(f"Bake type override: {forced_bake_type}")


def analyze_model_complexity(mesh_objects):
    """
    Analyze model complexity to determine optimal renderer.

    Returns:
        dict with complexity metrics
    """
    # Count total polygons
    total_polys = sum(len(obj.data.polygons) for obj in mesh_objects if obj.data.polygons)

    # Check for procedural shader nodes
    procedural_node_types = {
        'ShaderNodeTexNoise', 'ShaderNodeTexVoronoi', 'ShaderNodeTexWave',
        'ShaderNodeTexMusgrave', 'ShaderNodeTexMagic', 'ShaderNodeTexBrick',
        'ShaderNodeDisplacement', 'ShaderNodeVectorDisplacement'
    }

    has_procedural = False
    max_node_count = 0
    has_displacement = False
    has_subsurface = False

    for mat in bpy.data.materials:
        if mat.use_nodes:
            nodes = mat.node_tree.nodes
            node_count = len(nodes)
            max_node_count = max(max_node_count, node_count)

            for node in nodes:
                if node.type in procedural_node_types:
                    has_procedural = True
                if node.type in {'ShaderNodeDisplacement', 'ShaderNodeVectorDisplacement'}:
                    has_displacement = True
                if node.type == 'ShaderNodeBsdfPrincipled':
                    # Check if subsurface is used
                    if hasattr(node.inputs['Subsurface'], 'default_value'):
                        if node.inputs['Subsurface'].default_value > 0:
                            has_subsurface = True

    return {
        'poly_count': total_polys,
        'has_procedural': has_procedural,
        'max_node_count': max_node_count,
        'has_displacement': has_displacement,
        'has_subsurface': has_subsurface
    }


def analyze_material_for_bake_strategy(mat):
    """
    Analyze a single material to determine the optimal bake strategy.

    Returns a dict with:
        - strategy: 'DIFFUSE', 'COMBINED', 'EMIT', 'TEXTURE_DIRECT', or 'SIMPLE_COLOR'
        - reason: Human-readable explanation
        - base_color: Optional tuple (R,G,B) if simple color
        - texture_node: Optional texture node if using direct texture
    """
    if not mat or not mat.use_nodes:
        return {
            'strategy': 'DIFFUSE',
            'reason': 'No node tree, using simple diffuse bake',
            'base_color': None,
            'texture_node': None
        }

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Find the output node
    output_node = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
            output_node = node
            break

    if not output_node:
        return {
            'strategy': 'DIFFUSE',
            'reason': 'No output node found',
            'base_color': None,
            'texture_node': None
        }

    # Check what's connected to the surface input
    surface_input = output_node.inputs.get('Surface')
    if not surface_input or not surface_input.is_linked:
        return {
            'strategy': 'DIFFUSE',
            'reason': 'Nothing connected to surface output',
            'base_color': None,
            'texture_node': None
        }

    # Get the shader node connected to output
    shader_link = surface_input.links[0]
    shader_node = shader_link.from_node

    # Check for emission shader
    if shader_node.type == 'EMISSION':
        return {
            'strategy': 'EMIT',
            'reason': 'Emission shader detected',
            'base_color': None,
            'texture_node': None
        }

    # Check for complex shader setups (Mix Shader, Add Shader, Transparent)
    complex_shader_types = {'MIX_SHADER', 'ADD_SHADER', 'BSDF_TRANSPARENT', 'BSDF_GLASS', 'BSDF_REFRACTION'}
    if shader_node.type in complex_shader_types:
        return {
            'strategy': 'COMBINED',
            'reason': f'Complex shader: {shader_node.type}',
            'base_color': None,
            'texture_node': None
        }

    # Check for Principled BSDF (most common)
    if shader_node.type == 'BSDF_PRINCIPLED':
        base_color_input = shader_node.inputs.get('Base Color')

        if not base_color_input:
            return {
                'strategy': 'DIFFUSE',
                'reason': 'Principled BSDF without base color input',
                'base_color': None,
                'texture_node': None
            }

        # Check if base color is connected to an image texture
        if base_color_input.is_linked:
            color_link = base_color_input.links[0]
            color_source = color_link.from_node

            # Direct image texture connection
            if color_source.type == 'TEX_IMAGE':
                if color_source.image and color_source.image.filepath:
                    return {
                        'strategy': 'TEXTURE_DIRECT',
                        'reason': f'Direct image texture: {color_source.image.name}',
                        'base_color': None,
                        'texture_node': color_source
                    }

            # Check for procedural textures
            procedural_types = {'TEX_NOISE', 'TEX_VORONOI', 'TEX_WAVE', 'TEX_MUSGRAVE',
                               'TEX_MAGIC', 'TEX_BRICK', 'TEX_CHECKER', 'TEX_GRADIENT'}
            if color_source.type in procedural_types:
                return {
                    'strategy': 'DIFFUSE',
                    'reason': f'Procedural texture: {color_source.type}',
                    'base_color': None,
                    'texture_node': None
                }

            # Something else connected - use COMBINED to be safe
            return {
                'strategy': 'COMBINED',
                'reason': f'Complex color input: {color_source.type}',
                'base_color': None,
                'texture_node': None
            }
        else:
            # Simple solid color
            color = base_color_input.default_value[:3]
            return {
                'strategy': 'SIMPLE_COLOR',
                'reason': f'Solid color: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})',
                'base_color': color,
                'texture_node': None
            }

    # Fallback for other shader types - use COMBINED
    return {
        'strategy': 'COMBINED',
        'reason': f'Unknown shader type: {shader_node.type}',
        'base_color': None,
        'texture_node': None
    }


def analyze_all_materials_for_bake():
    """
    Analyze all materials in the scene and determine overall bake strategy.

    Returns:
        tuple: (primary_strategy, material_analyses)
            - primary_strategy: The bake type to use ('DIFFUSE', 'COMBINED', 'EMIT')
            - material_analyses: Dict mapping material name to analysis result
    """
    print("\n" + "="*60)
    print("[ANALYZE] ADAPTIVE BAKE STRATEGY ANALYSIS")
    print("="*60)

    analyses = {}
    strategy_counts = {
        'DIFFUSE': 0,
        'COMBINED': 0,
        'EMIT': 0,
        'TEXTURE_DIRECT': 0,
        'SIMPLE_COLOR': 0
    }

    for mat in bpy.data.materials:
        if not mat.users:  # Skip unused materials
            continue

        analysis = analyze_material_for_bake_strategy(mat)
        analyses[mat.name] = analysis
        strategy_counts[analysis['strategy']] += 1

        print(f"  Material '{mat.name}': {analysis['strategy']}")
        print(f"    -> {analysis['reason']}")

    print(f"\n  Strategy Summary:")
    print(f"    TEXTURE_DIRECT: {strategy_counts['TEXTURE_DIRECT']} (use image directly)")
    print(f"    SIMPLE_COLOR:   {strategy_counts['SIMPLE_COLOR']} (solid color)")
    print(f"    DIFFUSE:        {strategy_counts['DIFFUSE']} (simple bake)")
    print(f"    COMBINED:       {strategy_counts['COMBINED']} (complex bake)")
    print(f"    EMIT:           {strategy_counts['EMIT']} (emission bake)")

    # Determine primary strategy
    # Priority: If ANY material needs COMBINED, use COMBINED for all
    # If any needs EMIT but none need COMBINED, use EMIT
    # Otherwise use DIFFUSE
    if strategy_counts['COMBINED'] > 0:
        primary = 'COMBINED'
        reason = "Complex materials detected (Mix Shader, transparency, etc.)"
    elif strategy_counts['EMIT'] > 0:
        primary = 'EMIT'
        reason = "Emission shaders detected"
    elif strategy_counts['TEXTURE_DIRECT'] > 0 or strategy_counts['SIMPLE_COLOR'] > 0:
        # All materials are simple - we can use DIFFUSE
        primary = 'DIFFUSE'
        reason = "All materials are simple (image textures or solid colors)"
    else:
        primary = 'DIFFUSE'
        reason = "Default fallback"

    print(f"\n  [TARGET] PRIMARY BAKE STRATEGY: {primary}")
    print(f"     Reason: {reason}")
    print("="*60 + "\n")

    return primary, analyses


def choose_renderer(complexity_info):
    """
    Analyze model complexity and return renderer settings.

    NOTE: Texture baking ONLY works with Cycles - EEVEE does not support bpy.ops.object.bake()
    So we always return 'cycles', but use complexity analysis to optimize settings.

    Returns:
        tuple: ('cycles', is_simple_model) - always cycles, with complexity flag
    """
    score = 0
    reasons = []

    # Polygon count scoring
    if complexity_info['poly_count'] > 100000:
        score += 3
        reasons.append(f"High polygon count ({complexity_info['poly_count']:,})")
    elif complexity_info['poly_count'] > 50000:
        score += 2
        reasons.append(f"Moderate polygon count ({complexity_info['poly_count']:,})")
    elif complexity_info['poly_count'] > 10000:
        score += 1

    # Procedural shaders
    if complexity_info['has_procedural']:
        score += 3
        reasons.append("Procedural shaders detected")

    # Material complexity
    if complexity_info['max_node_count'] > 20:
        score += 2
        reasons.append(f"Complex materials ({complexity_info['max_node_count']} nodes)")
    elif complexity_info['max_node_count'] > 10:
        score += 1
        reasons.append(f"Moderate materials ({complexity_info['max_node_count']} nodes)")

    # Advanced features
    if complexity_info['has_displacement']:
        score += 2
        reasons.append("Displacement mapping")

    if complexity_info['has_subsurface']:
        score += 2
        reasons.append("Subsurface scattering")

    # Always use Cycles (EEVEE doesn't support baking), but track complexity
    is_simple = score <= 2

    if is_simple:
        print("\n" + "="*60)
        print("[STATS] MODEL ANALYSIS: Simple model")
        print(f"   Polygons: {complexity_info['poly_count']:,}")
        print(f"   Max nodes: {complexity_info['max_node_count']}")
        print(f"   Complexity score: {score}/10")
        print("\n[TARGET] DECISION: Using Cycles renderer (required for baking)")
        print("   Note: EEVEE does not support texture baking")
        print("   Optimized settings for simple model (fast bake)")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("[STATS] MODEL ANALYSIS: Complex model detected")
        print(f"   Polygons: {complexity_info['poly_count']:,}")
        print(f"   Max nodes: {complexity_info['max_node_count']}")
        print(f"   Complexity score: {score}/10")
        if reasons:
            print("   Reasons:")
            for reason in reasons:
                print(f"     - {reason}")
        print("\n[TARGET] DECISION: Using Cycles renderer (high quality)")
        if use_gpu:
            print("   GPU acceleration: Enabled")
            print("   Expected time: 2-5 minutes")
        else:
            print("   GPU acceleration: Disabled")
            print("   Expected time: 10-30 minutes")
            print("   [TIP] Add --use-gpu for 5-10x speedup!")
        print("="*60 + "\n")

    return ('cycles', is_simple)


def build_texture_index(search_dir: Path) -> dict:
    """
    Build an index of all texture files in the search directory.

    This allows O(1) lookup of textures by filename, regardless of
    where they are in the directory structure.

    Args:
        search_dir: Root directory to search

    Returns:
        Dictionary mapping lowercase filename -> full path
    """
    texture_extensions = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff', '.exr', '.hdr'}
    index = {}

    print(f"  Building texture index from: {search_dir}")

    # Walk the entire directory tree
    for root, dirs, files in os.walk(search_dir):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext in texture_extensions:
                full_path = Path(root) / filename
                # Use lowercase filename as key for case-insensitive matching
                key = filename.lower()
                if key not in index:
                    index[key] = full_path

    print(f"  Found {len(index)} texture files in working directory")
    return index


def fix_texture_paths():
    """
    Fix texture paths to ensure all images load correctly.

    This searches the entire working directory (blend file's folder and subfolders)
    for texture files by filename, regardless of the original path structure.

    This handles cases where:
    - Textures use relative paths that don't resolve correctly
    - Textures were in a different folder structure originally
    - User provided a folder with textures in various subfolders
    """
    print("\n" + "="*60)
    print("[FIX] TEXTURE PATH VALIDATION & AUTO-FIX")
    print("="*60)

    blend_file_path = Path(bpy.data.filepath)
    blend_dir = blend_file_path.parent

    print(f"  Blend file: {blend_file_path}")
    print(f"  Working directory: {blend_dir}")

    # Build index of all available textures in the working directory
    texture_index = build_texture_index(blend_dir)

    fixed_count = 0
    missing_count = 0
    valid_count = 0

    for img in bpy.data.images:
        # Skip generated/packed images
        if img.source != 'FILE':
            continue

        if not img.filepath:
            continue

        # Get the original path
        original_path = img.filepath

        # Extract just the filename (without any path)
        # Handle both forward and backward slashes
        filename = Path(original_path.replace('\\', '/')).name

        # Remove any .001, .002 suffixes that Blender adds for duplicates
        base_filename = filename
        if '.' in filename:
            parts = filename.rsplit('.', 2)
            if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 3:
                # This is a Blender duplicate suffix like .001
                base_filename = '.'.join(parts[:-1])

        # Try to resolve the path using Blender's method first
        try:
            abs_path = bpy.path.abspath(img.filepath)
            abs_path_obj = Path(abs_path)

            if abs_path_obj.exists():
                # Path is valid - make it absolute to ensure it works
                img.filepath = str(abs_path_obj.resolve())
                valid_count += 1
                print(f"  [OK] Valid: {img.name}")
                continue
        except Exception:
            pass  # Fall through to texture index lookup

        # Path doesn't exist - try to find by filename in our texture index
        found = False

        # Try exact filename match (case-insensitive)
        lookup_key = filename.lower()
        if lookup_key in texture_index:
            img.filepath = str(texture_index[lookup_key])
            fixed_count += 1
            found = True
            print(f"  [OK] Fixed: {img.name}")
            print(f"    -> {texture_index[lookup_key].name}")

        # Try base filename (without .001 suffix)
        if not found and base_filename != filename:
            lookup_key = base_filename.lower()
            if lookup_key in texture_index:
                img.filepath = str(texture_index[lookup_key])
                fixed_count += 1
                found = True
                print(f"  [OK] Fixed: {img.name} (matched base name)")
                print(f"    -> {texture_index[lookup_key].name}")

        if not found:
            missing_count += 1
            print(f"  [MISS] Missing: {img.name}")
            print(f"    Looking for: {filename}")

    print(f"\n  Summary: {valid_count} valid, {fixed_count} fixed, {missing_count} missing")

    if missing_count > 0:
        print("  [WARN] WARNING: Some textures are missing - bake may have incorrect colors!")
        print("  [TIP] Place texture files in a 'textures' subfolder next to your .blend file")
    else:
        print("  [OK] All textures found!")

    print("="*60 + "\n")

    return missing_count == 0


def bake_and_export():
    """Main baking function."""

    # 0. FIX TEXTURE PATHS FIRST
    # This ensures all image textures load correctly before we do anything else
    textures_ok = fix_texture_paths()
    if not textures_ok:
        print("[WARN] Some textures are missing - continuing anyway but output may be incorrect")

    # 1. SELECT AND PREPARE MESHES
    bpy.ops.object.select_all(action='DESELECT')
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    if not mesh_objects:
        print("ERROR: No mesh objects found in scene")
        sys.exit(1)

    print(f"Found {len(mesh_objects)} mesh object(s)")

    # ADAPTIVE RENDERER SELECTION
    # NOTE: Baking ONLY works with Cycles - EEVEE does not support bpy.ops.object.bake()
    # We use complexity analysis to optimize Cycles settings
    is_simple_model = False
    with Timer("Model analysis"):
        if renderer_mode == 'auto':
            print("\n[ANALYZE] Analyzing model complexity...")
            complexity_info = analyze_model_complexity(mesh_objects)
            _, is_simple_model = choose_renderer(complexity_info)
        elif renderer_mode == 'eevee':
            # User requested EEVEE but it doesn't support baking - use Cycles instead
            is_simple_model = True  # Use simple settings since user wanted fast mode
            print("\n[WARN] EEVEE does not support texture baking")
            print("[TARGET] Using Cycles renderer with fast settings instead")
        elif renderer_mode == 'cycles':
            print("\n[TARGET] Using Cycles renderer (user override)")
        else:
            print("\n[TARGET] Using Cycles renderer (default)")

    # ADAPTIVE BAKE STRATEGY ANALYSIS
    # Analyze materials to determine optimal bake type (or use forced type)
    with Timer("Material analysis"):
        if forced_bake_type != 'auto':
            # User forced a specific bake type
            adaptive_bake_type = forced_bake_type
            material_analyses = {}  # Skip analysis
            print(f"\n[TARGET] FORCED BAKE TYPE: {forced_bake_type}")
            print("   (Skipping adaptive material analysis)")
        else:
            adaptive_bake_type, material_analyses = analyze_all_materials_for_bake()

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
    with Timer("UV unwrapping"):
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
    
    # 3. SETUP BAKE TARGET ON EXISTING MATERIALS (PRESERVE ORIGINAL APPEARANCE)
    print("Setting up bake target on existing materials...")
    print("  IMPORTANT: Preserving original materials to capture their appearance")

    mesh = active_obj.data
    original_mat_count = len(mesh.materials)

    if original_mat_count == 0:
        # No materials - create a default one
        print("  No materials found - creating default material")
        mat = bpy.data.materials.new(name="DefaultMaterial")
        mat.use_nodes = True
        mesh.materials.append(mat)
        original_mat_count = 1

    print(f"  Found {original_mat_count} material(s) to bake from")

    # Store references to all materials for later bake target setup
    materials_to_bake = list(mesh.materials)

    # Ensure object is selected and active for baking
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='DESELECT')
    active_obj.select_set(True)
    bpy.context.view_layer.objects.active = active_obj
    print(f"  Object '{active_obj.name}' selected and ready for baking")

    # 4. CREATE IMAGE FOR BAKING AND ADD TO ALL MATERIALS
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

    # Add bake target node to EACH material
    # This is critical - Cycles bakes FROM the material appearance TO this image
    #
    # CRITICAL FIX: We must also ensure source textures read from the ORIGINAL UV layer (uv0)
    # because BakeUV is now the active layer but has completely different UV coordinates.
    # Without this, source textures sample garbage -> black output!

    # Find the original UV layer name (the one used for source textures)
    original_uv_name = "uv0"  # Default
    if active_obj.data.uv_layers:
        # First UV layer is typically the original mapping
        for uv_layer in active_obj.data.uv_layers:
            if uv_layer.name != "BakeUV" and uv_layer.name != "blend_ao":
                original_uv_name = uv_layer.name
                break
    print(f"  Source textures will use UV layer: '{original_uv_name}'")

    bake_nodes_added = 0
    source_uv_nodes_added = 0

    for mat in materials_to_bake:
        if mat is None:
            continue

        # Ensure material uses nodes
        if not mat.use_nodes:
            mat.use_nodes = True

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # CRITICAL: Add UV Map nodes to ALL source Image Texture nodes
        # This ensures they read from the original UV layout, not BakeUV
        for node in list(nodes):  # Use list() to allow modification during iteration
            if node.type == 'TEX_IMAGE' and node.image and node.name != "BakeTarget":
                # Check if this node already has a UV Map connected to its Vector input
                has_uv_input = False
                for link in links:
                    if link.to_node == node and link.to_socket.name == 'Vector':
                        has_uv_input = True
                        break

                if not has_uv_input:
                    # Create a UV Map node pointing to the original UV layer
                    uv_node = nodes.new(type='ShaderNodeUVMap')
                    uv_node.uv_map = original_uv_name
                    uv_node.location = (node.location.x - 200, node.location.y)

                    # Connect UV output to texture Vector input
                    links.new(uv_node.outputs['UV'], node.inputs['Vector'])
                    source_uv_nodes_added += 1

        # Check if bake node already exists
        existing_bake_node = None
        for node in nodes:
            if node.type == 'TEX_IMAGE' and node.image == img:
                existing_bake_node = node
                break

        if existing_bake_node:
            bake_node = existing_bake_node
        else:
            # Add new Image Texture node for baking target
            bake_node = nodes.new(type='ShaderNodeTexImage')
            bake_node.image = img
            bake_node.name = "BakeTarget"
            bake_node.label = "Bake Target"
            # Position it away from other nodes
            bake_node.location = (-600, 300)

        # Critical: Select ONLY the bake node and make it active in this material
        for node in nodes:
            node.select = False
        bake_node.select = True
        nodes.active = bake_node

        bake_nodes_added += 1
        print(f"    Added bake target to material: {mat.name}")

    print(f"  [OK] Added UV Map nodes to {source_uv_nodes_added} source textures (pointing to '{original_uv_name}')")

    print(f"  Bake target added to {bake_nodes_added} material(s)")
    
    # 5. EXECUTE BAKE
    with Timer("Baking"):
        print("Starting bake operation...")

        # Configure Cycles renderer (required for baking - EEVEE doesn't support bake)
        print("  Configuring Cycles renderer...")
        bpy.context.scene.render.engine = 'CYCLES'

        # Optimize samples based on model complexity
        if is_simple_model:
            bpy.context.scene.cycles.samples = min(samples, 16)  # Fewer samples for simple models
            print(f"  Using optimized samples for simple model: {bpy.context.scene.cycles.samples}")
        else:
            bpy.context.scene.cycles.samples = samples
            print(f"  Using samples: {samples}")

        # Configure GPU if requested
        if use_gpu:
            print("  Configuring GPU acceleration...")

            # Get Cycles preferences
            prefs = bpy.context.preferences.addons['cycles'].preferences

            # List of compute types to try (OptiX first for RTX cards, then CUDA, then others)
            compute_types = ['OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL']
            gpu_found = False

            for compute_type in compute_types:
                try:
                    print(f"    Trying {compute_type}...")
                    prefs.compute_device_type = compute_type

                    # CRITICAL: Must call get_devices() to refresh the device list
                    # This returns a tuple: (cuda_devices, opencl_devices) or similar
                    prefs.get_devices()

                    # Now iterate through the devices property (not the return value)
                    gpu_devices = []
                    for device in prefs.devices:
                        print(f"      Found: {device.name} (type={device.type})")
                        if device.type != 'CPU':
                            gpu_devices.append(device)

                    if gpu_devices:
                        # Enable GPU devices, disable CPU for GPU compute
                        for device in prefs.devices:
                            if device.type == 'CPU':
                                device.use = False  # Don't use CPU for GPU compute
                            else:
                                device.use = True
                                print(f"    [OK] Enabled: {device.name}")

                        bpy.context.scene.cycles.device = 'GPU'
                        gpu_found = True
                        print(f"  [OK] GPU acceleration enabled using {compute_type}")
                        break

                except Exception as e:
                    print(f"      {compute_type} not available: {e}")
                    continue

            if not gpu_found:
                print("  [WARN] No GPU found, falling back to CPU")
                print("  Available devices:")
                for device in prefs.devices:
                    print(f"    - {device.name} ({device.type})")
                bpy.context.scene.cycles.device = 'CPU'
        else:
            bpy.context.scene.cycles.device = 'CPU'
            print("  Using CPU")

        # Enable adaptive sampling for faster convergence
        bpy.context.scene.cycles.use_adaptive_sampling = True
        bpy.context.scene.cycles.adaptive_threshold = 0.01
        print("  Adaptive sampling: enabled")

        # Enable denoising if requested
        if use_denoise:
            bpy.context.scene.cycles.use_denoising = True
            # Use OptiX denoiser if available, otherwise OIDN
            if bpy.context.scene.cycles.device == 'GPU':
                bpy.context.scene.cycles.denoiser = 'OPTIX'
                print("  Denoising: OptiX")
            else:
                bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
                print("  Denoising: OpenImageDenoise")
        else:
            bpy.context.scene.cycles.use_denoising = False

        try:
            # Ensure the BakeUV layer is active
            if "BakeUV" in active_obj.data.uv_layers:
                active_obj.data.uv_layers.active = active_obj.data.uv_layers["BakeUV"]

            # CRITICAL FIX: Re-select the object right before baking
            # The object selection can be lost during renderer configuration
            bpy.ops.object.select_all(action='DESELECT')
            active_obj.select_set(True)
            bpy.context.view_layer.objects.active = active_obj

            # Verify selection
            if not active_obj.select_get():
                raise RuntimeError(f"Failed to select object '{active_obj.name}' for baking")
            if bpy.context.view_layer.objects.active != active_obj:
                raise RuntimeError(f"Failed to set '{active_obj.name}' as active object")

            print(f"  [OK] Object '{active_obj.name}' confirmed selected and active")
            print(f"  Starting bake operation...")
            print(f"  This may take several minutes depending on model complexity...")

            # Set up bake settings for COMBINED bake
            # COMBINED requires proper lighting - we need direct + indirect for full appearance
            bpy.context.scene.render.bake.use_pass_direct = True
            bpy.context.scene.render.bake.use_pass_indirect = True
            bpy.context.scene.render.bake.use_pass_color = True
            bpy.context.scene.render.bake.margin = 16
            bpy.context.scene.render.bake.use_clear = True

            # Ensure there's adequate lighting for COMBINED bake
            # Add a sun light if no lights exist in the scene
            lights = [obj for obj in bpy.data.objects if obj.type == 'LIGHT']
            if not lights:
                print(f"  Adding sun light for baking (no lights in scene)...")
                bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
                sun = bpy.context.active_object
                sun.name = "BakeSun"
                sun.data.energy = 3.0
                # Re-select the mesh for baking
                bpy.ops.object.select_all(action='DESELECT')
                active_obj.select_set(True)
                bpy.context.view_layer.objects.active = active_obj
            else:
                print(f"  Using existing lights ({len(lights)} found)")

            # CRITICAL FIX: bpy.ops.object.bake() requires a valid 3D View context
            # When running as a script (with or without -b flag), bpy.context.area is None
            # We need to use context.temp_override() to provide a proper context

            # Find or create a 3D View area for context override
            def get_3d_view_context():
                """Get a valid 3D View context for baking operations."""
                # First, try to find an existing 3D View area
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D':
                            for region in area.regions:
                                if region.type == 'WINDOW':
                                    return {
                                        'window': window,
                                        'screen': window.screen,
                                        'area': area,
                                        'region': region,
                                        'scene': bpy.context.scene,
                                        'view_layer': bpy.context.view_layer,
                                        'active_object': active_obj,
                                        'selected_objects': [active_obj],
                                    }

                # If no 3D View exists, we need to create one
                # This can happen in background mode or minimal UI
                print("  No 3D View found, creating temporary context...")

                # Get the first available window
                window = bpy.context.window_manager.windows[0]
                screen = window.screen

                # Find any area and change it to 3D View temporarily
                for area in screen.areas:
                    # Store original type to restore later
                    original_type = area.type
                    area.type = 'VIEW_3D'

                    for region in area.regions:
                        if region.type == 'WINDOW':
                            return {
                                'window': window,
                                'screen': screen,
                                'area': area,
                                'region': region,
                                'scene': bpy.context.scene,
                                'view_layer': bpy.context.view_layer,
                                'active_object': active_obj,
                                'selected_objects': [active_obj],
                                '_restore_type': original_type,  # For cleanup
                            }

                return None

            ctx = get_3d_view_context()

            if ctx is None:
                raise RuntimeError("Could not create valid 3D View context for baking")

            print(f"  [OK] 3D View context acquired")

            # Perform the bake with proper context override
            # IMPORTANT: Use COMBINED bake type instead of DIFFUSE
            # DIFFUSE only captures the diffuse shader contribution which may be black
            # for materials with complex shader setups (Mix Shader, Transparent BSDF, etc.)
            # COMBINED captures the full rendered appearance of the material

            # Check Blender version for API compatibility
            blender_version = bpy.app.version
            print(f"  Blender version: {blender_version[0]}.{blender_version[1]}.{blender_version[2]}")

            # Use adaptive bake type determined by material analysis
            print(f"  [TARGET] Using adaptive bake type: {adaptive_bake_type}")

            # DIAGNOSTIC: Dump full bake setup state before baking
            dump_bake_diagnostic(active_obj, img, adaptive_bake_type)

            # Configure pass filter for DIFFUSE bake to get pure color without lighting
            # This is critical - by default DIFFUSE includes direct/indirect lighting which can cause issues
            if adaptive_bake_type == 'DIFFUSE':
                bpy.context.scene.render.bake.use_pass_direct = False
                bpy.context.scene.render.bake.use_pass_indirect = False
                bpy.context.scene.render.bake.use_pass_color = True
                print("  [OK] DIFFUSE bake configured: Color only (no direct/indirect lighting)")

            if blender_version >= (3, 2, 0):
                # Blender 3.2+ uses temp_override() context manager
                print("  Using temp_override() for context (Blender 3.2+)")
                with bpy.context.temp_override(**ctx):
                    bpy.ops.object.bake(
                        type=adaptive_bake_type,
                        margin=16,
                        use_clear=True,
                        save_mode='INTERNAL',
                        use_selected_to_active=False,
                        cage_extrusion=0.0,
                        max_ray_distance=0.0,
                        normal_space='TANGENT',
                        target='IMAGE_TEXTURES'
                    )
            else:
                # Blender 3.1 and earlier use context override dict passed to operator
                print("  Using legacy context override (Blender < 3.2)")
                bpy.ops.object.bake(
                    ctx,  # Pass context dict as first argument
                    type=adaptive_bake_type,
                    margin=16,
                    use_clear=True,
                    save_mode='INTERNAL',
                    use_selected_to_active=False,
                    cage_extrusion=0.0,
                    max_ray_distance=0.0,
                    normal_space='TANGENT',
                    target='IMAGE_TEXTURES'
                )

            print("[OK] Bake complete!")

            # BAKE VALIDATION: Check if the bake produced actual content
            print("  Validating bake output...")
            pixels = list(img.pixels)
            total_pixels = len(pixels) // 4  # RGBA

            # Check for empty bake (all black or all same color)
            # Sample every 1000th pixel to quickly check
            sample_step = max(1, total_pixels // 1000)
            r_sum, g_sum, b_sum, a_sum = 0.0, 0.0, 0.0, 0.0
            r_var, g_var, b_var = 0.0, 0.0, 0.0
            sample_count = 0

            first_r, first_g, first_b = pixels[0], pixels[1], pixels[2]
            all_same = True

            for i in range(0, total_pixels, sample_step):
                idx = i * 4
                r, g, b, a = pixels[idx], pixels[idx+1], pixels[idx+2], pixels[idx+3]
                r_sum += r
                g_sum += g
                b_sum += b
                a_sum += a
                sample_count += 1

                if abs(r - first_r) > 0.01 or abs(g - first_g) > 0.01 or abs(b - first_b) > 0.01:
                    all_same = False

            avg_r = r_sum / sample_count if sample_count > 0 else 0
            avg_g = g_sum / sample_count if sample_count > 0 else 0
            avg_b = b_sum / sample_count if sample_count > 0 else 0
            avg_a = a_sum / sample_count if sample_count > 0 else 0

            # Check for problems
            is_all_black = avg_r < 0.01 and avg_g < 0.01 and avg_b < 0.01
            is_all_transparent = avg_a < 0.01

            if is_all_black:
                print(f"  [WARN] WARNING: Bake appears to be all BLACK (avg RGB: {avg_r:.3f}, {avg_g:.3f}, {avg_b:.3f})")
                print(f"      This may indicate the material has no diffuse color or lighting issues")
            elif is_all_transparent:
                print(f"  [WARN] WARNING: Bake appears to be all TRANSPARENT (avg alpha: {avg_a:.3f})")
            elif all_same:
                print(f"  [WARN] WARNING: Bake appears to be a SOLID COLOR (RGB: {first_r:.3f}, {first_g:.3f}, {first_b:.3f})")
            else:
                print(f"  [OK] Bake validation passed (avg RGB: {avg_r:.3f}, {avg_g:.3f}, {avg_b:.3f})")
                print(f"    Texture has color variation - bake looks valid")

        except Exception as e:
            print(f"ERROR during baking: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # 6. SAVE TEXTURE
    with Timer("Texture export"):
        texture_path = output_dir / "baked_texture.png"

        # Convert to absolute path for Blender
        texture_path_abs = texture_path.resolve()

        img.filepath_raw = str(texture_path_abs)
        img.file_format = 'PNG'

        try:
            img.save()
            print(f"[OK] Saved texture: {texture_path_abs}")
        except Exception as e:
            print(f"ERROR saving texture: {e}")
            sys.exit(1)

    # 7. REPLACE ALL MATERIALS WITH SINGLE BAKED MATERIAL FOR EXPORT
    print("Replacing materials with baked texture material...")

    # Create a new material that uses the baked texture
    export_mat = bpy.data.materials.new(name="BakedMaterial")
    export_mat.use_nodes = True

    nodes = export_mat.node_tree.nodes
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
    links = export_mat.node_tree.links
    links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    # Replace all materials with the baked material
    mesh = active_obj.data
    mesh.materials.clear()
    mesh.materials.append(export_mat)

    # Assign all faces to use the new material
    for poly in mesh.polygons:
        poly.material_index = 0

    print(f"  Replaced {len(materials_to_bake)} original material(s) with baked texture material")

    # 8. EXPORT OBJ WITH MTL
    with Timer("OBJ export"):
        obj_path = output_dir / f"{Path(bpy.data.filepath).stem}.obj"

        # Convert to absolute path for Blender
        obj_path_abs = obj_path.resolve()

        print(f"Exporting OBJ: {obj_path_abs}")

        # Determine Blender version for API compatibility
        blender_version = bpy.app.version

        try:
            if blender_version >= (4, 0, 0):
                # Blender 4.0+ uses bpy.ops.wm.obj_export
                bpy.ops.wm.obj_export(
                    filepath=str(obj_path_abs),
                    export_selected_objects=False,
                    export_materials=True,
                    export_uv=True,
                    path_mode='RELATIVE'
                )
                print(f"[OK] Exported OBJ (Blender 4.0+ API): {obj_path_abs}")
            elif blender_version >= (3, 2, 0):
                # Blender 3.2-3.6 uses export_scene.obj with path_mode
                bpy.ops.export_scene.obj(
                    filepath=str(obj_path_abs),
                    use_selection=False,
                    use_materials=True,
                    use_uvs=True,
                    path_mode='RELATIVE'
                )
                print(f"[OK] Exported OBJ (Blender 3.2+ API): {obj_path_abs}")
            else:
                # Blender 3.1 and earlier - path_mode not supported
                bpy.ops.export_scene.obj(
                    filepath=str(obj_path_abs),
                    use_selection=False,
                    use_materials=True,
                    use_uvs=True
                )
                print(f"[OK] Exported OBJ (Blender 3.1 legacy API): {obj_path_abs}")
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

