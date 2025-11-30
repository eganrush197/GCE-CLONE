"""
Script to create a simple test cube with procedural shader in Blender.
Run this with: blender --background --python create_test_cube.py
"""

import bpy
from pathlib import Path

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Create cube
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = "TestCube"

# Create material with procedural shader
mat = bpy.data.materials.new(name="ProceduralMaterial")
mat.use_nodes = True
cube.data.materials.append(mat)

# Get node tree
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# Clear default nodes
nodes.clear()

# Create nodes for procedural shader
# Output node
output_node = nodes.new(type='ShaderNodeOutputMaterial')
output_node.location = (400, 0)

# Principled BSDF
bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
bsdf.location = (0, 0)

# Checker texture for procedural pattern
checker = nodes.new(type='ShaderNodeTexChecker')
checker.location = (-400, 0)
checker.inputs['Scale'].default_value = 4.0
checker.inputs['Color1'].default_value = (0.8, 0.2, 0.2, 1.0)  # Red
checker.inputs['Color2'].default_value = (0.2, 0.2, 0.8, 1.0)  # Blue

# Connect nodes
links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

# UV unwrap the cube
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.unwrap(method='ANGLE_BASED')
bpy.ops.object.mode_set(mode='OBJECT')

# Save the file
output_path = Path(__file__).parent / "simple_cube.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

print(f"✓ Created test cube: {output_path}")
print(f"  - Cube with procedural checker shader (red/blue)")
print(f"  - UV unwrapped")
print(f"  - Ready for baking tests")

