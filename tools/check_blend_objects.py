# ABOUTME: Blender script to list all objects in a .blend file
# ABOUTME: Run with: blender -b file.blend -P this_script.py

import bpy

print('=' * 60)
print('OBJECTS IN SCENE:')
print('=' * 60)

mesh_count = 0
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        mesh_count += 1
        vert_count = len(obj.data.vertices)
        face_count = len(obj.data.polygons)
        mat_count = len(obj.data.materials)
        
        # Get bounding box
        bbox = obj.bound_box
        min_y = min(v[1] for v in bbox)
        max_y = max(v[1] for v in bbox)
        
        print(f'MESH: {obj.name}')
        print(f'  Vertices: {vert_count:,}')
        print(f'  Faces: {face_count:,}')
        print(f'  Y range: {min_y:.2f} to {max_y:.2f}')
        print(f'  Materials ({mat_count}):')
        for mat in obj.data.materials:
            if mat:
                print(f'    - {mat.name}')
        print()
    else:
        print(f'{obj.type}: {obj.name}')

print('=' * 60)
print(f'Total mesh objects: {mesh_count}')
print('=' * 60)

