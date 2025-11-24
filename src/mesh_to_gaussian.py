#!/usr/bin/env python3
# ABOUTME: Direct mesh-to-gaussian converter with LOD support
# ABOUTME: Converts OBJ/GLB meshes to gaussian splat PLY files

"""
Simplified Mesh to Gaussian Converter
No complex multi-view rendering - direct conversion with optimization
"""

import numpy as np
import trimesh
from dataclasses import dataclass
from typing import Tuple, Optional, List
import struct
import argparse
from pathlib import Path

# Try to import torch, but make it optional
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - optimization disabled")

# Internal data structure for individual gaussians during conversion
@dataclass
class _SingleGaussian:
    """Single gaussian splat representation (internal use only)"""
    position: np.ndarray  # xyz
    scales: np.ndarray     # 3D scale
    rotation: np.ndarray   # quaternion
    opacity: float
    sh_dc: np.ndarray      # Spherical harmonics DC term (RGB)
    sh_rest: Optional[np.ndarray] = None  # Higher order SH coefficients

class MeshToGaussianConverter:
    """Direct mesh to gaussian converter - no synthetic views needed"""
    
    def __init__(self, device='cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'):
        self.device = device
        if TORCH_AVAILABLE:
            print(f"Using device: {device}")
        else:
            print("PyTorch not available - using NumPy only mode")
        
    def load_mesh(self, path: str) -> trimesh.Trimesh:
        """Load and normalize mesh, with MTL color support for OBJ files"""
        
        # Check if it's an OBJ file with potential MTL
        if path.endswith('.obj'):
            mesh = self._load_obj_with_mtl(path)
        else:
            mesh = trimesh.load(path, force='mesh')
        
        # Center and scale to unit cube
        mesh.vertices -= mesh.vertices.mean(axis=0)
        scale = np.abs(mesh.vertices).max()
        if scale > 0:
            mesh.vertices /= scale
            
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
    
    def _load_obj_with_mtl(self, obj_path: str) -> trimesh.Trimesh:
        """Special OBJ loader that preserves MTL material colors"""
        from pathlib import Path

        # First load with trimesh
        mesh = trimesh.load(obj_path, force='mesh', process=False)

        # Check for MTL file
        mtl_path = Path(obj_path).with_suffix('.mtl')
        if not mtl_path.exists():
            return mesh

        print(f"Found MTL file: {mtl_path}")

        # Parse MTL for material colors
        materials = {}
        current_mat = None

        with open(mtl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == 'newmtl':
                    current_mat = parts[1]
                    materials[current_mat] = [0.7, 0.7, 0.7]

                elif parts[0] == 'Kd' and current_mat:
                    try:
                        materials[current_mat] = [float(parts[1]), float(parts[2]), float(parts[3])]
                    except:
                        pass

        # Now parse OBJ to map materials to faces
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
                    # Face found, assign current material color
                    color = materials.get(current_material, [0.7, 0.7, 0.7])

                    # Count vertices in this face (handles quads, tris, etc.)
                    # OBJ faces can be: f v1 v2 v3 (triangle) or f v1 v2 v3 v4 (quad)
                    # or f v1/vt1/vn1 v2/vt2/vn2 ... (with texture/normal indices)
                    num_vertices = len(parts) - 1  # Subtract 'f' command

                    # Trimesh triangulates: quads (4 verts) → 2 triangles, etc.
                    # For n-gon: (n-2) triangles
                    num_triangles = max(1, num_vertices - 2)

                    # Add color for each resulting triangle
                    for _ in range(num_triangles):
                        face_colors.append(color)

        # Apply face colors to mesh - but only if counts match!
        if face_colors and len(face_colors) == len(mesh.faces):
            face_colors = np.array(face_colors)
            # Ensure colors are in 0-255 range for trimesh
            if face_colors.max() <= 1.0:
                face_colors = (face_colors * 255).astype(np.uint8)

            # Add alpha channel
            face_colors = np.column_stack([face_colors, np.full(len(face_colors), 255)])

            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                face_colors=face_colors
            )
            print(f"✓ Applied {len(materials)} material colors to {len(face_colors)} faces")
        elif face_colors:
            print(f"⚠️  Warning: Face color count mismatch ({len(face_colors)} colors vs {len(mesh.faces)} faces)")
            print(f"   Attempting to use default color for all faces...")
            # Fallback: use first material color or gray for all faces
            default_color = list(materials.values())[0] if materials else [0.7, 0.7, 0.7]
            face_colors = np.array([default_color] * len(mesh.faces))
            if face_colors.max() <= 1.0:
                face_colors = (face_colors * 255).astype(np.uint8)
            face_colors = np.column_stack([face_colors, np.full(len(face_colors), 255)])
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=face_colors)
            print(f"   Using fallback color for all {len(mesh.faces)} faces")

        return mesh
    
    def mesh_to_gaussians(self, mesh: trimesh.Trimesh,
                         strategy: str = 'vertex',
                         samples_per_face: int = 1) -> List[_SingleGaussian]:
        """
        Convert mesh to initial gaussians
        Strategies:
        - 'vertex': One gaussian per vertex
        - 'face': Gaussians sampled on face centers
        - 'hybrid': Both vertices and faces
        - 'adaptive': Auto-select based on mesh (currently maps to hybrid)
        """
        gaussians = []

        # Map adaptive to hybrid for now
        if strategy == 'adaptive':
            strategy = 'hybrid'
            print(f"Using adaptive strategy -> hybrid")

        if strategy in ['vertex', 'hybrid']:
            # Create gaussians from vertices
            for i, vertex in enumerate(mesh.vertices):
                # Get vertex normal if available
                normal = mesh.vertex_normals[i] if hasattr(mesh, 'vertex_normals') else np.array([0, 0, 1])
                
                # Initial scale based on nearby vertices
                if i < len(mesh.vertices) - 1:
                    nearby_dists = np.linalg.norm(mesh.vertices - vertex, axis=1)
                    nearby_dists = nearby_dists[nearby_dists > 0]
                    scale = np.min(nearby_dists) * 0.5 if len(nearby_dists) > 0 else 0.01
                else:
                    scale = 0.01
                
                # Convert normal to quaternion (simplified - align Z axis with normal)
                # This is a hack but works for initialization
                quat = self._normal_to_quaternion(normal)
                
                # Get color from vertex colors or use default
                if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                    color = mesh.visual.vertex_colors[i][:3]
                    if color.max() > 1.0:
                        color = color / 255.0
                elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                    # Find a face containing this vertex and use its color
                    for face_idx, face in enumerate(mesh.faces):
                        if i in face:
                            color = mesh.visual.face_colors[face_idx][:3]
                            if color.max() > 1.0:
                                color = color / 255.0
                            break
                    else:
                        color = np.array([0.5, 0.5, 0.5])
                else:
                    color = np.array([0.5, 0.5, 0.5])
                
                gaussian = _SingleGaussian(
                    position=vertex,
                    scales=np.array([scale, scale, scale * 0.5]),  # Slightly flattened
                    rotation=quat,
                    opacity=0.9,
                    sh_dc=color - 0.5  # SH DC term centered at 0
                )
                gaussians.append(gaussian)
        
        if strategy in ['face', 'hybrid']:
            # Sample gaussians on faces
            for face_idx, face in enumerate(mesh.faces):
                for _ in range(samples_per_face):
                    # Random point on triangle
                    r1, r2 = np.random.random(2)
                    sqrt_r1 = np.sqrt(r1)
                    
                    w1 = 1 - sqrt_r1
                    w2 = sqrt_r1 * (1 - r2)
                    w3 = sqrt_r1 * r2
                    
                    point = (w1 * mesh.vertices[face[0]] + 
                           w2 * mesh.vertices[face[1]] + 
                           w3 * mesh.vertices[face[2]])
                    
                    # Face normal
                    v1 = mesh.vertices[face[1]] - mesh.vertices[face[0]]
                    v2 = mesh.vertices[face[2]] - mesh.vertices[face[0]]
                    normal = np.cross(v1, v2)
                    normal = normal / (np.linalg.norm(normal) + 1e-8)
                    
                    # Scale based on face area
                    area = np.linalg.norm(np.cross(v1, v2)) * 0.5
                    scale = np.sqrt(area) * 0.3
                    
                    quat = self._normal_to_quaternion(normal)
                    
                    # Interpolate vertex colors if available
                    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                        v_colors = mesh.visual.vertex_colors[face][:, :3]
                        # Normalize to 0-1 range
                        if v_colors.max() > 1.0:
                            v_colors = v_colors / 255.0
                        color = (w1 * v_colors[0] +
                               w2 * v_colors[1] + 
                               w3 * v_colors[2])
                    elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                        # Use face color
                        face_color = mesh.visual.face_colors[face_idx][:3]
                        if face_color.max() > 1.0:
                            color = face_color / 255.0
                        else:
                            color = face_color
                    else:
                        color = np.array([0.5, 0.5, 0.5])
                    
                    gaussian = _SingleGaussian(
                        position=point,
                        scales=np.array([scale, scale, scale * 0.3]),
                        rotation=quat,
                        opacity=0.7,
                        sh_dc=color - 0.5
                    )
                    gaussians.append(gaussian)
        
        print(f"Created {len(gaussians)} initial gaussians")
        return gaussians
    
    def _normal_to_quaternion(self, normal: np.ndarray) -> np.ndarray:
        """Convert normal vector to quaternion rotation"""
        # Simplified: Create rotation that aligns (0,0,1) with normal
        z = np.array([0, 0, 1])
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        if np.allclose(normal, z):
            return np.array([1, 0, 0, 0])  # Identity quaternion
        elif np.allclose(normal, -z):
            return np.array([0, 1, 0, 0])  # 180 degree rotation around X
        
        axis = np.cross(z, normal)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        angle = np.arccos(np.clip(np.dot(z, normal), -1, 1))
        
        # Axis-angle to quaternion
        s = np.sin(angle / 2)
        quat = np.array([
            np.cos(angle / 2),
            axis[0] * s,
            axis[1] * s,
            axis[2] * s
        ])
        return quat / (np.linalg.norm(quat) + 1e-8)
    
    def optimize_gaussians(self, gaussians: List[_SingleGaussian],
                          iterations: int = 100) -> List[_SingleGaussian]:
        """
        Simple optimization pass to improve gaussian placement
        This is a placeholder - in production you'd render and compare
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not available, skipping optimization")
            return gaussians
            
        if not torch.cuda.is_available():
            print("CUDA not available, skipping optimization")
            return gaussians
            
        print(f"Optimizing {len(gaussians)} gaussians for {iterations} iterations...")
        
        # Convert to tensors
        positions = torch.tensor(
            np.array([g.position for g in gaussians]), 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=True
        )
        
        scales = torch.tensor(
            np.array([g.scales for g in gaussians]),
            dtype=torch.float32,
            device=self.device, 
            requires_grad=True
        )
        
        opacities = torch.tensor(
            np.array([g.opacity for g in gaussians]),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        
        optimizer = torch.optim.Adam([positions, scales, opacities], lr=0.001)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Simple regularization losses (no rendering)
            # Encourage reasonable scales
            scale_loss = torch.mean(torch.abs(scales - 0.01))
            
            # Encourage opacity near 0.9
            opacity_loss = torch.mean((opacities - 0.9) ** 2)
            
            # Prevent gaussians from drifting too far
            position_loss = torch.mean(positions ** 2)
            
            # Total loss
            loss = scale_loss + opacity_loss * 0.1 + position_loss * 0.01
            
            loss.backward()
            optimizer.step()
            
            # Clamp values
            with torch.no_grad():
                scales.clamp_(0.001, 0.5)
                opacities.clamp_(0.01, 0.99)
            
            if i % 20 == 0:
                print(f"  Iteration {i}: loss = {loss.item():.4f}")
        
        # Convert back to gaussians
        optimized = []
        pos_np = positions.detach().cpu().numpy()
        scale_np = scales.detach().cpu().numpy()
        opacity_np = opacities.detach().cpu().numpy()
        
        for i, g in enumerate(gaussians):
            opt_g = _SingleGaussian(
                position=pos_np[i],
                scales=scale_np[i],
                rotation=g.rotation,
                opacity=float(opacity_np[i]),
                sh_dc=g.sh_dc,
                sh_rest=g.sh_rest
            )
            optimized.append(opt_g)
        
        return optimized

    def save_ply(self, gaussians: List[_SingleGaussian],
                 output_path: str):
        """Save gaussians to PLY format compatible with gaussian splatting viewers"""
        
        # Prepare data arrays
        positions = np.array([g.position for g in gaussians])
        scales = np.array([g.scales for g in gaussians])
        rotations = np.array([g.rotation for g in gaussians])
        opacities = np.array([g.opacity for g in gaussians])
        sh_dc = np.array([g.sh_dc for g in gaussians])
        
        # Convert scales to log scale (expected by viewers)
        log_scales = np.log(scales + 1e-8)
        
        # Prepare vertex data
        vertex_count = len(gaussians)
        
        with open(output_path, 'wb') as f:
            # PLY header
            f.write(b'ply\n')
            f.write(b'format binary_little_endian 1.0\n')
            f.write(f'element vertex {vertex_count}\n'.encode())
            
            # Position
            f.write(b'property float x\n')
            f.write(b'property float y\n')
            f.write(b'property float z\n')
            
            # Normals (unused but expected)
            f.write(b'property float nx\n')
            f.write(b'property float ny\n')
            f.write(b'property float nz\n')
            
            # Spherical harmonics DC
            f.write(b'property float f_dc_0\n')
            f.write(b'property float f_dc_1\n')
            f.write(b'property float f_dc_2\n')
            
            # Opacity
            f.write(b'property float opacity\n')
            
            # Scales
            f.write(b'property float scale_0\n')
            f.write(b'property float scale_1\n')
            f.write(b'property float scale_2\n')
            
            # Rotation
            f.write(b'property float rot_0\n')
            f.write(b'property float rot_1\n')
            f.write(b'property float rot_2\n')
            f.write(b'property float rot_3\n')
            
            f.write(b'end_header\n')
            
            # Write vertex data
            for i in range(vertex_count):
                # Position
                f.write(struct.pack('<fff', *positions[i]))
                
                # Normal (unused)
                f.write(struct.pack('<fff', 0, 0, 0))
                
                # SH DC
                f.write(struct.pack('<fff', *sh_dc[i]))
                
                # Opacity 
                f.write(struct.pack('<f', opacities[i]))
                
                # Scale
                f.write(struct.pack('<fff', *log_scales[i]))
                
                # Rotation
                f.write(struct.pack('<ffff', *rotations[i]))
        
        print(f"Saved {vertex_count} gaussians to {output_path}")

def main():
    from lod_generator import LODGenerator

    parser = argparse.ArgumentParser(description='Simple mesh to gaussian converter')
    parser.add_argument('input', help='Input mesh file (OBJ/GLB)')
    parser.add_argument('output', help='Output PLY file')
    parser.add_argument('--strategy', default='hybrid',
                       choices=['vertex', 'face', 'hybrid'],
                       help='Gaussian initialization strategy')
    parser.add_argument('--optimize', type=int, default=100,
                       help='Optimization iterations (0 to disable)')
    parser.add_argument('--lod', type=int, nargs='*',
                       default=[5000, 25000, 100000],
                       help='LOD levels to generate')

    args = parser.parse_args()

    # Initialize converter
    converter = MeshToGaussianConverter()

    # Load mesh
    mesh = converter.load_mesh(args.input)

    # Convert to gaussians
    gaussians = converter.mesh_to_gaussians(mesh, strategy=args.strategy)

    # Optimize if requested
    if args.optimize > 0 and TORCH_AVAILABLE and torch.cuda.is_available():
        gaussians = converter.optimize_gaussians(gaussians, iterations=args.optimize)
    elif args.optimize > 0:
        print("Optimization requested but PyTorch/CUDA not available")

    # Save full resolution
    base_name = Path(args.output).stem
    output_dir = Path(args.output).parent

    converter.save_ply(gaussians, args.output)

    # Generate LODs using LODGenerator
    if args.lod:
        lod_gen = LODGenerator(strategy='importance')
        for lod_count in args.lod:
            if lod_count < len(gaussians):
                lod_gaussians = lod_gen.generate_lod(gaussians, lod_count)
                lod_path = output_dir / f"{base_name}_lod_{lod_count}.ply"
                converter.save_ply(lod_gaussians, str(lod_path))

if __name__ == '__main__':
    main()