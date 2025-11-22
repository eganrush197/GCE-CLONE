# ABOUTME: Core mesh-to-gaussian converter implementation
# ABOUTME: Converts OBJ/GLB meshes to gaussian splats using direct geometric conversion

import numpy as np
import trimesh
from pathlib import Path
from typing import Union, Optional, List
from dataclasses import dataclass
from .gaussian_splat import GaussianSplat
from .lod_generator import LODGenerator


@dataclass
class ConversionConfig:
    """Configuration for mesh-to-gaussian conversion."""
    initialization_strategy: str = 'adaptive'  # 'vertex', 'face', 'hybrid', 'adaptive'
    target_gaussians: Optional[int] = None  # None = auto-determine
    samples_per_face: int = 10  # For face sampling
    scale_multiplier: float = 1.0  # Scale gaussians relative to local geometry
    opacity_default: float = 0.9  # Default opacity [0-1]
    optimize: bool = False  # Run quick optimization (requires PyTorch)
    optimization_iterations: int = 100  # Quick optimization iterations
    device: str = 'cpu'  # 'cpu' or 'cuda'


class MeshToGaussianConverter:
    """Converts 3D meshes to gaussian splat representations."""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        """
        Initialize converter.
        
        Args:
            config: Conversion configuration
        """
        self.config = config or ConversionConfig()
    
    def convert(self, mesh_path: Union[str, Path]) -> GaussianSplat:
        """
        Convert mesh to gaussian splat.
        
        Args:
            mesh_path: Path to OBJ or GLB mesh file
            
        Returns:
            GaussianSplat object
        """
        # Load mesh
        mesh = self._load_mesh(mesh_path)
        
        # Select strategy
        strategy = self._select_strategy(mesh)
        
        # Initialize gaussians
        if strategy == 'vertex':
            gaussians = self._initialize_from_vertices(mesh)
        elif strategy == 'face':
            gaussians = self._initialize_from_faces(mesh)
        elif strategy == 'hybrid':
            gaussians = self._initialize_hybrid(mesh)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Optional optimization
        if self.config.optimize:
            gaussians = self._optimize(gaussians, mesh)
        
        return gaussians
    
    def _load_mesh(self, mesh_path: Union[str, Path]) -> trimesh.Trimesh:
        """Load mesh from file."""
        mesh_path = Path(mesh_path)
        
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        
        mesh = trimesh.load(str(mesh_path), force='mesh')
        
        # Ensure mesh is a single Trimesh object
        if isinstance(mesh, trimesh.Scene):
            # Combine all geometries
            mesh = trimesh.util.concatenate(
                [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            )
        
        return mesh
    
    def _select_strategy(self, mesh: trimesh.Trimesh) -> str:
        """Auto-select initialization strategy based on mesh properties."""
        if self.config.initialization_strategy != 'adaptive':
            return self.config.initialization_strategy
        
        # Adaptive selection logic
        vertex_count = len(mesh.vertices)
        face_count = len(mesh.faces)
        
        # Low poly: use vertex strategy
        if vertex_count < 1000:
            return 'vertex'
        
        # High poly with textures: use face strategy
        if face_count > 10000 and mesh.visual.kind == 'texture':
            return 'face'
        
        # Default: hybrid
        return 'hybrid'
    
    def _initialize_from_vertices(self, mesh: trimesh.Trimesh) -> GaussianSplat:
        """Initialize gaussians at mesh vertices."""
        positions = mesh.vertices.copy()
        n = len(positions)
        
        # Extract colors
        colors = self._extract_vertex_colors(mesh)
        
        # Estimate scales from local geometry
        scales = self._estimate_scales_from_vertices(mesh)
        
        # Compute rotations from normals
        rotations = self._compute_rotations_from_normals(mesh.vertex_normals)
        
        # Default opacity (in log space for PLY)
        opacity = np.full(n, np.log(self.config.opacity_default / (1 - self.config.opacity_default)))
        
        return GaussianSplat(
            positions=positions,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacity=opacity
        )

    def _initialize_from_faces(self, mesh: trimesh.Trimesh) -> GaussianSplat:
        """Initialize gaussians by sampling points on faces."""
        # Sample points on mesh surface
        points, face_indices = trimesh.sample.sample_surface(
            mesh,
            count=self.config.target_gaussians or len(mesh.faces) * self.config.samples_per_face
        )

        n = len(points)
        positions = points

        # Extract colors at sampled points
        colors = self._extract_face_colors(mesh, face_indices)

        # Estimate scales from local face size
        scales = self._estimate_scales_from_faces(mesh, face_indices)

        # Compute rotations from face normals
        face_normals = mesh.face_normals[face_indices]
        rotations = self._compute_rotations_from_normals(face_normals)

        # Default opacity
        opacity = np.full(n, np.log(self.config.opacity_default / (1 - self.config.opacity_default)))

        return GaussianSplat(
            positions=positions,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacity=opacity
        )

    def _initialize_hybrid(self, mesh: trimesh.Trimesh) -> GaussianSplat:
        """Hybrid initialization: combine vertex and face sampling."""
        # Get vertex-based gaussians
        vertex_gaussians = self._initialize_from_vertices(mesh)

        # Get face-based gaussians (fewer samples)
        face_config = ConversionConfig(
            samples_per_face=max(1, self.config.samples_per_face // 2),
            target_gaussians=len(mesh.vertices) // 2
        )
        temp_converter = MeshToGaussianConverter(face_config)
        face_gaussians = temp_converter._initialize_from_faces(mesh)

        # Combine
        return GaussianSplat(
            positions=np.vstack([vertex_gaussians.positions, face_gaussians.positions]),
            scales=np.vstack([vertex_gaussians.scales, face_gaussians.scales]),
            rotations=np.vstack([vertex_gaussians.rotations, face_gaussians.rotations]),
            colors=np.vstack([vertex_gaussians.colors, face_gaussians.colors]),
            opacity=np.hstack([vertex_gaussians.opacity, face_gaussians.opacity])
        )

    def _extract_vertex_colors(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Extract colors from mesh vertices."""
        n = len(mesh.vertices)

        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            # Default gray
            colors = np.full((n, 3), 0.5)

        return colors.astype(np.float32)

    def _extract_face_colors(self, mesh: trimesh.Trimesh, face_indices: np.ndarray) -> np.ndarray:
        """Extract colors for sampled face points."""
        n = len(face_indices)

        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            # Average vertex colors of the face
            face_vertex_indices = mesh.faces[face_indices]
            vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0
            colors = vertex_colors[face_vertex_indices].mean(axis=1)
        else:
            # Default gray
            colors = np.full((n, 3), 0.5)

        return colors.astype(np.float32)

    def _estimate_scales_from_vertices(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Estimate gaussian scales from local vertex spacing."""
        from scipy.spatial import cKDTree

        tree = cKDTree(mesh.vertices)
        # Find 4 nearest neighbors (including self)
        distances, _ = tree.query(mesh.vertices, k=4)

        # Use mean distance to neighbors as scale estimate
        mean_distances = distances[:, 1:].mean(axis=1)

        # Isotropic scales (same in all directions), in log space
        scales = np.log(mean_distances[:, None] * self.config.scale_multiplier + 1e-8)
        scales = np.tile(scales, (1, 3))

        return scales.astype(np.float32)

    def _estimate_scales_from_faces(self, mesh: trimesh.Trimesh, face_indices: np.ndarray) -> np.ndarray:
        """Estimate gaussian scales from face areas."""
        # Get face areas
        face_areas = mesh.area_faces[face_indices]

        # Scale proportional to sqrt(area)
        scale_values = np.sqrt(face_areas) * self.config.scale_multiplier

        # Isotropic scales in log space
        scales = np.log(scale_values[:, None] + 1e-8)
        scales = np.tile(scales, (1, 3))

        return scales.astype(np.float32)

    def _compute_rotations_from_normals(self, normals: np.ndarray) -> np.ndarray:
        """Compute quaternion rotations to align gaussians with normals."""
        n = len(normals)

        # Normalize normals
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

        # Compute rotation from [0, 0, 1] to normal
        # Using quaternion from two vectors
        z_axis = np.array([0, 0, 1])

        quaternions = np.zeros((n, 4))

        for i in range(n):
            quaternions[i] = self._quaternion_from_vectors(z_axis, normals[i])

        return quaternions.astype(np.float32)

    def _quaternion_from_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Compute quaternion rotation from v1 to v2."""
        # Normalize
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)

        # Cross product and dot product
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)

        # Handle parallel vectors
        if dot > 0.9999:
            return np.array([1, 0, 0, 0])  # Identity quaternion
        elif dot < -0.9999:
            # 180 degree rotation around arbitrary perpendicular axis
            perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(v1, perp)
            axis = axis / np.linalg.norm(axis)
            return np.array([0, axis[0], axis[1], axis[2]])

        # Standard case
        w = 1 + dot
        q = np.array([w, cross[0], cross[1], cross[2]])
        q = q / np.linalg.norm(q)

        return q

    def _optimize(self, gaussians: GaussianSplat, mesh: trimesh.Trimesh) -> GaussianSplat:
        """Quick optimization of gaussian parameters (requires PyTorch)."""
        # TODO: Implement optional PyTorch-based optimization
        print("Warning: Optimization not yet implemented, returning unoptimized gaussians")
        return gaussians

    def generate_lods(self, gaussians: GaussianSplat, target_counts: List[int]) -> List[GaussianSplat]:
        """Generate multiple LOD levels."""
        lod_gen = LODGenerator(strategy='importance')
        return lod_gen.generate_lods(gaussians, target_counts)

