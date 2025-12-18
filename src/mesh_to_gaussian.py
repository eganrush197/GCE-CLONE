#!/usr/bin/env python3
# ABOUTME: Direct mesh-to-gaussian converter with LOD support
# ABOUTME: Converts OBJ/GLB meshes to gaussian splat PLY files

"""
Simplified Mesh to Gaussian Converter
No complex multi-view rendering - direct conversion with optimization
"""

import numpy as np
import trimesh
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List
import struct
import argparse
from pathlib import Path
from scipy.spatial import cKDTree
from collections import OrderedDict

# Try to import torch, but make it optional
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Note: Logger not yet initialized here, will log in __init__

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

    def __init__(self, device='cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu',
                 max_texture_cache_size: int = 10,
                 use_mipmaps: bool = True,
                 texture_filter: str = 'bilinear'):
        """
        Initialize converter.

        Args:
            device: Device to use ('cuda' or 'cpu')
            max_texture_cache_size: Maximum number of textures to cache (default: 10)
                                   Set to 0 to disable caching
            use_mipmaps: Whether to generate and use mipmaps for texture filtering (default: True)
            texture_filter: Texture filtering mode - 'nearest' or 'bilinear' (default: 'bilinear')
        """
        self.logger = logging.getLogger('gaussian_pipeline')
        self.device = device
        self.max_texture_cache_size = max_texture_cache_size
        self.use_mipmaps = use_mipmaps
        self.texture_filter = texture_filter

        # Initialize LRU texture cache (OrderedDict maintains insertion order)
        self._texture_cache = OrderedDict()

        if TORCH_AVAILABLE:
            self.logger.debug("Using device: %s", device)
        else:
            self.logger.info("PyTorch not available - using NumPy only mode")

        self.logger.debug("Texture cache size limit: %d", max_texture_cache_size)
        self.logger.debug("Texture filtering: %s (mipmaps: %s)", texture_filter, use_mipmaps)

    def _validate_mesh(self, mesh: trimesh.Trimesh) -> None:
        """
        Validate mesh has reasonable properties for conversion.

        Args:
            mesh: Mesh to validate

        Raises:
            ValueError: If mesh is invalid or degenerate
        """
        # Check for vertices
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh has no vertices")

        # Check for faces
        if len(mesh.faces) == 0:
            raise ValueError(
                "Mesh has no faces (point clouds are not supported). "
                "Please provide a mesh with faces."
            )

        # Check for NaN or Inf values
        if np.any(~np.isfinite(mesh.vertices)):
            raise ValueError(
                "Mesh contains NaN or Inf vertex coordinates. "
                "Please check your mesh file for corruption."
            )

        # Check for degenerate extent (all vertices at same point)
        extent = mesh.extents
        if np.any(extent < 1e-8):
            raise ValueError(
                f"Mesh has degenerate extent: {extent}. "
                "All vertices appear to be at the same location. "
                "Please check your mesh file."
            )

        self.logger.debug("Mesh validation passed: %d vertices, %d faces",
                         len(mesh.vertices), len(mesh.faces))

    def load_mesh(self, path: str) -> trimesh.Trimesh:
        """
        Load and normalize mesh, with MTL color support for OBJ files.

        Args:
            path: Path to mesh file (.obj, .glb, .fbx, etc.)

        Returns:
            Loaded and normalized mesh

        Raises:
            ValueError: If mesh is invalid or degenerate
            FileNotFoundError: If file doesn't exist
        """
        # Check if it's an OBJ file with potential MTL
        if path.endswith('.obj'):
            mesh = self._load_obj_with_mtl(path)
        else:
            mesh = trimesh.load(path, force='mesh')

        # Validate mesh before processing
        self._validate_mesh(mesh)

        # Center and scale to unit cube
        mesh.vertices -= mesh.vertices.mean(axis=0)
        scale = np.abs(mesh.vertices).max()

        # Check for zero scale (should be caught by validation, but double-check)
        if scale < 1e-8:
            raise ValueError(
                "Cannot normalize mesh: all vertices are at the same point after centering. "
                "This indicates a degenerate mesh."
            )

        mesh.vertices /= scale

        self.logger.debug("Loaded mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
        return mesh

    def _load_obj_with_mtl(self, obj_path: str) -> trimesh.Trimesh:
        """Special OBJ loader that preserves MTL material colors AND textures"""
        from pathlib import Path
        from PIL import Image

        # First load with trimesh
        mesh = trimesh.load(obj_path, force='mesh', process=False)

        # CRITICAL: Preserve UV coordinates before any visual changes
        # Store them as a mesh attribute so they survive visual type changes
        original_uvs = None
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            original_uvs = mesh.visual.uv.copy()
            self.logger.debug("Preserved %d UV coordinates from mesh", len(original_uvs))

        # Check for MTL file
        mtl_path = Path(obj_path).with_suffix('.mtl')
        if not mtl_path.exists():
            return mesh

        self.logger.debug("Found MTL file: %s", mtl_path)

        # Parse MTL for material colors AND texture maps
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
                        'Kd': [0.7, 0.7, 0.7],
                        'map_Kd': None
                    }

                elif parts[0] == 'Kd' and current_mat:
                    try:
                        materials[current_mat]['Kd'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                    except:
                        pass

                elif parts[0] == 'map_Kd' and current_mat:
                    # Store texture filename
                    materials[current_mat]['map_Kd'] = parts[1]

        # Load texture if referenced
        texture_image = None
        for mat_name, mat_data in materials.items():
            if mat_data['map_Kd']:
                texture_path = Path(obj_path).parent / mat_data['map_Kd']
                if texture_path.exists():
                    texture_image = Image.open(texture_path)
                    self.logger.debug("Loaded texture: %s (%s)", texture_path, texture_image.size)
                    break
                else:
                    self.logger.debug("Texture not found: %s", texture_path)

        # If we have a texture, use trimesh's UV texture system
        if texture_image is not None:
            # Convert mesh to have texture visuals
            # Use original_uvs since we preserved them at the start
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=original_uvs,
                image=texture_image
            )
            return mesh

        # Otherwise fall back to existing face color system
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
                    color = materials.get(current_material, {}).get('Kd', [0.7, 0.7, 0.7])

                    # Count vertices in this face (handles quads, tris, etc.)
                    # OBJ faces can be: f v1 v2 v3 (triangle) or f v1 v2 v3 v4 (quad)
                    # or f v1/vt1/vn1 v2/vt2/vn2 ... (with texture/normal indices)
                    num_vertices = len(parts) - 1  # Subtract 'f' command

                    # Trimesh triangulates: quads (4 verts) -> 2 triangles, etc.
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
            self.logger.debug("Applied %d material colors to %d faces", len(materials), len(face_colors))
        elif face_colors:
            self.logger.warning("Face color count mismatch (%d colors vs %d faces)", len(face_colors), len(mesh.faces))
            self.logger.warning("Attempting to use default color for all faces...")
            # Fallback: use first material color or gray for all faces
            default_color = list(materials.values())[0]['Kd'] if materials else [0.7, 0.7, 0.7]
            face_colors = np.array([default_color] * len(mesh.faces))
            if face_colors.max() <= 1.0:
                face_colors = (face_colors * 255).astype(np.uint8)
            face_colors = np.column_stack([face_colors, np.full(len(face_colors), 255)])
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=face_colors)
            self.logger.debug("Using fallback color for all %d faces", len(mesh.faces))

        # CRITICAL: Restore UV coordinates if we had them
        # ColorVisuals doesn't support UVs, so we store them as a custom attribute
        if original_uvs is not None:
            mesh.metadata['vertex_uvs'] = original_uvs
            self.logger.debug("Stored %d UV coordinates in mesh metadata", len(original_uvs))

        return mesh

    def _has_texture_visual(self, mesh: trimesh.Trimesh) -> bool:
        """
        Check if mesh has valid texture visual with all required attributes.

        Args:
            mesh: Mesh to check

        Returns:
            True if mesh has valid texture visual, False otherwise
        """
        try:
            # Check if mesh has visual attribute
            if not hasattr(mesh, 'visual') or mesh.visual is None:
                return False

            # Check if visual is TextureVisuals (not ColorVisuals)
            if not isinstance(mesh.visual, trimesh.visual.TextureVisuals):
                return False

            # Check if visual has UV coordinates
            if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
                return False

            # Check if visual has material
            if not hasattr(mesh.visual, 'material') or mesh.visual.material is None:
                return False

            # Check if material has image
            if not hasattr(mesh.visual.material, 'image') or mesh.visual.material.image is None:
                return False

            return True

        except (AttributeError, TypeError):
            # Any attribute access error means no valid texture
            return False

    def _sample_texture_color(self, mesh: trimesh.Trimesh, vertex_idx: int) -> np.ndarray:
        """
        Sample color from texture using UV coordinates for a vertex.

        Args:
            mesh: Mesh with texture
            vertex_idx: Index of vertex to sample

        Returns:
            RGB color array or None if no texture available
        """
        # Use comprehensive check
        if not self._has_texture_visual(mesh):
            return None

        try:
            # Get UV coordinate for this vertex
            uv = mesh.visual.uv[vertex_idx]
            return self._sample_texture_at_uv(mesh.visual.material.image, uv)
        except (IndexError, KeyError, AttributeError) as e:
            self.logger.debug("Texture sampling failed for vertex %d: %s", vertex_idx, e)
            return None

    def _sample_texture_at_uv(self, image, uv: np.ndarray) -> np.ndarray:
        """Sample color from texture image at given UV coordinate"""
        # UV coordinates are in [0, 1] range
        # Convert to pixel coordinates
        width, height = image.size

        # UV origin is bottom-left, but image origin is top-left
        # So we need to flip V coordinate
        u = uv[0]
        v = 1.0 - uv[1]

        # Clamp to [0, 1] range
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        # Convert to pixel coordinates
        x = int(u * (width - 1))
        y = int(v * (height - 1))

        # Sample the texture
        pixel = image.getpixel((x, y))

        # Convert to RGB array (handle RGBA or RGB)
        if isinstance(pixel, tuple):
            color = np.array(pixel[:3], dtype=np.float32) / 255.0
        else:
            # Grayscale
            color = np.array([pixel, pixel, pixel], dtype=np.float32) / 255.0

        return color

    def _sample_texture_batch(self, image, uvs: np.ndarray) -> np.ndarray:
        """
        Batch sample colors from texture image at given UV coordinates.

        Args:
            image: PIL Image
            uvs: Array of UV coordinates (n_samples, 2)

        Returns:
            colors: Array of RGB colors (n_samples, 3)
        """
        # Use image object as cache key
        cache_key = id(image)

        # Check if in cache (LRU: move to end if found)
        if cache_key in self._texture_cache:
            # Move to end (most recently used)
            self._texture_cache.move_to_end(cache_key)
            img_array = self._texture_cache[cache_key]
        else:
            # Convert PIL image to numpy array (H, W, C)
            img_array = np.array(image, dtype=np.float32) / 255.0

            # Handle different image modes
            if len(img_array.shape) == 2:
                # Grayscale - convert to RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                # RGBA - drop alpha channel
                img_array = img_array[:, :, :3]

            # Add to cache
            self._texture_cache[cache_key] = img_array

            # Evict oldest entry if cache is full (LRU eviction)
            if self.max_texture_cache_size > 0 and len(self._texture_cache) > self.max_texture_cache_size:
                # Remove oldest (first) entry
                oldest_key = next(iter(self._texture_cache))
                evicted = self._texture_cache.pop(oldest_key)

                # Calculate memory freed
                memory_mb = evicted.nbytes / (1024 * 1024)
                self.logger.debug("Evicted texture from cache (freed %.1f MB), cache size: %d/%d",
                                memory_mb, len(self._texture_cache), self.max_texture_cache_size)
        height, width = img_array.shape[:2]

        # UV coordinates are in [0, 1] range
        # UV origin is bottom-left, but image origin is top-left
        u = uvs[:, 0]
        v = 1.0 - uvs[:, 1]

        # Clamp to [0, 1] range
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        # Convert to pixel coordinates (vectorized)
        x = (u * (width - 1)).astype(int)
        y = (v * (height - 1)).astype(int)

        # Batch sample (vectorized indexing)
        colors = img_array[y, x, :]  # (n_samples, 3)

        return colors

    def _sample_texture_interpolated(self, mesh: trimesh.Trimesh, face: np.ndarray,
                                     w1: float, w2: float, w3: float) -> np.ndarray:
        """
        Sample color from texture using interpolated UV coordinates.

        Args:
            mesh: Mesh with texture
            face: Face vertex indices (3,)
            w1, w2, w3: Barycentric weights

        Returns:
            RGB color array or None if no texture available
        """
        # Use comprehensive check
        if not self._has_texture_visual(mesh):
            return None

        try:
            # Get UV coordinates for the three vertices of the face
            uv0 = mesh.visual.uv[face[0]]
            uv1 = mesh.visual.uv[face[1]]
            uv2 = mesh.visual.uv[face[2]]

            # Interpolate UV coordinates using barycentric weights
            uv_interpolated = w1 * uv0 + w2 * uv1 + w3 * uv2
        except (IndexError, KeyError, AttributeError) as e:
            self.logger.debug("Texture interpolation failed: %s", e)
            return None

        # Sample texture at interpolated UV
        return self._sample_texture_at_uv(mesh.visual.material.image, uv_interpolated)

    def _sample_colors_vectorized(self, mesh: trimesh.Trimesh,
                                   face_indices: np.ndarray,
                                   w1: np.ndarray,
                                   w2: np.ndarray,
                                   w3: np.ndarray) -> np.ndarray:
        """
        Vectorized color sampling for multiple face samples.

        Args:
            mesh: The mesh to sample from
            face_indices: Array of face indices (n_samples,)
            w1, w2, w3: Barycentric weights (n_samples,)

        Returns:
            colors: Array of RGB colors (n_samples, 3)
        """
        n_samples = len(face_indices)
        colors = np.zeros((n_samples, 3))

        # Try texture sampling first using comprehensive check
        if self._has_texture_visual(mesh):
            try:
                # OPTIMIZED: Fully vectorized texture sampling
                faces = mesh.faces[face_indices]  # (n_samples, 3)

                # Get UV coordinates for all samples at once
                uv0 = mesh.visual.uv[faces[:, 0]]  # (n_samples, 2)
                uv1 = mesh.visual.uv[faces[:, 1]]  # (n_samples, 2)
                uv2 = mesh.visual.uv[faces[:, 2]]  # (n_samples, 2)

                # Interpolate UVs (vectorized)
                uvs = (w1[:, None] * uv0 +
                      w2[:, None] * uv1 +
                      w3[:, None] * uv2)  # (n_samples, 2)

                # Batch texture sampling
                colors = self._sample_texture_batch(mesh.visual.material.image, uvs)
                return colors
            except (IndexError, KeyError, AttributeError) as e:
                self.logger.debug("Vectorized texture sampling failed: %s", e)
                # Fall through to vertex colors

        # Fall back to vertex colors (vectorized)
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            faces = mesh.faces[face_indices]  # (n_samples, 3)
            v_colors = mesh.visual.vertex_colors[faces][:, :, :3]  # (n_samples, 3, 3)

            # Normalize to 0-1 range if needed
            if v_colors.max() > 1.0:
                v_colors = v_colors / 255.0

            # Vectorized barycentric interpolation
            colors = (w1[:, None] * v_colors[:, 0, :] +
                     w2[:, None] * v_colors[:, 1, :] +
                     w3[:, None] * v_colors[:, 2, :])  # (n_samples, 3)
            return colors

        # Fall back to face colors (vectorized)
        if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
            face_colors = mesh.visual.face_colors[face_indices][:, :3]  # (n_samples, 3)
            if face_colors.max() > 1.0:
                colors = face_colors / 255.0
            else:
                colors = face_colors
            return colors

        # Default gray
        colors[:] = 0.5
        return colors

    def _compute_vertex_scales_fast(self, vertices: np.ndarray, n_vertices: int) -> np.ndarray:
        """
        Compute vertex scales using KD-tree for O(n log n) performance.

        This replaces the O(n^2) nested loop that computed distances to all vertices.
        For large meshes (100k+ vertices), this provides 100x+ speedup.

        Args:
            vertices: Vertex positions (n_vertices, 3)
            n_vertices: Number of vertices

        Returns:
            scales: Scale for each vertex (n_vertices,)
        """
        # Log progress for large meshes
        if n_vertices > 10000:
            self.logger.info("Computing vertex scales for %d vertices using KD-tree...", n_vertices)

        # Build KD-tree for fast nearest neighbor search
        # This takes O(n log n) time
        tree = cKDTree(vertices)

        # Query 2 nearest neighbors for each vertex (self + nearest)
        # This takes O(n log n) time total
        # k=2 because first neighbor is the vertex itself (distance 0)
        distances, indices = tree.query(vertices, k=2)

        # Extract distance to nearest neighbor (index 1, since index 0 is self)
        nearest_distances = distances[:, 1]

        # Scale is half the distance to nearest neighbor
        # Use 0.01 as minimum scale for isolated vertices
        scales = np.where(nearest_distances > 0, nearest_distances * 0.5, 0.01)

        if n_vertices > 10000:
            self.logger.info("Vertex scales computed (min: %.4f, max: %.4f, mean: %.4f)",
                           scales.min(), scales.max(), scales.mean())

        return scales

    def _sample_vertex_colors_vectorized(self, mesh: trimesh.Trimesh, n_vertices: int) -> np.ndarray:
        """
        Sample colors for all vertices using vectorized operations.

        This replaces the loop-based texture sampling for 10x+ speedup.

        Args:
            mesh: Mesh with visual data
            n_vertices: Number of vertices

        Returns:
            colors: RGB colors for each vertex (n_vertices, 3)
        """
        # Initialize with default gray
        colors = np.full((n_vertices, 3), 0.5, dtype=np.float32)

        # Try texture sampling first (vectorized)
        if self._has_texture_visual(mesh):
            try:
                # Get all UV coordinates at once
                uvs = mesh.visual.uv  # (n_vertices, 2)

                # Batch sample all texture colors
                colors = self._sample_texture_batch(mesh.visual.material.image, uvs)

                if n_vertices > 10000:
                    self.logger.info("Sampled %d vertex colors from texture (vectorized)", n_vertices)

                return colors

            except (IndexError, KeyError, AttributeError) as e:
                self.logger.debug("Vectorized texture sampling failed: %s, falling back to vertex colors", e)

        # Fall back to vertex colors if available
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            v_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32)

            # Normalize to 0-1 range if needed
            if v_colors.max() > 1.0:
                v_colors = v_colors / 255.0

            colors = v_colors

            if n_vertices > 10000:
                self.logger.info("Using %d vertex colors", n_vertices)

        return colors

    def clear_texture_cache(self):
        """
        Clear the texture cache to free memory.

        This is automatically called after each mesh conversion, but can be
        called manually if needed.
        """
        if self._texture_cache:
            total_memory = sum(arr.nbytes for arr in self._texture_cache.values()) / (1024 * 1024)
            self.logger.debug("Clearing texture cache (freeing %.1f MB)", total_memory)
            self._texture_cache.clear()

    def _generate_mipmaps(self, texture: np.ndarray, max_levels: int = 8) -> list:
        """
        Generate mipmap pyramid for a texture.

        Args:
            texture: Base texture array (H, W, C) or (H, W)
            max_levels: Maximum number of mipmap levels to generate

        Returns:
            List of mipmap levels [level0, level1, level2, ...]
            where level0 is the original texture
        """
        from PIL import Image

        mipmaps = [texture]  # Level 0 is the original

        current_texture = texture
        level = 0

        while level < max_levels - 1:
            height, width = current_texture.shape[:2]

            # Stop if texture is too small
            if width <= 1 or height <= 1:
                break

            # Calculate next mipmap size (half resolution)
            new_width = max(1, width // 2)
            new_height = max(1, height // 2)

            # Convert to PIL Image for high-quality downsampling
            if len(current_texture.shape) == 3:
                # Color texture
                img = Image.fromarray((current_texture * 255).astype(np.uint8))
            else:
                # Grayscale texture
                img = Image.fromarray((current_texture * 255).astype(np.uint8), mode='L')

            # Resize with LANCZOS filter (high quality)
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert back to numpy array
            mipmap = np.array(img_resized, dtype=np.float32) / 255.0

            mipmaps.append(mipmap)
            current_texture = mipmap
            level += 1

        self.logger.debug("Generated %d mipmap levels", len(mipmaps))
        return mipmaps

    def _load_material_textures(self, manifest: dict, use_mipmaps: bool = True) -> dict:
        """
        Load all textures referenced in the material manifest.

        Args:
            manifest: Material manifest from packed extraction
            use_mipmaps: Whether to generate mipmap pyramids for textures

        Returns:
            Dict mapping material name -> dict of loaded texture arrays or mipmap lists
        """
        from PIL import Image

        material_textures = {}

        for mat_name, mat_data in manifest.get('materials', {}).items():
            textures = {}

            # Load diffuse texture
            diffuse_entry = mat_data.get('diffuse')
            if diffuse_entry:
                try:
                    # Handle both old format (string path) and new format (dict with path and uv_layer)
                    diffuse_path = diffuse_entry['path'] if isinstance(diffuse_entry, dict) else diffuse_entry

                    img = Image.open(diffuse_path)
                    base_texture = np.array(img, dtype=np.float32) / 255.0
                    if len(base_texture.shape) == 2:
                        base_texture = np.stack([base_texture] * 3, axis=-1)

                    if use_mipmaps:
                        textures['diffuse'] = self._generate_mipmaps(base_texture)
                        self.logger.debug("  Loaded diffuse for %s (%d mipmap levels)",
                                        mat_name, len(textures['diffuse']))
                    else:
                        textures['diffuse'] = base_texture
                        self.logger.debug("  Loaded diffuse for %s", mat_name)
                except Exception as e:
                    self.logger.warning("Failed to load diffuse for %s: %s", mat_name, e)

            # Load transparency texture (or use diffuse alpha)
            transparency_entry = mat_data.get('transparency')
            if transparency_entry:
                try:
                    # Handle both old format (string path) and new format (dict with path and uv_layer)
                    transparency_path = transparency_entry['path'] if isinstance(transparency_entry, dict) else transparency_entry

                    img = Image.open(transparency_path)
                    base_texture = np.array(img, dtype=np.float32) / 255.0
                    if len(base_texture.shape) == 3:
                        base_texture = base_texture[:, :, 0]  # Use first channel

                    if use_mipmaps:
                        textures['transparency'] = self._generate_mipmaps(base_texture)
                        self.logger.debug("  Loaded transparency for %s (%d mipmap levels)",
                                        mat_name, len(textures['transparency']))
                    else:
                        textures['transparency'] = base_texture
                        self.logger.debug("  Loaded transparency for %s", mat_name)
                except Exception as e:
                    self.logger.warning("Failed to load transparency for %s: %s", mat_name, e)
            elif mat_data.get('diffuse_has_alpha') and 'diffuse' in textures:
                # Use alpha channel from diffuse
                try:
                    img = Image.open(mat_data['diffuse'])
                    if img.mode == 'RGBA':
                        base_texture = np.array(img, dtype=np.float32)[:, :, 3] / 255.0

                        if use_mipmaps:
                            textures['transparency'] = self._generate_mipmaps(base_texture)
                            self.logger.debug("  Using diffuse alpha for transparency: %s (%d mipmap levels)",
                                            mat_name, len(textures['transparency']))
                        else:
                            textures['transparency'] = base_texture
                            self.logger.debug("  Using diffuse alpha for transparency: %s", mat_name)
                except Exception as e:
                    self.logger.warning("Failed to extract alpha from diffuse for %s: %s", mat_name, e)

            # Load roughness texture
            roughness_entry = mat_data.get('roughness')
            if roughness_entry:
                try:
                    # Handle both old format (string path) and new format (dict with path and uv_layer)
                    roughness_path = roughness_entry['path'] if isinstance(roughness_entry, dict) else roughness_entry

                    img = Image.open(roughness_path)
                    base_texture = np.array(img, dtype=np.float32) / 255.0
                    if len(base_texture.shape) == 3:
                        base_texture = base_texture[:, :, 0]  # Use first channel

                    # Invert if it's a glossiness map
                    if mat_data.get('is_glossy', False):
                        base_texture = 1.0 - base_texture
                        label = "glossiness (inverted)"
                    else:
                        label = "roughness"

                    if use_mipmaps:
                        textures['roughness'] = self._generate_mipmaps(base_texture)
                        self.logger.debug("  Loaded %s for %s (%d mipmap levels)",
                                        label, mat_name, len(textures['roughness']))
                    else:
                        textures['roughness'] = base_texture
                        self.logger.debug("  Loaded %s for %s", label, mat_name)
                except Exception as e:
                    self.logger.warning("Failed to load roughness for %s: %s", mat_name, e)

            material_textures[mat_name] = textures

        return material_textures

    def _load_uv_layers(self, manifest: dict) -> dict:
        """
        Load UV layers from manifest.

        UV coordinates are stored per-loop (face corner) in Blender.

        Args:
            manifest: Material manifest from packed extraction

        Returns:
            Dict mapping UV layer name -> array of UV coordinates (n_loops, 2)
        """
        uv_layers = {}
        uv_layer_files = manifest.get('uv_layers', {})

        if not uv_layer_files:
            self.logger.debug("No UV layers in manifest")
            return {}

        for uv_layer_name, uv_path in uv_layer_files.items():
            try:
                uv_coords = np.load(uv_path)
                uv_layers[uv_layer_name] = uv_coords
                self.logger.info("Loaded UV layer '%s': %s", uv_layer_name, uv_coords.shape)
            except Exception as e:
                self.logger.warning("Failed to load UV layer '%s' from %s: %s",
                                  uv_layer_name, uv_path, e)

        return uv_layers

    def _load_vertex_colors(self, manifest: dict) -> Optional[np.ndarray]:
        """
        Load vertex colors from manifest.

        Vertex colors are stored per-loop (face corner) in Blender.

        Args:
            manifest: Material manifest from packed extraction

        Returns:
            Array of RGBA colors per loop (n_loops, 4), or None if not present
        """
        vertex_color_path = manifest.get('vertex_colors')

        if not vertex_color_path:
            self.logger.debug("No vertex colors in manifest")
            return None

        try:
            vertex_colors = np.load(vertex_color_path)
            self.logger.info("Loaded vertex colors: %s", vertex_colors.shape)
            return vertex_colors
        except Exception as e:
            self.logger.warning("Failed to load vertex colors from %s: %s", vertex_color_path, e)
            return None

    def _interpolate_vertex_color(self,
                                   vertex_colors: np.ndarray,
                                   face_idx: int,
                                   w1: float, w2: float, w3: float) -> np.ndarray:
        """
        Interpolate vertex color using barycentric coordinates.

        Args:
            vertex_colors: Per-loop RGBA colors (n_loops, 4)
            face_idx: Face index
            w1, w2, w3: Barycentric weights

        Returns:
            Interpolated RGBA color (4,)
        """
        # Each face has 3 loops (corners) for triangles
        # Loop indices are: face_idx * 3, face_idx * 3 + 1, face_idx * 3 + 2
        loop_start = face_idx * 3

        # Check bounds - if face_idx is out of range, return white (neutral for multiply)
        if loop_start + 2 >= len(vertex_colors):
            # Face index is beyond the vertex color data
            # This can happen if the mesh was triangulated after vertex colors were exported
            return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        color0 = vertex_colors[loop_start]
        color1 = vertex_colors[loop_start + 1]
        color2 = vertex_colors[loop_start + 2]

        # Interpolate using barycentric weights
        interpolated = w1 * color0 + w2 * color1 + w3 * color2

        return interpolated

    def _blend_vertex_color(self,
                            texture_color: np.ndarray,
                            vertex_color: np.ndarray,
                            blend_mode: str = 'multiply') -> np.ndarray:
        """
        Blend texture color with vertex color using specified blend mode.

        Args:
            texture_color: RGB color from texture (3,)
            vertex_color: RGBA color from vertex (4,)
            blend_mode: Blending mode - 'multiply', 'add', 'overlay', 'replace'

        Returns:
            Blended RGB color (3,)
        """
        # Extract RGB from vertex color (ignore alpha for now)
        vc_rgb = vertex_color[:3]

        if blend_mode == 'multiply':
            # Multiply mode: good for ambient occlusion
            # Darkens the texture based on vertex color
            return texture_color * vc_rgb

        elif blend_mode == 'add':
            # Add mode: good for highlights
            # Brightens the texture based on vertex color
            result = texture_color + vc_rgb
            return np.clip(result, 0.0, 1.0)

        elif blend_mode == 'overlay':
            # Overlay mode: good for contrast
            # Combines multiply and screen based on texture brightness
            # If texture < 0.5: multiply, else screen
            result = np.where(
                texture_color < 0.5,
                2.0 * texture_color * vc_rgb,
                1.0 - 2.0 * (1.0 - texture_color) * (1.0 - vc_rgb)
            )
            return np.clip(result, 0.0, 1.0)

        elif blend_mode == 'replace':
            # Replace mode: use vertex color only (ignore texture)
            return vc_rgb

        else:
            self.logger.warning("Unknown blend mode '%s', using multiply", blend_mode)
            return texture_color * vc_rgb

    def _calculate_mipmap_level(self, scales: np.ndarray, texture_size: tuple) -> np.ndarray:
        """
        Calculate appropriate mipmap level based on gaussian scale.

        The idea is that larger gaussians (which cover more screen space) should
        sample from lower-resolution mipmaps to avoid aliasing.

        Args:
            scales: Gaussian scales (N, 3) - we use the max of x,y scales
            texture_size: (width, height) of the base texture

        Returns:
            Mipmap levels (N,) as floats (can be fractional for trilinear filtering)
        """
        # Use the maximum of x,y scales (ignore z which is thickness)
        max_scale = np.maximum(scales[:, 0], scales[:, 1])

        # Heuristic: larger gaussians -> higher mipmap level (lower resolution)
        # A gaussian with scale 1.0 should use level 0 (full resolution)
        # A gaussian with scale 2.0 should use level 1 (half resolution)
        # etc.

        # Calculate level based on scale
        # log2(scale) gives us the mipmap level
        # Clamp to reasonable range
        levels = np.log2(np.maximum(max_scale, 0.1))  # Avoid log(0)
        levels = np.clip(levels, 0.0, 7.0)  # Max 8 mipmap levels

        return levels

    def _sample_texture_bilinear(self,
                                  texture,
                                  uvs: np.ndarray,
                                  num_channels: int = 3,
                                  mipmap_levels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample texture using bilinear interpolation or nearest-neighbor.
        Supports both single textures and mipmap pyramids.

        Args:
            texture: Texture array (H, W, C) or (H, W) for grayscale,
                    OR list of mipmap levels
            uvs: UV coordinates (N, 2) in range [0, 1]
            num_channels: Number of channels to sample (3 for RGB, 1 for grayscale)
            mipmap_levels: Optional mipmap levels to sample from (N,)

        Returns:
            Sampled colors (N, num_channels) or (N,) for grayscale
        """
        # Check if texture is a mipmap pyramid (list) or single texture (array)
        is_mipmap = isinstance(texture, list)

        if is_mipmap and mipmap_levels is not None:
            # Sample from appropriate mipmap level
            # For simplicity, we'll use nearest mipmap level (no trilinear filtering yet)
            level_indices = np.round(mipmap_levels).astype(int)
            level_indices = np.clip(level_indices, 0, len(texture) - 1)

            # Group samples by mipmap level for efficient processing
            unique_levels = np.unique(level_indices)

            n_samples = len(uvs)
            if num_channels == 3:
                result = np.zeros((n_samples, 3), dtype=np.float32)
            else:
                result = np.zeros(n_samples, dtype=np.float32)

            for level in unique_levels:
                mask = level_indices == level
                level_uvs = uvs[mask]

                # Sample from this mipmap level
                level_texture = texture[level]
                if self.texture_filter == 'nearest':
                    level_samples = self._sample_single_texture_nearest(
                        level_texture, level_uvs, num_channels
                    )
                else:
                    level_samples = self._sample_single_texture_bilinear(
                        level_texture, level_uvs, num_channels
                    )

                result[mask] = level_samples

            return result
        else:
            # Single texture or no mipmap levels specified
            base_texture = texture[0] if is_mipmap else texture
            if self.texture_filter == 'nearest':
                return self._sample_single_texture_nearest(base_texture, uvs, num_channels)
            else:
                return self._sample_single_texture_bilinear(base_texture, uvs, num_channels)

    def _sample_single_texture_nearest(self,
                                        texture: np.ndarray,
                                        uvs: np.ndarray,
                                        num_channels: int = 3) -> np.ndarray:
        """
        Sample a single texture using nearest-neighbor interpolation.

        Args:
            texture: Texture array (H, W, C) or (H, W) for grayscale
            uvs: UV coordinates (N, 2) in range [0, 1]
            num_channels: Number of channels to sample (3 for RGB, 1 for grayscale)

        Returns:
            Sampled colors (N, num_channels) or (N,) for grayscale
        """
        height, width = texture.shape[:2]

        # Clip UVs to valid range
        u = np.clip(uvs[:, 0], 0.0, 1.0)
        v = np.clip(1.0 - uvs[:, 1], 0.0, 1.0)  # Flip V

        # Convert to pixel coordinates and round to nearest
        x = np.round(u * (width - 1)).astype(int)
        y = np.round(v * (height - 1)).astype(int)

        # Clamp to valid range
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        # Sample texture
        if len(texture.shape) == 3:
            # Color texture (H, W, C)
            result = texture[y, x, :num_channels]
        else:
            # Grayscale texture (H, W)
            result = texture[y, x]

        return result

    def _sample_single_texture_bilinear(self,
                                         texture: np.ndarray,
                                         uvs: np.ndarray,
                                         num_channels: int = 3) -> np.ndarray:
        """
        Sample a single texture using bilinear interpolation.

        Args:
            texture: Texture array (H, W, C) or (H, W) for grayscale
            uvs: UV coordinates (N, 2) in range [0, 1]
            num_channels: Number of channels to sample (3 for RGB, 1 for grayscale)

        Returns:
            Sampled colors (N, num_channels) or (N,) for grayscale
        """
        height, width = texture.shape[:2]

        # Clip UVs to valid range
        u = np.clip(uvs[:, 0], 0.0, 1.0)
        v = np.clip(1.0 - uvs[:, 1], 0.0, 1.0)  # Flip V

        # Convert to pixel coordinates (continuous)
        x = u * (width - 1)
        y = v * (height - 1)

        # Get integer parts (floor)
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)

        # Get next pixel (ceiling), clamped to texture bounds
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)

        # Get fractional parts for interpolation
        fx = x - x0
        fy = y - y0

        # Sample four corners
        if len(texture.shape) == 3:
            # Color texture (H, W, C)
            c00 = texture[y0, x0, :num_channels]  # Top-left
            c10 = texture[y0, x1, :num_channels]  # Top-right
            c01 = texture[y1, x0, :num_channels]  # Bottom-left
            c11 = texture[y1, x1, :num_channels]  # Bottom-right
        else:
            # Grayscale texture (H, W)
            c00 = texture[y0, x0]
            c10 = texture[y0, x1]
            c01 = texture[y1, x0]
            c11 = texture[y1, x1]

        # Bilinear interpolation
        # Interpolate along x-axis
        c0 = c00 * (1 - fx[:, None] if len(texture.shape) == 3 else 1 - fx) + \
             c10 * (fx[:, None] if len(texture.shape) == 3 else fx)
        c1 = c01 * (1 - fx[:, None] if len(texture.shape) == 3 else 1 - fx) + \
             c11 * (fx[:, None] if len(texture.shape) == 3 else fx)

        # Interpolate along y-axis
        result = c0 * (1 - fy[:, None] if len(texture.shape) == 3 else 1 - fy) + \
                 c1 * (fy[:, None] if len(texture.shape) == 3 else fy)

        return result

    def _sample_multi_material_colors(self,
                                       face_indices: np.ndarray,
                                       uvs: np.ndarray,
                                       face_materials: list,
                                       material_textures: dict,
                                       vertex_colors: Optional[np.ndarray] = None,
                                       barycentric_weights: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                                       vertex_color_blend_mode: str = 'multiply',
                                       scales: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample colors, opacity, and roughness from multiple material textures.
        Optionally blend with vertex colors and use mipmapping based on scales.

        Args:
            face_indices: Array of face indices (n_samples,)
            uvs: Array of UV coordinates (n_samples, 2)
            face_materials: List of material names per face
            material_textures: Dict of loaded textures per material
            vertex_colors: Optional per-loop RGBA colors (n_loops, 4)
            barycentric_weights: Optional tuple of (w1, w2, w3) for vertex color interpolation
            vertex_color_blend_mode: Blending mode for vertex colors
            scales: Optional gaussian scales (n_samples,) for mipmap level calculation

        Returns:
            Tuple of (colors, opacities, roughness) arrays
        """
        n_samples = len(face_indices)
        colors = np.full((n_samples, 3), 0.5, dtype=np.float32)
        opacities = np.ones(n_samples, dtype=np.float32)
        roughness = np.full(n_samples, 0.5, dtype=np.float32)

        # Group samples by material for efficient batch processing
        material_groups = {}
        for i, face_idx in enumerate(face_indices):
            mat_name = face_materials[face_idx] if face_idx < len(face_materials) else None
            if mat_name not in material_groups:
                material_groups[mat_name] = []
            material_groups[mat_name].append(i)

        # Debug: log material group sizes
        self.logger.debug("Material groups for sampling:")
        for mat_name, indices in material_groups.items():
            self.logger.debug("  %s: %d samples", mat_name, len(indices))

        # Calculate mipmap levels if scales are provided
        mipmap_levels = None
        if scales is not None:
            # Create a dummy scales array with shape (n_samples, 3)
            # We use the base scale for x, y, and a smaller value for z
            scales_3d = np.column_stack([scales, scales, scales * 0.3])
            mipmap_levels = self._calculate_mipmap_level(scales_3d, (1024, 1024))  # Assume 1024x1024 base texture

        # Process each material group
        for mat_name, sample_indices in material_groups.items():
            if mat_name is None or mat_name not in material_textures:
                continue

            textures = material_textures[mat_name]
            sample_indices = np.array(sample_indices)
            sample_uvs = uvs[sample_indices]

            # Get mipmap levels for this group
            group_mipmap_levels = mipmap_levels[sample_indices] if mipmap_levels is not None else None

            # Sample diffuse color with bilinear filtering and mipmapping
            if 'diffuse' in textures:
                tex = textures['diffuse']
                sampled_colors = self._sample_texture_bilinear(
                    tex, sample_uvs, num_channels=3, mipmap_levels=group_mipmap_levels
                )
                colors[sample_indices] = sampled_colors

                # Debug: log color stats for this material
                avg_color = sampled_colors.mean(axis=0)
                is_green = avg_color[1] > avg_color[0]
                self.logger.debug("  %s avg color: RGB(%.3f, %.3f, %.3f) green=%s",
                                 mat_name, avg_color[0], avg_color[1], avg_color[2], is_green)

            # Sample transparency with bilinear filtering and mipmapping
            if 'transparency' in textures:
                tex = textures['transparency']
                sampled_opacity = self._sample_texture_bilinear(
                    tex, sample_uvs, num_channels=1, mipmap_levels=group_mipmap_levels
                )
                opacities[sample_indices] = sampled_opacity

            # Sample roughness with bilinear filtering and mipmapping
            if 'roughness' in textures:
                tex = textures['roughness']
                sampled_roughness = self._sample_texture_bilinear(
                    tex, sample_uvs, num_channels=1, mipmap_levels=group_mipmap_levels
                )
                roughness[sample_indices] = sampled_roughness

        # Apply vertex color blending if available
        if vertex_colors is not None and barycentric_weights is not None:
            w1, w2, w3 = barycentric_weights

            for i in range(n_samples):
                face_idx = face_indices[i]

                # Interpolate vertex color for this sample
                vc = self._interpolate_vertex_color(vertex_colors, face_idx, w1[i], w2[i], w3[i])

                # Blend with texture color
                colors[i] = self._blend_vertex_color(colors[i], vc, vertex_color_blend_mode)

        return colors, opacities, roughness

    def _mesh_to_gaussians_multi_material(self,
                                           mesh: trimesh.Trimesh,
                                           strategy: str,
                                           samples_per_face: int,
                                           manifest: dict) -> List[_SingleGaussian]:
        """
        Convert mesh to gaussians with multi-material texture sampling.

        This method handles meshes with multiple materials, sampling colors,
        opacity, and roughness from the appropriate texture for each face.

        Args:
            mesh: The mesh to convert
            strategy: Gaussian placement strategy
            samples_per_face: Number of samples per face
            manifest: Material manifest from packed extraction

        Returns:
            List of gaussians with proper colors and properties
        """
        self.logger.info("Using multi-material texture sampling")

        # Load all material textures
        material_textures = self._load_material_textures(manifest, use_mipmaps=self.use_mipmaps)
        face_materials = manifest.get('face_materials', [])

        self.logger.info("Loaded textures for %d materials", len(material_textures))

        # Load UV layers if available
        uv_layers = self._load_uv_layers(manifest)

        # Load vertex colors if available
        vertex_color_blend_mode = manifest.get('vertex_color_blend_mode', 'multiply')
        vertex_colors = None
        if vertex_color_blend_mode != 'none':
            vertex_colors = self._load_vertex_colors(manifest)

        gaussians = []

        # Transparency culling threshold (hardcoded as discussed)
        OPACITY_THRESHOLD = 0.1

        # Roughness scale modulation parameters
        ROUGHNESS_SCALE_MIN = 1.0
        ROUGHNESS_SCALE_MAX = 1.5

        try:
            # Map adaptive to hybrid
            if strategy == 'adaptive':
                strategy = 'hybrid'

            # For multi-material, we focus on face-based sampling
            # since that's where material assignments are defined
            if strategy in ['face', 'hybrid']:
                n_faces = len(mesh.faces)
                n_samples = n_faces * samples_per_face

                self.logger.info("Generating %d samples from %d faces", n_samples, n_faces)

                # Get all face vertices
                v0 = mesh.vertices[mesh.faces[:, 0]]
                v1 = mesh.vertices[mesh.faces[:, 1]]
                v2 = mesh.vertices[mesh.faces[:, 2]]

                # Compute face normals and areas
                edge1 = v1 - v0
                edge2 = v2 - v0
                face_normals = np.cross(edge1, edge2)
                face_areas = np.linalg.norm(face_normals, axis=1) * 0.5
                face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)

                # Generate barycentric coordinates
                r1 = np.random.random(n_samples)
                r2 = np.random.random(n_samples)
                sqrt_r1 = np.sqrt(r1)

                w1 = 1 - sqrt_r1
                w2 = sqrt_r1 * (1 - r2)
                w3 = sqrt_r1 * r2

                # Face indices for all samples
                face_indices = np.repeat(np.arange(n_faces), samples_per_face)

                # Compute positions
                positions = (w1[:, None] * v0[face_indices] +
                            w2[:, None] * v1[face_indices] +
                            w3[:, None] * v2[face_indices])

                # Compute normals
                normals = face_normals[face_indices]

                # Compute base scales
                scales_base = np.sqrt(face_areas[face_indices]) * 0.3

                # Get UV coordinates for texture sampling
                # Priority: UV layers from manifest > mesh.visual.uv > metadata
                uvs = None
                vertex_uvs = None  # Initialize for later use in rejection sampling

                if uv_layers:
                    # Use UV layers from manifest (per-loop storage)
                    # Get the default UV layer
                    default_uv_layer = manifest.get('uv_layer')
                    if default_uv_layer and default_uv_layer in uv_layers:
                        loop_uvs = uv_layers[default_uv_layer]

                        # For each face, get the 3 loop indices and interpolate UVs
                        # Loop indices for a face are: face_idx * 3, face_idx * 3 + 1, face_idx * 3 + 2
                        loop_idx0 = face_indices * 3
                        loop_idx1 = face_indices * 3 + 1
                        loop_idx2 = face_indices * 3 + 2

                        uv0 = loop_uvs[loop_idx0]
                        uv1 = loop_uvs[loop_idx1]
                        uv2 = loop_uvs[loop_idx2]

                        uvs = (w1[:, None] * uv0 +
                              w2[:, None] * uv1 +
                              w3[:, None] * uv2)
                    else:
                        self.logger.warning("Default UV layer '%s' not found in loaded UV layers", default_uv_layer)

                if uvs is None:
                    # Fallback to mesh UV coordinates (per-vertex storage)
                    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                        vertex_uvs = mesh.visual.uv
                    elif 'vertex_uvs' in mesh.metadata:
                        vertex_uvs = mesh.metadata['vertex_uvs']

                    if vertex_uvs is not None:
                        faces = mesh.faces[face_indices]
                        uv0 = vertex_uvs[faces[:, 0]]
                        uv1 = vertex_uvs[faces[:, 1]]
                        uv2 = vertex_uvs[faces[:, 2]]

                        uvs = (w1[:, None] * uv0 +
                              w2[:, None] * uv1 +
                              w3[:, None] * uv2)
                    else:
                        self.logger.warning("No UV coordinates found, using default colors")
                        uvs = np.zeros((n_samples, 2))

                # Sample colors, opacity, and roughness from materials
                colors, opacities, roughness = self._sample_multi_material_colors(
                    face_indices, uvs, face_materials, material_textures,
                    vertex_colors=vertex_colors,
                    barycentric_weights=(w1, w2, w3),
                    vertex_color_blend_mode=vertex_color_blend_mode,
                    scales=scales_base
                )

                # Apply roughness scale modulation
                # rougher surfaces -> larger gaussians
                scale_multiplier = ROUGHNESS_SCALE_MIN + roughness * (ROUGHNESS_SCALE_MAX - ROUGHNESS_SCALE_MIN)
                scales_base = scales_base * scale_multiplier

                # Convert normals to quaternions
                quaternions = self._normals_to_quaternions_vectorized(normals)

                # Build scales array
                scales = np.zeros((n_samples, 3))
                scales[:, 0] = scales_base
                scales[:, 1] = scales_base
                scales[:, 2] = scales_base * 0.3

                # Apply transparency culling with rejection resampling
                # For samples that fall on transparent parts, try to resample them
                valid_mask = opacities >= OPACITY_THRESHOLD
                n_invalid = n_samples - np.sum(valid_mask)

                MAX_RESAMPLE_ITERATIONS = 5
                if n_invalid > 0 and vertex_uvs is not None:
                    # Try rejection sampling for transparent samples
                    invalid_indices = np.where(~valid_mask)[0]

                    for iteration in range(MAX_RESAMPLE_ITERATIONS):
                        if len(invalid_indices) == 0:
                            break

                        # Generate new random barycentric coordinates for invalid samples
                        n_resample = len(invalid_indices)
                        r1_new = np.random.random(n_resample)
                        r2_new = np.random.random(n_resample)
                        sqrt_r1_new = np.sqrt(r1_new)

                        w1_new = 1 - sqrt_r1_new
                        w2_new = sqrt_r1_new * (1 - r2_new)
                        w3_new = sqrt_r1_new * r2_new

                        # Get the face indices for these samples
                        resample_face_indices = face_indices[invalid_indices]

                        # Compute new positions
                        positions[invalid_indices] = (
                            w1_new[:, None] * v0[resample_face_indices] +
                            w2_new[:, None] * v1[resample_face_indices] +
                            w3_new[:, None] * v2[resample_face_indices]
                        )

                        # Compute new UVs
                        faces_resample = mesh.faces[resample_face_indices]
                        uv0_new = vertex_uvs[faces_resample[:, 0]]
                        uv1_new = vertex_uvs[faces_resample[:, 1]]
                        uv2_new = vertex_uvs[faces_resample[:, 2]]

                        uvs[invalid_indices] = (
                            w1_new[:, None] * uv0_new +
                            w2_new[:, None] * uv1_new +
                            w3_new[:, None] * uv2_new
                        )

                        # Re-sample colors and opacity for these samples only
                        colors_new, opacities_new, roughness_new = self._sample_multi_material_colors(
                            resample_face_indices, uvs[invalid_indices], face_materials, material_textures
                        )

                        colors[invalid_indices] = colors_new
                        opacities[invalid_indices] = opacities_new
                        roughness[invalid_indices] = roughness_new

                        # Recompute scale multipliers for resampled points
                        scale_multiplier_new = ROUGHNESS_SCALE_MIN + roughness_new * (ROUGHNESS_SCALE_MAX - ROUGHNESS_SCALE_MIN)
                        scales[invalid_indices, 0] = scales_base[invalid_indices] * scale_multiplier_new
                        scales[invalid_indices, 1] = scales_base[invalid_indices] * scale_multiplier_new
                        scales[invalid_indices, 2] = scales_base[invalid_indices] * 0.3 * scale_multiplier_new

                        # Check which are now valid
                        new_valid = opacities[invalid_indices] >= OPACITY_THRESHOLD
                        invalid_indices = invalid_indices[~new_valid]

                    # Update valid mask after resampling
                    valid_mask = opacities >= OPACITY_THRESHOLD

                n_culled = n_samples - np.sum(valid_mask)

                if n_culled > 0:
                    self.logger.info("Culled %d samples (%.1f%%) due to low opacity (after %d resample iterations)",
                                   n_culled, 100.0 * n_culled / n_samples, MAX_RESAMPLE_ITERATIONS)

                # Debug: Check which colors survive culling
                surviving_colors = colors[valid_mask]
                if len(surviving_colors) > 0:
                    green_mask = surviving_colors[:, 1] > surviving_colors[:, 0]
                    n_green = green_mask.sum()
                    n_brown = len(surviving_colors) - n_green
                    self.logger.debug("Surviving samples: %d green, %d brown", n_green, n_brown)

                # Filter to valid samples only
                positions = positions[valid_mask]
                scales = scales[valid_mask]
                quaternions = quaternions[valid_mask]
                opacities = opacities[valid_mask]
                colors = colors[valid_mask]

                n_valid = len(positions)

                # Create gaussians
                face_gaussians = [None] * n_valid
                for i in range(n_valid):
                    face_gaussians[i] = _SingleGaussian(
                        position=positions[i],
                        scales=scales[i],
                        rotation=quaternions[i],
                        opacity=float(opacities[i]),
                        sh_dc=colors[i] - 0.5  # Convert to SH DC format
                    )
                gaussians.extend(face_gaussians)

            # Add vertex-based gaussians for hybrid strategy
            if strategy in ['vertex', 'hybrid']:
                n_vertices = len(mesh.vertices)

                # For vertices, we need to determine which material each vertex belongs to
                # Use the most common material among faces that share this vertex
                vertex_materials = [None] * n_vertices

                for face_idx, face in enumerate(mesh.faces):
                    mat_name = face_materials[face_idx] if face_idx < len(face_materials) else None
                    for v_idx in face:
                        if vertex_materials[v_idx] is None:
                            vertex_materials[v_idx] = mat_name

                # Get vertex UVs - check both visual.uv and metadata
                if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    vertex_uvs = mesh.visual.uv
                elif 'vertex_uvs' in mesh.metadata:
                    vertex_uvs = mesh.metadata['vertex_uvs']
                else:
                    vertex_uvs = np.zeros((n_vertices, 2))

                # Sample colors for vertices using bilinear filtering
                vertex_colors = np.full((n_vertices, 3), 0.5, dtype=np.float32)
                vertex_opacities = np.ones(n_vertices, dtype=np.float32)
                vertex_roughness = np.full(n_vertices, 0.5, dtype=np.float32)

                # Group vertices by material for efficient batch sampling
                material_vertex_groups = {}
                for i in range(n_vertices):
                    mat_name = vertex_materials[i]
                    if mat_name and mat_name in material_textures:
                        if mat_name not in material_vertex_groups:
                            material_vertex_groups[mat_name] = []
                        material_vertex_groups[mat_name].append(i)

                # Sample each material group
                for mat_name, vertex_indices in material_vertex_groups.items():
                    textures = material_textures[mat_name]
                    vertex_indices = np.array(vertex_indices)
                    mat_uvs = vertex_uvs[vertex_indices]

                    # Sample diffuse
                    if 'diffuse' in textures:
                        tex = textures['diffuse']
                        sampled = self._sample_texture_bilinear(tex, mat_uvs, num_channels=3)
                        vertex_colors[vertex_indices] = sampled

                    # Sample transparency
                    if 'transparency' in textures:
                        tex = textures['transparency']
                        sampled = self._sample_texture_bilinear(tex, mat_uvs, num_channels=1)
                        vertex_opacities[vertex_indices] = sampled

                    # Sample roughness
                    if 'roughness' in textures:
                        tex = textures['roughness']
                        sampled = self._sample_texture_bilinear(tex, mat_uvs, num_channels=1)
                        vertex_roughness[vertex_indices] = sampled

                # Compute vertex properties
                vertex_normals = mesh.vertex_normals if hasattr(mesh, 'vertex_normals') else np.tile([0, 0, 1], (n_vertices, 1))
                vertex_quaternions = self._normals_to_quaternions_vectorized(vertex_normals)
                vertex_scales = self._compute_vertex_scales_fast(mesh.vertices, n_vertices)

                # Apply roughness modulation
                scale_multiplier = ROUGHNESS_SCALE_MIN + vertex_roughness * (ROUGHNESS_SCALE_MAX - ROUGHNESS_SCALE_MIN)
                vertex_scales = vertex_scales * scale_multiplier

                # Apply transparency culling
                valid_mask = vertex_opacities >= OPACITY_THRESHOLD

                for i in range(n_vertices):
                    if not valid_mask[i]:
                        continue

                    gaussians.append(_SingleGaussian(
                        position=mesh.vertices[i],
                        scales=np.array([vertex_scales[i], vertex_scales[i], vertex_scales[i] * 0.5]),
                        rotation=vertex_quaternions[i],
                        opacity=float(vertex_opacities[i]),
                        sh_dc=vertex_colors[i] - 0.5
                    ))

            self.logger.info("Created %d gaussians with multi-material sampling", len(gaussians))
            return gaussians

        finally:
            self.clear_texture_cache()

    def mesh_to_gaussians(self, mesh: trimesh.Trimesh,
                         strategy: str = 'vertex',
                         samples_per_face: int = 1,
                         material_manifest: dict = None) -> List[_SingleGaussian]:
        """
        Convert mesh to initial gaussians
        Strategies:
        - 'vertex': One gaussian per vertex
        - 'face': Gaussians sampled on face centers
        - 'hybrid': Both vertices and faces
        - 'adaptive': Auto-select based on mesh (currently maps to hybrid)

        Args:
            mesh: The mesh to convert
            strategy: Gaussian placement strategy
            samples_per_face: Number of samples per face for face/hybrid strategies
            material_manifest: Optional manifest from packed texture extraction
                              Contains material-to-texture mappings for multi-material support
        """
        gaussians = []

        # If we have a material manifest, use multi-material sampling
        if material_manifest is not None:
            return self._mesh_to_gaussians_multi_material(
                mesh, strategy, samples_per_face, material_manifest
            )

        try:
            # Map adaptive to hybrid for now
            if strategy == 'adaptive':
                strategy = 'hybrid'
                self.logger.debug("Using adaptive strategy -> hybrid")

            if strategy in ['vertex', 'hybrid']:
                # OPTIMIZED: Vectorized vertex processing (Phase 2)
                n_vertices = len(mesh.vertices)

                # Pre-compute vertex normals (vectorized)
                if hasattr(mesh, 'vertex_normals'):
                    vertex_normals = mesh.vertex_normals
                else:
                    vertex_normals = np.tile([0, 0, 1], (n_vertices, 1))

                # OPTIMIZED: Vectorized quaternion conversion (Phase 2)
                vertex_quaternions = self._normals_to_quaternions_vectorized(vertex_normals)

                # Pre-compute scales using KD-tree (OPTIMIZED: O(n log n) instead of O(n^2))
                scales = self._compute_vertex_scales_fast(mesh.vertices, n_vertices)

                # Pre-compute colors (OPTIMIZED: Fully vectorized)
                colors = self._sample_vertex_colors_vectorized(mesh, n_vertices)

                # OPTIMIZED: Pre-allocate and batch create (Phase 2)
                vertex_gaussians = [None] * n_vertices
                for i in range(n_vertices):
                    vertex_gaussians[i] = _SingleGaussian(
                        position=mesh.vertices[i],
                        scales=np.array([scales[i], scales[i], scales[i] * 0.5]),
                        rotation=vertex_quaternions[i],
                        opacity=0.9,
                        sh_dc=colors[i] - 0.5
                    )
                gaussians.extend(vertex_gaussians)

            if strategy in ['face', 'hybrid']:
                # OPTIMIZED: Vectorized face sampling
                # Pre-compute all face properties
                n_faces = len(mesh.faces)
                n_samples = n_faces * samples_per_face

                # Get all face vertices at once
                v0 = mesh.vertices[mesh.faces[:, 0]]  # (n_faces, 3)
                v1 = mesh.vertices[mesh.faces[:, 1]]  # (n_faces, 3)
                v2 = mesh.vertices[mesh.faces[:, 2]]  # (n_faces, 3)

                # Pre-compute face normals and areas (vectorized)
                edge1 = v1 - v0  # (n_faces, 3)
                edge2 = v2 - v0  # (n_faces, 3)
                face_normals = np.cross(edge1, edge2)  # (n_faces, 3)
                face_areas = np.linalg.norm(face_normals, axis=1) * 0.5  # (n_faces,)
                face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)

                # Generate ALL barycentric coordinates at once
                r1 = np.random.random(n_samples)
                r2 = np.random.random(n_samples)
                sqrt_r1 = np.sqrt(r1)

                w1 = 1 - sqrt_r1
                w2 = sqrt_r1 * (1 - r2)
                w3 = sqrt_r1 * r2

                # Repeat face indices for all samples
                face_indices = np.repeat(np.arange(n_faces), samples_per_face)

                # Vectorized position interpolation
                positions = (w1[:, None] * v0[face_indices] +
                            w2[:, None] * v1[face_indices] +
                            w3[:, None] * v2[face_indices])  # (n_samples, 3)

                # Vectorized normal (same for all samples on a face)
                normals = face_normals[face_indices]  # (n_samples, 3)

                # Vectorized scale
                scales_base = np.sqrt(face_areas[face_indices]) * 0.3  # (n_samples,)

                # Vectorized color sampling
                colors = self._sample_colors_vectorized(mesh, face_indices, w1, w2, w3)

                # OPTIMIZED: Vectorized quaternion conversion (Phase 2)
                quaternions = self._normals_to_quaternions_vectorized(normals)  # (n_samples, 4)

                # OPTIMIZED: Pre-allocate scales array (Phase 2)
                scales = np.zeros((n_samples, 3))
                scales[:, 0] = scales_base
                scales[:, 1] = scales_base
                scales[:, 2] = scales_base * 0.3

                # OPTIMIZED: Batch create gaussians (Phase 2)
                # Pre-allocate list for better performance
                face_gaussians = [None] * n_samples
                for i in range(n_samples):
                    face_gaussians[i] = _SingleGaussian(
                        position=positions[i],
                        scales=scales[i],
                        rotation=quaternions[i],
                        opacity=0.7,
                        sh_dc=colors[i] - 0.5
                    )
                gaussians.extend(face_gaussians)

            self.logger.debug("Created %d initial gaussians", len(gaussians))
            return gaussians

        finally:
            # Clear texture cache after processing to prevent memory leaks
            # Cache will be rebuilt for next mesh if needed
            self.clear_texture_cache()

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

    def _normals_to_quaternions_vectorized(self, normals: np.ndarray) -> np.ndarray:
        """
        Vectorized conversion of normal vectors to quaternion rotations.

        Args:
            normals: Array of normal vectors (n_samples, 3)

        Returns:
            quaternions: Array of quaternions (n_samples, 4) in [w, x, y, z] format
        """
        n_samples = normals.shape[0]
        z = np.array([0, 0, 1])

        # Normalize all normals at once
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-8)

        # Compute dot products with z-axis (vectorized)
        dots = normals @ z  # (n_samples,)

        # Compute rotation axes (vectorized cross product)
        axes = np.cross(z, normals)  # (n_samples, 3)
        axis_norms = np.linalg.norm(axes, axis=1, keepdims=True)
        axes = axes / (axis_norms + 1e-8)

        # Compute rotation angles (vectorized)
        angles = np.arccos(np.clip(dots, -1, 1))  # (n_samples,)

        # Convert to quaternions (vectorized)
        half_angles = angles / 2
        s = np.sin(half_angles)  # (n_samples,)
        c = np.cos(half_angles)  # (n_samples,)

        quaternions = np.zeros((n_samples, 4))
        quaternions[:, 0] = c  # w
        quaternions[:, 1] = axes[:, 0] * s  # x
        quaternions[:, 2] = axes[:, 1] * s  # y
        quaternions[:, 3] = axes[:, 2] * s  # z

        # Handle special cases (normals aligned with z-axis)
        # Identity quaternion for normals close to +z
        aligned_pos = dots > 0.9999
        quaternions[aligned_pos] = [1, 0, 0, 0]

        # 180 degree rotation for normals close to -z
        aligned_neg = dots < -0.9999
        quaternions[aligned_neg] = [0, 1, 0, 0]

        # Normalize quaternions
        quat_norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        quaternions = quaternions / (quat_norms + 1e-8)

        return quaternions

    def optimize_gaussians(self, gaussians: List[_SingleGaussian],
                          iterations: int = 100) -> List[_SingleGaussian]:
        """
        Simple optimization pass to improve gaussian placement
        This is a placeholder - in production you'd render and compare
        """
        if not TORCH_AVAILABLE:
            self.logger.info("PyTorch not available, skipping optimization")
            return gaussians

        if not torch.cuda.is_available():
            self.logger.info("CUDA not available, skipping optimization")
            return gaussians

        self.logger.info("Optimizing %d gaussians for %d iterations...", len(gaussians), iterations)

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
                self.logger.debug("Iteration %d: loss = %.4f", i, loss.item())

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
                 output_path: str,
                 compress: bool = False):
        """
        Save gaussians to PLY format compatible with gaussian splatting viewers.

        Args:
            gaussians: List of gaussian splats to save
            output_path: Output file path (.ply or .ply.gz)
            compress: If True, compress with gzip (or auto-detect from .gz extension)

        File Format:
            - 17 float properties per vertex (68 bytes)
            - Position (x, y, z)
            - Normals (nx, ny, nz) - currently unused, set to (0, 0, 0)
            - SH DC (f_dc_0, f_dc_1, f_dc_2) - spherical harmonics degree 0
            - Opacity (opacity)
            - Scales (scale_0, scale_1, scale_2) - stored in LOG SPACE
            - Rotation (rot_0, rot_1, rot_2, rot_3) - quaternion (w, x, y, z)

        Note: RGB properties removed in Phase 1 optimization (saves 3 bytes per gaussian).
              RGB can be computed from SH DC: RGB = SH_DC + 0.5
        """
        import gzip

        # Auto-detect compression from extension
        if output_path.endswith('.gz'):
            compress = True

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

        # Choose file handle (compressed or uncompressed)
        if compress:
            f = gzip.open(output_path, 'wb')
            self.logger.info("Saving %d gaussians to %s (gzip compressed)", vertex_count, output_path)
        else:
            f = open(output_path, 'wb')
            self.logger.info("Saving %d gaussians to %s", vertex_count, output_path)

        try:
            # PLY header with documentation comments
            f.write(b'ply\n')
            f.write(b'format binary_little_endian 1.0\n')
            f.write(b'comment Generated by gaussian-mesh-converter\n')
            f.write(b'comment Phase 1 optimized format (68 bytes per vertex)\n')
            f.write(b'comment ===== DATA ENCODING NOTES =====\n')
            f.write(b'comment Scales: Stored in LOG SPACE - use exp(scale) to get linear values\n')
            f.write(b'comment Rotation: Quaternion format (w, x, y, z)\n')
            f.write(b'comment SH DC: Spherical harmonics degree 0 - RGB = SH_DC + 0.5\n')
            f.write(b'comment Normals: Currently unused, set to (0, 0, 0)\n')
            f.write(f'element vertex {vertex_count}\n'.encode())

            # Position
            f.write(b'property float x\n')
            f.write(b'property float y\n')
            f.write(b'property float z\n')

            # Normals (unused but kept for compatibility)
            f.write(b'property float nx\n')
            f.write(b'property float ny\n')
            f.write(b'property float nz\n')

            # Spherical harmonics DC (degree 0)
            f.write(b'property float f_dc_0\n')
            f.write(b'property float f_dc_1\n')
            f.write(b'property float f_dc_2\n')

            # Opacity
            f.write(b'property float opacity\n')

            # Scales (log space)
            f.write(b'property float scale_0\n')
            f.write(b'property float scale_1\n')
            f.write(b'property float scale_2\n')

            # Rotation (quaternion: w, x, y, z)
            f.write(b'property float rot_0\n')
            f.write(b'property float rot_1\n')
            f.write(b'property float rot_2\n')
            f.write(b'property float rot_3\n')

            f.write(b'end_header\n')

            # Write vertex data (17 floats = 68 bytes per vertex)
            for i in range(vertex_count):
                # Position (3 floats)
                f.write(struct.pack('<fff', *positions[i]))

                # Normal (3 floats - unused, set to zero)
                f.write(struct.pack('<fff', 0, 0, 0))

                # SH DC (3 floats)
                f.write(struct.pack('<fff', *sh_dc[i]))

                # Opacity (1 float)
                f.write(struct.pack('<f', opacities[i]))

                # Scale (3 floats - log space)
                f.write(struct.pack('<fff', *log_scales[i]))

                # Rotation (4 floats - quaternion)
                f.write(struct.pack('<ffff', *rotations[i]))

        finally:
            f.close()

        # Log file size info
        file_size = Path(output_path).stat().st_size
        bytes_per_vertex = file_size / vertex_count if vertex_count > 0 else 0
        self.logger.info("File size: %.2f MB (%.1f bytes/vertex)",
                        file_size / (1024 * 1024), bytes_per_vertex)

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
        logging.getLogger('gaussian_pipeline').warning("Optimization requested but PyTorch/CUDA not available")

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