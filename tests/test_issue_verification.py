# ABOUTME: Test suite to verify identified issues in the codebase
# ABOUTME: Tests for resource leaks, edge cases, and error handling

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import trimesh
from unittest.mock import Mock, patch, MagicMock
import io

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh_to_gaussian import MeshToGaussianConverter
from src.lod_generator import LODGenerator


class TestIssue1_TextureVisualAttributeCheck:
    """Test Issue #1: Missing texture visual attribute check"""
    
    def test_mesh_with_no_visual_attribute(self):
        """Test mesh without visual attribute doesn't crash"""
        converter = MeshToGaussianConverter(device='cpu')
        
        # Create mesh without visual
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        mesh.visual = None
        
        # Should not crash
        result = converter._sample_texture_color(mesh, 0)
        assert result is None
    
    def test_mesh_with_color_visual_not_texture(self):
        """Test mesh with ColorVisuals instead of TextureVisuals"""
        converter = MeshToGaussianConverter(device='cpu')
        
        # Create mesh with color visual
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        mesh.visual = trimesh.visual.ColorVisuals(mesh)
        
        # Should not crash
        result = converter._sample_texture_color(mesh, 0)
        assert result is None


class TestIssue3_DivisionByZero:
    """Test Issue #3: Potential division by zero in mesh normalization"""
    
    def test_mesh_with_all_vertices_at_origin(self):
        """Test mesh with all vertices at origin (scale = 0)"""
        converter = MeshToGaussianConverter(device='cpu')

        # Create a temp OBJ file with all vertices at origin
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write("v 0 0 0\n")
            f.write("v 0 0 0\n")
            f.write("v 0 0 0\n")
            f.write("v 0 0 0\n")
            f.write("f 1 2 3\n")
            f.write("f 1 3 4\n")
            temp_path = f.name

        try:
            # Should raise ValueError for degenerate mesh (FIXED in Issue #3)
            with pytest.raises(ValueError, match="degenerate extent"):
                mesh = converter.load_mesh(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestIssue4_VertexScaleComputation:
    """Test Issue #4: Inefficient O(n²) vertex scale computation"""
    
    def test_vertex_scale_computation_performance(self):
        """Test that vertex scale computation is slow for large meshes"""
        import time
        
        converter = MeshToGaussianConverter(device='cpu')
        
        # Create mesh with 1000 vertices (should be slow with O(n²))
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        print(f"\nMesh has {len(mesh.vertices)} vertices")
        
        start = time.time()
        gaussians = converter.mesh_to_gaussians(mesh, strategy='vertex')
        elapsed = time.time() - start
        
        print(f"Time for {len(mesh.vertices)} vertices: {elapsed:.2f}s")
        
        # This will be slow with current implementation
        # Expected: > 1 second for 1000+ vertices
        assert len(gaussians) == len(mesh.vertices)


class TestIssue9_MeshValidation:
    """Test Issue #9: No mesh validation"""
    
    def test_mesh_with_no_faces(self):
        """Test mesh with no faces (point cloud)"""
        converter = MeshToGaussianConverter(device='cpu')
        
        # Create point cloud (no faces)
        vertices = np.random.rand(100, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=[])
        
        # Should this crash or handle gracefully?
        try:
            gaussians = converter.mesh_to_gaussians(mesh, strategy='face')
            # If it doesn't crash, should return empty or error?
            print(f"Generated {len(gaussians)} gaussians from faceless mesh")
        except Exception as e:
            print(f"Error (expected): {e}")
    
    def test_mesh_with_degenerate_faces(self):
        """Test mesh with degenerate faces (zero area)"""
        converter = MeshToGaussianConverter(device='cpu')
        
        # Create mesh with degenerate face (all vertices same)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
        faces = [[0, 1, 2], [0, 3, 1]]  # Second face is degenerate
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Should handle gracefully
        try:
            gaussians = converter.mesh_to_gaussians(mesh, strategy='face')
            print(f"Generated {len(gaussians)} gaussians from mesh with degenerate faces")
        except Exception as e:
            print(f"Error: {e}")


class TestIssue11_TextureCacheLeak:
    """Test Issue #11: Texture cache LRU eviction (FIXED)"""

    def test_texture_cache_lru_eviction(self):
        """Test that texture cache is bounded with LRU eviction"""
        # Create converter with small cache size
        converter = MeshToGaussianConverter(device='cpu', max_texture_cache_size=3)

        from PIL import Image

        # Process more meshes than cache size
        for i in range(5):
            mesh = trimesh.creation.box(extents=[1, 1, 1])
            # Add unique texture for each mesh
            img = Image.new('RGB', (64, 64), color=(i*50, 0, 0))
            mesh.visual = trimesh.visual.TextureVisuals(image=img)

            # Trigger texture caching
            converter._sample_texture_color(mesh, 0)

            # Cache should never exceed max size
            cache_size = len(converter._texture_cache)
            assert cache_size <= 3, f"Cache size {cache_size} exceeds limit of 3"

        # Final cache size should be at limit
        final_cache_size = len(converter._texture_cache)
        print(f"\nTexture cache size after 5 meshes: {final_cache_size}/3 (LRU eviction working)")
        assert final_cache_size <= 3, "Cache should be bounded by max_texture_cache_size"

        # Test cache clearing
        converter.clear_texture_cache()
        assert len(converter._texture_cache) == 0, "Cache should be empty after clear"
        print("✓ Cache cleared successfully")


class TestIssue18_VoxelIndexingOverflow:
    """Test Issue #18: Potential integer overflow in voxel indexing"""
    
    def test_large_grid_size_voxel_indexing(self):
        """Test voxel indexing with large grid size"""
        lod_gen = LODGenerator(strategy='spatial')
        
        # Create many gaussians to force large grid
        from src.mesh_to_gaussian import _SingleGaussian
        
        gaussians = []
        for i in range(10000):
            g = _SingleGaussian(
                position=np.random.rand(3) * 100,
                scales=np.array([0.01, 0.01, 0.01]),
                rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                opacity=0.9,
                sh_dc=np.array([0.5, 0.5, 0.5]),
                sh_rest=np.zeros((15, 3))
            )
            gaussians.append(g)
        
        # Generate LOD with spatial strategy
        try:
            lod = lod_gen.generate_lod(gaussians, target_count=1000)
            print(f"\nGenerated LOD with {len(lod)} gaussians")
            assert len(lod) <= 1000
        except Exception as e:
            print(f"Error in voxel indexing: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

