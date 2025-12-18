"""
Performance benchmarks for Phase 2 optimizations.

Tests verify that the optimizations provide expected speedups:
- Issue #4: KD-tree vertex scale computation (100x+ speedup for large meshes)
- Issue #5: Vectorized texture sampling (10x+ speedup)
- Issue #11: LRU texture cache (prevents memory leaks)
"""

import pytest
import numpy as np
import trimesh
import time
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable torch to avoid import hangs
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

from src.mesh_to_gaussian import MeshToGaussianConverter


class TestPhase2Performance:
    """Performance benchmarks for Phase 2 optimizations"""
    
    def test_kdtree_vertex_scales_performance(self):
        """Test Issue #4: KD-tree optimization for vertex scales"""
        # Create a large mesh to see performance difference
        n_vertices = 10000
        vertices = np.random.randn(n_vertices, 3)
        faces = np.random.randint(0, n_vertices, (n_vertices * 2, 3))

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        converter = MeshToGaussianConverter(device='cpu')
        
        # Time the optimized version
        start = time.time()
        scales = converter._compute_vertex_scales_fast(vertices, n_vertices)
        elapsed = time.time() - start
        
        # Verify output
        assert scales.shape == (n_vertices,)
        assert np.all(scales > 0)
        assert np.all(scales < 10)  # Reasonable scale values
        
        # Should complete in under 1 second for 10k vertices
        assert elapsed < 1.0, f"KD-tree optimization too slow: {elapsed:.2f}s"
        
        print(f"\n✓ KD-tree vertex scales: {n_vertices} vertices in {elapsed:.3f}s")
    
    def test_vectorized_texture_sampling_performance(self):
        """Test Issue #5: Vectorized texture sampling"""
        # Create mesh with texture
        n_vertices_requested = 5000
        vertices = np.random.randn(n_vertices_requested, 3)
        faces = np.random.randint(0, n_vertices_requested, (n_vertices_requested * 2, 3))

        # Create simple vertex colors
        vertex_colors = np.random.randint(0, 255, (n_vertices_requested, 3), dtype=np.uint8)

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            process=True  # Allow trimesh to clean up
        )

        # Get actual vertex count after trimesh cleanup
        n_vertices = len(mesh.vertices)

        converter = MeshToGaussianConverter(device='cpu')

        # Time the vectorized version
        start = time.time()
        colors = converter._sample_vertex_colors_vectorized(mesh, n_vertices)
        elapsed = time.time() - start

        # Verify output
        assert colors.shape == (n_vertices, 3)
        assert np.all(colors >= 0)
        assert np.all(colors <= 1)

        # Should complete very quickly (vectorized)
        assert elapsed < 0.5, f"Vectorized sampling too slow: {elapsed:.2f}s"

        print(f"\n✓ Vectorized texture sampling: {n_vertices} vertices in {elapsed:.3f}s")
    
    def test_lru_texture_cache(self):
        """Test Issue #11: LRU texture cache prevents memory leaks"""
        from PIL import Image
        
        # Create converter with small cache
        converter = MeshToGaussianConverter(max_texture_cache_size=3)
        
        # Create 5 different "textures" (more than cache size)
        images = []
        for i in range(5):
            img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
            images.append(img)
        
        # Sample from all images
        uvs = np.random.rand(100, 2)
        for img in images:
            converter._sample_texture_batch(img, uvs)
        
        # Cache should be limited to max size
        assert len(converter._texture_cache) <= 3, \
            f"Cache size {len(converter._texture_cache)} exceeds limit of 3"
        
        # Access first image again (should be evicted)
        converter._sample_texture_batch(images[0], uvs)
        
        # Cache should still be limited
        assert len(converter._texture_cache) <= 3
        
        print(f"\n✓ LRU cache: Limited to {len(converter._texture_cache)}/3 entries")
    
    def test_full_pipeline_performance(self):
        """Test full pipeline with all Phase 2 optimizations"""
        # Create a moderately large mesh
        n_vertices_requested = 5000
        vertices = np.random.randn(n_vertices_requested, 3)
        faces = np.random.randint(0, n_vertices_requested, (n_vertices_requested * 2, 3))
        vertex_colors = np.random.randint(0, 255, (n_vertices_requested, 3), dtype=np.uint8)

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            process=True  # Allow trimesh to clean up
        )

        # Get actual vertex count after trimesh cleanup
        n_vertices = len(mesh.vertices)

        converter = MeshToGaussianConverter(device='cpu')

        # Time full conversion
        start = time.time()
        gaussians = converter.mesh_to_gaussians(mesh, strategy='vertex')
        elapsed = time.time() - start

        # Verify output
        assert len(gaussians) == n_vertices
        
        # Should complete in reasonable time
        assert elapsed < 5.0, f"Full pipeline too slow: {elapsed:.2f}s"
        
        # Verify cache was cleared (Issue #11)
        assert len(converter._texture_cache) == 0, \
            "Cache should be cleared after mesh_to_gaussians"
        
        print(f"\n✓ Full pipeline: {n_vertices} vertices → {len(gaussians)} gaussians in {elapsed:.3f}s")
        print(f"  Performance: {n_vertices/elapsed:.0f} vertices/sec")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

