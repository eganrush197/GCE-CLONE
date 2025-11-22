# ABOUTME: Test suite for mesh-to-gaussian converter
# ABOUTME: Tests core conversion functionality with synthetic meshes

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh_to_gaussian import MeshToGaussianConverter, ConversionConfig
from src.gaussian_splat import GaussianSplat
from src.lod_generator import LODGenerator
import trimesh


@pytest.fixture
def simple_cube():
    """Create a simple cube mesh for testing."""
    return trimesh.creation.box(extents=[2, 2, 2])


@pytest.fixture
def simple_sphere():
    """Create a simple sphere mesh for testing."""
    return trimesh.creation.icosphere(subdivisions=2, radius=1.0)


@pytest.fixture
def converter():
    """Create a basic converter."""
    return MeshToGaussianConverter()


class TestGaussianSplat:
    """Test GaussianSplat data structure."""
    
    def test_creation(self):
        """Test creating a gaussian splat."""
        n = 100
        splat = GaussianSplat(
            positions=np.random.randn(n, 3),
            scales=np.random.randn(n, 3),
            rotations=np.random.randn(n, 4),
            colors=np.random.rand(n, 3),
            opacity=np.random.randn(n)
        )
        
        assert splat.count == n
        assert splat.positions.shape == (n, 3)
        # Quaternions should be normalized
        norms = np.linalg.norm(splat.rotations, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_subset(self):
        """Test creating a subset of gaussians."""
        n = 100
        splat = GaussianSplat(
            positions=np.random.randn(n, 3),
            scales=np.random.randn(n, 3),
            rotations=np.random.randn(n, 4),
            colors=np.random.rand(n, 3),
            opacity=np.random.randn(n)
        )
        
        indices = np.array([0, 10, 20, 30])
        subset = splat.subset(indices)
        
        assert subset.count == 4
        np.testing.assert_array_equal(subset.positions, splat.positions[indices])


class TestMeshToGaussianConverter:
    """Test mesh-to-gaussian conversion."""
    
    def test_vertex_strategy(self, simple_cube, converter):
        """Test vertex-based initialization."""
        config = ConversionConfig(initialization_strategy='vertex')
        converter = MeshToGaussianConverter(config)

        # Save cube to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            temp_path = Path(f.name)

        try:
            simple_cube.export(temp_path)

            gaussians = converter.convert(temp_path)

            # Should have one gaussian per vertex
            assert gaussians.count == len(simple_cube.vertices)
            assert gaussians.positions.shape == (len(simple_cube.vertices), 3)
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def test_face_strategy(self, simple_sphere, converter):
        """Test face-based initialization."""
        config = ConversionConfig(
            initialization_strategy='face',
            samples_per_face=5
        )
        converter = MeshToGaussianConverter(config)

        # Save sphere to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            temp_path = Path(f.name)

        try:
            simple_sphere.export(temp_path)

            gaussians = converter.convert(temp_path)

            # Should have multiple gaussians
            assert gaussians.count > 0
            assert gaussians.positions.shape[1] == 3
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def test_color_extraction(self, simple_cube):
        """Test color extraction from mesh."""
        converter = MeshToGaussianConverter()
        
        # Cube should have default colors
        colors = converter._extract_vertex_colors(simple_cube)
        
        assert colors.shape == (len(simple_cube.vertices), 3)
        assert np.all(colors >= 0) and np.all(colors <= 1)


class TestLODGenerator:
    """Test LOD generation."""
    
    def test_importance_pruning(self):
        """Test importance-based pruning."""
        # Create test gaussians
        n = 1000
        gaussians = GaussianSplat(
            positions=np.random.randn(n, 3),
            scales=np.random.randn(n, 3),
            rotations=np.random.randn(n, 4),
            colors=np.random.rand(n, 3),
            opacity=np.random.randn(n)
        )
        
        lod_gen = LODGenerator(strategy='importance')
        lods = lod_gen.generate_lods(gaussians, [100, 500])
        
        assert len(lods) == 2
        assert lods[0].count <= 500
        assert lods[1].count <= 100
    
    def test_opacity_pruning(self):
        """Test opacity-based pruning."""
        n = 1000
        gaussians = GaussianSplat(
            positions=np.random.randn(n, 3),
            scales=np.random.randn(n, 3),
            rotations=np.random.randn(n, 4),
            colors=np.random.rand(n, 3),
            opacity=np.random.randn(n)
        )
        
        lod_gen = LODGenerator(strategy='opacity')
        lod = lod_gen._prune_to_count(gaussians, 100)
        
        assert lod.count == 100

