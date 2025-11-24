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

from src.mesh_to_gaussian import MeshToGaussianConverter
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
    return MeshToGaussianConverter(device='cpu')


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
        # Save cube to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            temp_path = Path(f.name)

        try:
            simple_cube.export(temp_path)

            # Load mesh and convert with vertex strategy
            mesh = converter.load_mesh(str(temp_path))
            gaussians = converter.mesh_to_gaussians(mesh, strategy='vertex')

            # Should have one gaussian per vertex
            assert len(gaussians) == len(simple_cube.vertices)
            assert len(gaussians) > 0
            # Check first gaussian has correct structure
            assert hasattr(gaussians[0], 'position')
            assert hasattr(gaussians[0], 'scales')
            assert hasattr(gaussians[0], 'rotation')
            assert hasattr(gaussians[0], 'opacity')
            assert hasattr(gaussians[0], 'sh_dc')
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    def test_face_strategy(self, simple_sphere, converter):
        """Test face-based initialization."""
        # Save sphere to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            temp_path = Path(f.name)

        try:
            simple_sphere.export(temp_path)

            # Load mesh and convert with face strategy
            mesh = converter.load_mesh(str(temp_path))
            gaussians = converter.mesh_to_gaussians(mesh, strategy='face', samples_per_face=5)

            # Should have multiple gaussians (faces * samples_per_face)
            assert len(gaussians) > 0
            assert len(gaussians) >= len(simple_sphere.faces)
            # Check structure
            assert hasattr(gaussians[0], 'position')
            assert hasattr(gaussians[0], 'sh_dc')
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    def test_hybrid_strategy(self, simple_cube, converter):
        """Test hybrid strategy (vertex + face)."""
        # Save cube to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            temp_path = Path(f.name)

        try:
            simple_cube.export(temp_path)

            # Load mesh and convert with hybrid strategy
            mesh = converter.load_mesh(str(temp_path))
            gaussians = converter.mesh_to_gaussians(mesh, strategy='hybrid')

            # Should have more gaussians than just vertices
            assert len(gaussians) > len(simple_cube.vertices)
            assert len(gaussians) > 0
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()


class TestLODGenerator:
    """Test LOD generation."""

    def test_importance_pruning(self, simple_cube, converter):
        """Test importance-based pruning."""
        # Create test gaussians from a real mesh
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            temp_path = Path(f.name)

        try:
            simple_cube.export(temp_path)
            mesh = converter.load_mesh(str(temp_path))
            gaussians = converter.mesh_to_gaussians(mesh, strategy='hybrid')

            # Generate LODs
            lod_gen = LODGenerator(strategy='importance')
            lods = lod_gen.generate_lods(gaussians, [10, 20])

            assert len(lods) == 2
            assert len(lods[0]) <= 20  # First LOD (sorted descending)
            assert len(lods[1]) <= 10  # Second LOD
            assert len(lods[1]) <= len(lods[0])  # Smaller LOD has fewer gaussians
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_opacity_pruning(self, simple_sphere, converter):
        """Test opacity-based pruning."""
        # Create test gaussians from a real mesh
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            temp_path = Path(f.name)

        try:
            simple_sphere.export(temp_path)
            mesh = converter.load_mesh(str(temp_path))
            gaussians = converter.mesh_to_gaussians(mesh, strategy='face', samples_per_face=3)

            # Generate single LOD with opacity strategy
            lod_gen = LODGenerator(strategy='opacity')
            target_count = min(50, len(gaussians) // 2)
            lod = lod_gen.generate_lod(gaussians, target_count)

            assert len(lod) == target_count
            assert len(lod) < len(gaussians)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_spatial_pruning(self, simple_cube, converter):
        """Test spatial-based pruning."""
        # Create test gaussians from a real mesh
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            temp_path = Path(f.name)

        try:
            simple_cube.export(temp_path)
            mesh = converter.load_mesh(str(temp_path))
            gaussians = converter.mesh_to_gaussians(mesh, strategy='hybrid')

            # Generate LOD with spatial strategy
            lod_gen = LODGenerator(strategy='spatial')
            target_count = min(15, len(gaussians) // 2)
            lod = lod_gen.generate_lod(gaussians, target_count)

            assert len(lod) <= target_count
            assert len(lod) > 0
        finally:
            if temp_path.exists():
                temp_path.unlink()

