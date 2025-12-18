# ABOUTME: Tests for UV texture sampling from image files
# ABOUTME: Validates texture loading, UV interpolation, and color extraction

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh_to_gaussian import MeshToGaussianConverter
from PIL import Image


def test_load_obj_with_texture_map():
    """Test that OBJ with map_Kd texture reference loads the image"""
    converter = MeshToGaussianConverter(device='cpu')

    # Use the test output from baker tests
    test_obj = Path("test_output/simple_cube.obj")

    if not test_obj.exists():
        pytest.skip("Test asset not found - run baker tests first")

    mesh = converter.load_mesh(str(test_obj))

    # Check that texture was loaded
    assert hasattr(mesh.visual, 'material'), "Mesh should have material"
    assert mesh.visual.material.image is not None, "Material should have texture image"
    assert isinstance(mesh.visual.material.image, Image.Image), "Texture should be PIL Image"


def test_gaussians_sample_from_texture():
    """Test that gaussians get colors from texture, not just Kd"""
    import trimesh

    converter = MeshToGaussianConverter(device='cpu')

    # Use the test output from baker tests
    test_obj = Path("test_output/simple_cube.obj")

    if not test_obj.exists():
        pytest.skip("Test asset not found - run baker tests first")

    mesh = converter.load_mesh(str(test_obj))

    # Instead of creating a new mesh, just sample a small number of gaussians
    # from the original mesh to keep the test fast
    np.random.seed(42)  # For reproducibility

    # Use face strategy with only 100 faces for speed
    sample_face_count = min(100, len(mesh.faces))
    sample_face_indices = np.random.choice(len(mesh.faces), size=sample_face_count, replace=False)

    # Create a small mesh with just the sampled faces
    sampled_faces = mesh.faces[sample_face_indices]

    # Get unique vertices used by these faces
    unique_verts = np.unique(sampled_faces.flatten())

    # Create vertex mapping (old index -> new index)
    vert_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_verts)}

    # Remap faces to new vertex indices
    new_faces = np.array([[vert_map[v] for v in face] for face in sampled_faces])

    # Create small mesh with texture visuals preserved
    small_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[unique_verts],
        faces=new_faces,
        visual=trimesh.visual.TextureVisuals(
            uv=mesh.visual.uv[unique_verts] if hasattr(mesh.visual, 'uv') else None,
            image=mesh.visual.material.image if hasattr(mesh.visual, 'material') else None
        )
    )

    # Generate gaussians from the small mesh (only ~100 vertices = fast!)
    gaussians = converter.mesh_to_gaussians(small_mesh, strategy='vertex')

    # Check that colors vary (not all same Kd value)
    colors = np.array([g.sh_dc for g in gaussians])
    color_variance = np.var(colors, axis=0).sum()

    # Once texture sampling is implemented, colors should vary
    assert color_variance > 0.01, "Colors should vary across gaussians (sampled from texture)"

