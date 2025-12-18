# ABOUTME: Test suite for Blender baker functionality
# ABOUTME: Tests subprocess management, validation, and baking workflow

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stage1_baker import BlenderBaker


# Blender executable path (Windows default)
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 3.1\blender.exe"


@pytest.fixture
def baker():
    """Create BlenderBaker instance with correct Blender path."""
    return BlenderBaker(blender_executable=BLENDER_EXE)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_baker_"))
    yield temp_dir
    # Cleanup after test
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def test_blender_baker_initialization():
    """Test that BlenderBaker initializes and validates Blender executable."""
    baker = BlenderBaker(blender_executable=BLENDER_EXE)
    
    assert baker.blender_exe == BLENDER_EXE
    assert baker.script_path.exists(), "Bake script should exist"
    assert baker.script_path.name == "bake_and_export.py"


def test_blender_baker_invalid_executable():
    """Test that BlenderBaker raises error for invalid Blender path."""
    with pytest.raises(FileNotFoundError):
        BlenderBaker(blender_executable="nonexistent_blender.exe")


def test_blender_baker_script_exists(baker):
    """Test that the Blender Python script exists."""
    assert baker.script_path.exists()

    # Check script has required content
    # Use UTF-8 encoding to handle any non-ASCII characters in the script
    script_content = baker.script_path.read_text(encoding='utf-8')
    assert "bake_and_export" in script_content
    assert "bpy.ops.object.bake" in script_content


@pytest.mark.skipif(not Path(BLENDER_EXE).exists(), reason="Blender not installed")
def test_bake_simple_cube(baker, temp_output_dir):
    """
    Test baking a simple procedural cube.
    
    This test requires a test asset: test_assets/simple_cube.blend
    If the asset doesn't exist, the test will be skipped.
    """
    # Check if test asset exists
    test_asset = Path(__file__).parent.parent / "test_assets" / "simple_cube.blend"
    
    if not test_asset.exists():
        pytest.skip(f"Test asset not found: {test_asset}")
    
    # Bake the cube
    obj_path, texture_path = baker.bake(
        str(test_asset),
        output_dir=str(temp_output_dir),
        texture_resolution=512,  # Small for fast testing
        timeout=120
    )
    
    # Verify outputs exist
    assert obj_path.exists(), "OBJ file should be created"
    assert texture_path.exists(), "Texture file should be created"
    
    # Verify file sizes
    assert obj_path.stat().st_size > 100, "OBJ should not be empty"
    assert texture_path.stat().st_size > 1000, "Texture should not be empty"
    
    # Verify OBJ has content
    obj_content = obj_path.read_text()
    assert "v " in obj_content, "OBJ should have vertices"
    assert "vt " in obj_content, "OBJ should have UV coordinates"
    assert "f " in obj_content, "OBJ should have faces"
    
    # Verify MTL file exists
    mtl_path = obj_path.with_suffix('.mtl')
    assert mtl_path.exists(), "MTL file should be created"
    
    # Verify MTL references texture
    mtl_content = mtl_path.read_text()
    assert "map_Kd" in mtl_content, "MTL should reference texture"
    assert "baked_texture.png" in mtl_content, "MTL should reference correct texture file"


@pytest.mark.skipif(not Path(BLENDER_EXE).exists(), reason="Blender not installed")
def test_bake_nonexistent_file(baker, temp_output_dir):
    """Test that baking a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        baker.bake("nonexistent.blend", output_dir=str(temp_output_dir))


@pytest.mark.skipif(not Path(BLENDER_EXE).exists(), reason="Blender not installed")
def test_bake_with_temp_directory(baker):
    """Test that baking creates temp directory when output_dir is None."""
    test_asset = Path(__file__).parent.parent / "test_assets" / "simple_cube.blend"
    
    if not test_asset.exists():
        pytest.skip(f"Test asset not found: {test_asset}")
    
    # Bake without specifying output directory
    obj_path, texture_path = baker.bake(
        str(test_asset),
        texture_resolution=512,
        timeout=120
    )
    
    # Verify outputs exist
    assert obj_path.exists()
    assert texture_path.exists()
    
    # Verify temp directory was created
    assert "blender_bake_" in str(obj_path.parent)
    
    # Cleanup
    baker.cleanup_temp(obj_path.parent)
    assert not obj_path.parent.exists(), "Temp directory should be cleaned up"


def test_cleanup_temp(baker, temp_output_dir):
    """Test that cleanup_temp removes temporary directories."""
    # Create a fake temp directory
    fake_temp = temp_output_dir / "blender_bake_test123"
    fake_temp.mkdir()
    
    # Create a dummy file
    (fake_temp / "test.txt").write_text("test")
    
    # Cleanup
    baker.cleanup_temp(fake_temp)
    
    # Verify cleanup
    assert not fake_temp.exists(), "Temp directory should be removed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

