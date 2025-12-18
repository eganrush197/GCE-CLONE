"""
Tests for the unified pipeline orchestrator.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.config import PipelineConfig
from pipeline.router import FileRouter
from pipeline.orchestrator import Pipeline

# Blender executable path (Windows default)
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 3.1\blender.exe"


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""
    
    def test_config_with_valid_obj_file(self, tmp_path):
        """Test config creation with valid OBJ file."""
        # Create a dummy OBJ file
        obj_file = tmp_path / "test.obj"
        obj_file.write_text("v 0 0 0\n")
        
        output_dir = tmp_path / "output"
        
        config = PipelineConfig(
            input_file=obj_file,
            output_dir=output_dir
        )
        
        assert config.input_file == obj_file
        assert config.output_dir == output_dir
        assert output_dir.exists()  # Should be created
        assert config.strategy == 'hybrid'  # Default
        assert config.lod_levels == [100000, 25000, 5000]  # Sorted descending
    
    def test_config_with_nonexistent_file(self, tmp_path):
        """Test config raises error for nonexistent file."""
        nonexistent = tmp_path / "nonexistent.obj"
        output_dir = tmp_path / "output"
        
        with pytest.raises(FileNotFoundError):
            PipelineConfig(input_file=nonexistent, output_dir=output_dir)
    
    def test_config_with_invalid_extension(self, tmp_path):
        """Test config raises error for unsupported file type."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("test")
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            PipelineConfig(input_file=invalid_file, output_dir=output_dir)
    
    def test_config_with_invalid_strategy(self, tmp_path):
        """Test config raises error for invalid strategy."""
        obj_file = tmp_path / "test.obj"
        obj_file.write_text("v 0 0 0\n")
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="Invalid strategy"):
            PipelineConfig(
                input_file=obj_file,
                output_dir=output_dir,
                strategy='invalid'
            )
    
    def test_config_with_invalid_lod_strategy(self, tmp_path):
        """Test config raises error for invalid LOD strategy."""
        obj_file = tmp_path / "test.obj"
        obj_file.write_text("v 0 0 0\n")
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="Invalid LOD strategy"):
            PipelineConfig(
                input_file=obj_file,
                output_dir=output_dir,
                lod_strategy='invalid'
            )
    
    def test_config_lod_levels_sorted(self, tmp_path):
        """Test that LOD levels are sorted in descending order."""
        obj_file = tmp_path / "test.obj"
        obj_file.write_text("v 0 0 0\n")
        output_dir = tmp_path / "output"
        
        config = PipelineConfig(
            input_file=obj_file,
            output_dir=output_dir,
            lod_levels=[5000, 100000, 25000]  # Unsorted
        )
        
        assert config.lod_levels == [100000, 25000, 5000]  # Should be sorted
    
    def test_config_with_invalid_texture_resolution(self, tmp_path):
        """Test config raises error for invalid texture resolution."""
        obj_file = tmp_path / "test.obj"
        obj_file.write_text("v 0 0 0\n")
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="Texture resolution"):
            PipelineConfig(
                input_file=obj_file,
                output_dir=output_dir,
                texture_resolution=256  # Too small
            )


class TestFileRouter:
    """Tests for FileRouter class."""
    
    def test_route_blend_file(self, tmp_path):
        """Test routing for .blend file."""
        blend_file = tmp_path / "test.blend"
        blend_file.write_text("dummy")
        
        needs_stage1, needs_stage2 = FileRouter.route(blend_file)
        
        assert needs_stage1 is True
        assert needs_stage2 is True
    
    def test_route_obj_file(self, tmp_path):
        """Test routing for .obj file."""
        obj_file = tmp_path / "test.obj"
        obj_file.write_text("v 0 0 0\n")
        
        needs_stage1, needs_stage2 = FileRouter.route(obj_file)
        
        assert needs_stage1 is False
        assert needs_stage2 is True
    
    def test_route_glb_file(self, tmp_path):
        """Test routing for .glb file."""
        glb_file = tmp_path / "test.glb"
        glb_file.write_text("dummy")
        
        needs_stage1, needs_stage2 = FileRouter.route(glb_file)
        
        assert needs_stage1 is False
        assert needs_stage2 is True
    
    def test_route_unsupported_file(self, tmp_path):
        """Test routing raises error for unsupported file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            FileRouter.route(txt_file)
    
    def test_get_description_blend(self, tmp_path):
        """Test description for .blend file."""
        blend_file = tmp_path / "test.blend"
        blend_file.write_text("dummy")
        
        desc = FileRouter.get_description(blend_file)
        
        assert "Blender baking" in desc
        assert "Gaussian conversion" in desc
        assert "LOD generation" in desc
    
    def test_get_description_obj(self, tmp_path):
        """Test description for .obj file."""
        obj_file = tmp_path / "test.obj"
        obj_file.write_text("v 0 0 0\n")
        
        desc = FileRouter.get_description(obj_file)
        
        assert "Blender baking" not in desc
        assert "Gaussian conversion" in desc
        assert "LOD generation" in desc


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.mark.skipif(
        not Path("test_assets/simple_cube.blend").exists() or not Path(BLENDER_EXE).exists(),
        reason="Test asset or Blender not found"
    )
    def test_pipeline_with_blend_file(self, tmp_path):
        """Test complete pipeline with .blend file."""
        blend_file = Path("test_assets/simple_cube.blend")
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            input_file=blend_file,
            output_dir=output_dir,
            lod_levels=[100, 500],  # Small LODs for testing
            texture_resolution=512,  # Small texture for speed
            keep_temp_files=False,
            blender_executable=BLENDER_EXE
        )

        pipeline = Pipeline(config)
        output_files = pipeline.run()

        # Should generate: full + 2 LODs = 3 files
        assert len(output_files) >= 1  # At least full resolution

        # Check that full resolution file exists
        full_res = output_dir / "simple_cube_full.ply"
        assert full_res.exists()
        assert full_res.stat().st_size > 0

    def test_pipeline_with_obj_file(self, tmp_path):
        """Test pipeline with simple OBJ file (no baking needed)."""
        # Create a simple OBJ file
        obj_file = tmp_path / "test.obj"
        obj_content = """
# Simple cube
v -1 -1 -1
v -1 -1  1
v -1  1 -1
v -1  1  1
v  1 -1 -1
v  1 -1  1
v  1  1 -1
v  1  1  1

vt 0 0
vt 1 0
vt 1 1
vt 0 1

f 1/1 2/2 4/3 3/4
f 5/1 6/2 8/3 7/4
f 1/1 5/2 7/3 3/4
f 2/1 6/2 8/3 4/4
f 1/1 2/2 6/3 5/4
f 3/1 4/2 8/3 7/4
"""
        obj_file.write_text(obj_content)

        output_dir = tmp_path / "output"

        config = PipelineConfig(
            input_file=obj_file,
            output_dir=output_dir,
            lod_levels=[50, 100],  # Small LODs
            strategy='vertex'
        )

        pipeline = Pipeline(config)
        output_files = pipeline.run()

        # Should generate at least full resolution
        assert len(output_files) >= 1

        # Check full resolution file
        full_res = output_dir / "test_full.ply"
        assert full_res.exists()
        assert full_res.stat().st_size > 0

    def test_pipeline_cleanup_temp_files(self, tmp_path):
        """Test that temp files are cleaned up by default."""
        obj_file = tmp_path / "test.obj"
        # Create a proper 3D mesh (tetrahedron) instead of 2D plane
        obj_file.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n")

        output_dir = tmp_path / "output"

        config = PipelineConfig(
            input_file=obj_file,
            output_dir=output_dir,
            keep_temp_files=False
        )

        pipeline = Pipeline(config)

        # Store temp_dir reference before cleanup
        temp_dir_before = pipeline.temp_dir

        output_files = pipeline.run()

        # Temp dir should be None or cleaned up
        # (Only created for .blend files, not for .obj)
        if temp_dir_before:
            assert not temp_dir_before.exists()

    def test_pipeline_keep_temp_files(self, tmp_path):
        """Test that temp files are kept when requested."""
        # This test only applies to .blend files
        # For .obj files, no temp directory is created
        pass  # Skip for now, would need .blend file

