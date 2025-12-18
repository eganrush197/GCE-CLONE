# ABOUTME: Main pipeline orchestrator
# ABOUTME: Coordinates Stage 1, Stage 2, and LOD generation with progress tracking

import tempfile
import shutil
import sys
import logging
from pathlib import Path
from typing import List, Tuple
import time

from .config import PipelineConfig
from .router import FileRouter
from utils.logging_utils import Timer, TimingStats


class Pipeline:
    """Main pipeline orchestrator for unified gaussian conversion."""

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.temp_dir = None
        self.logger = logging.getLogger('gaussian_pipeline')
        self.timing_stats = []  # Track timing for each stage

        # Import stages dynamically to avoid circular imports
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # Lazy initialization - only create baker if needed
        self._baker = None
        self._converter = None
        self._lod_gen = None

    @property
    def baker(self):
        """Lazy initialization of BlenderBaker."""
        if self._baker is None:
            from stage1_baker.baker import BlenderBaker
            self._baker = BlenderBaker(blender_executable=self.config.blender_executable)
        return self._baker

    @property
    def converter(self):
        """Lazy initialization of MeshToGaussianConverter."""
        if self._converter is None:
            from mesh_to_gaussian import MeshToGaussianConverter
            self._converter = MeshToGaussianConverter(
                device=self.config.device,
                use_mipmaps=self.config.use_mipmaps,
                texture_filter=self.config.texture_filter
            )
        return self._converter

    @property
    def lod_gen(self):
        """Lazy initialization of LODGenerator."""
        if self._lod_gen is None:
            from lod_generator import LODGenerator
            self._lod_gen = LODGenerator(strategy=self.config.lod_strategy)
        return self._lod_gen

    @property
    def packed_extractor(self):
        """Lazy initialization of PackedExtractor."""
        if not hasattr(self, '_packed_extractor') or self._packed_extractor is None:
            from stage1_baker.packed_extractor import PackedExtractor
            self._packed_extractor = PackedExtractor(blender_executable=self.config.blender_executable)
        return self._packed_extractor
    
    def run(self) -> List[Path]:
        """Execute the complete pipeline."""
        start_time = time.time()

        self.logger.info("="*70)
        self.logger.info("UNIFIED GAUSSIAN PIPELINE")
        self.logger.info("="*70)
        self.logger.info("Input: %s", self.config.input_file)
        self.logger.info("Output: %s", self.config.output_dir)
        self.logger.info("Strategy: %s", self.config.strategy)
        self.logger.info("LOD levels: %s", self.config.lod_levels)
        if self.config.use_packed:
            self.logger.info("Mode: PACKED TEXTURE EXTRACTION")
            self.logger.info("UV Layer: %s", self.config.uv_layer)
        self.logger.info("="*70)

        try:
            # 1. Route file
            needs_stage1, needs_stage2 = FileRouter.route(self.config.input_file)
            processing_path = FileRouter.get_description(self.config.input_file)

            self.logger.info("")
            self.logger.info("Processing path: %s", processing_path)
            self.logger.info("")

            # 2. Stage 1: Blender baking or packed extraction (if needed)
            manifest = None
            if self.config.use_packed:
                # Use packed texture extraction mode
                obj_path, manifest = self._run_packed_extraction()
            elif needs_stage1:
                obj_path, texture_path = self._run_stage1()
            else:
                obj_path = self.config.input_file
                texture_path = None

            # 3. Stage 2: Gaussian conversion
            gaussians = self._run_stage2(obj_path, manifest=manifest)

            # 4. Generate and save LODs
            output_files = self._generate_lods(gaussians)

            # 5. Cleanup
            self._cleanup()

            # 6. Print summary
            self._print_summary(start_time, output_files)

            return output_files

        except Exception as e:
            self.logger.error("")
            self.logger.error("PIPELINE FAILED: %s", e)
            import traceback
            self.logger.error("Traceback:\n%s", traceback.format_exc())
            self._cleanup()
            raise
    
    def _run_stage1(self) -> Tuple[Path, Path]:
        """Execute Stage 1: Blender baking."""
        self.logger.info("")
        self.logger.info("-"*70)
        self.logger.info("STAGE 1: BLENDER BAKING")
        self.logger.info("-"*70)

        # Log folder-based input if applicable
        if self.config.input_folder:
            self.logger.info("Folder-based input: %s", self.config.input_folder.name)

        with Timer("Stage 1: Blender Baking", self.logger) as timer:
            # Create temp directory
            self.temp_dir = Path(tempfile.mkdtemp(prefix="gaussian_pipeline_"))

            # Bake - pass input_folder for folder-based input
            obj_path, texture_path = self.baker.bake(
                str(self.config.input_file),
                output_dir=str(self.temp_dir),
                texture_resolution=self.config.texture_resolution,
                timeout=self.config.bake_timeout,
                use_gpu=self.config.use_gpu,
                samples=self.config.bake_samples,
                renderer=self.config.renderer,
                bake_type=self.config.bake_type,
                denoise=self.config.denoise,
                input_folder=str(self.config.input_folder) if self.config.input_folder else None
            )

        # Track timing
        stage1_stats = TimingStats("Stage 1: Blender Baking", timer.elapsed)
        self.timing_stats.append(stage1_stats)

        return obj_path, texture_path

    def _run_packed_extraction(self) -> Tuple[Path, dict]:
        """Execute packed texture extraction (alternative to Stage 1 baking)."""
        self.logger.info("")
        self.logger.info("-"*70)
        self.logger.info("STAGE 1: PACKED TEXTURE EXTRACTION")
        self.logger.info("-"*70)

        with Timer("Stage 1: Packed Extraction", self.logger) as timer:
            # Create temp directory
            self.temp_dir = Path(tempfile.mkdtemp(prefix="gaussian_packed_"))

            # Extract packed textures
            obj_path, manifest = self.packed_extractor.extract(
                str(self.config.input_file),
                output_dir=str(self.temp_dir),
                uv_layer=self.config.uv_layer,
                timeout=self.config.bake_timeout
            )

            # Add manifest location for path resolution (Fix 1D)
            manifest['_manifest_path'] = str(self.temp_dir / "material_manifest.json")

            # Add vertex color blend mode to manifest
            manifest['vertex_color_blend_mode'] = self.config.vertex_color_blend_mode

        # Track timing
        stage1_stats = TimingStats("Stage 1: Packed Extraction", timer.elapsed)
        self.timing_stats.append(stage1_stats)

        self.logger.info("Extracted %d materials", len(manifest.get('materials', {})))

        return obj_path, manifest

    def _run_stage2(self, obj_path: Path, manifest: dict = None) -> List:
        """Execute Stage 2: Gaussian conversion."""
        self.logger.info("")
        self.logger.info("-"*70)
        self.logger.info("STAGE 2: GAUSSIAN CONVERSION")
        self.logger.info("-"*70)

        with Timer("Stage 2: Gaussian Conversion", self.logger) as timer:
            # Load mesh
            with Timer("Mesh loading", self.logger) as load_timer:
                mesh = self.converter.load_mesh(str(obj_path))

            # Convert to gaussians
            with Timer("Gaussian generation", self.logger) as gen_timer:
                gaussians = self.converter.mesh_to_gaussians(
                    mesh,
                    strategy=self.config.strategy,
                    samples_per_face=10,
                    material_manifest=manifest
                )

        # Track timing
        stage2_stats = TimingStats("Stage 2: Gaussian Conversion", timer.elapsed)
        stage2_stats.add_substep("Mesh loading", load_timer.elapsed)
        stage2_stats.add_substep("Gaussian generation", gen_timer.elapsed)
        self.timing_stats.append(stage2_stats)

        self.logger.info("Generated %d gaussians", len(gaussians))

        return gaussians
    
    def _generate_lods(self, gaussians: List) -> List[Path]:
        """Generate LOD levels and save PLY files."""
        self.logger.info("")
        self.logger.info("-"*70)
        self.logger.info("LOD GENERATION")
        self.logger.info("-"*70)

        with Timer("LOD Generation", self.logger) as timer:
            output_files = []
            base_name = self.config.input_file.stem

            # Determine file extension based on compression setting
            ext = '.ply.gz' if self.config.compress else '.ply'

            # Save full resolution
            full_res_path = self.config.output_dir / f"{base_name}_full{ext}"
            self.converter.save_ply(gaussians, str(full_res_path), compress=self.config.compress)
            output_files.append(full_res_path)
            self.logger.info("Full resolution: %d gaussians", len(gaussians))

            # Generate LODs
            for lod_count in self.config.lod_levels:
                if lod_count >= len(gaussians):
                    continue

                lod_gaussians = self.lod_gen.generate_lod(gaussians, lod_count)
                lod_path = self.config.output_dir / f"{base_name}_lod{lod_count}{ext}"
                self.converter.save_ply(lod_gaussians, str(lod_path), compress=self.config.compress)
                output_files.append(lod_path)

                self.logger.info("LOD %d: %d gaussians", lod_count, len(lod_gaussians))

        # Track timing
        lod_stats = TimingStats("LOD Generation", timer.elapsed)
        self.timing_stats.append(lod_stats)

        return output_files

    def _cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            if self.config.keep_temp_files:
                self.logger.info("")
                self.logger.info("Temp files kept at: %s", self.temp_dir)
            else:
                try:
                    shutil.rmtree(self.temp_dir)
                    self.logger.debug("Cleaned up temp files")
                except PermissionError:
                    # Windows sometimes locks files - not critical, just log a warning
                    self.logger.warning("Could not fully clean up temp directory (files may be locked): %s", self.temp_dir)
                except Exception as e:
                    self.logger.warning("Error during cleanup (non-critical): %s", e)

    def _print_summary(self, start_time: float, output_files: List[Path]):
        """Print performance summary."""
        total_time = time.time() - start_time

        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("PIPELINE COMPLETE in %.1fs", total_time)
        self.logger.info("="*70)
        self.logger.info("")

        # Timing breakdown
        if self.timing_stats:
            self.logger.info("TIMING BREAKDOWN:")
            for stat in self.timing_stats:
                self.logger.info(stat.format_tree(total_time))
            self.logger.info("")
            self.logger.info("Total: %.1fs", total_time)
            self.logger.info("")

        # Output files
        self.logger.info("OUTPUT FILES:")
        for f in output_files:
            size_mb = f.stat().st_size / 1e6
            self.logger.info("  %s (%.1f MB)", f.name, size_mb)

        self.logger.info("")
        self.logger.info("="*70)

