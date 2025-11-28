# Unified Gaussian Pipeline - Implementation Specification
## Part 4: Phase 3 - Pipeline Orchestrator & Phase 4 - FBX Support

**Version:** 1.0  
**Date:** November 27, 2025

---

## Phase 3: Pipeline Orchestrator

**Estimated Duration:** 1-2 weeks  
**Priority:** MEDIUM - Unifies system  
**Assigned Team Size:** 2 developers  
**Dependencies:** Phases 1 and 2 must be complete

### Overview

This phase creates the unified pipeline that intelligently routes files to the correct processing stages and provides a clean user interface.

### Implementation Files

#### File 1: `src/pipeline/config.py`

```python
# ABOUTME: Configuration dataclass for pipeline settings
# ABOUTME: Validates user inputs and provides defaults

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the unified gaussian pipeline."""
    
    input_file: Path
    output_dir: Path
    lod_levels: List[int] = field(default_factory=lambda: [5000, 25000, 100000])
    strategy: str = 'hybrid'
    lod_strategy: str = 'importance'
    texture_resolution: int = 4096
    blender_executable: str = 'blender'
    keep_temp_files: bool = False
    device: str = 'cpu'
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert paths
        self.input_file = Path(self.input_file)
        self.output_dir = Path(self.output_dir)
        
        # Validate input file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        # Validate file extension
        valid_extensions = {'.blend', '.obj', '.glb', '.fbx'}
        if self.input_file.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported file type: {self.input_file.suffix}\n"
                f"Supported: {valid_extensions}"
            )
        
        # Validate strategy
        valid_strategies = {'vertex', 'face', 'hybrid', 'adaptive'}
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        
        # Validate LOD strategy
        valid_lod_strategies = {'importance', 'opacity', 'spatial'}
        if self.lod_strategy not in valid_lod_strategies:
            raise ValueError(f"Invalid LOD strategy: {self.lod_strategy}")
        
        # Validate LOD levels
        if not self.lod_levels:
            raise ValueError("At least one LOD level required")
        
        if any(level <= 0 for level in self.lod_levels):
            raise ValueError("LOD levels must be positive integers")
        
        # Sort LOD levels (highest to lowest)
        self.lod_levels = sorted(self.lod_levels, reverse=True)
        
        # Validate texture resolution
        if self.texture_resolution < 512 or self.texture_resolution > 8192:
            raise ValueError("Texture resolution must be between 512 and 8192")
        
        # Validate device
        if self.device not in {'cpu', 'cuda'}:
            raise ValueError(f"Invalid device: {self.device}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
```

#### File 2: `src/pipeline/router.py`

```python
# ABOUTME: File routing logic for pipeline stages
# ABOUTME: Determines which processing stages are needed for each file type

from pathlib import Path
from typing import Tuple


class FileRouter:
    """Determines which pipeline stages are needed for a given input file."""
    
    @staticmethod
    def route(input_file: Path) -> Tuple[bool, bool]:
        """
        Determine which stages are needed.
        
        Returns:
            Tuple of (needs_stage1, needs_stage2)
        """
        suffix = input_file.suffix.lower()
        
        if suffix == '.blend':
            # Blender files need both stages
            return (True, True)
        
        elif suffix in {'.obj', '.glb', '.fbx'}:
            # Mesh files go straight to Stage 2
            return (False, True)
        
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    @staticmethod
    def get_description(input_file: Path) -> str:
        """Get human-readable description of processing path."""
        needs_stage1, needs_stage2 = FileRouter.route(input_file)
        
        if needs_stage1 and needs_stage2:
            return "Blender baking → Gaussian conversion → LOD generation"
        elif needs_stage2:
            return "Gaussian conversion → LOD generation"
        else:
            return "Unknown processing path"
```

#### File 3: `src/pipeline/orchestrator.py`

```python
# ABOUTME: Main pipeline orchestrator
# ABOUTME: Coordinates Stage 1, Stage 2, and LOD generation with progress tracking

import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple
import time

from .config import PipelineConfig
from .router import FileRouter


class Pipeline:
    """Main pipeline orchestrator for unified gaussian conversion."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.temp_dir = None
        
        # Import stages dynamically
        from stage1_baker import BlenderBaker
        from stage2_converter.mesh_to_gaussian import MeshToGaussianConverter
        from stage2_converter.lod_generator import LODGenerator
        
        self.baker = BlenderBaker(blender_executable=config.blender_executable)
        self.converter = MeshToGaussianConverter(device=config.device)
        self.lod_gen = LODGenerator(strategy=config.lod_strategy)
    
    def run(self) -> List[Path]:
        """Execute the complete pipeline."""
        start_time = time.time()
        
        print("="*70)
        print("UNIFIED GAUSSIAN PIPELINE")
        print("="*70)
        print(f"Input: {self.config.input_file}")
        print(f"Output: {self.config.output_dir}")
        print(f"Strategy: {self.config.strategy}")
        print(f"LOD levels: {self.config.lod_levels}")
        print("="*70)
        
        try:
            # 1. Route file
            needs_stage1, needs_stage2 = FileRouter.route(self.config.input_file)
            processing_path = FileRouter.get_description(self.config.input_file)
            
            print(f"\nProcessing path: {processing_path}\n")
            
            # 2. Stage 1: Blender baking (if needed)
            if needs_stage1:
                obj_path, texture_path = self._run_stage1()
            else:
                obj_path = self.config.input_file
                texture_path = None
            
            # 3. Stage 2: Gaussian conversion
            gaussians = self._run_stage2(obj_path)
            
            # 4. Generate and save LODs
            output_files = self._generate_lods(gaussians)
            
            # 5. Cleanup
            self._cleanup()
            
            elapsed = time.time() - start_time
            
            print("\n" + "="*70)
            print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
            print(f"Generated {len(output_files)} output files")
            print("="*70)
            
            return output_files
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            self._cleanup()
            raise
    
    def _run_stage1(self) -> Tuple[Path, Path]:
        """Execute Stage 1: Blender baking."""
        print("\n" + "-"*70)
        print("STAGE 1: BLENDER BAKING")
        print("-"*70)
        
        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gaussian_pipeline_"))
        
        # Bake
        obj_path, texture_path = self.baker.bake(
            str(self.config.input_file),
            output_dir=str(self.temp_dir),
            texture_resolution=self.config.texture_resolution,
            timeout=600
        )
        
        print(f"✓ Stage 1 complete")
        
        return obj_path, texture_path
    
    def _run_stage2(self, obj_path: Path) -> List:
        """Execute Stage 2: Gaussian conversion."""
        print("\n" + "-"*70)
        print("STAGE 2: GAUSSIAN CONVERSION")
        print("-"*70)
        
        # Load mesh
        mesh = self.converter.load_mesh(str(obj_path))
        
        # Convert to gaussians
        gaussians = self.converter.mesh_to_gaussians(
            mesh,
            strategy=self.config.strategy,
            samples_per_face=10
        )
        
        print(f"✓ Stage 2 complete: {len(gaussians)} gaussians")
        
        return gaussians
    
    def _generate_lods(self, gaussians: List) -> List[Path]:
        """Generate LOD levels and save PLY files."""
        print("\n" + "-"*70)
        print("LOD GENERATION")
        print("-"*70)
        
        output_files = []
        base_name = self.config.input_file.stem
        
        # Save full resolution
        full_res_path = self.config.output_dir / f"{base_name}_full.ply"
        self.converter.save_ply(gaussians, str(full_res_path))
        output_files.append(full_res_path)
        print(f"✓ Full resolution: {len(gaussians)} gaussians")
        
        # Generate LODs
        for lod_count in self.config.lod_levels:
            if lod_count >= len(gaussians):
                continue
            
            lod_gaussians = self.lod_gen.generate_lod(gaussians, lod_count)
            lod_path = self.config.output_dir / f"{base_name}_lod{lod_count}.ply"
            self.converter.save_ply(lod_gaussians, str(lod_path))
            output_files.append(lod_path)
            
            print(f"✓ LOD {lod_count}: {len(lod_gaussians)} gaussians")
        
        return output_files
    
    def _cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            if self.config.keep_temp_files:
                print(f"\n📁 Temp files kept: {self.temp_dir}")
            else:
                shutil.rmtree(self.temp_dir)
                print(f"\n🗑️  Cleaned up temp files")
```

#### File 4: `cli.py` (Unified CLI)

```python
#!/usr/bin/env python3
# ABOUTME: Unified command-line interface for gaussian pipeline
# ABOUTME: Provides user-friendly access to all pipeline features

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import Pipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description='Unified Gaussian Splat Pipeline',
        epilog="""
Examples:
  python cli.py model.blend ./output
  python cli.py model.obj ./output --lod 5000 25000 100000
  python cli.py tree.blend ./output --strategy hybrid --keep-temp
        """
    )
    
    parser.add_argument('input', type=str, help='Input file (.blend, .obj, .glb, .fbx)')
    parser.add_argument('output_dir', type=str, help='Output directory for PLY files')
    parser.add_argument('--lod', type=int, nargs='+', default=[5000, 25000, 100000])
    parser.add_argument('--strategy', type=str, default='hybrid',
                       choices=['vertex', 'face', 'hybrid', 'adaptive'])
    parser.add_argument('--lod-strategy', type=str, default='importance',
                       choices=['importance', 'opacity', 'spatial'])
    parser.add_argument('--texture-resolution', type=int, default=4096)
    parser.add_argument('--blender', type=str, default='blender')
    parser.add_argument('--keep-temp', action='store_true')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    try:
        config = PipelineConfig(
            input_file=args.input,
            output_dir=args.output_dir,
            lod_levels=args.lod,
            strategy=args.strategy,
            lod_strategy=args.lod_strategy,
            texture_resolution=args.texture_resolution,
            blender_executable=args.blender,
            keep_temp_files=args.keep_temp,
            device=args.device
        )
        
        pipeline = Pipeline(config)
        output_files = pipeline.run()
        
        print(f"\n✅ Success! Generated {len(output_files)} files")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

### Usage Examples

```bash
# Convert Blender file
python cli.py tree.blend ./output --lod 5000 25000 100000

# Convert OBJ with custom settings
python cli.py model.obj ./output --strategy face --lod-strategy spatial

# High-quality baking
python cli.py tree.blend ./output --texture-resolution 8192 --keep-temp
```

---

## Phase 4: FBX Support

**Estimated Duration:** 3-5 days  
**Priority:** LOW - Optional enhancement  
**Assigned Team Size:** 1 developer  
**Dependencies:** Phase 3 must be complete

### Implementation

**Modify:** `src/pipeline/router.py`

```python
elif suffix == '.fbx':
    # FBX files go to Stage 2
    # TODO: Future enhancement - detect if FBX has procedural materials
    return (False, True)
```

### Acceptance Criteria

- [ ] FBX files load without errors
- [ ] FBX → PLY conversion works
- [ ] Tests pass for FBX input
- [ ] Documentation updated

---

**Continue to Part 5 for Testing Strategy and Deployment**
