#!/usr/bin/env python3
# ABOUTME: Unified command-line interface for gaussian pipeline
# ABOUTME: Provides user-friendly access to all pipeline features

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import Pipeline, PipelineConfig
from utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description='Unified Gaussian Splat Pipeline - Convert 3D meshes to Gaussian Splats',
        epilog="""
Examples:
  # Single file input
  python cli.py model.blend ./output
  python cli.py model.obj ./output --lod 5000 25000 100000

  # Folder-based input (recommended for models with textures)
  python cli.py MyModel/ ./output --lod 100000

  # The folder should contain:
  #   MyModel/
  #   +-- model.blend          (your .blend file)
  #   +-- textures/            (texture files)
  #       +-- diffuse.png
  #       +-- normal.png

  # Advanced options
  python cli.py tree.blend ./output --strategy hybrid --use-gpu
  python cli.py character.glb ./output --texture-resolution 2048

  # Packed texture mode (for .blend files with embedded textures)
  python cli.py packed-tree.blend ./output --packed --uv-layer uv0
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('input', type=str,
                       help='Input file (.blend, .obj, .glb, .fbx) or folder containing model + textures')
    parser.add_argument('output_dir', type=str, 
                       help='Output directory for PLY files')
    
    # Optional arguments
    parser.add_argument('--lod', type=int, nargs='+', 
                       default=[5000, 25000, 100000],
                       help='LOD levels (gaussian counts). Default: 5000 25000 100000')
    parser.add_argument('--strategy', type=str, default='hybrid',
                       choices=['vertex', 'face', 'hybrid', 'adaptive'],
                       help='Gaussian sampling strategy. Default: hybrid')
    parser.add_argument('--lod-strategy', type=str, default='importance',
                       choices=['importance', 'opacity', 'spatial'],
                       help='LOD generation strategy. Default: importance')
    parser.add_argument('--texture-resolution', type=int, default=4096,
                       help='Texture baking resolution (for .blend files). Default: 4096')
    parser.add_argument('--blender', type=str, default=None,
                       help='Path to Blender executable. Default: auto-detect')
    parser.add_argument('--timeout', type=int, default=1800,
                       help='Blender baking timeout in seconds. Default: 1800 (30 min)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU for Blender baking (5-10x faster)')
    parser.add_argument('--samples', type=int, default=32,
                       help='Cycles render samples. Default: 32. Lower = faster but noisier')
    parser.add_argument('--renderer', type=str, default='auto',
                       choices=['auto', 'cycles', 'eevee'],
                       help='Rendering engine. Default: auto (intelligent detection)')
    parser.add_argument('--bake-type', type=str, default='auto',
                       choices=['auto', 'DIFFUSE', 'COMBINED', 'EMIT'],
                       help='Force bake type. Default: auto (adaptive detection)')
    parser.add_argument('--denoise', action='store_true',
                       help='Enable denoising (allows lower sample counts)')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep temporary files (for debugging)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Computation device. Default: cpu')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging (DEBUG level)')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode - only show warnings and errors')

    # Packed texture mode arguments
    parser.add_argument('--packed', action='store_true',
                       help='Use packed texture extraction mode (for .blend files with embedded textures)')
    parser.add_argument('--uv-layer', type=str, default='uv0',
                       help='UV layer name to use for texture sampling. Default: uv0')

    # Vertex color arguments
    parser.add_argument('--vertex-color-blend', type=str, default='multiply',
                       choices=['multiply', 'add', 'overlay', 'replace', 'none'],
                       help='Vertex color blending mode. Default: multiply. Use "none" to disable vertex colors')

    # Texture filtering arguments (Phase B)
    parser.add_argument('--no-mipmaps', action='store_true',
                       help='Disable mipmap generation for textures (faster but lower quality)')
    parser.add_argument('--texture-filter', type=str, default='bilinear',
                       choices=['nearest', 'bilinear'],
                       help='Texture filtering mode. Default: bilinear (higher quality)')

    # Phase 1 optimization arguments
    parser.add_argument('--compress', action='store_true',
                       help='Compress output PLY files with gzip (5-10x smaller, .ply.gz extension)')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(verbose=args.verbose, quiet=args.quiet)
    
    try:
        # Create configuration
        config = PipelineConfig(
            input_file=args.input,
            output_dir=args.output_dir,
            lod_levels=args.lod,
            strategy=args.strategy,
            lod_strategy=args.lod_strategy,
            texture_resolution=args.texture_resolution,
            blender_executable=args.blender,
            bake_timeout=args.timeout,
            use_gpu=args.use_gpu,
            bake_samples=args.samples,
            renderer=args.renderer,
            bake_type=args.bake_type,
            denoise=args.denoise,
            keep_temp_files=args.keep_temp,
            device=args.device,
            use_packed=args.packed,
            uv_layer=args.uv_layer,
            vertex_color_blend_mode=args.vertex_color_blend,
            use_mipmaps=not args.no_mipmaps,
            texture_filter=args.texture_filter,
            compress=args.compress
        )
        
        # Run pipeline
        pipeline = Pipeline(config)
        output_files = pipeline.run()
        
        # Success!
        logger.info("")
        logger.info("Success! Generated %d files:", len(output_files))
        for f in output_files:
            logger.info("   - %s", f)

        sys.exit(0)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Invalid configuration: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

