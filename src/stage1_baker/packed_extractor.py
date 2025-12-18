# ABOUTME: Python wrapper for Blender packed texture extraction
# ABOUTME: Manages headless Blender execution for extracting packed textures

import subprocess
import logging
import json
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict
import time
import sys
import platform

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging_utils import ProgressTracker, Timer


class PackedExtractor:
    """
    Extracts packed textures from .blend files.
    
    This class provides a Python interface to invoke Blender headlessly,
    execute the extraction script, and return the resulting OBJ + textures + manifest.
    
    Usage:
        extractor = PackedExtractor(blender_executable="C:/Program Files/Blender Foundation/Blender 4.0/blender.exe")
        obj_path, manifest = extractor.extract("model.blend", output_dir="./temp", uv_layer="uv0")
    """
    
    @staticmethod
    def _find_blender() -> str:
        """Find Blender installation (reuses logic from baker.py)."""
        from .baker import BlenderBaker
        return BlenderBaker._find_blender()
    
    def __init__(self, blender_executable: str = None):
        """
        Initialize packed texture extractor.
        
        Args:
            blender_executable: Path to Blender executable.
                               Default: None (auto-detect)
        """
        self.logger = logging.getLogger('gaussian_pipeline')
        
        if blender_executable is None:
            self.blender_exe = self._find_blender()
        else:
            self.blender_exe = blender_executable
        
        self._validate_blender()
        
        # Path to the Blender Python script
        self.script_path = Path(__file__).parent / "blender_scripts" / "extract_packed.py"
        
        if not self.script_path.exists():
            raise FileNotFoundError(f"Extract script not found: {self.script_path}")
    
    def _validate_blender(self):
        """Verify Blender executable exists and is callable."""
        try:
            result = subprocess.run(
                [self.blender_exe, "--version"],
                capture_output=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"Blender executable failed: {self.blender_exe}")
            
            version_info = result.stdout.decode('utf-8')
            version_line = [line for line in version_info.split('\n') if 'Blender' in line][0]
            self.logger.debug("Found %s", version_line.strip())
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Blender not found: {self.blender_exe}\n"
                f"Please install Blender or specify correct path."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Blender --version timed out")
    
    def extract(self,
                blend_file: str,
                output_dir: Optional[str] = None,
                uv_layer: str = 'uv0',
                timeout: int = 300) -> Tuple[Path, Dict]:
        """
        Extract packed textures from .blend file.
        
        Args:
            blend_file: Path to input .blend file
            output_dir: Output directory (default: temp directory)
            uv_layer: UV layer name to use (default: 'uv0')
            timeout: Max extraction time in seconds (default: 300)
        
        Returns:
            Tuple of (obj_path, manifest_dict)
        
        Raises:
            FileNotFoundError: If blend file doesn't exist
            RuntimeError: If extraction fails
        """
        blend_path = Path(blend_file).resolve()
        
        if not blend_path.exists():
            raise FileNotFoundError(f"Blend file not found: {blend_file}")
        
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            import tempfile
            output_path = Path(tempfile.mkdtemp(prefix="packed_extract_"))
        
        self.logger.info("Input: %s", blend_path.name)
        self.logger.info("Output: %s", output_path)
        self.logger.info("UV layer: %s", uv_layer)
        
        # Expected output files
        obj_path = output_path / f"{blend_path.stem}.obj"
        manifest_path = output_path / "material_manifest.json"
        
        # Build Blender command
        cmd = [
            str(self.blender_exe),
            "-b",                           # Background mode
            str(blend_path),                # Input file
            "-P", str(self.script_path),    # Python script
            "--",                           # Separator for script args
            str(output_path),               # Output directory
            uv_layer                        # UV layer name
        ]
        
        # Execute Blender
        start_time = time.time()
        log_file = output_path / "blender_extract.log"
        
        try:
            self.logger.info("Starting Blender process...")
            cmd_str = ' '.join(f'"{c}"' if ' ' in str(c) else str(c) for c in cmd)
            self.logger.debug("Command: %s", cmd_str)
            
            # Start Blender process
            if platform.system() == 'Windows':
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    encoding='utf-8',
                    errors='replace'
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding='utf-8',
                    errors='replace'
                )

            # Collect output
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                # Log progress indicators
                if '[OK]' in line or '[UNPACK]' in line or '[SAVE]' in line:
                    self.logger.info(line.strip())

            process.wait(timeout=timeout)

            # Save log (use UTF-8 to handle any unicode from Blender)
            with open(log_file, 'w', encoding='utf-8') as f:
                f.writelines(output_lines)

            elapsed = time.time() - start_time

            if process.returncode != 0:
                self.logger.error("Blender extraction failed (code %d)", process.returncode)
                self.logger.error("See log: %s", log_file)
                raise RuntimeError(f"Blender extraction failed with code {process.returncode}")

            self.logger.info("Extraction complete in %.1fs", elapsed)

        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError(f"Blender extraction timed out after {timeout}s")

        # Verify outputs exist
        if not obj_path.exists():
            raise RuntimeError(f"OBJ file not created: {obj_path}")

        if not manifest_path.exists():
            raise RuntimeError(f"Manifest not created: {manifest_path}")

        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        self.logger.info("Loaded manifest with %d materials", len(manifest.get('materials', {})))

        return obj_path, manifest


def extract_packed_textures(blend_file: str,
                            output_dir: str,
                            uv_layer: str = 'uv0',
                            blender_exe: str = None) -> Tuple[Path, Dict]:
    """
    Convenience function for packed texture extraction.

    Args:
        blend_file: Path to .blend file
        output_dir: Output directory
        uv_layer: UV layer name (default: 'uv0')
        blender_exe: Optional Blender executable path

    Returns:
        Tuple of (obj_path, manifest_dict)
    """
    extractor = PackedExtractor(blender_executable=blender_exe)
    return extractor.extract(blend_file, output_dir, uv_layer)

