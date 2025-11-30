# ABOUTME: Python wrapper for Blender subprocess operations
# ABOUTME: Manages headless Blender execution for shader baking

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional
import time


class BlenderBaker:
    """
    Manages Blender subprocess for baking procedural shaders to textures.
    
    This class provides a Python interface to invoke Blender headlessly,
    execute baking scripts, and return the resulting OBJ + texture files.
    
    Usage:
        baker = BlenderBaker(blender_executable="C:/Program Files/Blender Foundation/Blender 3.1/blender.exe")
        obj_path, texture_path = baker.bake("model.blend", output_dir="./temp")
    """
    
    def __init__(self, blender_executable: str = "blender"):
        """
        Initialize Blender baker.
        
        Args:
            blender_executable: Path to Blender executable.
                               Default: "blender" (assumes in PATH)
        """
        self.blender_exe = blender_executable
        self._validate_blender()
        
        # Path to the Blender Python script
        self.script_path = Path(__file__).parent / "blender_scripts" / "bake_and_export.py"
        
        if not self.script_path.exists():
            raise FileNotFoundError(f"Bake script not found: {self.script_path}")
    
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
            # Extract version number (e.g., "Blender 3.1.0")
            version_line = [line for line in version_info.split('\n') if 'Blender' in line][0]
            print(f"✓ Found {version_line.strip()}")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Blender not found: {self.blender_exe}\n"
                f"Please install Blender or specify correct path."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Blender --version timed out")
    
    def bake(self, 
             blend_file: str, 
             output_dir: Optional[str] = None,
             texture_resolution: int = 4096,
             timeout: int = 600) -> Tuple[Path, Path]:
        """
        Bake procedural shaders from .blend file to texture + OBJ.
        
        Args:
            blend_file: Path to input .blend file
            output_dir: Output directory (default: temp directory)
            texture_resolution: Texture size in pixels (default: 4096)
            timeout: Max baking time in seconds (default: 600)
            
        Returns:
            Tuple of (obj_path, texture_path)
            
        Raises:
            FileNotFoundError: If blend file doesn't exist
            RuntimeError: If baking fails
            subprocess.TimeoutExpired: If baking exceeds timeout
        """
        blend_path = Path(blend_file)
        
        if not blend_path.exists():
            raise FileNotFoundError(f"Blend file not found: {blend_file}")
        
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(tempfile.mkdtemp(prefix="blender_bake_"))
        
        print(f"Baking {blend_path.name} to {output_path}")
        print(f"Texture resolution: {texture_resolution}x{texture_resolution}")
        
        # Expected output files
        obj_path = output_path / f"{blend_path.stem}.obj"
        texture_path = output_path / "baked_texture.png"
        
        # Build Blender command
        cmd = [
            str(self.blender_exe),
            "-b",                           # Background mode
            str(blend_path),                # Input file
            "-P", str(self.script_path),    # Python script
            "--",                           # Separator for script args
            str(output_path),               # Output directory
            str(texture_resolution)         # Texture size
        ]
        
        # Execute Blender
        start_time = time.time()

        # Create log file for Blender output
        log_file = output_path / "blender_bake.log"

        try:
            print("Starting Blender baker (this may take 1-5 minutes)...")
            print(f"Blender output will be logged to: {log_file}")

            # Start Blender process with output redirected to log file
            log = open(log_file, 'w')

            # On Windows, create new process group to avoid Ctrl+C propagation
            import sys
            if sys.platform == 'win32':
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT
                )

            # Poll for completion by checking output files
            poll_interval = 1.0  # Check every second
            max_polls = int(timeout / poll_interval)

            for i in range(max_polls):
                # Check if output files exist
                obj_exists = obj_path.exists()
                tex_exists = texture_path.exists()

                if i % 5 == 0:  # Print status every 5 seconds
                    print(f"  Polling... ({i}s) OBJ:{obj_exists} TEX:{tex_exists}")

                if obj_exists and tex_exists:
                    elapsed = time.time() - start_time
                    print(f"✓ Output files detected in {elapsed:.1f}s")

                    # Close log file
                    log.close()

                    # Give Blender a moment to finish writing
                    time.sleep(0.5)

                    # Terminate if still running
                    if process.poll() is None:
                        print("  Terminating Blender process...")
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()

                    break

                try:
                    time.sleep(poll_interval)
                except KeyboardInterrupt:
                    # Blender sends Ctrl+C on startup, ignore it
                    pass
            else:
                # Timeout reached
                log.close()
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

                raise subprocess.TimeoutExpired(
                    cmd, timeout,
                    f"Blender baking exceeded {timeout}s timeout"
                )

            elapsed = time.time() - start_time
            print(f"✓ Baking complete in {elapsed:.1f}s")
            
            # Validate output
            self._validate_output(obj_path, texture_path)

            return obj_path, texture_path

        except subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired(
                cmd, timeout,
                f"Blender baking exceeded {timeout}s timeout"
            )

    def _validate_output(self, obj_path: Path, texture_path: Path):
        """
        Verify that baking produced valid output files.

        Raises:
            RuntimeError: If output is invalid
        """
        # Check OBJ exists
        if not obj_path.exists():
            raise RuntimeError(f"OBJ not created: {obj_path}")

        # Check texture exists
        if not texture_path.exists():
            raise RuntimeError(f"Texture not created: {texture_path}")

        # Check file sizes
        obj_size = obj_path.stat().st_size
        texture_size = texture_path.stat().st_size

        if obj_size < 100:
            raise RuntimeError(f"OBJ too small ({obj_size} bytes), likely empty")

        if texture_size < 1000:
            raise RuntimeError(f"Texture too small ({texture_size} bytes), likely failed")

        print(f"✓ OBJ: {obj_size / 1e6:.1f} MB")
        print(f"✓ Texture: {texture_size / 1e6:.1f} MB")

    def cleanup_temp(self, path: Path):
        """Delete temporary baking directory."""
        if path.exists() and "blender_bake_" in str(path):
            shutil.rmtree(path)
            print(f"✓ Cleaned up temp dir: {path}")

