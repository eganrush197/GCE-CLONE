# ABOUTME: Python wrapper for Blender subprocess operations
# ABOUTME: Manages headless Blender execution for shader baking

import subprocess
import tempfile
import shutil
import logging
import threading
from pathlib import Path
from typing import Tuple, Optional
import time
import sys
import platform

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging_utils import ProgressTracker, Timer, ProcessMonitor


class BlenderBaker:
    """
    Manages Blender subprocess for baking procedural shaders to textures.

    This class provides a Python interface to invoke Blender headlessly,
    execute baking scripts, and return the resulting OBJ + texture files.

    Usage:
        baker = BlenderBaker(blender_executable="C:/Program Files/Blender Foundation/Blender 3.1/blender.exe")
        obj_path, texture_path = baker.bake("model.blend", output_dir="./temp")
    """

    @staticmethod
    def _find_blender_windows() -> Optional[str]:
        """
        Automatically find Blender installation on Windows.

        Returns:
            Path to blender.exe if found, None otherwise
        """
        common_paths = [
            r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.5\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.4\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.3\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.1\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.0\blender.exe",
        ]

        for path in common_paths:
            if Path(path).exists():
                return path

        return None

    @staticmethod
    def _find_blender() -> str:
        """
        Automatically find Blender installation.

        Returns:
            Path to Blender executable
        """
        # Try Windows-specific paths first
        if sys.platform == 'win32':
            blender_path = BlenderBaker._find_blender_windows()
            if blender_path:
                return blender_path

        # Fall back to assuming it's in PATH
        return "blender"

    def __init__(self, blender_executable: str = None):
        """
        Initialize Blender baker.

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
            self.logger.debug("Found %s", version_line.strip())
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Blender not found: {self.blender_exe}\n"
                f"Please install Blender or specify correct path."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Blender --version timed out")

    def _copy_folder_contents(self, source_folder: Path, dest_folder: Path):
        """
        Copy all contents of source folder to destination folder.

        This preserves the directory structure and copies all files,
        including textures, blend files, and any subdirectories.

        Args:
            source_folder: Source folder to copy from
            dest_folder: Destination folder to copy to
        """
        # Count files for logging
        file_count = 0
        dir_count = 0

        for item in source_folder.iterdir():
            dest_item = dest_folder / item.name

            if item.is_dir():
                # Recursively copy directories
                self.logger.debug("Copying directory: %s/", item.name)
                shutil.copytree(item, dest_item, dirs_exist_ok=True)
                dir_count += 1
            else:
                # Copy files
                self.logger.debug("Copying file: %s", item.name)
                shutil.copy2(item, dest_item)
                file_count += 1

        self.logger.info("  Copied %d files and %d directories", file_count, dir_count)

    def bake(self,
             blend_file: str,
             output_dir: Optional[str] = None,
             texture_resolution: int = 4096,
             timeout: int = 600,
             use_gpu: bool = False,
             samples: int = 32,
             renderer: str = 'auto',
             bake_type: str = 'auto',
             denoise: bool = False,
             input_folder: Optional[str] = None) -> Tuple[Path, Path]:
        """
        Bake procedural shaders from .blend file to texture + OBJ.

        Args:
            blend_file: Path to input .blend file
            output_dir: Output directory (default: temp directory)
            texture_resolution: Texture size in pixels (default: 4096)
            timeout: Max baking time in seconds (default: 600)
            use_gpu: Use GPU acceleration if available (default: False)
            samples: Number of samples for Cycles (default: 32)
            renderer: Rendering engine - 'auto', 'cycles', or 'eevee' (default: 'auto')
            bake_type: Bake type - 'auto', 'DIFFUSE', 'COMBINED', 'EMIT' (default: 'auto')
            denoise: Enable denoising (default: False)
            input_folder: Optional folder containing .blend + textures (for folder-based input)

        Returns:
            Tuple of (obj_path, texture_path)

        Raises:
            FileNotFoundError: If blend file doesn't exist
            RuntimeError: If baking fails
            subprocess.TimeoutExpired: If baking exceeds timeout
        """
        blend_path = Path(blend_file).resolve()

        if not blend_path.exists():
            raise FileNotFoundError(f"Blend file not found: {blend_file}")

        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(tempfile.mkdtemp(prefix="blender_bake_"))

        # CRITICAL: Work on a COPY of the blend file to protect the original
        # This prevents any possibility of corrupting the user's source file
        work_dir = output_path / "_blender_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Determine the source folder for textures
        if input_folder:
            # Folder-based input: copy the ENTIRE folder structure
            source_folder = Path(input_folder).resolve()
            self.logger.info("Folder-based input detected: %s", source_folder.name)
            self.logger.info("Copying entire folder structure to working directory...")

            # Copy all contents of the folder to work_dir
            self._copy_folder_contents(source_folder, work_dir)

            # The blend file path within the work directory
            work_blend_path = work_dir / blend_path.name
        else:
            # Single file input: copy blend + nearby textures
            work_blend_path = work_dir / blend_path.name
            self.logger.info("Creating working copy of blend file...")

            # Copy the blend file
            shutil.copy2(blend_path, work_blend_path)

            # Also copy any texture files that might be referenced with relative paths
            # Look for common texture directories
            blend_dir = blend_path.parent

            # Standard texture directory names to look for
            standard_tex_dirs = ["textures", "Textures", "tex", "Tex", "images", "Images",
                                 "maps", "Maps", "materials", "Materials"]

            for tex_dir_name in standard_tex_dirs:
                tex_dir = blend_dir / tex_dir_name
                if tex_dir.exists() and tex_dir.is_dir():
                    dest_tex_dir = work_dir / tex_dir_name
                    self.logger.info("Copying texture directory: %s", tex_dir_name)
                    shutil.copytree(tex_dir, dest_tex_dir, dirs_exist_ok=True)

            # Also look for any directories containing "texture" in their name (case-insensitive)
            # This catches folders like "TREE-MODEL-COPY-TEXTURES", "MyModel_textures", etc.
            for item in blend_dir.iterdir():
                if item.is_dir():
                    item_lower = item.name.lower()
                    # Skip if already copied by standard names
                    if item.name in standard_tex_dirs:
                        continue
                    # Check if directory name contains texture-related keywords
                    if any(keyword in item_lower for keyword in ['texture', 'tex', 'material', 'map', 'image']):
                        dest_tex_dir = work_dir / item.name
                        if not dest_tex_dir.exists():
                            self.logger.info("Copying texture directory: %s", item.name)
                            shutil.copytree(item, dest_tex_dir, dirs_exist_ok=True)

            # Copy any loose image files in the same directory as the blend file
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tga', '*.bmp', '*.tif', '*.tiff', '*.exr']:
                for img_file in blend_dir.glob(ext):
                    if img_file.name != blend_path.name:  # Don't copy if somehow a blend has these extensions
                        dest_img = work_dir / img_file.name
                        if not dest_img.exists():
                            self.logger.debug("Copying texture: %s", img_file.name)
                            shutil.copy2(img_file, dest_img)

        self.logger.info("Input: %s (original)", blend_path.name)
        self.logger.info("Working directory: %s", work_dir)
        self.logger.info("Output: %s", output_path)
        self.logger.info("Texture resolution: %dx%d", texture_resolution, texture_resolution)

        # Expected output files
        obj_path = output_path / f"{blend_path.stem}.obj"
        texture_path = output_path / "baked_texture.png"

        # Clean up any existing output files from previous runs
        # This prevents false detection of "completed" files
        for old_file in [obj_path, texture_path]:
            if old_file.exists():
                self.logger.debug("Removing old output file: %s", old_file.name)
                old_file.unlink()

        # Build Blender command - use the WORKING COPY, not the original
        # NOTE: We now use "-b" (background mode) with low-level baking API
        # This avoids viewport context issues while still allowing baking to work
        cmd = [
            str(self.blender_exe),
            "-b",                           # Background mode
            str(work_blend_path),           # WORKING COPY (not original!)
            "-P", str(self.script_path),    # Python script
            "--",                           # Separator for script args
            str(output_path),               # Output directory
            str(texture_resolution),        # Texture size
            str(use_gpu).lower(),           # GPU flag
            str(samples),                   # Sample count
            renderer,                       # Renderer mode ('auto', 'cycles', 'eevee')
            str(denoise).lower(),           # Denoise flag
            bake_type                       # Bake type ('auto', 'DIFFUSE', 'COMBINED', 'EMIT')
        ]
        
        # Execute Blender
        start_time = time.time()

        # Create log file for Blender output
        log_file = output_path / "blender_bake.log"

        try:
            self.logger.info("Starting Blender process...")
            cmd_str = ' '.join(f'"{c}"' if ' ' in str(c) else str(c) for c in cmd)
            self.logger.info("Command: %s", cmd_str)
            self.logger.info("Log file: %s", log_file)

            # Start Blender process with output piped for real-time streaming
            # On Windows, create new process group to avoid Ctrl+C propagation
            if platform.system() == 'win32':
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1
                )

            # Open log file for writing using context manager to ensure it's always closed
            with open(log_file, 'w', encoding='utf-8', errors='replace') as log_file_handle:
                # Stream Blender output in real-time using a thread
                output_lines = []
                def stream_output():
                    for line in process.stdout:
                        line = line.rstrip()
                        if line:
                            output_lines.append(line)
                            log_file_handle.write(line + '\n')
                            log_file_handle.flush()
                            # Log Blender output with [Blender] prefix
                            self.logger.info("[Blender] %s", line)

                output_thread = threading.Thread(target=stream_output, daemon=True)
                output_thread.start()

                # Smart polling with progress tracker
                progress = ProgressTracker("Baking", timeout, update_interval=30, logger=self.logger)

                # Process monitor to detect if Blender is stuck
                monitor = ProcessMonitor(
                    process.pid,
                    name="Blender",
                    check_interval=60,  # Check every 60s
                    stuck_threshold=1.0,  # < 1% CPU = stuck
                    stuck_duration=300,  # Warn after 5min of low CPU
                    logger=self.logger
                )

                poll_interval = 1.0  # Check every second
                error_detected = False

                try:
                    while True:
                        # Check if process has exited (could be error or success)
                        if process.poll() is not None:
                            # Process ended - check if it was successful
                            obj_exists = obj_path.exists()
                            tex_exists = texture_path.exists()

                            if obj_exists and tex_exists:
                                # Success!
                                elapsed = time.time() - start_time
                                self.logger.info("Output files detected in %.1fs", elapsed)
                                break
                            else:
                                # Process ended but no output - check for errors
                                # Read log file content to include in error message
                                log_content = ""
                                if log_file.exists():
                                    try:
                                        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                                            log_lines = f.readlines()
                                            # Get last 20 lines or all if less
                                            log_content = ''.join(log_lines[-20:])
                                    except Exception as e:
                                        log_content = f"(Could not read log file: {e})"

                                # Look for error messages in output
                                error_lines = [line for line in output_lines if 'ERROR' in line.upper() or 'Error:' in line]
                                if error_lines:
                                    error_msg = '\n'.join(error_lines[-3:])  # Last 3 errors
                                    raise RuntimeError(
                                        f"Blender baking failed:\n{error_msg}\n\n"
                                        f"Log file content (last 20 lines):\n{log_content}"
                                    )
                                else:
                                    raise RuntimeError(
                                        f"Blender process ended without producing output files.\n\n"
                                        f"Log file content (last 20 lines):\n{log_content}"
                                    )

                        # Check if output files exist AND have reasonable sizes
                        # Files might be created empty before being written to
                        obj_exists = obj_path.exists()
                        tex_exists = texture_path.exists()

                        if obj_exists and tex_exists:
                            # Check file sizes - don't terminate until files have content
                            obj_size = obj_path.stat().st_size
                            tex_size = texture_path.stat().st_size

                            # OBJ should be at least 100 bytes, texture at least 1KB
                            if obj_size >= 100 and tex_size >= 1000:
                                elapsed = time.time() - start_time
                                self.logger.info("Output files detected in %.1fs (OBJ: %d bytes, Texture: %d bytes)",
                                               elapsed, obj_size, tex_size)

                                # Give Blender a moment to finish writing
                                time.sleep(2.0)

                                # Double-check sizes haven't changed (file still being written)
                                new_obj_size = obj_path.stat().st_size
                                new_tex_size = texture_path.stat().st_size

                                if new_obj_size != obj_size or new_tex_size != tex_size:
                                    # Files are still being written, wait more
                                    self.logger.debug("Files still being written, waiting...")
                                    time.sleep(5.0)

                                # Terminate if still running
                                if process.poll() is None:
                                    self.logger.debug("Terminating Blender process...")
                                    process.terminate()
                                    try:
                                        process.wait(timeout=5)
                                    except subprocess.TimeoutExpired:
                                        process.kill()

                                break
                            else:
                                # Files exist but are too small - still being written
                                self.logger.debug("Files exist but still being written (OBJ: %d bytes, Texture: %d bytes)",
                                                obj_size, tex_size)

                        # Check for timeout
                        if progress.is_timeout():
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()

                            raise subprocess.TimeoutExpired(
                                cmd, timeout,
                                f"Blender baking exceeded {timeout}s timeout"
                            )

                        # Monitor process health
                        status = monitor.check_and_warn()
                        if status['is_stuck']:
                            self.logger.warning(
                                "Blender may be stuck. Check Task Manager to verify CPU/GPU usage."
                            )

                        # Log progress milestones
                        progress.check_and_log()

                        try:
                            time.sleep(poll_interval)
                        except KeyboardInterrupt:
                            # Blender sends Ctrl+C on startup, ignore it
                            pass

                finally:
                    # Ensure output thread finishes writing before file is closed
                    output_thread.join(timeout=5)

                    # Wait for Blender to fully exit
                    try:
                        return_code = process.wait(timeout=5)
                        self.logger.info("Blender process exited with code: %d", return_code)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        self.logger.warning("Blender process killed after timeout")

            # File handle is now guaranteed to be closed

            # Validate output
            self._validate_output(obj_path, texture_path)

            # Clean up working directory (keep the outputs, delete the temp copy)
            try:
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                    self.logger.debug("Cleaned up working directory")
            except Exception as e:
                self.logger.warning("Could not clean up work directory: %s", e)

            return obj_path, texture_path

        except subprocess.TimeoutExpired:
            # Clean up on timeout too
            try:
                if work_dir.exists():
                    shutil.rmtree(work_dir)
            except Exception:
                pass
            raise subprocess.TimeoutExpired(
                cmd, timeout,
                f"Blender baking exceeded {timeout}s timeout"
            )

    def _wait_for_file_stable(self, file_path: Path, timeout: float = 10.0,
                              check_interval: float = 0.5) -> bool:
        """
        Wait for a file to exist and become stable (not being written to).

        A file is considered stable when its size doesn't change for 2 consecutive checks.
        This prevents TOCTOU race conditions where we detect a file exists but it's
        still being written.

        Args:
            file_path: Path to file to check
            timeout: Maximum time to wait in seconds
            check_interval: Time between stability checks in seconds

        Returns:
            True if file exists and is stable, False if timeout
        """
        start_time = time.time()
        last_size = None
        stable_count = 0

        while time.time() - start_time < timeout:
            if not file_path.exists():
                time.sleep(check_interval)
                continue

            try:
                current_size = file_path.stat().st_size

                if last_size is not None and current_size == last_size:
                    stable_count += 1
                    # Require 2 consecutive stable checks
                    if stable_count >= 2:
                        self.logger.debug("File stable: %s (%d bytes)", file_path.name, current_size)
                        return True
                else:
                    stable_count = 0

                last_size = current_size
                time.sleep(check_interval)

            except (OSError, PermissionError) as e:
                # File might be locked, wait and retry
                self.logger.debug("File access error (retrying): %s", e)
                time.sleep(check_interval)

        return False

    def _validate_output(self, obj_path: Path, texture_path: Path):
        """
        Verify that baking produced valid output files.

        This includes waiting for files to be stable (fully written) to avoid
        race conditions where files exist but are still being written.

        Raises:
            RuntimeError: If output is invalid or files not stable
        """
        # Wait for OBJ to be stable
        if not self._wait_for_file_stable(obj_path, timeout=10.0):
            if obj_path.exists():
                raise RuntimeError(
                    f"OBJ file exists but is not stable (still being written): {obj_path}"
                )
            else:
                raise RuntimeError(f"OBJ not created: {obj_path}")

        # Wait for texture to be stable
        if not self._wait_for_file_stable(texture_path, timeout=10.0):
            if texture_path.exists():
                raise RuntimeError(
                    f"Texture file exists but is not stable (still being written): {texture_path}"
                )
            else:
                raise RuntimeError(f"Texture not created: {texture_path}")

        # Check file sizes
        obj_size = obj_path.stat().st_size
        texture_size = texture_path.stat().st_size

        if obj_size < 100:
            raise RuntimeError(f"OBJ too small ({obj_size} bytes), likely empty")

        if texture_size < 1000:
            raise RuntimeError(f"Texture too small ({texture_size} bytes), likely failed")

        # Try to validate texture is readable
        try:
            from PIL import Image
            with Image.open(texture_path) as img:
                # Just verify we can open it
                _ = img.size
            self.logger.debug("Texture is valid image")
        except Exception as e:
            raise RuntimeError(f"Texture file exists but is not a valid image: {e}")

        self.logger.info("OBJ: %.1f MB", obj_size / 1e6)
        self.logger.info("Texture: %.1f MB", texture_size / 1e6)

    def cleanup_temp(self, path: Path):
        """Delete temporary baking directory."""
        if path.exists() and "blender_bake_" in str(path):
            shutil.rmtree(path)
            self.logger.debug("Cleaned up temp dir: %s", path)

