# ABOUTME: Configuration dataclass for pipeline settings
# ABOUTME: Validates user inputs and provides defaults

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# Supported texture subfolder names (searched in order)
TEXTURE_FOLDER_NAMES = [
    'textures',
    'Textures',
    'tex',
    'Tex',
    'maps',
    'Maps',
    'images',
    'Images',
    'materials',
    'Materials',
]


@dataclass
class PipelineConfig:
    """Configuration for the unified gaussian pipeline."""

    input_file: Path
    output_dir: Path
    lod_levels: List[int] = field(default_factory=lambda: [5000, 25000, 100000])
    strategy: str = 'hybrid'
    lod_strategy: str = 'importance'
    texture_resolution: int = 4096
    blender_executable: Optional[str] = None  # Auto-detect if None
    bake_timeout: int = 1800  # 30 minutes default (increased from 600s)
    use_gpu: bool = False  # Use GPU for baking
    bake_samples: int = 32  # Cycles samples
    renderer: str = 'auto'  # 'auto', 'cycles', or 'eevee'
    bake_type: str = 'auto'  # 'auto', 'DIFFUSE', 'COMBINED', 'EMIT'
    denoise: bool = False  # Enable denoising
    keep_temp_files: bool = False
    device: str = 'cpu'
    compress: bool = False  # Phase 1: Compress PLY files with gzip

    # Packed texture mode
    use_packed: bool = False  # Use packed texture extraction instead of baking
    uv_layer: str = 'uv0'  # UV layer name for texture sampling
    vertex_color_blend_mode: str = 'multiply'  # Vertex color blending mode
    use_mipmaps: bool = True  # Generate and use mipmaps for texture filtering
    texture_filter: str = 'bilinear'  # Texture filtering mode: 'nearest' or 'bilinear'

    # Folder-based input support
    input_folder: Optional[Path] = None  # Set if input was a folder

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert paths
        self.input_file = Path(self.input_file)
        self.output_dir = Path(self.output_dir)

        # Check if input exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input not found: {self.input_file}")

        # Handle folder-based input
        if self.input_file.is_dir():
            self.input_folder = self.input_file
            self.input_file = self._find_blend_in_folder(self.input_folder)
        else:
            # Single file input - check if there's a parent folder with textures
            self.input_folder = None

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

    def _find_blend_in_folder(self, folder: Path) -> Path:
        """
        Find a .blend file in the given folder.

        Args:
            folder: Path to folder to search

        Returns:
            Path to the .blend file

        Raises:
            FileNotFoundError: If no .blend file found
            ValueError: If multiple .blend files found
        """
        blend_files = list(folder.glob('*.blend'))

        if not blend_files:
            # Also check for other supported formats
            for ext in ['.obj', '.glb', '.fbx']:
                files = list(folder.glob(f'*{ext}'))
                if files:
                    if len(files) == 1:
                        return files[0]
                    else:
                        raise ValueError(
                            f"Multiple {ext} files found in folder:\n"
                            + "\n".join(f"  - {f.name}" for f in files) +
                            "\nPlease specify which file to use."
                        )

            raise FileNotFoundError(
                f"No .blend, .obj, .glb, or .fbx files found in folder: {folder}"
            )

        if len(blend_files) == 1:
            return blend_files[0]

        # Multiple .blend files - try to find the main one
        # Prefer files without "backup" or "autosave" in name
        main_files = [f for f in blend_files
                      if 'backup' not in f.name.lower()
                      and 'autosave' not in f.name.lower()
                      and not f.name.startswith('.')]

        if len(main_files) == 1:
            return main_files[0]

        raise ValueError(
            f"Multiple .blend files found in folder:\n"
            + "\n".join(f"  - {f.name}" for f in blend_files) +
            "\nPlease specify which file to use or remove extras."
        )

    def get_texture_search_paths(self) -> List[Path]:
        """
        Get list of paths to search for textures.

        Returns:
            List of directories to search for texture files
        """
        search_paths = []

        # If input was a folder, search the whole folder structure
        if self.input_folder:
            # Add the folder itself (for loose textures)
            search_paths.append(self.input_folder)

            # Add known texture subfolder names
            for subfolder_name in TEXTURE_FOLDER_NAMES:
                subfolder = self.input_folder / subfolder_name
                if subfolder.exists():
                    search_paths.append(subfolder)

            # Also search any subdirectory that exists
            for subdir in self.input_folder.iterdir():
                if subdir.is_dir() and subdir not in search_paths:
                    search_paths.append(subdir)
        else:
            # Single file input - search relative to the file
            file_dir = self.input_file.parent
            search_paths.append(file_dir)

            for subfolder_name in TEXTURE_FOLDER_NAMES:
                subfolder = file_dir / subfolder_name
                if subfolder.exists():
                    search_paths.append(subfolder)

        return search_paths

