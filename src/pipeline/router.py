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
        
        Args:
            input_file: Path to input file
            
        Returns:
            Tuple of (needs_stage1, needs_stage2)
            - needs_stage1: True if Blender baking is required
            - needs_stage2: True if Gaussian conversion is required
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
        """
        Get human-readable description of processing path.
        
        Args:
            input_file: Path to input file
            
        Returns:
            Description string of the processing pipeline
        """
        needs_stage1, needs_stage2 = FileRouter.route(input_file)
        
        if needs_stage1 and needs_stage2:
            return "Blender baking -> Gaussian conversion -> LOD generation"
        elif needs_stage2:
            return "Gaussian conversion -> LOD generation"
        else:
            return "Unknown processing path"

