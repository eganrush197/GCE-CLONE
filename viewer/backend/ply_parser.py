#!/usr/bin/env python3
# ABOUTME: Binary PLY parser for Gaussian Splat files
# ABOUTME: Reads 20-property PLY format and converts to JSON-serializable format

import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Generator, Callable, Optional
import logging


class PLYParser:
    """
    Parse binary PLY files containing Gaussian Splat data.

    Supports two formats:

    1. Legacy format (20 properties, 71 bytes):
       - Position (x, y, z)
       - Normals (nx, ny, nz)
       - SH DC terms (f_dc_0, f_dc_1, f_dc_2)
       - Opacity
       - Scales (scale_0, scale_1, scale_2) - stored in LOG SPACE
       - Rotation quaternion (rot_0, rot_1, rot_2, rot_3)
       - RGB colors (red, green, blue) - uchar

    2. Phase 1 optimized format (17 properties, 68 bytes):
       - Same as legacy but WITHOUT RGB properties
       - RGB computed from SH DC: RGB = SH_DC + 0.5

    The parser auto-detects which format based on property count.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('gaussian_viewer.ply_parser')
    
    def parse_header(self, file_handle) -> Tuple[int, List[Tuple[str, str]]]:
        """
        Parse PLY header to extract vertex count and properties.
        
        Returns:
            Tuple of (vertex_count, properties_list)
            properties_list is [(name, type), ...]
        """
        properties = []
        vertex_count = 0
        
        while True:
            line = file_handle.readline().decode('ascii').strip()
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))
            elif line == 'end_header':
                break
        
        return vertex_count, properties
    
    def parse_ply(self, file_path: str) -> Dict:
        """
        Parse PLY file and return gaussian data.
        
        Args:
            file_path: Path to PLY file
            
        Returns:
            Dictionary with gaussian data:
            {
                'filename': str,
                'vertex_count': int,
                'gaussians': {
                    'positions': [[x, y, z], ...],
                    'scales': [[sx, sy, sz], ...],
                    'rotations': [[w, x, y, z], ...],
                    'colors': [[r, g, b], ...],
                    'opacities': [o1, o2, ...],
                    'normals': [[nx, ny, nz], ...]
                }
            }
        """
        file_path = Path(file_path)
        self.logger.info(f"Parsing PLY file: {file_path}")
        
        with open(file_path, 'rb') as f:
            vertex_count, properties = self.parse_header(f)
            
            self.logger.info(f"Vertex count: {vertex_count}")
            self.logger.debug(f"Properties: {properties}")
            
            # Calculate bytes per vertex
            type_sizes = {'float': 4, 'uchar': 1, 'int': 4, 'double': 8}
            bytes_per_vertex = sum(type_sizes.get(t, 4) for _, t in properties)

            # Detect format based on property count
            has_rgb = len(properties) == 20  # Legacy format with RGB

            if has_rgb:
                self.logger.debug("Detected legacy format (20 properties, 71 bytes)")
                unpack_format = '<17f3B'
            else:
                self.logger.debug("Detected Phase 1 optimized format (17 properties, 68 bytes)")
                unpack_format = '<17f'

            # Read all binary data
            binary_data = f.read(vertex_count * bytes_per_vertex)

            # Parse vertices
            positions = np.zeros((vertex_count, 3), dtype=np.float32)
            normals = np.zeros((vertex_count, 3), dtype=np.float32)
            sh_dc = np.zeros((vertex_count, 3), dtype=np.float32)
            opacities = np.zeros(vertex_count, dtype=np.float32)
            scales = np.zeros((vertex_count, 3), dtype=np.float32)
            rotations = np.zeros((vertex_count, 4), dtype=np.float32)

            for i in range(vertex_count):
                offset = i * bytes_per_vertex
                vertex_data = binary_data[offset:offset + bytes_per_vertex]

                # Unpack based on detected format
                unpacked = struct.unpack(unpack_format, vertex_data)

                # Map to arrays
                positions[i] = unpacked[0:3]      # x, y, z
                normals[i] = unpacked[3:6]        # nx, ny, nz
                sh_dc[i] = unpacked[6:9]          # f_dc_0, f_dc_1, f_dc_2
                opacities[i] = unpacked[9]        # opacity
                scales[i] = unpacked[10:13]       # scale_0, scale_1, scale_2 (LOG SPACE!)
                rotations[i] = unpacked[13:17]    # rot_0, rot_1, rot_2, rot_3
                # Note: RGB properties removed in Phase 1 format
            
            # Convert scales from log space to linear space
            scales = np.exp(scales)
            
            # Convert SH DC to RGB colors (f_dc + 0.5, clamped to [0, 1])
            colors = np.clip(sh_dc + 0.5, 0.0, 1.0)
            
            self.logger.info(f"Successfully parsed {vertex_count} gaussians")
            
            return {
                'filename': file_path.name,
                'vertex_count': vertex_count,
                'gaussians': {
                    'positions': positions.tolist(),
                    'scales': scales.tolist(),
                    'rotations': rotations.tolist(),
                    'colors': colors.tolist(),
                    'opacities': opacities.tolist(),
                    'normals': normals.tolist()
                }
            }
    
    def get_file_info(self, file_path: str) -> Dict:
        """
        Get basic info about PLY file without parsing all data.
        
        Returns:
            {
                'name': str,
                'path': str,
                'size': int (bytes),
                'vertex_count': int
            }
        """
        file_path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            vertex_count, _ = self.parse_header(f)
        
        return {
            'name': file_path.name,
            'path': str(file_path),
            'size': file_path.stat().st_size,
            'vertex_count': vertex_count
        }

    def parse_ply_chunked(
        self,
        file_path: str,
        chunk_size: int = 50000,
        progress_callback: Optional[Callable[[int, int, int], None]] = None
    ) -> Generator[Dict, None, None]:
        """
        Parse PLY file in chunks for streaming/progressive loading.

        Args:
            file_path: Path to PLY file
            chunk_size: Number of vertices per chunk
            progress_callback: Optional callback(current, chunk_size, total) for progress updates

        Yields:
            Dictionary chunks with gaussian data for each chunk
        """
        file_path = Path(file_path)
        self.logger.info(f"Parsing PLY file in chunks: {file_path}, chunk_size={chunk_size}")

        with open(file_path, 'rb') as f:
            vertex_count, properties = self.parse_header(f)

            self.logger.info(f"Total vertices: {vertex_count}, will process in chunks of {chunk_size}")

            # Calculate bytes per vertex
            type_sizes = {'float': 4, 'uchar': 1, 'int': 4, 'double': 8}
            bytes_per_vertex = sum(type_sizes.get(t, 4) for _, t in properties)

            # Detect format based on property count
            has_rgb = len(properties) == 20  # Legacy format with RGB

            if has_rgb:
                self.logger.debug("Detected legacy format (20 properties, 71 bytes)")
                unpack_format = '<17f3B'
            else:
                self.logger.debug("Detected Phase 1 optimized format (17 properties, 68 bytes)")
                unpack_format = '<17f'

            # Process in chunks
            vertices_processed = 0
            chunk_index = 0

            while vertices_processed < vertex_count:
                # Calculate chunk size (last chunk might be smaller)
                current_chunk_size = min(chunk_size, vertex_count - vertices_processed)

                # Read chunk of binary data
                chunk_bytes = current_chunk_size * bytes_per_vertex
                binary_data = f.read(chunk_bytes)

                if not binary_data:
                    break

                # Parse chunk
                positions = []
                normals = []
                colors = []
                opacities = []
                scales = []
                rotations = []

                for i in range(current_chunk_size):
                    offset = i * bytes_per_vertex
                    vertex_data = binary_data[offset:offset + bytes_per_vertex]

                    # Unpack based on detected format
                    unpacked = struct.unpack(unpack_format, vertex_data)

                    # Extract components
                    pos = unpacked[0:3]
                    norm = unpacked[3:6]
                    sh_dc = unpacked[6:9]  # Keep as-is for proper gaussian splatting
                    opacity = unpacked[9]
                    scale = np.exp(np.array(unpacked[10:13], dtype=np.float32))  # Convert from log space
                    rot = unpacked[13:17]
                    # Note: RGB properties removed in Phase 1 format

                    # Keep SH DC as-is (don't convert to RGB)
                    # Conversion will happen in shader for proper rendering
                    sh_dc_list = list(sh_dc)

                    positions.append(list(pos))
                    normals.append(list(norm))
                    colors.append(sh_dc_list)  # Now contains SH DC, not RGB
                    opacities.append(opacity)
                    scales.append(scale.tolist())
                    rotations.append(list(rot))

                vertices_processed += current_chunk_size

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(vertices_processed, current_chunk_size, vertex_count)

                # Yield chunk data
                chunk_data = {
                    'positions': positions,
                    'normals': normals,
                    'sh_dc': colors,  # SH DC terms (not RGB colors!)
                    'opacities': opacities,
                    'scales': scales,
                    'rotations': rotations,
                    'count': current_chunk_size
                }

                self.logger.debug(f"Yielding chunk {chunk_index}: {current_chunk_size} vertices ({vertices_processed}/{vertex_count})")
                chunk_index += 1

                yield chunk_data

            self.logger.info(f"Completed parsing {vertices_processed} vertices in {chunk_index} chunks")

