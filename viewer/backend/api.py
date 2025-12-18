#!/usr/bin/env python3
# ABOUTME: FastAPI routes for Gaussian Splat Viewer
# ABOUTME: Provides REST endpoints and WebSocket for file operations

import asyncio
import logging
from pathlib import Path
from typing import List, Set
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from .ply_parser import PLYParser
from .file_watcher import FileWatcher


# Request/Response models
class LODGenerateRequest(BaseModel):
    source_file: str
    target_count: int
    strategy: str = 'importance'


class LODGenerateResponse(BaseModel):
    success: bool
    lod_file: str
    path: str
    vertex_count: int
    reduction_ratio: float


class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    vertex_count: int


class FilesResponse(BaseModel):
    files: List[FileInfo]
    lod_files: List[FileInfo]


# API Router
router = APIRouter(prefix="/api")

# Global state
ply_parser = PLYParser()
file_watcher: FileWatcher = None
websocket_clients: Set[WebSocket] = set()
logger = logging.getLogger('gaussian_viewer.api')


def initialize_watcher(watch_dir: str, loop: asyncio.AbstractEventLoop):
    """Initialize file watcher with event loop."""
    global file_watcher
    file_watcher = FileWatcher(watch_dir)
    file_watcher.start(broadcast_file_event, loop)


async def broadcast_file_event(event_type: str, file_path: str):
    """Broadcast file event to all connected WebSocket clients."""
    message = {
        'type': event_type,
        'file': {
            'name': Path(file_path).name,
            'path': file_path
        }
    }
    
    # Send to all connected clients
    disconnected = set()
    for client in websocket_clients:
        try:
            await client.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            disconnected.add(client)
    
    # Remove disconnected clients
    websocket_clients.difference_update(disconnected)


@router.get("/files", response_model=FilesResponse)
async def list_files(watch_dir: str = "output_clouds"):
    """
    List all PLY files in the watched directory.
    
    Returns:
        FilesResponse with main files and LOD files
    """
    watch_path = Path(watch_dir)
    lod_path = watch_path / "LOD_output"
    
    # Get main files
    main_files = []
    if watch_path.exists():
        for ply_file in watch_path.glob("*.ply"):
            try:
                info = ply_parser.get_file_info(str(ply_file))
                main_files.append(FileInfo(**info))
            except Exception as e:
                logger.error(f"Error reading {ply_file}: {e}")
    
    # Get LOD files
    lod_files = []
    if lod_path.exists():
        for ply_file in lod_path.glob("*.ply"):
            try:
                info = ply_parser.get_file_info(str(ply_file))
                lod_files.append(FileInfo(**info))
            except Exception as e:
                logger.error(f"Error reading {ply_file}: {e}")
    
    return FilesResponse(files=main_files, lod_files=lod_files)


@router.get("/load/{filename}")
async def load_file(filename: str, watch_dir: str = "output_clouds", chunk_size: int = 100000):
    """
    Load gaussian data from PLY file with chunked streaming.

    Args:
        filename: Name of file in output_clouds/ or LOD_output/
        chunk_size: Number of points to load per chunk (default: 100k)

    Returns:
        Gaussian data dictionary
    """
    watch_path = Path(watch_dir)

    # Try main directory first
    file_path = watch_path / filename
    if not file_path.exists():
        # Try LOD directory
        file_path = watch_path / "LOD_output" / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    try:
        # For now, still load all at once but we'll add streaming in next step
        data = ply_parser.parse_ply(str(file_path))
        return data
    except Exception as e:
        logger.error(f"Error parsing {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing file: {str(e)}")


@router.get("/load-stream/{filename}")
async def load_file_stream(filename: str, watch_dir: str = "output_clouds"):
    """
    Stream gaussian data from PLY file in chunks with progress updates.
    DEPRECATED: Use /load-binary for better performance.

    Args:
        filename: Name of file in output_clouds/ or LOD_output/

    Returns:
        Server-Sent Events stream with progress and data chunks
    """
    from fastapi.responses import StreamingResponse
    import json

    watch_path = Path(watch_dir)

    # Try main directory first
    file_path = watch_path / filename
    if not file_path.exists():
        # Try LOD directory
        file_path = watch_path / "LOD_output" / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    async def generate_chunks():
        """Generate SSE chunks with progress updates."""
        try:
            # Get file info first
            file_size = file_path.stat().st_size
            vertex_count, _ = ply_parser.parse_header(open(file_path, 'rb'))

            # Send initial metadata
            yield f"data: {json.dumps({'type': 'metadata', 'filename': filename, 'vertex_count': vertex_count, 'file_size': file_size})}\n\n"

            # Parse file in chunks
            chunk_size = 50000  # 50k points per chunk
            data = ply_parser.parse_ply_chunked(str(file_path), chunk_size, progress_callback=lambda p, c, t: None)

            for chunk_idx, chunk_data in enumerate(data):
                progress = ((chunk_idx + 1) * chunk_size) / vertex_count * 100
                progress = min(progress, 100)

                # Send chunk data
                chunk_payload = {
                    'type': 'chunk',
                    'chunk_index': chunk_idx,
                    'progress': progress,
                    'data': chunk_data
                }
                yield f"data: {json.dumps(chunk_payload)}\n\n"

            # Send completion
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"Error streaming {filename}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate_chunks(), media_type="text/event-stream")


@router.get("/load-binary/{filename}")
async def load_file_binary(filename: str, watch_dir: str = "output_clouds"):
    """
    Stream gaussian data as binary chunks for maximum performance.

    Binary format per chunk:
    - Header (24 bytes):
      - Magic number (4 bytes): 0x47535053 ('GSPS' = Gaussian Splat Point Stream)
      - Chunk index (4 bytes uint32)
      - Point count in chunk (4 bytes uint32)
      - Total points in file (4 bytes uint32)
      - Progress percentage (4 bytes float32)
      - Reserved (4 bytes)
    - Data (point_count * 56 bytes):
      - Position XYZ (12 bytes: 3x float32)
      - SH DC RGB (12 bytes: 3x float32) - Spherical Harmonic DC terms
      - Scale XYZ (12 bytes: 3x float32) - Already converted from log space
      - Rotation WXYZ (16 bytes: 4x float32) - Quaternion
      - Opacity (4 bytes: 1x float32)

    Total per gaussian: 14 floats = 56 bytes

    Args:
        filename: Name of file in output_clouds/ or LOD_output/

    Returns:
        Binary stream with chunked gaussian data
    """
    from fastapi.responses import StreamingResponse
    import struct
    import numpy as np

    watch_path = Path(watch_dir)

    # Try main directory first
    file_path = watch_path / filename
    if not file_path.exists():
        # Try LOD directory
        file_path = watch_path / "LOD_output" / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    async def generate_binary_chunks():
        """Generate binary chunks with positions and colors only."""
        try:
            # Get file info first
            file_size = file_path.stat().st_size
            with open(file_path, 'rb') as f:
                vertex_count, _ = ply_parser.parse_header(f)

            logger.info(f"Starting binary stream for {filename}: {vertex_count} points, {file_size} bytes")

            # Send initial metadata as JSON header (for compatibility)
            import json
            metadata = json.dumps({
                'type': 'metadata',
                'filename': filename,
                'vertex_count': vertex_count,
                'file_size': file_size,
                'format': 'binary'
            })
            metadata_bytes = metadata.encode('utf-8')
            metadata_header = struct.pack('<I', len(metadata_bytes))  # 4-byte length prefix
            yield metadata_header + metadata_bytes

            # Parse file in chunks
            chunk_size = 50000  # 50k points per chunk
            chunk_data_generator = ply_parser.parse_ply_chunked(str(file_path), chunk_size)

            for chunk_idx, chunk_data in enumerate(chunk_data_generator):
                current_chunk_size = chunk_data['count']
                progress = min(((chunk_idx + 1) * chunk_size) / vertex_count * 100, 100.0)

                # Build binary chunk header (24 bytes)
                magic = 0x47535053  # 'GSPS'
                header = struct.pack(
                    '<IIIIfI',
                    magic,
                    chunk_idx,
                    current_chunk_size,
                    vertex_count,
                    progress,
                    0  # reserved
                )

                # Convert all properties to numpy arrays
                positions = np.array(chunk_data['positions'], dtype=np.float32)
                sh_dc = np.array(chunk_data['sh_dc'], dtype=np.float32)
                scales = np.array(chunk_data['scales'], dtype=np.float32)
                rotations = np.array(chunk_data['rotations'], dtype=np.float32)
                opacities = np.array(chunk_data['opacities'], dtype=np.float32)

                # Diagnostic: Log data stats for first chunk
                if chunk_idx == 0:
                    logger.info(f"First chunk gaussian diagnostics:")
                    logger.info(f"  Positions shape: {positions.shape}")
                    logger.info(f"  SH DC shape: {sh_dc.shape}")
                    logger.info(f"  Scales shape: {scales.shape}")
                    logger.info(f"  Rotations shape: {rotations.shape}")
                    logger.info(f"  Opacities shape: {opacities.shape}")
                    logger.info(f"  First gaussian:")
                    logger.info(f"    Position: {positions[0]}")
                    logger.info(f"    SH DC: {sh_dc[0]}")
                    logger.info(f"    Scale: {scales[0]}")
                    logger.info(f"    Rotation: {rotations[0]}")
                    logger.info(f"    Opacity: {opacities[0]}")

                # Interleave all properties: [x,y,z, sh0,sh1,sh2, sx,sy,sz, rw,rx,ry,rz, opacity, ...]
                # Total: 14 floats per gaussian
                interleaved = np.empty((current_chunk_size, 14), dtype=np.float32)
                interleaved[:, 0:3] = positions      # Position (3)
                interleaved[:, 3:6] = sh_dc          # SH DC (3)
                interleaved[:, 6:9] = scales         # Scales (3)
                interleaved[:, 9:13] = rotations     # Rotation quaternion (4)
                interleaved[:, 13] = opacities       # Opacity (1)

                # Convert to bytes
                data_bytes = interleaved.tobytes()

                # Yield header + data
                yield header + data_bytes

                logger.debug(f"Sent binary chunk {chunk_idx}: {current_chunk_size} points, {len(data_bytes)} bytes ({progress:.1f}%)")

            logger.info(f"Completed binary stream for {filename}")

        except Exception as e:
            logger.error(f"Error in binary stream for {filename}: {e}")
            # Send error as JSON
            import json
            error_msg = json.dumps({'type': 'error', 'message': str(e)})
            error_bytes = error_msg.encode('utf-8')
            error_header = struct.pack('<I', len(error_bytes))
            yield error_header + error_bytes

    return StreamingResponse(
        generate_binary_chunks(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={filename}.stream",
            "X-Content-Type-Options": "nosniff"
        }
    )


@router.post("/generate-lod", response_model=LODGenerateResponse)
async def generate_lod(request: LODGenerateRequest, watch_dir: str = "output_clouds"):
    """
    Generate LOD from full-resolution file.
    
    This endpoint will be fully implemented in Phase 3.
    For now, it returns a placeholder response.
    """
    # TODO: Implement in Phase 3
    raise HTTPException(status_code=501, detail="LOD generation will be implemented in Phase 3")


@router.websocket("/watch")
async def websocket_watch(websocket: WebSocket):
    """
    WebSocket endpoint for real-time file change notifications.
    
    Clients connect to this endpoint to receive notifications when
    PLY files are added, modified, or deleted.
    """
    await websocket.accept()
    websocket_clients.add(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(websocket_clients)}")
    
    try:
        # Keep connection alive
        while True:
            # Wait for messages from client (heartbeat)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_clients.discard(websocket)
        logger.info(f"WebSocket client removed. Total clients: {len(websocket_clients)}")

