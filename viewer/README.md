# Gaussian Splat Viewer

Web-based viewer for Gaussian Splat PLY files with real-time LOD generation and file watching.

## Features

### Phase 1: Backend Foundation ✅ COMPLETE

- **FastAPI Server**: Modern async web framework
- **PLY Parser**: Binary PLY reader for gaussian format
- **File Watcher**: Real-time monitoring of `output_clouds/` directory
- **REST API**: List files, load gaussian data
- **WebSocket API**: Real-time file change notifications
- **CORS Support**: For local development

### Phase 2: Basic Renderer ✅ COMPLETE

- Three.js WebGL renderer
- Orbit camera controls
- File selection UI
- Gaussian visualization with custom shaders

### Phase 3: LOD & Streaming ✅ COMPLETE

- Binary data streaming for performance
- LOD level support
- Real-time file loading

### Current Features

- View Gaussian Splat PLY files in browser
- Pan, zoom, and orbit controls
- Binary streaming for large files
- Real-time file watching with WebSocket updates

## Installation

### 1. Install Dependencies

```bash
# Install viewer dependencies
pip install fastapi uvicorn[standard] watchdog python-multipart

# Or install all requirements
pip install -r ../requirements.txt
```

### 2. Prepare Output Directory

The viewer watches the `output_clouds/` directory for PLY files:

```bash
# Create directories (already done if you ran Phase 1)
mkdir -p output_clouds/LOD_output

# Copy your PLY files to output_clouds/
cp output_dir/*.ply output_clouds/
```

## Usage

### Start the Server

```bash
# From the viewer/ directory
python server.py

# Or with custom options
python server.py --host 0.0.0.0 --port 8000 --watch-dir ../output_clouds
```

### Access the Viewer

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc

### API Endpoints

#### List Files
```bash
curl http://localhost:8000/api/files
```

Returns all PLY files in `output_clouds/` and `output_clouds/LOD_output/`

#### Load File
```bash
curl http://localhost:8000/api/load/packed-tree_full.ply
```

Returns gaussian data (positions, scales, rotations, colors, opacities)

#### WebSocket Watch
```javascript
const ws = new WebSocket('ws://localhost:8000/api/watch');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('File event:', data.type, data.file);
};
```

Receives real-time notifications when PLY files are added, modified, or deleted.

## Architecture

```
viewer/
├── server.py              # FastAPI entry point
├── backend/
│   ├── __init__.py
│   ├── ply_parser.py      # Binary PLY reader
│   ├── file_watcher.py    # Directory monitoring
│   └── api.py             # FastAPI routes
└── static/                # Frontend (Phase 2+)
    ├── index.html
    ├── css/
    └── js/
```

## PLY File Format

The viewer expects binary PLY files with 20 properties (71 bytes per vertex):

- **Position**: x, y, z (float32)
- **Normals**: nx, ny, nz (float32)
- **SH DC**: f_dc_0, f_dc_1, f_dc_2 (float32)
- **Opacity**: opacity (float32)
- **Scales**: scale_0, scale_1, scale_2 (float32, **log space**)
- **Rotation**: rot_0, rot_1, rot_2, rot_3 (float32, quaternion)
- **RGB**: red, green, blue (uint8)

**Note**: Scales are stored in log space in the PLY file and automatically converted to linear space by the parser.

## Development

### Enable Auto-Reload

```bash
python server.py --reload
```

The server will automatically restart when code changes are detected.

### Logging

Logs are written to stdout with timestamps:

```
2024-12-10 10:30:00 - gaussian_viewer - INFO - Starting Gaussian Splat Viewer...
2024-12-10 10:30:00 - gaussian_viewer.file_watcher - INFO - Starting file watcher on: output_clouds
2024-12-10 10:30:00 - gaussian_viewer - INFO - Gaussian Splat Viewer started successfully!
```

## Testing

Test the API endpoints:

```bash
# List files
curl http://localhost:8000/api/files | jq

# Load a file
curl http://localhost:8000/api/load/packed-tree_full.ply | jq '.vertex_count'

# Check server health
curl http://localhost:8000/
```

## Known Issues

- Large files (>1M gaussians) may cause browser memory issues
- See `docs/TROUBLESHOOTING.md` for viewer-specific issues

## Troubleshooting

### Port Already in Use

```bash
# Use a different port
python server.py --port 8001
```

### No PLY Files Found

Make sure PLY files are in the `output_clouds/` directory:

```bash
ls -la output_clouds/*.ply
```

### WebSocket Connection Failed

Check that the server is running and accessible:

```bash
curl http://localhost:8000/
```

## License

Part of the Gaussian Mesh Converter project.

