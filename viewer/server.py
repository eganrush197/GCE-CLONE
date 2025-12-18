#!/usr/bin/env python3
# ABOUTME: FastAPI server for Gaussian Splat Viewer
# ABOUTME: Serves static files and provides REST/WebSocket API

import asyncio
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

from backend.api import router, initialize_watcher


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gaussian_viewer')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Gaussian Splat Viewer...")

    # Get event loop
    loop = asyncio.get_event_loop()

    # Initialize file watcher
    watch_dir = Path(__file__).parent.parent / "output_clouds"
    logger.info(f"Watching directory: {watch_dir}")
    initialize_watcher(str(watch_dir), loop)

    logger.info("Gaussian Splat Viewer started successfully!")
    logger.info("API documentation available at: http://localhost:8000/docs")

    yield

    # Shutdown
    logger.info("Shutting down Gaussian Splat Viewer...")

    # Stop file watcher
    from backend.api import file_watcher
    if file_watcher:
        file_watcher.stop()

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Gaussian Splat Viewer",
    description="Web-based viewer for Gaussian Splat PLY files",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression middleware for better performance
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # Only compress responses larger than 1KB
    compresslevel=6     # Balance between speed and compression ratio (1-9)
)

# Include API routes
app.include_router(router)

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return {
            "message": "Gaussian Splat Viewer API",
            "status": "running",
            "docs": "/docs",
            "note": "Frontend not yet implemented. See /docs for API documentation."
        }


def main():
    """Run the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gaussian Splat Viewer Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--watch-dir', default='output_clouds', 
                       help='Directory to watch for PLY files')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("Gaussian Splat Viewer")
    logger.info("="*70)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Watch directory: {args.watch_dir}")
    logger.info("="*70)
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()

