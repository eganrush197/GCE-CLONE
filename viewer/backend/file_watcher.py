#!/usr/bin/env python3
# ABOUTME: File system watcher for monitoring PLY files
# ABOUTME: Uses watchdog to detect file changes and notify clients via WebSocket

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


class PLYFileHandler(FileSystemEventHandler):
    """Handler for PLY file system events."""
    
    def __init__(self, callback: Callable, loop: asyncio.AbstractEventLoop):
        """
        Initialize handler.
        
        Args:
            callback: Async function to call on file events
            loop: Event loop to schedule callbacks
        """
        self.callback = callback
        self.loop = loop
        self.logger = logging.getLogger('gaussian_viewer.file_watcher')
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if not event.is_directory and event.src_path.endswith('.ply'):
            self.logger.info(f"File created: {event.src_path}")
            asyncio.run_coroutine_threadsafe(
                self.callback('file_added', event.src_path),
                self.loop
            )
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if not event.is_directory and event.src_path.endswith('.ply'):
            self.logger.info(f"File modified: {event.src_path}")
            asyncio.run_coroutine_threadsafe(
                self.callback('file_modified', event.src_path),
                self.loop
            )
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if not event.is_directory and event.src_path.endswith('.ply'):
            self.logger.info(f"File deleted: {event.src_path}")
            asyncio.run_coroutine_threadsafe(
                self.callback('file_deleted', event.src_path),
                self.loop
            )


class FileWatcher:
    """
    Watch directory for PLY file changes.
    
    Monitors a directory and its subdirectories for .ply file changes,
    and notifies registered callbacks.
    """
    
    def __init__(self, watch_dir: str):
        """
        Initialize file watcher.
        
        Args:
            watch_dir: Directory to watch for PLY files
        """
        self.watch_dir = Path(watch_dir)
        self.observer: Optional[Observer] = None
        self.logger = logging.getLogger('gaussian_viewer.file_watcher')
        
        if not self.watch_dir.exists():
            self.watch_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created watch directory: {self.watch_dir}")
    
    def start(self, callback: Callable, loop: asyncio.AbstractEventLoop):
        """
        Start watching directory.
        
        Args:
            callback: Async function to call on file events
                     Signature: async def callback(event_type: str, file_path: str)
            loop: Event loop for scheduling callbacks
        """
        if self.observer is not None:
            self.logger.warning("Watcher already started")
            return
        
        self.logger.info(f"Starting file watcher on: {self.watch_dir}")
        
        event_handler = PLYFileHandler(callback, loop)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.watch_dir), recursive=True)
        self.observer.start()
        
        self.logger.info("File watcher started")
    
    def stop(self):
        """Stop watching directory."""
        if self.observer is None:
            return
        
        self.logger.info("Stopping file watcher")
        self.observer.stop()
        self.observer.join()
        self.observer = None
        self.logger.info("File watcher stopped")
    
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self.observer is not None and self.observer.is_alive()

