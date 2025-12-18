"""
Logging utilities for the Gaussian pipeline.

Provides structured logging with timestamps, timing, and progress tracking.
"""

import logging
import time
from typing import Optional, Dict, List
from dataclasses import dataclass, field

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[37m',     # White
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """
    Configure logging for the pipeline.
    
    Args:
        verbose: Enable DEBUG level logging
        quiet: Only show WARNING and above
        
    Returns:
        Configured logger
    """
    # Determine log level
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Configure root logger
    logger = logging.getLogger('gaussian_pipeline')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


class Timer:
    """Context manager for timing operations with automatic logging."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed
            logger: Logger to use (defaults to gaussian_pipeline logger)
        """
        self.name = name
        self.logger = logger or logging.getLogger('gaussian_pipeline')
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        self.logger.info(f"[TIMER] {self.name} started...")
        return self
    
    def __exit__(self, *args):
        """Stop timing and log result."""
        self.elapsed = time.time() - self.start_time
        self.logger.info(f"[OK] {self.name} complete in {self.elapsed:.1f}s")


@dataclass
class TimingStats:
    """Track timing statistics for pipeline stages."""
    
    name: str
    elapsed: float
    substeps: List['TimingStats'] = field(default_factory=list)
    
    def add_substep(self, name: str, elapsed: float):
        """Add a substep timing."""
        self.substeps.append(TimingStats(name, elapsed))
    
    def get_percentage(self, total: float) -> float:
        """Get percentage of total time."""
        return (self.elapsed / total * 100) if total > 0 else 0
    
    def format_tree(self, total_time: float, indent: int = 0) -> str:
        """Format as a tree structure."""
        lines = []
        prefix = "  " * indent
        pct = self.get_percentage(total_time)
        
        # Format time with appropriate precision
        if self.elapsed < 1:
            time_str = f"{self.elapsed*1000:.0f}ms"
        else:
            time_str = f"{self.elapsed:.1f}s"
        
        # Main line
        dots = "." * max(1, 50 - len(prefix) - len(self.name))
        lines.append(f"{prefix}{self.name} {dots} {time_str:>8} ({pct:>5.1f}%)")
        
        # Substeps
        for substep in self.substeps:
            lines.extend(substep.format_tree(total_time, indent + 1).split('\n'))
        
        return '\n'.join(lines)


class ProgressTracker:
    """Track and log progress with smart milestone updates."""
    
    def __init__(self, 
                 name: str,
                 timeout: int,
                 update_interval: int = 30,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize progress tracker.
        
        Args:
            name: Name of the operation
            timeout: Maximum time allowed (seconds)
            update_interval: How often to log updates (seconds)
            logger: Logger to use
        """
        self.name = name
        self.timeout = timeout
        self.update_interval = update_interval
        self.logger = logger or logging.getLogger('gaussian_pipeline')
        self.start_time = time.time()
        self.last_update = self.start_time
    
    def check_and_log(self) -> float:
        """
        Check elapsed time and log if interval passed.
        
        Returns:
            Elapsed time in seconds
        """
        now = time.time()
        elapsed = now - self.start_time
        
        # Log milestone updates
        if now - self.last_update >= self.update_interval:
            pct = (elapsed / self.timeout * 100) if self.timeout > 0 else 0
            self.logger.info(
                f"[TIMER] {self.name}: {elapsed:.0f}s elapsed "
                f"(timeout: {self.timeout}s, {pct:.0f}%)"
            )
            self.last_update = now

        return elapsed

    def is_timeout(self) -> bool:
        """
        Check if timeout has been reached.

        Returns:
            True if timeout exceeded, False otherwise
        """
        elapsed = time.time() - self.start_time
        return elapsed >= self.timeout

class ProcessMonitor:
    """Monitor subprocess CPU/memory usage to detect if process is stuck."""

    def __init__(self,
                 process_pid: int,
                 name: str = "Process",
                 check_interval: int = 60,
                 stuck_threshold: float = 1.0,
                 stuck_duration: int = 300,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize process monitor.

        Args:
            process_pid: PID of process to monitor
            name: Name of the process for logging
            check_interval: How often to check CPU usage (seconds)
            stuck_threshold: CPU % below which process is considered stuck (default: 1%)
            stuck_duration: How long process must be stuck before warning (seconds, default: 300s = 5min)
            logger: Logger to use
        """
        self.process_pid = process_pid
        self.name = name
        self.check_interval = check_interval
        self.stuck_threshold = stuck_threshold
        self.stuck_duration = stuck_duration
        self.logger = logger or logging.getLogger('gaussian_pipeline')

        self.last_check = time.time()
        self.low_cpu_start = None
        self.last_cpu_percent = 0.0

        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available - process monitoring disabled")
            self.psutil_process = None
        else:
            try:
                self.psutil_process = psutil.Process(process_pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                self.logger.warning(f"Could not attach to process {process_pid}: {e}")
                self.psutil_process = None

    def check_and_warn(self) -> Dict[str, float]:
        """
        Check process status and warn if appears stuck.

        Returns:
            Dict with cpu_percent, memory_mb, and is_stuck status
        """
        now = time.time()

        # Only check at intervals
        if now - self.last_check < self.check_interval:
            return {
                'cpu_percent': self.last_cpu_percent,
                'memory_mb': 0.0,
                'is_stuck': False
            }

        self.last_check = now

        if not self.psutil_process:
            return {'cpu_percent': 0.0, 'memory_mb': 0.0, 'is_stuck': False}

        try:
            # Get CPU usage (averaged over check_interval)
            cpu_percent = self.psutil_process.cpu_percent(interval=1.0)
            memory_info = self.psutil_process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            self.last_cpu_percent = cpu_percent

            # Check if CPU usage is very low (potentially stuck)
            if cpu_percent < self.stuck_threshold:
                if self.low_cpu_start is None:
                    self.low_cpu_start = now
                else:
                    stuck_time = now - self.low_cpu_start
                    if stuck_time >= self.stuck_duration:
                        self.logger.warning(
                            f"[WARN] {self.name} appears stuck: "
                            f"CPU usage {cpu_percent:.1f}% for {stuck_time:.0f}s"
                        )
                        return {
                            'cpu_percent': cpu_percent,
                            'memory_mb': memory_mb,
                            'is_stuck': True
                        }
            else:
                # Reset stuck timer if CPU usage is normal
                self.low_cpu_start = None
                self.logger.debug(
                    f"{self.name} active: CPU {cpu_percent:.1f}%, Memory {memory_mb:.0f}MB"
                )

            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'is_stuck': False
            }

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process ended or access denied
            return {'cpu_percent': 0.0, 'memory_mb': 0.0, 'is_stuck': False}


