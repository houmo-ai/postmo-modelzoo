import os
import time
import threading
from contextlib import contextmanager
from typing import Optional, Callable, Dict, Any

import psutil


class ProcessMemoryMonitor:
    """Monitors the memory usage of the current Python process in real-time."""

    def __init__(
        self,
        interval: float = 2.0,
        log_file: Optional[str] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        quiet: bool = False,
    ):
        """
        Args:
            interval: Time between measurements in seconds.
            log_file: Path to log file. If None, prints to console.
            callback: Optional callback function called with memory info dict.
            quiet: If True, suppress console output (still writes to log_file if provided).
        """
        self._process = psutil.Process(os.getpid())
        self._interval = interval
        self._log_file = log_file
        self._callback = callback
        self._quiet = quiet
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_rss_mb = 0.0
        self._file_handle: Optional[Any] = None

    @property
    def peak_memory_mb(self) -> float:
        """Returns peak RSS memory in MB."""
        return self._peak_rss_mb

    @property
    def is_running(self) -> bool:
        """Returns whether monitoring is active."""
        return self._thread is not None and self._thread.is_alive()

    def get_memory_info(self) -> Dict[str, float]:
        """Returns current memory usage including all child processes.

        Returns:
            {'rss_mb': float, 'percent': float, 'main_rss_mb': float, 'children_rss_mb': float}
        """
        try:
            main_mem = self._process.memory_info()
            main_rss = main_mem.rss
            main_percent = self._process.memory_percent()
            children_rss = 0.0

            # Recursively get memory of all child processes
            for child in self._process.children(recursive=True):
                try:
                    children_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            total_rss = main_rss + children_rss
            return {
                "rss_mb": total_rss / (1024 * 1024),
                "percent": main_percent,  # System percent for main process only
                "main_rss_mb": main_rss / (1024 * 1024),
                "children_rss_mb": children_rss / (1024 * 1024),
            }
        except psutil.NoSuchProcess:
            return {
                "rss_mb": 0.0,
                "percent": 0.0,
                "main_rss_mb": 0.0,
                "children_rss_mb": 0.0,
            }

    def start(self) -> "ProcessMemoryMonitor":
        """Starts monitoring in a daemon thread."""
        if self.is_running:
            return self

        self._stop_event.clear()
        self._peak_rss_mb = 0.0

        # Initial sample
        mem_info = self.get_memory_info()
        self._peak_rss_mb = mem_info["rss_mb"]

        if self._log_file:
            os.makedirs(os.path.dirname(self._log_file) or ".", exist_ok=True)
            self._file_handle = open(self._log_file, "a", encoding="utf-8")

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[MemoryMonitor] Started (interval: {self._interval}s)")
        return self

    def stop(self) -> float:
        """Stops monitoring and returns peak RSS in MB."""
        if not self.is_running:
            return self._peak_rss_mb

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._interval + 1)
            self._thread = None

        # Final sample to ensure we capture peak memory
        mem_info = self.get_memory_info()
        self._peak_rss_mb = max(self._peak_rss_mb, mem_info["rss_mb"])

        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

        print(f"[MemoryMonitor] Stopped. Peak RSS: {self._peak_rss_mb:.2f} MB")
        return self._peak_rss_mb

    def _monitor_loop(self):
        """Internal monitoring loop."""
        while True:
            mem_info = self.get_memory_info()
            self._peak_rss_mb = max(self._peak_rss_mb, mem_info["rss_mb"])

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            # Show breakdown: main + children
            msg = (
                f"{timestamp} - Total RSS: {mem_info['rss_mb']:.2f} MB "
                f"(main: {mem_info['main_rss_mb']:.2f} MB, "
                f"children: {mem_info['children_rss_mb']:.2f} MB), "
                f"System: {mem_info['percent']:.2f}%"
            )

            if self._file_handle:
                self._file_handle.write(msg + "\n")
                self._file_handle.flush()
            elif not self._quiet:
                print(msg)

            if self._callback:
                try:
                    self._callback(mem_info)
                except Exception:
                    pass

            if self._stop_event.wait(self._interval):
                break

    def __enter__(self) -> "ProcessMemoryMonitor":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


@contextmanager
def memory_monitor(
    interval: float = 2.0,
    log_file: Optional[str] = None,
):
    """Context manager for memory monitoring."""
    monitor = ProcessMemoryMonitor(interval, log_file)
    try:
        monitor.start()
        yield monitor
    finally:
        monitor.stop()
