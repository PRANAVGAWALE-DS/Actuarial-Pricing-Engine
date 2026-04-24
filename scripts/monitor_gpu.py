#!/usr/bin/env python
"""
Real-time GPU monitoring and usage tracking.

Displays GPU memory usage, utilization, temperature, and other metrics
using PyTorch and system APIs.
"""

import argparse
import logging
import os
import time
import warnings
from datetime import datetime

import psutil

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor GPU usage in real-time."""

    def __init__(self, interval: float = 1.0, max_duration: float | None = None):
        """
        Initialize GPU monitor.

        Args:
            interval: Update interval in seconds
            max_duration: Maximum monitoring duration in seconds (None for infinite)
        """
        self.interval = interval
        self.max_duration = max_duration
        self.torch_available = False
        self.tf_available = False
        self.running = False
        self.start_time = None
        self._check_frameworks()

    def _check_frameworks(self) -> None:
        """Check available deep learning frameworks."""
        try:
            import torch

            self.torch_available = True
            self.torch = torch
        except ImportError:
            logger.warning("PyTorch not available")

        try:
            import tensorflow as tf

            self.tf_available = True
            self.tf = tf
        except ImportError:
            logger.warning("TensorFlow not available")

    def _get_torch_gpu_stats(self) -> dict[str, any]:
        """Get GPU statistics from PyTorch."""
        stats = {}

        if not self.torch_available or not self.torch.cuda.is_available():
            return stats

        try:
            num_gpus = self.torch.cuda.device_count()

            for i in range(num_gpus):
                props = self.torch.cuda.get_device_properties(i)
                allocated = self.torch.cuda.memory_allocated(i) / 1024**3
                reserved = self.torch.cuda.memory_reserved(i) / 1024**3
                total = props.total_memory / 1024**3

                stats[f"gpu_{i}"] = {
                    "name": props.name,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "total_gb": total,
                    "utilization_percent": ((allocated / total * 100) if total > 0 else 0),
                    "free_gb": total - allocated,
                }
        except Exception as e:
            logger.debug(f"Error getting PyTorch GPU stats: {e}")

        return stats

    def _get_torch_cuda_info(self) -> dict[str, any]:
        """Get CUDA information from PyTorch."""
        info = {}

        if not self.torch_available:
            return info

        try:
            info["cuda_available"] = self.torch.cuda.is_available()
            info["cuda_version"] = self.torch.version.cuda
            info["pytorch_version"] = self.torch.__version__
            info["cudnn_version"] = self.torch.backends.cudnn.version()
        except Exception as e:
            logger.debug(f"Error getting CUDA info: {e}")

        return info

    def _get_cpu_stats(self) -> dict[str, any]:
        """Get CPU statistics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / 1024**3,
                "memory_total_gb": memory.total / 1024**3,
                "memory_available_gb": memory.available / 1024**3,
            }
        except Exception as e:
            logger.debug(f"Error getting CPU stats: {e}")
            return {}

    def _print_header(self) -> None:
        """Print monitoring header."""
        print("\n" + "=" * 80)
        print("GPU MONITORING - Press Ctrl+C to stop")
        print("=" * 80)

    def _print_torch_stats(self, stats: dict) -> None:
        """Print PyTorch GPU statistics."""
        if not stats:
            return

        print("\nPyTorch GPU Status:")
        print("-" * 80)

        for gpu_name, gpu_stats in stats.items():
            if not isinstance(gpu_stats, dict):
                continue

            print(f"\n{gpu_name.upper()}: {gpu_stats.get('name', 'Unknown')}")
            allocated = gpu_stats.get("allocated_gb", 0)
            total = gpu_stats.get("total_gb", 0)
            utilization = gpu_stats.get("utilization_percent", 0)
            free = gpu_stats.get("free_gb", 0)

            # Progress bar
            bar_width = 40
            filled = int(bar_width * utilization / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            print(f"  Memory: [{bar}] {utilization:5.1f}%")
            print(f"    {allocated:6.2f}GB / {total:6.2f}GB allocated | {free:6.2f}GB free")

    def _print_cpu_stats(self, stats: dict) -> None:
        """Print CPU statistics."""
        if not stats:
            return

        print("\nSystem Resources:")
        print("-" * 80)

        cpu_percent = stats.get("cpu_percent", 0)
        mem_percent = stats.get("memory_percent", 0)
        mem_used = stats.get("memory_used_gb", 0)
        mem_total = stats.get("memory_total_gb", 0)

        # Progress bars
        bar_width = 30
        cpu_filled = int(bar_width * cpu_percent / 100)
        mem_filled = int(bar_width * mem_percent / 100)

        cpu_bar = "█" * cpu_filled + "░" * (bar_width - cpu_filled)
        mem_bar = "█" * mem_filled + "░" * (bar_width - mem_filled)

        print(f"CPU:    [{cpu_bar}] {cpu_percent:5.1f}%")
        print(f"Memory: [{mem_bar}] {mem_percent:5.1f}% ({mem_used:.2f}GB / {mem_total:.2f}GB)")

    def _print_elapsed_time(self) -> None:
        """Print elapsed time."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            print(f"\nMonitoring time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def _check_duration(self) -> bool:
        """Check if max duration has been exceeded."""
        if self.max_duration is None:
            return False

        elapsed = time.time() - self.start_time
        return elapsed >= self.max_duration

    def monitor(self) -> None:
        """Start monitoring GPU in real-time."""
        self.running = True
        self.start_time = time.time()
        self._print_header()

        try:
            iteration = 0
            while self.running:
                # Fetch statistics
                gpu_stats = self._get_torch_gpu_stats()
                cpu_stats = self._get_cpu_stats()

                # Clear screen and print (simple approach for cross-platform compatibility)
                os.system("cls" if os.name == "nt" else "clear")

                # Print header with timestamp
                self._print_header()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\nTimestamp: {timestamp}")
                print(f"Iteration: {iteration}")

                # Print statistics
                if gpu_stats:
                    self._print_torch_stats(gpu_stats)
                else:
                    print("\nNo GPU information available")

                self._print_cpu_stats(cpu_stats)
                self._print_elapsed_time()

                # Check duration limit
                if self._check_duration():
                    logger.info(f"Monitoring duration limit ({self.max_duration}s) reached")
                    break

                iteration += 1
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
        finally:
            self.running = False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor GPU and system resource usage")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Update interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Maximum monitoring duration in seconds (default: infinite)",
    )

    args = parser.parse_args()

    # Print system information
    logger.info("GPU Monitoring Starting...")
    logger.info("=" * 60)

    try:
        import torch

        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
    except ImportError:
        logger.info("PyTorch: Not installed")

    logger.info("=" * 60)
    logger.info("")

    # Start monitoring
    monitor = GPUMonitor(interval=args.interval, max_duration=args.duration)
    monitor.monitor()


if __name__ == "__main__":
    main()
