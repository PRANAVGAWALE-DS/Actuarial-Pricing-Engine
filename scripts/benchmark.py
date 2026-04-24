#!/usr/bin/env python
"""
Benchmark GPU vs CPU performance for model training and inference.

Measures execution time and resource utilization for training and inference
operations on both GPU and CPU devices.
"""

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run performance benchmarks for GPU vs CPU."""

    def __init__(self):
        """Initialize benchmark runner."""
        self.results = {}
        self.torch_available = False
        self.tf_available = False
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

    def benchmark_torch(self, device: str = "cpu") -> dict[str, float]:
        """Benchmark PyTorch operations."""
        if not self.torch_available:
            logger.warning("PyTorch not available, skipping PyTorch benchmark")
            return {}

        logger.info(f"Running PyTorch benchmark on {device}...")
        results = {}

        try:
            torch_device = self.torch.device(device)

            # Matrix multiplication benchmark
            matrix_size = 4096
            iterations = 10

            # Warmup
            a = self.torch.randn(matrix_size, matrix_size, device=torch_device)
            b = self.torch.randn(matrix_size, matrix_size, device=torch_device)
            _ = self.torch.matmul(a, b)

            if device == "cuda":
                self.torch.cuda.synchronize()

            # Actual benchmark
            start_time = time.time()
            for _ in range(iterations):
                _ = self.torch.matmul(a, b)
                if device == "cuda":
                    self.torch.cuda.synchronize()
            elapsed = time.time() - start_time

            results["matmul_time"] = elapsed / iterations
            results["matmul_throughput"] = iterations / elapsed

            logger.info(
                f"  Matrix multiplication: {results['matmul_time']*1000:.2f}ms per operation"
            )

            # Convolution benchmark
            try:
                conv_iterations = 5
                batch_size = 32
                in_channels = 3
                out_channels = 64
                kernel_size = 3

                input_tensor = self.torch.randn(
                    batch_size, in_channels, 256, 256, device=torch_device
                )
                conv_layer = self.torch.nn.Conv2d(in_channels, out_channels, kernel_size).to(
                    torch_device
                )

                # Warmup
                _ = conv_layer(input_tensor)
                if device == "cuda":
                    self.torch.cuda.synchronize()

                # Benchmark
                start_time = time.time()
                for _ in range(conv_iterations):
                    _ = conv_layer(input_tensor)
                    if device == "cuda":
                        self.torch.cuda.synchronize()
                elapsed = time.time() - start_time

                results["conv2d_time"] = elapsed / conv_iterations
                results["conv2d_throughput"] = conv_iterations / elapsed

                logger.info(f"  Conv2D: {results['conv2d_time']*1000:.2f}ms per operation")
            except Exception as e:
                logger.warning(f"Convolution benchmark failed: {e}")

        except Exception as e:
            logger.error(f"PyTorch benchmark failed: {e}")

        return results

    def benchmark_tensorflow(self, device: str = "cpu") -> dict[str, float]:
        """Benchmark TensorFlow operations."""
        if not self.tf_available:
            logger.warning("TensorFlow not available, skipping TensorFlow benchmark")
            return {}

        logger.info(f"Running TensorFlow benchmark on {device}...")
        results = {}

        try:
            # Use GPU if available and requested
            if device == "gpu":
                gpus = self.tf.config.list_physical_devices("GPU")
                if not gpus:
                    logger.warning("No GPU detected for TensorFlow")
                    return {}

            with self.tf.device(f"/{device.upper()}:0" if device == "gpu" else "/CPU:0"):
                # Matrix multiplication benchmark
                matrix_size = 4096
                iterations = 10

                a = self.tf.random.normal((matrix_size, matrix_size))
                b = self.tf.random.normal((matrix_size, matrix_size))

                # Warmup
                _ = self.tf.matmul(a, b)

                # Benchmark
                start_time = time.time()
                for _ in range(iterations):
                    _ = self.tf.matmul(a, b)
                elapsed = time.time() - start_time

                results["matmul_time"] = elapsed / iterations
                results["matmul_throughput"] = iterations / elapsed

                logger.info(
                    f"  Matrix multiplication: {results['matmul_time']*1000:.2f}ms per operation"
                )

        except Exception as e:
            logger.error(f"TensorFlow benchmark failed: {e}")

        return results

    def benchmark_numpy(self) -> dict[str, float]:
        """Benchmark NumPy operations (CPU only)."""
        logger.info("Running NumPy benchmark...")
        results = {}

        try:
            # Matrix multiplication benchmark
            matrix_size = 4096
            iterations = 10

            a = np.random.randn(matrix_size, matrix_size).astype(np.float32)
            b = np.random.randn(matrix_size, matrix_size).astype(np.float32)

            # Warmup
            _ = np.matmul(a, b)

            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                _ = np.matmul(a, b)
            elapsed = time.time() - start_time

            results["matmul_time"] = elapsed / iterations
            results["matmul_throughput"] = iterations / elapsed

            logger.info(
                f"  Matrix multiplication: {results['matmul_time']*1000:.2f}ms per operation"
            )

        except Exception as e:
            logger.error(f"NumPy benchmark failed: {e}")

        return results

    def run_all_benchmarks(self, compare_devices: bool = False) -> None:
        """Run all available benchmarks."""
        logger.info("=" * 60)
        logger.info("Starting Benchmark Suite")
        logger.info("=" * 60)

        # NumPy baseline (CPU)
        self.results["numpy"] = self.benchmark_numpy()

        # PyTorch benchmarks
        if self.torch_available:
            self.results["pytorch_cpu"] = self.benchmark_torch("cpu")
            if compare_devices and self.torch.cuda.is_available():
                self.results["pytorch_gpu"] = self.benchmark_torch("cuda")

        # TensorFlow benchmarks
        if self.tf_available:
            self.results["tensorflow_cpu"] = self.benchmark_tensorflow("cpu")
            if compare_devices:
                self.results["tensorflow_gpu"] = self.benchmark_tensorflow("gpu")

    def print_results(self) -> None:
        """Print benchmark results."""
        logger.info("=" * 60)
        logger.info("Benchmark Results")
        logger.info("=" * 60)

        for framework, metrics in self.results.items():
            if metrics:
                logger.info(f"\n{framework}:")
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")

    def save_results(self, output_path: str = "reports/benchmark_results.json") -> None:
        """Save benchmark results to JSON file."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")

    def print_summary(self) -> None:
        """Print performance summary and speedup analysis."""
        logger.info("\n" + "=" * 60)
        logger.info("Performance Summary")
        logger.info("=" * 60)

        if "pytorch_cpu" in self.results and "pytorch_gpu" in self.results:
            cpu_time = self.results["pytorch_cpu"].get("matmul_time", 0)
            gpu_time = self.results["pytorch_gpu"].get("matmul_time", 0)
            if cpu_time > 0 and gpu_time > 0:
                speedup = cpu_time / gpu_time
                logger.info(f"\nPyTorch GPU Speedup: {speedup:.2f}x")

        if "tensorflow_cpu" in self.results and "tensorflow_gpu" in self.results:
            cpu_time = self.results["tensorflow_cpu"].get("matmul_time", 0)
            gpu_time = self.results["tensorflow_gpu"].get("matmul_time", 0)
            if cpu_time > 0 and gpu_time > 0:
                speedup = cpu_time / gpu_time
                logger.info(f"TensorFlow GPU Speedup: {speedup:.2f}x")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark GPU vs CPU performance")
    parser.add_argument(
        "--compare-devices", action="store_true", help="Compare GPU and CPU performance"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/benchmark_results.json",
        help="Output file for benchmark results",
    )

    args = parser.parse_args()

    # Check environment
    logger.info("System Information:")
    logger.info(f"  OS: {os.name}")

    try:
        import torch

        logger.info(f"  PyTorch: {torch.__version__}")
        logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.info("  PyTorch: Not installed")

    try:
        import tensorflow as tf

        logger.info(f"  TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        logger.info(f"  GPU Devices: {len(gpus)}")
    except ImportError:
        logger.info("  TensorFlow: Not installed")

    logger.info("")

    # Run benchmarks
    runner = BenchmarkRunner()
    runner.run_all_benchmarks(compare_devices=args.compare_devices)
    runner.print_results()
    runner.print_summary()
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
