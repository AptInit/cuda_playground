#!/usr/bin/env python3
"""
CUDA GEMM Kernel Auto-Tuner

This script explores the parameter space of BlkCfg template parameters
for the kernel_grid_v3() GEMM kernel to find optimal configurations.

Usage:
    python autotune.py                    # Full parameter space exploration
    python autotune.py --dry-run          # Print configs without building
    python autotune.py --max-configs 5    # Test only 5 configurations
    python autotune.py --reset            # Clear checkpoint and start fresh
"""

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime

# Optional tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback if tqdm is not installed."""
        total = kwargs.get('total', None)
        desc = kwargs.get('desc', '')
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc}: {i+1}/{total}", end='', flush=True)
            yield item
        print()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BlkCfg:
    """Kernel configuration parameters."""
    grd: int    # Subgrid size for L2 cache reuse (power of 4)
    blkN: int   # Block dimension N (blockDim.x)
    blkM: int   # Block dimension M (blockDim.y)
    blkK: int   # Block dimension K (tile in K)
    thN: int    # Thread tile size in N
    thM: int    # Thread tile size in M

    def to_cpp_type(self) -> str:
        """Generate C++ BlkCfg type string."""
        return f"BlkCfg<{self.grd},{self.blkN},{self.blkM},{self.blkK},{self.thN},{self.thM}>"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> 'BlkCfg':
        return BlkCfg(**d)

    def __hash__(self):
        return hash((self.grd, self.blkN, self.blkM, self.blkK, self.thN, self.thM))


@dataclass
class BuildResult:
    """Result of a build attempt."""
    success: bool
    elapsed_seconds: float
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


@dataclass
class BenchResult:
    """Result of a benchmark run."""
    success: bool
    time_ms: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


@dataclass
class ConfigResult:
    """Complete result for a single configuration."""
    params: dict
    status: str  # "success", "build_failed", "run_failed"
    time_ms: Optional[float] = None
    build_time_s: Optional[float] = None
    error_log: Optional[str] = None
    timestamp: str = ""


# =============================================================================
# Parameter Space
# =============================================================================

class ParameterSpace:
    """Generates valid BlkCfg configurations."""

    # Default parameter ranges
    DEFAULT_GRD = [4, 16, 64]
    DEFAULT_BLK_N = [8, 16, 32]
    DEFAULT_BLK_M = [8, 16, 32]
    DEFAULT_BLK_K = [16, 32, 64]
    DEFAULT_TH_N = [2, 4, 8]
    DEFAULT_TH_M = [2, 4, 8]

    MAX_THREADS_PER_BLOCK = 1024

    def __init__(
        self,
        grd_values: list[int] = None,
        blkN_values: list[int] = None,
        blkM_values: list[int] = None,
        blkK_values: list[int] = None,
        thN_values: list[int] = None,
        thM_values: list[int] = None,
    ):
        self.grd_values = grd_values or self.DEFAULT_GRD
        self.blkN_values = blkN_values or self.DEFAULT_BLK_N
        self.blkM_values = blkM_values or self.DEFAULT_BLK_M
        self.blkK_values = blkK_values or self.DEFAULT_BLK_K
        self.thN_values = thN_values or self.DEFAULT_TH_N
        self.thM_values = thM_values or self.DEFAULT_TH_M

    def is_valid(self, cfg: BlkCfg) -> bool:
        """Check if a configuration satisfies all constraints."""
        # Constraint: blkN * blkM <= MAX_THREADS_PER_BLOCK
        if cfg.blkN * cfg.blkM > self.MAX_THREADS_PER_BLOCK:
            return False

        # Constraint: blkK % blkN == 0 (from demo3.cu line 39)
        if cfg.blkK % cfg.blkN != 0:
            return False

        # Constraint: blkK % blkM == 0 (from demo3.cu line 90)
        if cfg.blkK % cfg.blkM != 0:
            return False

        # Constraint: blkK % thM == 0 (for compute loop, line 91)
        if cfg.blkK % cfg.thM != 0:
            return False

        return True

    def generate(self) -> Iterator[BlkCfg]:
        """Generate all valid configurations."""
        for grd in self.grd_values:
            for blkN in self.blkN_values:
                for blkM in self.blkM_values:
                    for blkK in self.blkK_values:
                        for thN in self.thN_values:
                            for thM in self.thM_values:
                                cfg = BlkCfg(grd, blkN, blkM, blkK, thN, thM)
                                if self.is_valid(cfg):
                                    yield cfg

    def count_valid(self) -> int:
        """Count total valid configurations."""
        return sum(1 for _ in self.generate())


# =============================================================================
# Build Manager
# =============================================================================

class BuildManager:
    """Handles CMake configuration and building."""

    def __init__(self, source_dir: Path, build_dir: Path):
        self.source_dir = source_dir.resolve()
        self.build_dir = build_dir.resolve()
        self.system = platform.system()
        self._cmake_path = self._find_cmake()
        self._generator = self._detect_generator()
        self._configured = False

    def _find_cmake(self) -> str:
        """Find CMake executable, preferring system cmake for compatibility."""
        # Prefer system cmake (more stable, known compatibility)
        cmake_path = shutil.which("cmake")
        if cmake_path:
            return cmake_path

        # Fall back to python -m cmake
        try:
            result = subprocess.run(
                [sys.executable, "-m", "cmake", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return f"{sys.executable} -m cmake"
        except Exception:
            pass

        raise RuntimeError(
            "CMake not found. Install with: pip install cmake"
        )

    def _detect_generator(self) -> str:
        """Detect the best CMake generator for this platform."""
        if self.system == "Windows":
            # For CUDA on Windows, Visual Studio generator is preferred because:
            # 1. It handles MSVC environment setup automatically
            # 2. Ninja requires running from a VS Developer Command Prompt
            # Check for VS 2022 first, then 2019
            for vs_version in ["Visual Studio 17 2022", "Visual Studio 16 2019"]:
                # Test if this generator works by checking for vswhere or just try it
                # For simplicity, assume VS 2022 is installed if on Windows
                return vs_version
        else:
            # Linux/macOS: prefer Ninja, fall back to Make
            if shutil.which("ninja"):
                return "Ninja"
            return "Unix Makefiles"

    def _run_cmake(self, args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
        """Run cmake with the given arguments."""
        if self._cmake_path.startswith(sys.executable):
            # Using python -m cmake
            cmd = [sys.executable, "-m", "cmake"] + args
        else:
            cmd = [self._cmake_path] + args

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.build_dir if self.build_dir.exists() else self.source_dir
        )

    def configure(self, bench_config: Optional[str] = None) -> BuildResult:
        """Configure the CMake project."""
        import time
        start = time.time()

        # Create build directory
        self.build_dir.mkdir(parents=True, exist_ok=True)

        args = [
            "-S", str(self.source_dir),
            "-B", str(self.build_dir),
            "-G", self._generator,
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        if bench_config:
            args.append(f"-DBENCH_CONFIG={bench_config}")

        try:
            result = self._run_cmake(args)
            elapsed = time.time() - start

            if result.returncode != 0:
                return BuildResult(
                    success=False,
                    elapsed_seconds=elapsed,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error_message=f"CMake configure failed with code {result.returncode}"
                )

            self._configured = True
            return BuildResult(
                success=True,
                elapsed_seconds=elapsed,
                stdout=result.stdout,
                stderr=result.stderr
            )

        except subprocess.TimeoutExpired as e:
            return BuildResult(
                success=False,
                elapsed_seconds=time.time() - start,
                error_message=f"CMake configure timed out after {e.timeout}s"
            )
        except Exception as e:
            return BuildResult(
                success=False,
                elapsed_seconds=time.time() - start,
                error_message=f"CMake configure error: {e}"
            )

    def build(self, timeout: int = 300) -> BuildResult:
        """Build the project."""
        import time
        start = time.time()

        args = ["--build", str(self.build_dir), "--config", "Release"]

        # Use parallel builds
        args.extend(["--parallel"])

        try:
            result = self._run_cmake(args, timeout=timeout)
            elapsed = time.time() - start

            if result.returncode != 0:
                return BuildResult(
                    success=False,
                    elapsed_seconds=elapsed,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error_message=f"Build failed with code {result.returncode}"
                )

            return BuildResult(
                success=True,
                elapsed_seconds=elapsed,
                stdout=result.stdout,
                stderr=result.stderr
            )

        except subprocess.TimeoutExpired as e:
            return BuildResult(
                success=False,
                elapsed_seconds=time.time() - start,
                error_message=f"Build timed out after {e.timeout}s"
            )
        except Exception as e:
            return BuildResult(
                success=False,
                elapsed_seconds=time.time() - start,
                error_message=f"Build error: {e}"
            )

    def configure_and_build(self, cfg: BlkCfg) -> BuildResult:
        """Configure and build with a specific BlkCfg."""
        import time
        total_start = time.time()

        # Always reconfigure to change BENCH_CONFIG
        config_result = self.configure(cfg.to_cpp_type())
        if not config_result.success:
            return config_result

        build_result = self.build()
        build_result.elapsed_seconds = time.time() - total_start
        return build_result

    def get_executable_path(self) -> Path:
        """Get path to the built executable."""
        if self.system == "Windows":
            # Visual Studio puts binaries in Release/Debug subdirs
            if self._generator.startswith("Visual Studio"):
                return self.build_dir / "Release" / "cuda_playground.exe"
            else:
                return self.build_dir / "cuda_playground.exe"
        else:
            return self.build_dir / "cuda_playground"


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Runs the benchmark executable and parses results."""

    # Pattern to match "AVG: X.XX ms"
    TIME_PATTERN = re.compile(r"AVG:\s*([\d.]+)\s*ms")

    def __init__(self, executable: Path):
        self.executable = executable

    def run(self, timeout: int = 120) -> BenchResult:
        """Run the benchmark and extract timing."""
        if not self.executable.exists():
            return BenchResult(
                success=False,
                error_message=f"Executable not found: {self.executable}"
            )

        try:
            result = subprocess.run(
                [str(self.executable)],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                return BenchResult(
                    success=False,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error_message=f"Benchmark failed with code {result.returncode}"
                )

            # Parse timing from stdout
            match = self.TIME_PATTERN.search(result.stdout)
            if match:
                time_ms = float(match.group(1))
                return BenchResult(
                    success=True,
                    time_ms=time_ms,
                    stdout=result.stdout,
                    stderr=result.stderr
                )
            else:
                return BenchResult(
                    success=False,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error_message="Could not parse timing from output"
                )

        except subprocess.TimeoutExpired:
            return BenchResult(
                success=False,
                error_message=f"Benchmark timed out after {timeout}s"
            )
        except Exception as e:
            return BenchResult(
                success=False,
                error_message=f"Benchmark error: {e}"
            )


# =============================================================================
# Checkpoint Manager
# =============================================================================

class Checkpoint:
    """Manages saving/loading auto-tuning progress."""

    VERSION = 1

    def __init__(self, path: Path):
        self.path = path
        self.data = {
            "version": self.VERSION,
            "platform": platform.system(),
            "started_at": datetime.now().isoformat(),
            "configs_tested": []
        }
        self._tested_set: set[BlkCfg] = set()
        self._load()

    def _load(self):
        """Load checkpoint from disk if it exists."""
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    self.data = json.load(f)
                # Rebuild the tested set
                for result in self.data.get("configs_tested", []):
                    cfg = BlkCfg.from_dict(result["params"])
                    self._tested_set.add(cfg)
                print(f"Loaded checkpoint with {len(self._tested_set)} completed configs")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")

    def save(self):
        """Save checkpoint to disk."""
        self.data["updated_at"] = datetime.now().isoformat()
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def is_tested(self, cfg: BlkCfg) -> bool:
        """Check if a configuration has already been tested."""
        return cfg in self._tested_set

    def add_result(self, result: ConfigResult):
        """Add a result and save checkpoint."""
        self.data["configs_tested"].append(asdict(result))
        cfg = BlkCfg.from_dict(result.params)
        self._tested_set.add(cfg)
        self.save()

    def get_results(self) -> list[ConfigResult]:
        """Get all results."""
        return [
            ConfigResult(**r) for r in self.data.get("configs_tested", [])
        ]

    def get_best_results(self, n: int = 10) -> list[ConfigResult]:
        """Get top N results sorted by time."""
        results = [r for r in self.get_results() if r.status == "success" and r.time_ms]
        results.sort(key=lambda r: r.time_ms)
        return results[:n]

    def reset(self):
        """Clear checkpoint data."""
        if self.path.exists():
            self.path.unlink()
        self.data = {
            "version": self.VERSION,
            "platform": platform.system(),
            "started_at": datetime.now().isoformat(),
            "configs_tested": []
        }
        self._tested_set.clear()


# =============================================================================
# Auto-Tuner
# =============================================================================

class AutoTuner:
    """Main auto-tuning orchestrator."""

    def __init__(
        self,
        source_dir: Path,
        build_dir: Path,
        checkpoint_path: Path,
        dry_run: bool = False,
        max_configs: Optional[int] = None,
        verbose: bool = False
    ):
        self.source_dir = source_dir
        self.build_dir = build_dir
        self.dry_run = dry_run
        self.max_configs = max_configs
        self.verbose = verbose

        self.param_space = ParameterSpace()
        self.checkpoint = Checkpoint(checkpoint_path)
        self.build_manager = BuildManager(source_dir, build_dir)

    def run(self):
        """Run the auto-tuning process."""
        # Generate all valid configs
        all_configs = list(self.param_space.generate())
        total = len(all_configs)
        print(f"Total valid configurations: {total}")

        # Filter out already tested
        configs_to_test = [c for c in all_configs if not self.checkpoint.is_tested(c)]
        remaining = len(configs_to_test)
        print(f"Remaining to test: {remaining}")

        if self.max_configs:
            configs_to_test = configs_to_test[:self.max_configs]
            print(f"Limited to: {len(configs_to_test)}")

        if self.dry_run:
            print("\n=== DRY RUN: Configurations to test ===")
            for i, cfg in enumerate(configs_to_test[:20]):
                print(f"  {i+1}. {cfg.to_cpp_type()}")
            if len(configs_to_test) > 20:
                print(f"  ... and {len(configs_to_test) - 20} more")
            return

        # Get current best if resuming
        best_results = self.checkpoint.get_best_results(1)
        best_time = best_results[0].time_ms if best_results else float('inf')

        # Run tests
        success_count = 0
        fail_count = 0

        for cfg in tqdm(configs_to_test, desc="Auto-tuning", total=len(configs_to_test)):
            result = self._test_config(cfg)

            if result.status == "success":
                success_count += 1
                
                if result.time_ms < best_time:
                    best_time = result.time_ms
                    print(f"\n[NEW BEST] {result.time_ms:.3f} ms - {cfg.to_cpp_type()}")
                
                if self.verbose:
                    print(f"\n  {cfg.to_cpp_type()} -> {result.time_ms:.3f} ms")
            else:
                fail_count += 1
                if self.verbose:
                    print(f"\n  {cfg.to_cpp_type()} -> {result.status}")

        # Print summary
        self._print_summary(success_count, fail_count)

    def _test_config(self, cfg: BlkCfg) -> ConfigResult:
        """Test a single configuration."""
        timestamp = datetime.now().isoformat()

        # Build
        build_result = self.build_manager.configure_and_build(cfg)

        if not build_result.success:
            error_log = self._format_error_log(build_result)
            result = ConfigResult(
                params=cfg.to_dict(),
                status="build_failed",
                build_time_s=build_result.elapsed_seconds,
                error_log=error_log,
                timestamp=timestamp
            )
            self.checkpoint.add_result(result)
            return result

        # Run benchmark
        runner = BenchmarkRunner(self.build_manager.get_executable_path())
        bench_result = runner.run()

        if not bench_result.success:
            error_log = self._format_error_log(bench_result)
            result = ConfigResult(
                params=cfg.to_dict(),
                status="run_failed",
                build_time_s=build_result.elapsed_seconds,
                error_log=error_log,
                timestamp=timestamp
            )
            self.checkpoint.add_result(result)
            return result

        # Success
        result = ConfigResult(
            params=cfg.to_dict(),
            status="success",
            time_ms=bench_result.time_ms,
            build_time_s=build_result.elapsed_seconds,
            timestamp=timestamp
        )
        self.checkpoint.add_result(result)
        return result

    def _format_error_log(self, result) -> str:
        """Format error information for logging."""
        lines = []
        if result.error_message:
            lines.append(f"Error: {result.error_message}")
        if result.stdout:
            lines.append(f"--- stdout ---\n{result.stdout[-2000:]}")
        if result.stderr:
            lines.append(f"--- stderr ---\n{result.stderr[-2000:]}")
        return "\n".join(lines)

    def _print_summary(self, success_count: int, fail_count: int):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("AUTO-TUNING SUMMARY")
        print("=" * 60)
        print(f"Successful: {success_count}")
        print(f"Failed: {fail_count}")

        best = self.checkpoint.get_best_results(10)
        if best:
            print(f"\nTop {len(best)} configurations:")
            print("-" * 60)
            for i, r in enumerate(best):
                cfg = BlkCfg.from_dict(r.params)
                print(f"  {i+1}. {r.time_ms:7.3f} ms  {cfg.to_cpp_type()}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CUDA GEMM Kernel Auto-Tuner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without building"
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of configurations to test"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear checkpoint and start fresh"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Build directory (default: build_autotune)"
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    source_dir = script_dir
    build_dir = args.build_dir or (script_dir / "build_autotune")
    checkpoint_path = script_dir / "autotune_results.json"

    # Handle reset
    if args.reset:
        print("Resetting checkpoint...")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if build_dir.exists():
            shutil.rmtree(build_dir)
        print("Done.")
        if not args.dry_run and not args.max_configs:
            return

    # Run auto-tuner
    tuner = AutoTuner(
        source_dir=source_dir,
        build_dir=build_dir,
        checkpoint_path=checkpoint_path,
        dry_run=args.dry_run,
        max_configs=args.max_configs,
        verbose=args.verbose
    )
    tuner.run()


if __name__ == "__main__":
    main()
