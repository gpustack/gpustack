import argparse
import platform
import sys

from gpustack import __version__, __git_commit__


def setup_info_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "info",
        help="Print environment information.",
        description="Print system and environment information useful for debugging.",
    )
    parser.set_defaults(func=run)


def run(args):
    print("GPUStack Environment Information")
    print("-" * 30)
    print(f"GPUStack Version: {__version__} ({__git_commit__})")
    print(
        f"OS:               {platform.system()} {platform.release()} ({platform.machine()})"
    )
    print(f"Python Version:   {sys.version.split()[0]}")
    print(f"Platform:         {platform.platform()}")

    # Check for some common dependencies
    try:
        import torch

        print(f"PyTorch Version:  {torch.__version__}")
    except ImportError:
        print("PyTorch Version:  Not installed or not found in current environment")

    try:
        import vllm

        print(f"vLLM Version:     {vllm.__version__}")
    except ImportError:
        pass

    try:
        from gpustack_runtime import __version__ as runtime_version

        print(f"Runtime Version:  {runtime_version}")
    except ImportError:
        pass

    print("-" * 30)
    print("Accelerators (GPUs/NPUs):")

    try:
        from gpustack.detectors.runtime.runtime import Runtime

        detector = Runtime()
        devices = detector.gather_gpu_info()

        if not devices:
            print("  No supported accelerators detected.")
        else:
            for dev in devices:
                print(f"  - Vendor: {dev.vendor}, Name: {dev.name}, Index: {dev.index}")
                if dev.memory:
                    mem_mb = dev.memory.total // (1024 * 1024)
                    print(f"    Memory: {mem_mb} MiB")
                if dev.core and dev.core.total > 0:
                    print(f"    Cores:  {dev.core.total}")
    except Exception as e:
        print(f"  Error detecting accelerators: {e}")

    print("-" * 30)
