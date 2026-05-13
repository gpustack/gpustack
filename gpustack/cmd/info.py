import argparse
import glob
import json
import os
import platform
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

from gpustack import __git_commit__, __version__


RUNTIME_ENV_NAMES = [
    "GPUSTACK_RUNTIME_DETECT",
    "GPUSTACK_RUNTIME_DETECT_NO_PCI_CHECK",
    "GPUSTACK_RUNTIME_DETECT_NO_TOOLKIT_CALL",
    "GPUSTACK_RUNTIME_DETECT_NO_HEALTH_CHECK",
    "GPUSTACK_RUNTIME_DETECT_PHYSICAL_INDEX_PRIORITY",
]

ACCELERATOR_DEVICE_PATTERNS = [
    "/dev/accel/*",
    "/dev/dri/card*",
    "/dev/dri/renderD*",
    "/dev/kfd",
    "/dev/*gpu*",
    "/dev/*npu*",
    "/dev/*xpu*",
    "/dev/*ppu*",
    "/dev/*smi*",
]


def setup_info_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "info",
        help="Print container-visible accelerator diagnostics.",
        description="Print container-visible runtime and accelerator diagnostics.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print diagnostics as JSON.",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed errors for failed probes.",
        default=False,
    )
    parser.set_defaults(func=run)


def get_package_version(distribution_name: str) -> Optional[str]:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None
    except Exception as e:
        return f"error reading metadata: {e}"


def run(args):
    report = collect_info(verbose=args.verbose)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(report, verbose=args.verbose)


def collect_info(verbose: bool = False) -> dict[str, Any]:
    return {
        "gpustack": collect_gpustack_info(),
        "system": collect_system_info(),
        "container": collect_container_info(),
        "environment": collect_accelerator_envs(),
        "runtime": collect_runtime_detection(verbose=verbose),
        "device_files": collect_accelerator_device_files(),
        "detector": collect_detector_info(verbose=verbose),
    }


def collect_gpustack_info() -> dict[str, Any]:
    return {
        "version": __version__,
        "git_commit": __git_commit__,
        "gpustack_runtime": get_package_version("gpustack-runtime"),
        "gpustack_runner": get_package_version("gpustack-runner"),
        "torch": get_package_version("torch"),
        "vllm": get_package_version("vllm"),
    }


def collect_system_info() -> dict[str, str]:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def collect_container_info() -> dict[str, Any]:
    cgroup = read_file("/proc/1/cgroup")
    container_runtime = "unknown"
    if os.path.exists("/.dockerenv"):
        container_runtime = "docker"
    elif cgroup:
        lowered = cgroup.lower()
        if "kubepods" in lowered:
            container_runtime = "kubernetes"
        elif "containerd" in lowered:
            container_runtime = "containerd"
        elif "docker" in lowered:
            container_runtime = "docker"

    return {
        "detected": container_runtime != "unknown",
        "runtime": container_runtime,
        "dockerenv": os.path.exists("/.dockerenv"),
        "cgroup": first_non_empty_line(cgroup),
    }


def collect_accelerator_envs() -> dict[str, Optional[str]]:
    names = set(RUNTIME_ENV_NAMES)
    for name in os.environ:
        if name.endswith("VISIBLE_DEVICES") or name in (
            "CUDA_DEVICE_ORDER",
            "GPU_DEVICE_ORDINAL",
        ):
            names.add(name)
    return {name: os.environ.get(name) for name in sorted(names)}


def collect_runtime_detection(verbose: bool = False) -> dict[str, Any]:
    try:
        from gpustack_runtime.detector import (
            available_backends,
            available_manufacturers,
            detect_backend,
            supported_manufacturers,
        )

        return {
            "ok": True,
            "available_manufacturers": [
                str(manufacturer) for manufacturer in available_manufacturers()
            ],
            "available_backends": [str(backend) for backend in available_backends()],
            "supported_manufacturers": [
                str(manufacturer) for manufacturer in supported_manufacturers()
            ],
            "detected_backends": normalize_detected_backends(
                detect_backend(fast=False)
            ),
        }
    except Exception as e:
        runtime = {"ok": False, "error": str(e)}
        if verbose:
            runtime["traceback"] = traceback.format_exc()
        return runtime


def collect_accelerator_device_files() -> dict[str, Any]:
    matches_by_pattern = {}
    all_matches = set()
    for pattern in ACCELERATOR_DEVICE_PATTERNS:
        matches = sorted(glob.glob(pattern))
        matches_by_pattern[pattern] = matches
        all_matches.update(matches)
    return {
        "patterns": matches_by_pattern,
        "matches": sorted(all_matches),
    }


def collect_detector_info(verbose: bool = False) -> dict[str, Any]:
    try:
        from gpustack.detectors.runtime.runtime import Runtime

        raw_devices = Runtime().gather_gpu_info()
        filtered_devices = filter_worker_usable_devices(raw_devices)
        return {
            "ok": True,
            "raw_count": len(raw_devices),
            "worker_usable_count": len(filtered_devices),
            "raw_devices": [format_gpu_device(device) for device in raw_devices],
            "worker_usable_devices": [
                format_gpu_device(device) for device in filtered_devices
            ],
        }
    except Exception as e:
        detector = {"ok": False, "error": str(e)}
        if verbose:
            detector["traceback"] = traceback.format_exc()
        return detector


def filter_worker_usable_devices(devices) -> list:
    return [
        device
        for device in devices
        if device.memory and device.memory.total and device.memory.total > 0
    ]


def format_gpu_device(device) -> dict[str, Any]:
    memory_total = getattr(getattr(device, "memory", None), "total", None)
    memory_used = getattr(getattr(device, "memory", None), "used", None)
    core_total = getattr(getattr(device, "core", None), "total", None)
    return {
        "vendor": device.vendor,
        "type": device.type,
        "index": device.index,
        "device_index": device.device_index,
        "device_chip_index": device.device_chip_index,
        "name": device.name,
        "uuid": device.uuid,
        "driver_version": device.driver_version,
        "runtime_version": device.runtime_version,
        "compute_capability": device.compute_capability,
        "arch_family": device.arch_family,
        "memory_total_mib": bytes_to_mib(memory_total),
        "memory_used_mib": bytes_to_mib(memory_used),
        "core_total": core_total,
    }


def print_human(report: dict[str, Any], verbose: bool = False):
    print("GPUStack Info")
    print("-" * 30)
    print("GPUStack:")
    print(f"  version:          {report['gpustack']['version']}")
    print(f"  git_commit:       {report['gpustack']['git_commit']}")
    print(
        f"  gpustack-runtime: {format_optional(report['gpustack']['gpustack_runtime'])}"
    )
    print(
        f"  gpustack-runner:  {format_optional(report['gpustack']['gpustack_runner'])}"
    )
    print(f"  torch:            {format_optional(report['gpustack']['torch'])}")
    print(f"  vllm:             {format_optional(report['gpustack']['vllm'])}")

    print("-" * 30)
    print("System:")
    print(
        f"  os:               {report['system']['os']} ({report['system']['machine']})"
    )
    print(f"  python:           {report['system']['python']}")
    print(f"  platform:         {report['system']['platform']}")

    print("-" * 30)
    print("Container:")
    print(f"  detected:         {yes_no(report['container']['detected'])}")
    print(f"  runtime:          {report['container']['runtime']}")
    print(f"  /.dockerenv:      {yes_no(report['container']['dockerenv'])}")
    if report["container"]["cgroup"]:
        print(f"  cgroup:           {report['container']['cgroup']}")

    print("-" * 30)
    print("Environment:")
    if report["environment"]:
        for name, value in report["environment"].items():
            print(f"  {name}: {format_optional(value)}")
    else:
        print("  no accelerator-related environment variables found")

    print("-" * 30)
    print("GPUStack Runtime:")
    runtime = report["runtime"]
    if runtime["ok"]:
        print(
            f"  available manufacturers: {format_list(runtime['available_manufacturers'])}"
        )
        print(
            f"  available backends:      {format_list(runtime['available_backends'])}"
        )
        print(
            f"  supported manufacturers: {format_list(runtime['supported_manufacturers'])}"
        )
        print(f"  detected backends:       {format_list(runtime['detected_backends'])}")
    else:
        print(f"  error: {runtime['error']}")
        if verbose and runtime.get("traceback"):
            print(runtime["traceback"])

    print("-" * 30)
    print("Accelerator Device Files:")
    if report["device_files"]["matches"]:
        for path in report["device_files"]["matches"]:
            print(f"  {path}")
    else:
        print("  no accelerator-like device files found")

    print("-" * 30)
    print("GPUStack Detector:")
    detector = report["detector"]
    if not detector["ok"]:
        print(f"  error: {detector['error']}")
        if verbose and detector.get("traceback"):
            print(detector["traceback"])
        print("-" * 30)
        return

    print(f"  raw devices:             {detector['raw_count']}")
    print(f"  worker usable devices:   {detector['worker_usable_count']}")
    devices = detector["worker_usable_devices"] or detector["raw_devices"]
    for device in devices:
        memory = format_optional(device["memory_total_mib"], suffix=" MiB")
        print(
            "  - "
            f"vendor={device['vendor']} type={device['type']} "
            f"index={device['index']} name={device['name']} memory={memory}"
        )
        if device.get("driver_version") or device.get("runtime_version"):
            print(
                "    "
                f"driver={format_optional(device['driver_version'])} "
                f"runtime={format_optional(device['runtime_version'])}"
            )
    print("-" * 30)


def read_file(path: str) -> str:
    try:
        with open(path, "r") as file:
            return file.read()
    except Exception:
        return ""


def first_non_empty_line(value: str) -> str:
    for line in value.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def bytes_to_mib(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return value // (1024 * 1024)


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def format_optional(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return "<unset>"
    return f"{value}{suffix}"


def format_list(values: list[str]) -> str:
    if not values:
        return "<none>"
    return ", ".join(values)


def normalize_detected_backends(backends: str | list[str]) -> list[str]:
    if isinstance(backends, str):
        return [backends] if backends else []
    return [str(backend) for backend in backends if backend]
