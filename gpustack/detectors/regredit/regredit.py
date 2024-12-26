from typing import Dict, Any
import logging
import subprocess
import json

from gpustack.detectors.base import GPUDetector
from gpustack.schemas.workers import (
    GPUDevicesInfo,
    GPUDeviceInfo,
    VendorEnum,
    GPUCoreInfo,
    MemoryInfo,
)
from gpustack.utils import platform

logger = logging.getLogger(__name__)


class Regredit(GPUDetector):
    def is_available(self) -> bool:
        return platform.system() == "windows"

    def gather_gpu_info(self) -> GPUDevicesInfo:
        devices = self.get_gpu_from_regredit()
        if not devices:
            return []

        memory_usage = self.get_memory_usage_from_counter()
        gpu_usage = self.get_gpu_usage_from_counter()

        for i, _ in enumerate(devices):
            key = adapter_luid_to_string(int(devices[i].uuid))
            memory_used = memory_usage.get(key, 0)
            gpu_util = gpu_usage.get(key, 0)

            devices[i].memory.used = memory_used
            devices[i].core = GPUCoreInfo(
                total=0, utilization_rate=gpu_util  # Not available
            )

            if devices[i].memory.total > 0:
                devices[i].memory.utilization_rate = (
                    memory_used / devices[i].memory.total
                ) * 100

        return devices

    def get_memory_usage_from_counter(self) -> Dict[str, float]:
        counter = "\\GPU Adapter Memory(*)\\Dedicated Usage"
        return self.get_cooked_value_from_counter(counter)

    def get_gpu_usage_from_counter(self) -> Dict[str, float]:
        counter = "\\GPU Engine(*)\\Utilization Percentage"
        return self.get_cooked_value_from_counter(counter)

    def get_cooked_value_from_counter(self, counter: str) -> Dict[str, float]:
        command = (
            'powershell.exe -Command '
            '"Get-Counter -Counter \\"{}\\" | '
            'Select-Object -ExpandProperty CounterSamples | '
            'Select-Object InstanceName,CookedValue | '
            'ConvertTo-Json"'
        ).format(counter)

        # example output:
        # [
        #     {
        #         "InstanceName":  "luid_0x00000000_0x00010860_phys_0",
        #         "CookedValue":  0
        #     },
        #     {
        #         "InstanceName":  "luid_0x00000000_0x0050090f_phys_0",
        #         "CookedValue":  2630623232
        #     }
        # ]

        result = None
        try:
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            if result.returncode != 0:
                raise Exception(f"Unexpected return code: {result.returncode}")

            output = result.stdout
            if output == "" or output is None:
                raise Exception(f"Output is empty, return code: {result.returncode}")

        except Exception as e:
            error_message = f"Failed to execute {command}: {e}"
            if result:
                error_message += f", stdout: {result.stdout}, stderr: {result.stderr}"
            raise Exception(error_message)

        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            raise Exception(
                f"Failed to parse the output of {command}: {e}, output: {output}"
            )

        usage = {}
        for item in data:
            instance_name = item['InstanceName']
            cooked_value = item['CookedValue']
            usage[instance_name] = cooked_value

        return usage

    def get_gpu_from_regredit(self) -> GPUDevicesInfo:
        results = {}
        key_path = r"SOFTWARE\Microsoft\DirectX"
        get_registry_items(key_path, results, is_top_level=True)
        devices = []
        index = 0
        for _, subvalue in results.items():
            vendor_id = subvalue.get("VendorId", 0)
            vendor = vendor_from_vendor_id(vendor_id)
            if vendor == "Unknown":
                continue
            type = platform.device_type_from_vendor(vendor)

            luid = subvalue.get("AdapterLuid", "Unknown")
            name = subvalue.get("Description", "Unknown")
            memory_total = subvalue.get("DedicatedVideoMemory", 0)
            if memory_total == 0:
                continue

            device_info = GPUDeviceInfo(
                uuid=str(luid),
                index=index,
                name=name,
                vendor=vendor,
                memory=MemoryInfo(
                    total=memory_total,
                    is_unified_memory=False,
                ),
                type=type,
            )
            devices.append(device_info)
            index += 1

        return devices


def vendor_from_vendor_id(vendor_id: int) -> str:
    # https://devicehunt.com/all-pci-vendors
    vendor_map = {
        0x1002: VendorEnum.AMD.value,
        0x1022: VendorEnum.AMD.value,
        0x1DD8: VendorEnum.AMD.value,
        0x106B: VendorEnum.Apple.value,
        0x0955: VendorEnum.NVIDIA.value,
        0x10DE: VendorEnum.NVIDIA.value,
        0x12D2: VendorEnum.NVIDIA.value,
        0x1ED5: VendorEnum.MTHREADS.value,
    }
    vendor = vendor_map.get(vendor_id, "Unknown")
    return vendor


def get_registry_items(
    key_path: str, results: Dict[str, Any], is_top_level: bool = False
) -> None:
    import winreg

    try:
        try:
            registry_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
        except FileNotFoundError:
            logging.debug(f"Registry path not found: {key_path}")
            return

        num_subkeys, num_values, _ = winreg.QueryInfoKey(registry_key)

        current_dict = {}
        # iterate all subkeys
        for i in range(num_subkeys):
            subkey_name = winreg.EnumKey(registry_key, i)
            subkey_path = f"{key_path}\\{subkey_name}"
            subkey_dict = {}
            get_registry_items(subkey_path, subkey_dict)
            current_dict.update(subkey_dict)

        # iterate all values
        if num_subkeys == 0:
            for i in range(num_values):
                value_name, value_data, _ = winreg.EnumValue(registry_key, i)
                current_dict[value_name] = value_data

        if not is_top_level:
            results[key_path] = current_dict
        else:
            results.update(current_dict)

        winreg.CloseKey(registry_key)

    except Exception as e:
        raise Exception(f"Failed to get registry items {key_path}: {e}")


def adapter_luid_to_string(adapter_luid):
    high_part = adapter_luid >> 32
    low_part = adapter_luid & 0xFFFFFFFF
    return f"luid_0x{high_part:08x}_0x{low_part:08x}_phys_0"
