import json
import logging
import subprocess
from typing import Dict
from gpustack.detectors.base import GPUDetector
from gpustack.schemas.workers import (
    GPUCoreInfo,
    GPUDeviceInfo,
    GPUDevicesInfo,
    MemoryInfo,
    VendorEnum,
)
from gpustack.utils import platform
from gpustack.utils.command import is_command_available
from gpustack.utils.convert import safe_float, safe_int

logger = logging.getLogger(__name__)


class RocmSMI(GPUDetector):
    def is_available(self) -> bool:
        rocm_smi_available = is_command_available("rocm-smi")
        if not rocm_smi_available:
            return False

        rocminfo_available = is_command_available("rocminfo")
        if not rocminfo_available:
            return False

        return True

    def gather_gpu_info(self) -> GPUDevicesInfo:
        rocminfo_command = self._command_rocminfo()
        rocminfo_result = self._run_command(rocminfo_command)
        rocminfo_devices = self.decode_rocminfo(rocminfo_result)

        rocm_smi_command = self._command_rocm_smi()
        rocm_smi_result = self._run_command(rocm_smi_command)
        rocm_smi_devices = self.decode_rocm_smi(rocm_smi_result)

        for device in rocm_smi_devices:
            for device_uuid, rocminfo_device in rocminfo_devices.items():
                if device.uuid == device_uuid or device.uuid in rocminfo_device.get(
                    "Chip ID"
                ):
                    device.name = rocminfo_device.get("Marketing Name")
                    device.core.total = safe_int(rocminfo_device.get("Compute Unit", 0))
                    device.labels = {
                        "llvm": rocminfo_device.get("LLVM Target Name", "")
                    }
                    break

        return rocm_smi_devices

    def decode_rocm_smi(self, result) -> GPUDevicesInfo:
        """
        result example:
        $rocm-smi -i --showmeminfo vram --showpower --showserial --showuse --showtemp --showproductname --json
        {
            "card0": {
                "Device ID": "0x747e",
                "Device Rev": "0xc8",
                "Temperature (Sensor edge) (C)": "36.0",
                "Temperature (Sensor junction) (C)": "41.0",
                "Temperature (Sensor memory) (C)": "44.0",
                "Average Graphics Package Power (W)": "4.0",
                "GPU use (%)": "0",
                "Serial Number": "5c88007d760374f3",
                "VRAM Total Memory (B)": "17163091968",
                "VRAM Total Used Memory (B)": "283090944",
                "Card series": "0x747e",
                "Card model": "0x7801",
                "Card vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
                "Card SKU": "EXT94393"
            }
        }
        """

        devices = []
        parsed_json = json.loads(result)
        for key in parsed_json.keys():
            info = parsed_json.get(key)

            index = safe_int(key.removeprefix("card"))
            uuid = (
                info.get("Device ID")
                if "N/A" in info.get("Serial Number")
                else info.get("Serial Number")
            )
            name = info.get("Device Name")

            memory_total = safe_int(info.get("VRAM Total Memory (B)", 0))
            memory_used = safe_int(info.get("VRAM Total Used Memory (B)", 0))
            utilization_gpu = safe_float(info.get("GPU use (%)", 0))
            temperature_gpu = safe_float(info.get("Temperature (Sensor memory) (C)", 0))

            device = GPUDeviceInfo(
                index=index,
                name=name,
                uuid=uuid,
                vendor=VendorEnum.AMD.value,
                memory=MemoryInfo(
                    is_unified_memory=False,
                    used=memory_used,
                    total=memory_total,
                    utilization_rate=(
                        (memory_used / memory_total) * 100 if memory_total > 0 else 0
                    ),
                ),
                core=GPUCoreInfo(
                    utilization_rate=utilization_gpu,
                    total=0,  # Total cores information is not provided by rocm-smi
                ),
                temperature=temperature_gpu,
                type=platform.DeviceTypeEnum.ROCM.value,
            )
            devices.append(device)
        return devices

    def decode_rocminfo(self, result) -> Dict[str, Dict[str, str]]:
        """
        results example:
        $rocminfo
        ROCk module version 6.2.4 is loaded
        =====================
        HSA System Attributes
        =====================
        Runtime Version:         1.13
        Runtime Ext Version:     1.4
        System Timestamp Freq.:  1000.000000MHz
        Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
        Machine Model:           LARGE
        System Endianness:       LITTLE
        Mwaitx:                  DISABLED
        DMAbuf Support:          YES

        ==========
        HSA Agents
        ==========
        *******
        Agent 1
        *******
        Name:                    Intel(R) Core(TM) i5-14600KF
        Uuid:                    CPU-XX
        Marketing Name:          Intel(R) Core(TM) i5-14600KF
        Vendor Name:             CPU
        Feature:                 None specified
        Profile:                 FULL_PROFILE
        Float Round Mode:        NEAR
        Max Queue Number:        0(0x0)
        Queue Min Size:          0(0x0)
        Queue Max Size:          0(0x0)
        Queue Type:              MULTI
        Node:                    0
        Device Type:             CPU
        Cache Info:
            L1:                      49152(0xc000) KB
        Chip ID:                 0(0x0)
        ASIC Revision:           0(0x0)
        Cacheline Size:          64(0x40)
        Max Clock Freq. (MHz):   5300
        BDFID:                   0
        Internal Node ID:        0
        Compute Unit:            20
        SIMDs per CU:            0
        Shader Engines:          0
        Shader Arrs. per Eng.:   0
        WatchPts on Addr. Ranges:1
        Features:                None
        Pool Info:
            Pool 1
            Segment:                 GLOBAL; FLAGS: FINE GRAINED
            Size:                    65613932(0x3e9306c) KB
            Allocatable:             TRUE
            Alloc Granule:           4KB
            Alloc Recommended Granule:4KB
            Alloc Alignment:         4KB
            Accessible by all:       TRUE
            Pool 2
            Segment:                 GLOBAL; FLAGS: KERNARG, FINE GRAINED
            Size:                    65613932(0x3e9306c) KB
            Allocatable:             TRUE
            Alloc Granule:           4KB
            Alloc Recommended Granule:4KB
            Alloc Alignment:         4KB
            Accessible by all:       TRUE
            Pool 3
            Segment:                 GLOBAL; FLAGS: COARSE GRAINED
            Size:                    65613932(0x3e9306c) KB
            Allocatable:             TRUE
            Alloc Granule:           4KB
            Alloc Recommended Granule:4KB
            Alloc Alignment:         4KB
            Accessible by all:       TRUE
        ISA Info:
        *******
        Agent 2
        *******
        Name:                    gfx1101
        Uuid:                    GPU-5c88007d760374f3
        Marketing Name:          AMD Radeon RX 7800 XT
        Vendor Name:             AMD
        Feature:                 KERNEL_DISPATCH
        Profile:                 BASE_PROFILE
        Float Round Mode:        NEAR
        Max Queue Number:        128(0x80)
        Queue Min Size:          64(0x40)
        Queue Max Size:          131072(0x20000)
        Queue Type:              MULTI
        Node:                    1
        Device Type:             GPU
        Cache Info:
            L1:                      32(0x20) KB
            L2:                      4096(0x1000) KB
            L3:                      65536(0x10000) KB
        Chip ID:                 29822(0x747e)
        ASIC Revision:           0(0x0)
        Cacheline Size:          64(0x40)
        Max Clock Freq. (MHz):   2254
        BDFID:                   768
        Internal Node ID:        1
        Compute Unit:            60
        SIMDs per CU:            2
        Shader Engines:          3
        Shader Arrs. per Eng.:   2
        WatchPts on Addr. Ranges:4
        Coherent Host Access:    FALSE
        Features:                KERNEL_DISPATCH
        Fast F16 Operation:      TRUE
        Wavefront Size:          32(0x20)
        Workgroup Max Size:      1024(0x400)
        Workgroup Max Size per Dimension:
            x                        1024(0x400)
            y                        1024(0x400)
            z                        1024(0x400)
        Max Waves Per CU:        32(0x20)
        Max Work-item Per CU:    1024(0x400)
        Grid Max Size:           4294967295(0xffffffff)
        Grid Max Size per Dimension:
            x                        4294967295(0xffffffff)
            y                        4294967295(0xffffffff)
            z                        4294967295(0xffffffff)
        Max fbarriers/Workgrp:   32
        Packet Processor uCode:: 546
        SDMA engine uCode::      20
        IOMMU Support::          None
        Pool Info:
            Pool 1
            Segment:                 GLOBAL; FLAGS: COARSE GRAINED
            Size:                    16760832(0xffc000) KB
            Allocatable:             TRUE
            Alloc Granule:           4KB
            Alloc Recommended Granule:2048KB
            Alloc Alignment:         4KB
            Accessible by all:       FALSE
            Pool 2
            Segment:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
            Size:                    16760832(0xffc000) KB
            Allocatable:             TRUE
            Alloc Granule:           4KB
            Alloc Recommended Granule:2048KB
            Alloc Alignment:         4KB
            Accessible by all:       FALSE
            Pool 3
            Segment:                 GROUP
            Size:                    64(0x40) KB
            Allocatable:             FALSE
            Alloc Granule:           0KB
            Alloc Recommended Granule:0KB
            Alloc Alignment:         0KB
            Accessible by all:       FALSE
        ISA Info:
            ISA 1
            Name:                    amdgcn-amd-amdhsa--gfx1101
            Machine Models:          HSA_MACHINE_MODEL_LARGE
            Profiles:                HSA_PROFILE_BASE
            Default Rounding Mode:   NEAR
            Default Rounding Mode:   NEAR
            Fast f16:                TRUE
            Workgroup Max Size:      1024(0x400)
            Workgroup Max Size per Dimension:
                x                        1024(0x400)
                y                        1024(0x400)
                z                        1024(0x400)
            Grid Max Size:           4294967295(0xffffffff)
            Grid Max Size per Dimension:
                x                        4294967295(0xffffffff)
                y                        4294967295(0xffffffff)
                z                        4294967295(0xffffffff)
            FBarrier Max Size:       32
        *** Done ***
        """

        collecting_keyworks = [
            "Uuid",
            "Marketing Name",
            "Vendor Name",
            "Device Type",
            "Chip ID",
            "Compute Unit",
            "amdgcn-amd-amdhsa",
        ]

        valid_lines = [
            line.strip()
            for line in result.splitlines()
            if any(keyword in line for keyword in collecting_keyworks)
        ]

        uuid = ""
        devices = {}
        for line in valid_lines:
            for keyword in collecting_keyworks:
                if keyword in line:
                    contents = line.split(":")
                    if len(contents) < 2:
                        continue
                    key = contents[0].strip()
                    value = contents[-1].strip()
                    if key == "Uuid":
                        value = value.removeprefix("GPU-")
                        uuid = value
                        devices[uuid] = {}
                    if "amdgcn-amd-amdhsa" in value:
                        key = "LLVM Target Name"
                        value = value.removeprefix("amdgcn-amd-amdhsa--")
                    devices[uuid][key] = value
                    break

        return devices

    def _run_command(self, command):
        result = None
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True, encoding="utf-8"
            )
            output = result.stdout

            if result.returncode != 0:
                raise Exception(f"Unexpected return code: {result.returncode}")

            if output == "" or output is None:
                raise Exception(f"Output is empty, return code: {result.returncode}")

            return output
        except Exception as e:
            raise Exception(
                f"Failed to execute {command}: {e},"
                f" stdout: {result.stdout}, stderr: {result.stderr}"
            )

    def _command_rocm_smi(self):
        executable_command = [
            "rocm-smi",
            "--showid",
            "--showmeminfo",
            "vram",
            "--showpower",
            "--showserial",
            "--showuse",
            "--showtemp",
            "--showproductname",
            "--json",
        ]
        return executable_command

    def _command_rocminfo(self):
        executable_command = [
            "rocminfo",
        ]
        return executable_command
