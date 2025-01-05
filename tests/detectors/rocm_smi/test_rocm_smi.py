import os

from gpustack.detectors.rocm_smi.rocm_smi import RocmSMI
from gpustack.schemas.workers import GPUCoreInfo, GPUDeviceInfo, MemoryInfo, VendorEnum
from gpustack.utils.platform import DeviceTypeEnum


def test_decode_gpu_devices():
    files = [
        "hygon-gpu-24.04.3",
        "amd-gpu-6.2.4",
    ]

    expected_outputs = [
        {
            "gpus": [
                gpu_device(
                    "7100fa8866904061",
                    "K1xx_AI",
                    0,
                    VendorEnum.Hygon.value,
                    65520,
                    2,
                    12,
                    0.0,
                    44,
                    "gfx928",
                ),
            ],
        },
        {
            "gpus": [
                gpu_device(
                    "5c88007d760374f3",
                    "AMD Radeon RX 7800 XT",
                    0,
                    VendorEnum.AMD.value,
                    16368,
                    11077,
                    60,
                    100.0,
                    54,
                    "gfx1101",
                ),
            ],
        },
    ]

    for i, file in enumerate(files):
        info_output, smi_output = command_output(file)
        rocm_smi = RocmSMI()
        info_devices = rocm_smi.decode_rocminfo(info_output)
        smi_devices = rocm_smi.decode_rocm_smi(smi_output)

        gpus = rocm_smi.inject_gpu_info(info_devices, smi_devices)
        assert expected_outputs[i].get("gpus") == gpus


def command_output(file: str) -> tuple[str, str]:
    info_output = ""
    smi_output = ""

    current_dir = os.path.dirname(__file__)

    info_file = os.path.join(current_dir, "data", f"{file}-rocminfo.txt")
    with open(info_file, 'r') as f:
        info_output = f.read()

    smi_file = os.path.join(current_dir, "data", f"{file}-rocm-smi.json")
    with open(smi_file, 'r') as f:
        smi_output = f.read()

    return info_output, smi_output


def gpu_device(
    uuid: str,
    name: str,
    index: int,
    vendor: VendorEnum,
    mem_total_in_mib: int,
    mem_used_in_mib: int,
    core_total: int,
    core_util: float,
    temp: float,
    llvm: str,
) -> GPUDeviceInfo:
    mem_total = mem_total_in_mib * 1024 * 1024
    mem_used = mem_used_in_mib * 1024 * 1024
    device_type = DeviceTypeEnum.ROCM.value
    if vendor == VendorEnum.Hygon.value:
        device_type = DeviceTypeEnum.DCU.value
    return GPUDeviceInfo(
        uuid=uuid,
        index=index,
        name=name,
        vendor=vendor,
        core=GPUCoreInfo(
            total=core_total,
            utilization_rate=core_util,
        ),
        memory=MemoryInfo(
            total=mem_total,
            used=mem_used,
            utilization_rate=mem_used / mem_total * 100 if mem_total > 0 else 0,
        ),
        temperature=temp,
        type=device_type,
        labels={"llvm": llvm},
    )
