import os

from gpustack.detectors.npu_smi.npu_smi import NPUSMI
from gpustack.schemas.workers import GPUCoreInfo, GPUDeviceInfo, MemoryInfo, VendorEnum


def test_decode_gpu_devices():
    files = [
        "ai-server_atlas-800-inference-server_model-3000_version-1.0.0-1.0.10.txt",
        "ai-server_atlas-800-inference-server_model-3000_version-1.0.11-1.0.15.txt",
        "ai-server_atlas-800-training-server_model-9010_version-6.0.RC1.txt",
        "ai-server_atlas-800-training-server_model-9010_version-24.1.RC2.txt",
    ]

    expected_outputs = [
        {
            "mapping": {
                (215, 0): 0,
                (215, 1): 1,
                (215, 2): 2,
                (215, 3): 3,
            },
            "gpus": [
                gpu_device("310", 0, 2703, 8192, 0.0, 56),
                gpu_device("310", 1, 2703, 8192, 0.0, 57),
                gpu_device("310", 2, 2703, 8192, 0.0, 57),
                gpu_device("310", 3, 2703, 8192, 0.0, 55),
            ],
        },
        {
            "mapping": {
                (6, 0): 0,
                (6, 1): 1,
                (6, 2): 2,
                (6, 3): 3,
            },
            "gpus": [
                gpu_device("310", 0, 2703, 8192, 0.0, 73),
                gpu_device("310", 1, 2867, 8192, 0.0, 74),
                gpu_device("310", 2, 2867, 8192, 0.0, 70),
                gpu_device("310", 3, 2867, 8192, 0.0, 65),
            ],
        },
        {
            "mapping": {
                (0, 0): 0,
                (1, 0): 1,
                (2, 0): 2,
                (3, 0): 3,
                (4, 0): 4,
                (5, 0): 5,
                (6, 0): 6,
                (7, 0): 7,
            },
            "gpus": [
                gpu_device("910A", 0, 0, 32768, 0.0, 43),
                gpu_device("910A", 1, 0, 32768, 0.0, 38),
                gpu_device("910A", 2, 0, 32768, 0.0, 34),
                gpu_device("910A", 3, 0, 32768, 0.0, 42),
                gpu_device("910A", 4, 0, 32768, 0.0, 44),
                gpu_device("910A", 5, 0, 32768, 0.0, 33),
                gpu_device("910A", 6, 0, 32768, 0.0, 35),
                gpu_device("910A", 7, 0, 32768, 0.0, 40),
            ],
        },
        {
            "mapping": {
                (0, 0): 0,
                (1, 0): 1,
                (2, 0): 2,
                (3, 0): 3,
                (4, 0): 4,
                (5, 0): 5,
                (6, 0): 6,
                (7, 0): 7,
            },
            "gpus": [
                gpu_device("xxx", 0, 0, 32768, 0.0, 42),
                gpu_device("xxx", 1, 0, 32768, 0.0, 36),
                gpu_device("xxx", 2, 0, 32768, 0.0, 35),
                gpu_device("xxx", 3, 0, 32768, 0.0, 39),
                gpu_device("xxx", 4, 0, 32768, 0.0, 39),
                gpu_device("xxx", 5, 0, 32768, 0.0, 37),
                gpu_device("xxx", 6, 0, 32768, 0.0, 40),
                gpu_device("xxx", 7, 0, 32768, 0.0, 42),
            ],
        },
    ]

    for i, file in enumerate(files):
        info_output, mapping_output = command_output(file)
        npu_smi = NPUSMI()
        mapping = npu_smi.decode_gpu_device_mapping(mapping_output)
        gpus = npu_smi.decode_gpu_devices(info_output, mapping)

        assert expected_outputs[i].get("mapping") == mapping
        assert expected_outputs[i].get("gpus") == gpus


def command_output(file: str) -> tuple[str, str]:
    info_output = ""
    mapping_output = ""

    current_dir = os.path.dirname(__file__)

    info_file = os.path.join(current_dir, "data", file)
    with open(info_file, 'r') as f:
        info_output = f.read()

    base_name, ext = os.path.splitext(file)
    mapping_file = base_name + "_mapping" + ext
    mapping_file = os.path.join(current_dir, "data", mapping_file)
    with open(mapping_file, 'r') as f:
        mapping_output = f.read()

    return info_output, mapping_output


def gpu_device(
    name: str,
    index: int,
    mem_used_in_mib: int,
    mem_total_in_mib: int,
    core_util: float,
    temp: float,
) -> GPUDeviceInfo:
    mem_total = mem_total_in_mib * 1024 * 1024
    mem_used = mem_used_in_mib * 1024 * 1024
    return GPUDeviceInfo(
        index=index,
        name=name,
        vendor=VendorEnum.Huawei,
        core=GPUCoreInfo(
            total=0,
            utilization_rate=core_util,
        ),
        memory=MemoryInfo(
            total=mem_total,
            used=mem_used,
            utilization_rate=mem_used / mem_total * 100 if mem_total > 0 else 0,
        ),
        temperature=temp,
    )
