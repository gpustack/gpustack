import asyncio
import logging
from typing import Dict, List
from gpustack.detectors.base import GPUDetector
from gpustack.schemas.workers import (
    GPUCoreInfo,
    GPUDeviceInfo,
    GPUDevicesInfo,
    MemoryInfo,
    VendorEnum,
    GPUNetworkInfo,
)
from gpustack.utils import platform
from gpustack.utils.command import is_command_available
from gpustack.utils.convert import safe_float, safe_int
import gpustack.logging as glogging

logger = logging.getLogger(__name__)


class NPUSMI(GPUDetector):
    def is_available(self) -> bool:
        return is_command_available("npu-smi")

    def gather_gpu_info(self) -> GPUDevicesInfo:
        return asyncio.run(_gather_gpu_info())


async def _gather_gpu_info() -> GPUDevicesInfo:
    """
    Gather the NPU chips' info in asynchronous way.
    """

    devs = []

    # Get all NPU chips' mapping: {Chip Logic ID -> (NPU ID, Chip ID)}.
    all_npu_chips_mapping = await get_all_npu_chips_mapping()

    # Store all NPU chips' common info and usages info: {NPU ID -> {Chip ID -> {<key> -> <value>}}}.
    all_npu_chips_infos = {}

    # Iterate through each NPU chip and gather its info.
    for chip_logic_id, (npu_id, chip_id) in all_npu_chips_mapping.items():
        # Get NPU chips' info for this NPU ID: {Chip ID -> {<key> -> <value>}}.
        npu_chips_info = all_npu_chips_infos.get(npu_id, None)
        if npu_chips_info is None:
            # Gather the NPU chips' info for this NPU ID.
            npu_chips_info = await get_npu_chips_info(npu_id)
            all_npu_chips_infos[npu_id] = npu_chips_info

        # Get the NPU chip info for this Chip ID: {<key> -> <value>}.
        npu_chip_info = npu_chips_info.get(chip_id, {})
        if logger.isEnabledFor(glogging.TRACE_LEVEL):
            logger.trace(
                f"Gathered NPU chip {chip_logic_id} ({npu_id}, {chip_id}) info: {npu_chip_info}"
            )

        # Tidy up the device info.
        # - Name
        dev_name = npu_chip_info.get("Chip Name", "unknown")
        # - UUID
        dev_uuid = ""
        for key in ["VDie ID", "Die ID"]:
            if key not in npu_chip_info:
                continue
            dev_uuid = npu_chip_info.get(key, "")
            if dev_uuid:
                break
        # - Core
        dev_core = GPUCoreInfo(
            total=safe_int(npu_chip_info.get("Aicore Count", 0)),
            utilization_rate=safe_float(npu_chip_info.get("Aicore Usage Rate(%)", 0)),
        )
        # - Memory
        dev_memory = MemoryInfo(
            is_unified_memory=False,
            total=0,
            utilization_rate=0.0,
        )
        for key in [
            ["HBM Capacity(MB)", "HBM Usage Rate(%)"],
            ["DDR Capacity(MB)", "DDR Usage Rate(%)"],
        ]:
            if not all(k in npu_chip_info for k in key):
                continue
            total = safe_int(npu_chip_info.get(key[0], 0)) * 1024 * 1024
            utilization_rate = safe_float(npu_chip_info.get(key[1], 0))
            if total > 0 and utilization_rate >= 0:
                dev_memory.total = total
                dev_memory.utilization_rate = utilization_rate
                break
        dev_memory.used = round(dev_memory.total * dev_memory.utilization_rate / 100)
        # - Network
        dev_network = GPUNetworkInfo(
            status=npu_chip_info.get("link status", "down").lower(),
            inet=npu_chip_info.get("ipaddr", ""),
            netmask=npu_chip_info.get("netmask", ""),
            mac=npu_chip_info.get("mac addr", ""),
            gateway=npu_chip_info.get("default gateway", ""),
            iface=npu_chip_info.get("Iface", None),
            mtu=safe_int(npu_chip_info.get("mtu", None)),
        )
        # - Temperature
        dev_temperature = safe_float(npu_chip_info.get("Temperature(C)", 0))

        # Add to list.
        devs.append(
            GPUDeviceInfo(
                index=chip_logic_id,
                device_index=npu_id,
                device_chip_index=chip_id,
                vendor=VendorEnum.Huawei.value,
                type=platform.DeviceTypeEnum.NPU.value,
                name=dev_name,
                uuid=dev_uuid,
                core=dev_core,
                memory=dev_memory,
                network=dev_network,
                temperature=dev_temperature,
            )
        )

    return devs


async def get_all_npu_chips_mapping() -> Dict[int, tuple[int, int]]:
    """
    Get all NPU chips' mapping,
    return a dictionary with the mapping: {Chip Logic ID -> (NPU ID, Chip ID)},
    see:
    - 910B: https://support.huawei.com/enterprise/zh/doc/EDOC1100438695/875cdf74.
    - 310P: https://support.huawei.com/enterprise/zh/doc/EDOC1100368198/78616140.
    """

    output = await _async_run_command(["npu-smi", "info", "-m"])
    """
    # Example output 1:

    NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
    0                              0                              0                              Ascend xxx
    1                              0                              1                              Ascend xxx
    2                              0                              2                              Ascend xxx
    3                              0                              3                              Ascend xxx

    # Example output 2:

    NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
    4                              0                              0                              Ascend xxx
    4                              1                              -                              Mcu
    """

    return _parse_all_npu_chips_mapping(output)


def _parse_all_npu_chips_mapping(output: str) -> Dict[int, tuple[int, int]]:
    """
    Parse the output of `npu-smi info -m to a dictionary`
    to a directory: {Chip Logic ID -> (NPU ID, Chip ID)}.
    """

    mapping = {}

    for line in output.split('\n'):
        # Ignore empty lines
        if not line.strip():
            continue
        # Ignore header
        if line.startswith("NPU ID"):
            continue

        # Gather NPU ID, Chip ID and Chip Logic ID
        arr = line.split(maxsplit=3)
        if len(arr) < 3:
            continue

        # Check if NPU ID, Chip ID and Chip Logic ID are valid integers
        npu_id_str = arr[0].strip()
        chip_id_str = arr[1].strip()
        logic_id_str = arr[2].strip()
        if (
            not npu_id_str.isdecimal()
            or not chip_id_str.isdecimal()
            or not logic_id_str.isdecimal()
        ):
            continue

        # Add to mapping
        mapping[int(logic_id_str)] = (int(npu_id_str), int(chip_id_str))

    return mapping


async def get_npu_chips_info(npu_id: int) -> Dict[int, Dict[str, str]]:
    """
    Collect the NPU chips' info after calling the following functions:
        - get_npu_chips_common_info
        - get_npu_chips_usages_info
        - get_npu_network_info
        - get_npu_chip_board_info
    return a dictionary with the chip info: {<key> -> <value>}.
    """

    npu_id = str(npu_id)

    # Gather the NPU chips' * info for this NPU ID.
    common_info, usages_info, network_info = await asyncio.gather(
        get_npu_chips_common_info(npu_id),
        get_npu_chips_usages_info(npu_id),
        get_npu_network_info(npu_id),
    )

    info = {}

    for chip_id in common_info.keys():
        # Gather the NPU chip board info for this Chip ID.
        board_info = await get_npu_chip_board_info(npu_id, str(chip_id))

        # Merge the NPU chip * info for this Chip ID.
        chip_info = common_info.get(chip_id, {})
        chip_info.update(usages_info.get(chip_id, {}))
        chip_info.update(board_info)
        chip_info.update(network_info)

        # Add to info
        info[chip_id] = chip_info

    return info


async def get_npu_chips_common_info(npu_id: str) -> Dict[int, Dict[str, str]]:
    """
    Get NPU chips' common info,
    return a dictionary with the NPU chip common info: {Chip ID -> {<key> -> <value>}},
    see:
    - 910B: https://support.huawei.com/enterprise/zh/doc/EDOC1100438695/37eb6c60.
    - 310P: https://support.huawei.com/enterprise/zh/doc/EDOC1100368198/6eb85cfb.

    Note:
    - When enabling virtual NPU,
      the "HBM Usage Rate(%)" and "Aicore Usage Rate(%)" are 0.
    """

    output = await _async_run_command(["npu-smi", "info", "-t", "common", "-i", npu_id])
    """
    # Example output 1:

    NPU ID                         : 0
    Chip Count                     : 1

    Chip ID                        : 0
    Memory Usage Rate(%)           : 6
    HBM Usage Rate(%)              : 0
    Aicore Usage Rate(%)           : 0
    Aicore Freq(MHZ)               : 1000
    Aicore curFreq(MHZ)            : 1000
    Aicore Count                   : 32
    Temperature(C)                 : 46
    NPU Real-time Power(W)         : 69.0

    # Example output 2:

    NPU ID                         : 1
    Chip Count                     : 1

    Chip ID                        : 0
    Memory Usage Rate(%)           : 4
    Aicore Usage Rate(%)           : 0
    Aicore Freq(MHZ)               : 1080
    Aicore curFreq(MHZ)            : 960
    Temperature(C)                 : 43

    Chip Name                      : mcu
    Temperature(C)                 : 41
    NPU Real-time Power(W)         : 13.4
    """

    return _parse_npu_chips_common_info(output)


def _parse_npu_chips_common_info(output: str) -> Dict[int, Dict[str, str]]:
    """
    Parse the output of `npu-smi info -t common -i npu_id`
    to a dictionary: {Chip ID -> {<key> -> <value>}}.
    """

    info = {}

    chip_id = -1
    for line in output.split('\n'):
        # Gather key and value
        kvs = _parse_line_to_dict(line)
        if not kvs:
            continue

        for key, value in kvs.items():
            if key in ["NPU ID", "Chip Count"]:
                continue

            # Init info if "Chip ID" is valid,
            # otherwise lock the chip_id to -1, which always invalid.
            if key == "Chip ID":
                if value.isdecimal():
                    chip_id = int(value)
                    info[chip_id] = {}
                else:
                    chip_id = -1
                continue

            chip_info = info.get(chip_id, None)
            if chip_info is None:
                continue

            # Add to chip_info
            chip_info[key] = value

    return info


async def get_npu_chips_usages_info(npu_id: str) -> Dict[int, Dict[str, str]]:
    """
    Get the NPU chips' usages info,
    return a dictionary with the chip usages info: {Chip ID -> {<key> -> <value>}},
    see:
    - 910B: https://support.huawei.com/enterprise/zh/doc/EDOC1100438695/2e670e83.
    - 310P: https://support.huawei.com/enterprise/zh/doc/EDOC1100368198/97b70cda.

    Note:
    - When enabling virtual NPU,
        the "Aicore Usage Rate(%)", "Aicpu Usage Rate(%)", "Ctrlcpu Usage Rate(%)", "HBM Usage Rate(%)", and "DDR Bandwidth Usage Rate(%)" are 0.
    - When enabling virtual NPU and without DVPP resources,
        the "DVPP VDEC Usage Rate(%)", "DVPP VPC Usage Rate(%)", "DVPP VENC Usage Rate(%)", "DVPP JPEGE Usage Rate(%)" and "DVPP JPEGD Usage Rate(%)" are 0.
    - When profiling,
        the "Aicore Usage Rate(%)" is 0.
    """

    output = await _async_run_command(["npu-smi", "info", "-t", "usages", "-i", npu_id])
    """
    # Example output 1:

    NPU ID                         : 0
    Chip Count                     : 1

    DDR Capacity(MB)               : 15171
    DDR Usage Rate(%)              : 3
    DDR Hugepages Total(page)      : 0
    DDR Hugepages Usage Rate(%)    : 0
    HBM Capacity(MB)               : 32768
    HBM Usage Rate(%)              : 0
    Aicore Usage Rate(%)           : 0
    Aicpu Usage Rate(%)            : 0
    Ctrlcpu Usage Rate(%)          : 9
    DDR Bandwidth Usage Rate(%)    : 0
    HBM Bandwidth Usage Rate(%)    : 0
    Chip ID                        : 0

    # Example output 2:

    NPU ID                         : 1
    Chip Count                     : 1

    DDR Capacity(MB)               : 21534
    DDR Usage Rate(%)              : 4
    DDR Hugepages Total(page)      : 0
    DDR Hugepages Usage Rate(%)    : 0
    Aicore Usage Rate(%)           : 0
    Aicpu Usage Rate(%)            : 0
    Ctrlcpu Usage Rate(%)          : 13
    Vectorcore Usage Rate(%)       : 0
    DDR Bandwidth Usage Rate(%)    : 51
    DVPP VDEC Usage Rate(%)        : 0
    DVPP VPC Usage Rate(%)         : 0
    DVPP VENC Usage Rate(%)        : 0
    DVPP JPEGE Usage Rate(%)       : 0
    DVPP JPEGD Usage Rate(%)       : 0
    Chip ID                        : 0
    """

    return _parse_npu_chips_usages_info(output)


def _parse_npu_chips_usages_info(output: str) -> Dict[int, Dict[str, str]]:
    """
    Parse the output of `npu-smi info -t usages -i npu_id`
    to a dictionary: {Chip ID -> {<key> -> <value>}}.
    """

    info = {}

    chip_info = {}
    for line in output.split('\n'):
        # Gather key and value
        kvs = _parse_line_to_dict(line)
        if not kvs:
            continue

        for key, value in kvs.items():
            if key in ["NPU ID", "Chip Count"]:
                continue

            # Add to info if "Chip ID" is valid,
            # and reset chip_info not matter "Chip ID" is valid or not.
            if key == "Chip ID":
                if value.isdecimal():
                    chip_id = int(value)
                    info[chip_id] = chip_info
                chip_info = {}
                continue

            # Add to chip_info
            chip_info[key] = value

    return info


async def get_npu_network_info(npu_id: str) -> Dict[str, str]:
    """
    Get the NPU network info,
    return a dictionary with the network info: {<key> -> <value>},
    see:
    - 910B: https://support.huawei.com/enterprise/zh/doc/EDOC1100439048/426cffd9.
    """

    hccn_tool = "/usr/local/Ascend/driver/tools/hccn_tool"
    if not is_command_available(hccn_tool):
        hccn_tool = "hccn_tool"
        if not is_command_available(hccn_tool):
            # hccn_tool is not available, return an empty dictionary
            return {}

    info = {}

    # Check if the network is available
    output = await _async_run_command([hccn_tool, "-i", npu_id, "-link", "-g"])
    """
    Example output:

    link status: DOWN
    """

    kvs = _parse_line_to_dict(output)
    if not kvs:
        return {}
    info.update(kvs)

    if info.get("link status", "down").lower() != "up":
        return info

    output = await asyncio.gather(
        _async_run_command([hccn_tool, "-i", npu_id, "-ip", "-g"]),
        _async_run_command([hccn_tool, "-i", npu_id, "-mac", "-g"]),
        _async_run_command([hccn_tool, "-i", npu_id, "-gateway", "-g"]),
        _async_run_command([hccn_tool, "-i", npu_id, "-mtu", "-g"]),
    )
    output = '\n'.join(output)
    """
    Example output:

    ipaddr:192.168.6.10
    netmask:255.255.255.0

    mac addr: xx:xx:xx:xx:xx:xx

    default gateway:192.168.6.1, Iface:eth0
    """

    for line in output.split('\n'):
        # Gather key and value
        kvs = _parse_line_to_dict(line)
        if not kvs:
            continue

        # Add to info
        info.update(kvs)

    return info


async def get_npu_chip_board_info(npu_id: str, chip_id: str) -> Dict[str, str]:
    """
    Get the NPU chip's board info,
    return a dictionary with the chip board info: {<key> -> <value>},
    see:
    - 910B: https://support.huawei.com/enterprise/zh/doc/EDOC1100438695/66dc0fff.
    - 310P: https://support.huawei.com/enterprise/zh/doc/EDOC1100368198/95c5adf7.

    Note:
    - When enabling virtual NPU,
      the "Firmware Version" is unavailable.
    - When enabling virtual NPU,
      the "Chip ID" is the ID of MCU, and the "Board ID" is unavailable.
    """

    output = await _async_run_command(
        ["npu-smi", "info", "-t", "board", "-i", npu_id, "-c", chip_id]
    )
    """
    # Example output 1:

    NPU ID                         : 0
    Chip ID                        : 0
    Chip Type                      : Ascend
    Chip Name                      : xxx
    Chip Version                   : V1
    Board ID                       : 0x02
    PCB ID                         : NA
    BOM ID                         : 1
    VDie ID                        : 901421D4 02103514 1711E613 6DA7040A 00102001
    NDie ID                        : 05821994 20902110 1B11E613 6DA7040A 70102001
    Chip Position ID               : 1
    PCIe Bus Info                  : 0000:81:00.0
    Firmware Version               : 7.1.0.4.218

    # Example output 2:

    NPU ID                         : 1
    Chip ID                        : 0
    Chip Type                      : Ascend
    Chip Name                      : xxx
    Chip Version                   : V1
    Board ID                       : 0x64
    Die ID                         : 409921D4 2140A100 E54BDDD4 07CC040A 9B00301F
    Chip Position ID               : 0
    PCIe Bus Info                  : 0000:81:00.0
    Firmware Version               : 7.1.0.4
    """

    info = {}

    for line in output.split('\n'):
        # Gather key and value
        kvs = _parse_line_to_dict(line)
        if not kvs:
            continue

        # Add to info
        info.update(kvs)

    return info


def _parse_line_to_dict(line: str) -> Dict[str, str]:
    """
    Parse a string line to an array: {<key> -> <value>},
    with characters: ',' and ':' as delimiters.
    """

    line = line.strip()
    if not line:
        return {}

    ret = {}

    for item in line.split(','):
        item = item.strip()
        if not item:
            continue

        arr = item.split(':', maxsplit=1)
        if len(arr) < 2:
            continue

        ret[arr[0].strip()] = arr[1].strip()

    return ret


async def _async_run_command(command: List[str]) -> str:
    """
    Run a command and return the output,
    return an empty string if the command fails.
    """

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"Exit {process.returncode}: {stderr}")
        return str(stdout, encoding="utf-8")
    except Exception as e:
        if logger.isEnabledFor(glogging.TRACE_LEVEL):
            error_message = f"Failed to execute {command}: {e}"
            logger.warning(error_message)

    return ""
