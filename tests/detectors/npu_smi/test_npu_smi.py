import pytest

from gpustack.detectors.npu_smi.npu_smi import (
    _parse_all_npu_chips_mapping,
    _parse_npu_chips_common_info,
    _parse_npu_chips_usages_info,
    _parse_line_to_dict,
)


@pytest.mark.parametrize(
    "output, expected",
    [
        # 910B
        (
            """
NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
0                              0                              0                              Ascend xxx
0                              1                              -                              Mcu
1                              0                              1                              Ascend xxx
1                              1                              -                              Mcu
2                              0                              2                              Ascend xxx
2                              1                              -                              Mcu
3                              0                              3                              Ascend xxx
3                              1                              -                              Mcu
4                              0                              4                              Ascend xxx
4                              1                              -                              Mcu
5                              0                              5                              Ascend xxx
5                              1                              -                              Mcu
6                              0                              6                              Ascend xxx
6                              1                              -                              Mcu
7                              0                              7                              Ascend xxx
7                              1                              -                              Mcu
            """,
            {
                0: (0, 0),
                1: (1, 0),
                2: (2, 0),
                3: (3, 0),
                4: (4, 0),
                5: (5, 0),
                6: (6, 0),
                7: (7, 0),
            },
        ),
        # 310P
        (
            """
NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
1                              0                              0                              Ascend xxx
1                              1                              1                              Ascend xxx
1                              2                              -                              Mcu
2                              0                              2                              Ascend xxx
2                              1                              3                              Ascend xxx
2                              2                              -                              Mcu
4                              0                              4                              Ascend xxx
4                              1                              5                              Ascend xxx
4                              2                              -                              Mcu
5                              0                              6                              Ascend xxx
5                              1                              7                              Ascend xxx
5                              2                              -                              Mcu
            """,
            {
                0: (1, 0),
                1: (1, 1),
                2: (2, 0),
                3: (2, 1),
                4: (4, 0),
                5: (4, 1),
                6: (5, 0),
                7: (5, 1),
            },
        ),
    ],
)
def test__parse_all_npu_chips_mapping(output, expected):
    actual = _parse_all_npu_chips_mapping(output)
    assert actual == expected, f"Expected {expected}, but got {actual}"


@pytest.mark.parametrize(
    "output, expected",
    [
        # 910B
        (
            """
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

Chip Name                      : mcu
Temperature(C)                 : 48
NPU Real-time Power(W)         : 45.2
            """,
            {
                0: {
                    "Memory Usage Rate(%)": "6",
                    "HBM Usage Rate(%)": "0",
                    "Aicore Usage Rate(%)": "0",
                    "Aicore Freq(MHZ)": "1000",
                    "Aicore curFreq(MHZ)": "1000",
                    "Aicore Count": "32",
                    "Temperature(C)": "46",
                    "NPU Real-time Power(W)": "69.0",
                },
            },
        ),
        # 310P
        (
            """
NPU ID                         : 1
Chip Count                     : 2

Chip ID                        : 0
Memory Usage Rate(%)           : 3
Aicore Usage Rate(%)           : 0
Aicore Freq(MHZ)               : 1080
Aicore curFreq(MHZ)            : 960
Temperature(C)                 : 60

Chip ID                        : 1
Memory Usage Rate(%)           : 3
Aicore Usage Rate(%)           : 0
Aicore Freq(MHZ)               : 1080
Aicore curFreq(MHZ)            : 960
Temperature(C)                 : 59

Chip Name                      : mcu
Temperature(C)                 : 48
NPU Real-time Power(W)         : 45.2
            """,
            {
                0: {
                    "Memory Usage Rate(%)": "3",
                    "Aicore Usage Rate(%)": "0",
                    "Aicore Freq(MHZ)": "1080",
                    "Aicore curFreq(MHZ)": "960",
                    "Temperature(C)": "60",
                },
                1: {
                    "Memory Usage Rate(%)": "3",
                    "Aicore Usage Rate(%)": "0",
                    "Aicore Freq(MHZ)": "1080",
                    "Aicore curFreq(MHZ)": "960",
                    "Temperature(C)": "59",
                },
            },
        ),
    ],
)
def test__parse_npu_chips_common_info(output, expected):
    actual = _parse_npu_chips_common_info(output)
    assert actual == expected, f"Expected {expected}, but got {actual}"


@pytest.mark.parametrize(
    "output, expected",
    [
        # 910B
        (
            """
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
            """,
            {
                0: {
                    "DDR Capacity(MB)": "15171",
                    "DDR Usage Rate(%)": "3",
                    "DDR Hugepages Total(page)": "0",
                    "DDR Hugepages Usage Rate(%)": "0",
                    "HBM Capacity(MB)": "32768",
                    "HBM Usage Rate(%)": "0",
                    "Aicore Usage Rate(%)": "0",
                    "Aicpu Usage Rate(%)": "0",
                    "Ctrlcpu Usage Rate(%)": "9",
                    "DDR Bandwidth Usage Rate(%)": "0",
                    "HBM Bandwidth Usage Rate(%)": "0",
                },
            },
        ),
        # 310P
        (
            """
NPU ID                         : 1
Chip Count                     : 2

DDR Capacity(MB)               : 44280
DDR Usage Rate(%)              : 3
DDR Hugepages Total(page)      : 0
DDR Hugepages Usage Rate(%)    : 0
Aicore Usage Rate(%)           : 0
Aicpu Usage Rate(%)            : 0
Ctrlcpu Usage Rate(%)          : 2
Vectorcore Usage Rate(%)       : 0
DDR Bandwidth Usage Rate(%)    : 51
DVPP VDEC Usage Rate(%)        : 0
DVPP VPC Usage Rate(%)         : 0
DVPP VENC Usage Rate(%)        : 0
DVPP JPEGE Usage Rate(%)       : 0
DVPP JPEGD Usage Rate(%)       : 0
Chip ID                        : 0

DDR Capacity(MB)               : 43693
DDR Usage Rate(%)              : 3
DDR Hugepages Total(page)      : 0
DDR Hugepages Usage Rate(%)    : 0
Aicore Usage Rate(%)           : 0
Aicpu Usage Rate(%)            : 0
Ctrlcpu Usage Rate(%)          : 3
Vectorcore Usage Rate(%)       : 0
DDR Bandwidth Usage Rate(%)    : 51
DVPP VDEC Usage Rate(%)        : 0
DVPP VPC Usage Rate(%)         : 0
DVPP VENC Usage Rate(%)        : 0
DVPP JPEGE Usage Rate(%)       : 0
DVPP JPEGD Usage Rate(%)       : 0
Chip ID                        : 1
           """,
            {
                0: {
                    "DDR Capacity(MB)": "44280",
                    "DDR Usage Rate(%)": "3",
                    "DDR Hugepages Total(page)": "0",
                    "DDR Hugepages Usage Rate(%)": "0",
                    "Aicore Usage Rate(%)": "0",
                    "Aicpu Usage Rate(%)": "0",
                    "Ctrlcpu Usage Rate(%)": "2",
                    "Vectorcore Usage Rate(%)": "0",
                    "DDR Bandwidth Usage Rate(%)": "51",
                    "DVPP VDEC Usage Rate(%)": "0",
                    "DVPP VPC Usage Rate(%)": "0",
                    "DVPP VENC Usage Rate(%)": "0",
                    "DVPP JPEGE Usage Rate(%)": "0",
                    "DVPP JPEGD Usage Rate(%)": "0",
                },
                1: {
                    "DDR Capacity(MB)": "43693",
                    "DDR Usage Rate(%)": "3",
                    "DDR Hugepages Total(page)": "0",
                    "DDR Hugepages Usage Rate(%)": "0",
                    "Aicore Usage Rate(%)": "0",
                    "Aicpu Usage Rate(%)": "0",
                    "Ctrlcpu Usage Rate(%)": "3",
                    "Vectorcore Usage Rate(%)": "0",
                    "DDR Bandwidth Usage Rate(%)": "51",
                    "DVPP VDEC Usage Rate(%)": "0",
                    "DVPP VPC Usage Rate(%)": "0",
                    "DVPP VENC Usage Rate(%)": "0",
                    "DVPP JPEGE Usage Rate(%)": "0",
                    "DVPP JPEGD Usage Rate(%)": "0",
                },
            },
        ),
    ],
)
def test__parse_npu_chips_usages_info(output, expected):
    actual = _parse_npu_chips_usages_info(output)
    assert actual == expected, f"Expected {expected}, but got {actual}"


@pytest.mark.parametrize(
    "output, expected",
    [
        # 910B, network is down.
        (
            """
link status: DOWN
            """,
            {
                "link status": "DOWN",
            },
        ),
        # 910B, default gateway and interface.
        (
            """

    default gateway:192.168.6.1, Iface:eth0

            """,
            {
                "default gateway": "192.168.6.1",
                "Iface": "eth0",
            },
        ),
    ],
)
def test__parse_line_to_dict(output, expected):
    actual = _parse_line_to_dict(output)
    assert actual == expected, f"Expected {expected}, but got {actual}"
