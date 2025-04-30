from typing import Any

from gpustack.detectors.npu_smi.npu_smi import (
    _parse_all_npu_chips_mapping,
    _parse_npu_chips_common_info,
    _parse_npu_chips_usages_info,
    _parse_line_to_dict,
)


class _TestCase:
    input: str
    expected: Any

    def __init__(self, input: str, expected: Any):
        self.input = input
        self.expected = expected


def test__parse_all_npu_chips_mapping():
    cases = [
        _TestCase(
            input="""
NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
0                              0                              0                              Ascend xxx
1                              0                              1                              Ascend xxx
2                              0                              2                              Ascend xxx
3                              0                              3                              Ascend xxx
""",
            expected={
                0: (0, 0),
                1: (1, 0),
                2: (2, 0),
                3: (3, 0),
            },
        ),
        _TestCase(
            input="""
NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
0                              0                              0                              Ascend xxx
0                              1                              -                              MCU
1                              0                              1                              Ascend xxx
1                              1                              -                              MCU
""",
            expected={
                0: (0, 0),
                1: (1, 0),
            },
        ),
    ]

    for case in cases:
        actual = _parse_all_npu_chips_mapping(case.input)
        assert actual == case.expected, f"Expected {case.expected}, but got {actual}"


def test__parse_npu_chips_common_info():
    cases = [
        _TestCase(
            input="""
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
""",
            expected={
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
        _TestCase(
            input="""
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

    Chip ID                        : 1
    Memory Usage Rate(%)           : 5
    HBM Usage Rate(%)              : 0
    Aicore Usage Rate(%)           : 0
    Aicore Freq(MHZ)               : 1000
    Aicore curFreq(MHZ)            : 1000
    Aicore Count                   : 32
    Temperature(C)                 : 35
    NPU Real-time Power(W)         : 74.0
            """,
            expected={
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
                1: {
                    "Memory Usage Rate(%)": "5",
                    "HBM Usage Rate(%)": "0",
                    "Aicore Usage Rate(%)": "0",
                    "Aicore Freq(MHZ)": "1000",
                    "Aicore curFreq(MHZ)": "1000",
                    "Aicore Count": "32",
                    "Temperature(C)": "35",
                    "NPU Real-time Power(W)": "74.0",
                },
            },
        ),
    ]

    for case in cases:
        actual = _parse_npu_chips_common_info(case.input)
        assert actual == case.expected, f"Expected {case.expected}, but got {actual}"


def test__parse_npu_chips_usages_info():
    cases = [
        _TestCase(
            input="""
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
            expected={
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
        _TestCase(
            input="""
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

    DDR Capacity(MB)               : 15171
    DDR Usage Rate(%)              : 5
    DDR Hugepages Total(page)      : 0
    DDR Hugepages Usage Rate(%)    : 0
    HBM Capacity(MB)               : 32768
    HBM Usage Rate(%)              : 5
    Aicore Usage Rate(%)           : 0
    Aicpu Usage Rate(%)            : 0
    Ctrlcpu Usage Rate(%)          : 9
    DDR Bandwidth Usage Rate(%)    : 0
    HBM Bandwidth Usage Rate(%)    : 0
    Chip ID                        : 1
            """,
            expected={
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
                1: {
                    "DDR Capacity(MB)": "15171",
                    "DDR Usage Rate(%)": "5",
                    "DDR Hugepages Total(page)": "0",
                    "DDR Hugepages Usage Rate(%)": "0",
                    "HBM Capacity(MB)": "32768",
                    "HBM Usage Rate(%)": "5",
                    "Aicore Usage Rate(%)": "0",
                    "Aicpu Usage Rate(%)": "0",
                    "Ctrlcpu Usage Rate(%)": "9",
                    "DDR Bandwidth Usage Rate(%)": "0",
                    "HBM Bandwidth Usage Rate(%)": "0",
                },
            },
        ),
    ]

    for case in cases:
        actual = _parse_npu_chips_usages_info(case.input)
        assert actual == case.expected, f"Expected {case.expected}, but got {actual}"


def test__parse_line_to_dict():
    cases = [
        _TestCase(
            input="""
link status: DOWN
                """,
            expected={
                "link status": "DOWN",
            },
        ),
        _TestCase(
            input="""

    default gateway:192.168.6.1, Iface:eth0

            """,
            expected={
                "default gateway": "192.168.6.1",
                "Iface": "eth0",
            },
        ),
    ]

    for case in cases:
        actual = _parse_line_to_dict(case.input)
        assert actual == case.expected, f"Expected {case.expected}, but got {actual}"
