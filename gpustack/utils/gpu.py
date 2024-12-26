import re

from typing import Tuple

pattern = r"^(?P<worker_name>.+):(?P<device>cuda|npu|rocm|musa|mps):(?P<gpu_index>\d+)$"


def parse_gpu_id(input: str) -> Tuple[bool, dict]:
    """
    Parse the input string to check if it matches the format worker_name:device:gpu_index.

    Args:
        input_string (str): The input string to parse.

    Returns:
        tuple: (bool, dict)
               - If matched, the first value is True, and the second value is a dictionary
                 containing worker_name, device, and gpu_index.
               - If not matched, the first value is False, and the second value is None.
    """

    match = re.match(pattern, input)
    if match:
        return True, match.groupdict()
    return False, None
