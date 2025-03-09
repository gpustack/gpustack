BYTES_TO_KIB = 1024
BYTES_TO_MIB = 1024 * 1024
BYTES_TO_GIB = 1024 * 1024 * 1024


def byte_to_unit(byte: int, unit: int) -> float:
    return round(byte / unit, 2)


def byte_to_kib(byte: int) -> float:
    return byte_to_unit(byte, BYTES_TO_KIB)


def byte_to_mib(byte: int) -> float:
    return byte_to_unit(byte, BYTES_TO_MIB)


def byte_to_gib(byte: int) -> float:
    return byte_to_unit(byte, BYTES_TO_GIB)
