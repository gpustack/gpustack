from packaging import version


def in_range(version_str: str, range_str: str) -> bool:
    conditions = [cond.strip() for cond in range_str.split(",")]
    ver = version.parse(version_str)
    for cond in conditions:
        if cond.startswith(">="):
            if ver < version.parse(cond[2:]):
                return False
        elif cond.startswith("<="):
            if ver > version.parse(cond[2:]):
                return False
        elif cond.startswith(">"):
            if ver <= version.parse(cond[1:]):
                return False
        elif cond.startswith("<"):
            if ver >= version.parse(cond[1:]):
                return False
        else:  # exact match
            if ver != version.parse(cond):
                return False
    return True
