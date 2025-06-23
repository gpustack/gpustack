from __future__ import annotations

from typing import List


def get_attr(obj, path: str | List[str]):
    """
    Get an attribute from an object using a dot-separated path.

    :param obj: The object to get the attribute from.
    :param path: A string representing the path to the attribute (e.g., 'a.b.c', 'a.b.0.c').
    :return: The value of the attribute.
    """

    if obj is None:
        return None

    if isinstance(path, str):
        return get_attr(obj, path.split('.'))

    p = path[0]
    if (p.isdigit() or p == "-1") and isinstance(obj, list):
        obj = obj[int(p)]
    elif isinstance(obj, dict):
        obj = obj.get(p, None)
    else:
        obj = getattr(obj, p, None)
    return get_attr(obj, path[1:]) if len(path) > 1 else obj


def set_attr(obj, path: str | List[str], value):
    """
    Set an attribute on an object using a dot-separated path.

    :param obj: The object to set the attribute on.
    :param path: A string representing the path to the attribute (e.g., 'a.b.c', 'a.b.0.c').
    :param value: The value to set the attribute to.
    """

    if isinstance(path, str):
        set_attr(obj, path.split('.'), value)
        return

    if obj is None:
        return None

    if len(path) > 1:
        obj = get_attr(obj, path[:-1])

    if obj is None:
        return None

    p = path[-1]
    if (p.isdigit() or p == "-1") and isinstance(obj, list):
        while len(obj) <= int(p):
            obj.append(None)
        obj[int(p)] = value
    elif isinstance(obj, dict):
        obj[p] = value
    elif hasattr(obj, p):
        setattr(obj, p, value)
