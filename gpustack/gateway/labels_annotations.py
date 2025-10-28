from typing import Dict, Optional

managed_labels = {"gpustack.ai/managed": "true"}


def match_labels(
    obj_labels: Optional[Dict[str, str]], expected_labels: Dict[str, str]
) -> bool:
    if not obj_labels:
        return False
    return all(obj_labels.get(k) == v for k, v in expected_labels.items())
