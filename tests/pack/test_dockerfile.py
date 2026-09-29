import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "pack" / "Dockerfile"


def test_the_image_excludes_mig_declarations_from_mirroring():
    # Mirrored deployment copies the worker container's env into every model
    # pod it deploys. NVIDIA_MIG_CONFIG_DEVICES and NVIDIA_MIG_MONITOR_DEVICES
    # (declared for the worker's MIG visibility) are only legal in a
    # privileged container: the NVIDIA runtime's CDI modifier rejects a
    # non-privileged model pod carrying them. The image must name both in the
    # mirrored-environment ignore list, with ";" — the separator the runtime
    # parses this variable with.
    content = DOCKERFILE.read_text(encoding="utf-8")
    match = re.search(
        r'GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_ENVIRONMENTS="([^"]*)"',
        content,
    )
    assert match is not None
    ignored = match.group(1).split(";")
    assert "NVIDIA_MIG_CONFIG_DEVICES" in ignored
    assert "NVIDIA_MIG_MONITOR_DEVICES" in ignored
