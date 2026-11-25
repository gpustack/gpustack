import json
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


BOOTSTRAP_SCRIPT = Path(__file__).resolve().parents[2] / "gpustack/k8s/bootstrap.sh"
RENDERED_AT = "20261125120000000000"
MOCK_COMMANDS = r"""
set -euo pipefail
NAMESPACE=gpustack-system
BOOTSTRAP_CONFIGMAP=gpustack-bootstrap
waits=0

log() { :; }
exit_if_overtaken() { :; }

kubectl() {
  [[ "$1 $2" == 'get pods' ]] || return 1
  if (( waits == 0 )); then
    printf '%s\n' "$initial_pods"
  else
    printf '%s\n' "$remaining_pods"
  fi
}

sleep() {
  printf 'sleep %s\n' "$1"
  (( waits += 1 ))
}
"""


def pod(job: str, state: dict) -> dict:
    return {
        "metadata": {
            "name": f"{job}-abcde",
            "ownerReferences": [{"name": job, "kind": "Job", "controller": True}],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "install", "state": state}],
        },
    }


def test_waits_for_running_install_in_another_bootstrap():
    bash = shutil.which("bash")
    if bash is None or shutil.which("jq") is None:
        pytest.skip("bash and jq are required")

    script = BOOTSTRAP_SCRIPT.read_text(encoding="utf-8")
    start = script.index("wait_for_previous_bootstrap() {")
    helper = script[start : script.index("\n}\n", start) + 2]

    running = {"running": {"startedAt": "2026-11-25T11:00:00Z"}}
    remaining = [
        pod(f"gpustack-bootstrap-{RENDERED_AT}", running),
        pod("gpustack-bootstrap-custom", running),
        pod("gpustack-bootstrap-20261125100000000000", {"terminated": {"exitCode": 1}}),
    ]
    initial = [pod("gpustack-bootstrap-20261125110000000000", running), *remaining]
    result = subprocess.run(
        [bash, "-s"],
        input="\n".join(
            (
                MOCK_COMMANDS,
                helper,
                f"RENDERED_AT={RENDERED_AT}",
                f"initial_pods={shlex.quote(json.dumps({'items': initial}))}",
                f"remaining_pods={shlex.quote(json.dumps({'items': remaining}))}",
                "wait_for_previous_bootstrap\n",
            )
        ),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )

    assert result.stdout.splitlines() == ["sleep 5"]
