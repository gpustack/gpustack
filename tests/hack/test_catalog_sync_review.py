import base64
import copy
import os
import signal
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from hack.catalog_sync import hubs, publish, run, supervise, validate
from hack.catalog_sync.models import CATALOGS, digest


@pytest.mark.parametrize(
    "body",
    [
        None,
        {"Code": 200},
        {"Code": 200, "Data": None},
        {"Code": 200, "Data": []},
        {"Code": 200, "Data": {}},
        {"Code": 200, "Data": {"Files": None}},
        {"Code": 200, "Data": {"Files": {}}},
        {"Code": 200, "Data": {"Files": [None]}},
        {"Code": 200, "Data": {"Files": [{"Size": 12}]}},
    ],
)
def test_malformed_modelscope_listing_is_an_explicit_error(body):
    client = Mock()
    client.json.return_value = body
    with pytest.raises(ValueError, match="ModelScope"):
        hubs.collect(client, "model_scope", "org/model")


def test_empty_modelscope_listing_is_distinct_from_malformed_response():
    client = Mock()
    client.json.return_value = {"Code": 200, "Data": {"Files": []}}
    assert hubs._manifest(client, "model_scope", "org/model", None) == (
        "master",
        {},
        None,
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "[",
        "draft_models: []",
        "model_sets: null",
        "model_sets: {}",
        "model_sets: [null]",
        "model_sets: [{description: test}]",
        "model_sets: [{name: null}]",
        "model_sets: [{name: test, specs: null}]",
        "model_sets: [{name: test, specs: [null]}]",
    ],
)
def test_bad_catalog_structure_fails_at_read_and_render(tmp_path, text):
    for path in CATALOGS.values():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    with pytest.raises(ValueError, match="[Cc]atalog"):
        run._existing(tmp_path)
    with pytest.raises(ValueError, match="[Cc]atalog"):
        validate.replace_cards(text, [])


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "git_head", lambda root: "a" * 40)
    path = CATALOGS["huggingface"]
    result = {
        "request": {"base_sha": "a" * 40},
        "files": {path: base64.b64encode(b"model_sets: []\n").decode()},
        "before": {path: None},
        "body": "Reviewed public evidence",
    }
    result["sha256"] = digest(result)
    return result


class FakeGitHub:
    def __init__(self, changed_on=None):
        self.calls = []
        self.reads = 0
        self.changed_on = changed_on

    def call(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path == "git/ref/heads/main":
            self.reads += 1
            return {
                "object": {"sha": ("b" if self.reads == self.changed_on else "a") * 40}
            }
        if method == "GET":
            return {"tree": {"sha": "base-tree"}}
        return {"sha": path, "html_url": "https://github.com/org/repo/pull/1"}


def test_publish_uses_frozen_parent_and_only_validated_files(tmp_path, bundle):
    api = FakeGitHub()
    publish.publish(
        tmp_path, bundle, bundle["sha256"], "automation/model-catalog-1-1", api
    )
    commit = next(body for method, path, body in api.calls if path == "git/commits")
    assert commit["parents"] == [bundle["request"]["base_sha"]]
    tree = next(body for method, path, body in api.calls if path == "git/trees")
    assert tree["base_tree"] == "base-tree"
    assert [item["path"] for item in tree["tree"]] == list(bundle["files"])
    blob = next(body for method, path, body in api.calls if path == "git/blobs")
    assert blob == {
        "content": next(iter(bundle["files"].values())),
        "encoding": "base64",
    }
    assert api.calls[-1][1] == "pulls" and api.calls[-1][2]["draft"] is True
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("changed_on", [1, 2, 3])
def test_publish_stops_if_base_advances(tmp_path, bundle, changed_on):
    api = FakeGitHub(changed_on)
    with pytest.raises(ValueError, match="main changed"):
        publish.publish(
            tmp_path, bundle, bundle["sha256"], "automation/model-catalog-1-1", api
        )
    assert not any(path == "pulls" for method, path, body in api.calls)
    if changed_on == 1:
        assert all(method == "GET" for method, path, body in api.calls)


def test_publish_checks_digest_before_api_calls(tmp_path, bundle):
    api = FakeGitHub()
    bundle["body"] = "Tampered body"
    with pytest.raises(ValueError, match="digest"):
        publish.publish(
            tmp_path, bundle, bundle["sha256"], "automation/model-catalog-1-1", api
        )
    assert not api.calls


def test_verify_does_not_apply_files(tmp_path, bundle):
    run.apply(tmp_path, copy.deepcopy(bundle), bundle["sha256"], write=False)
    assert not list(tmp_path.iterdir())


def test_supervisor_preserves_exit_status_and_output():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            supervise.__file__,
            sys.executable,
            "-c",
            "print('research'); raise SystemExit(7)",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 7 and result.stdout.strip() == "research"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux child adoption uses procfs")
@pytest.mark.parametrize("terminate", [False, True])
def test_linux_supervisor_reaps_detached_descendants(tmp_path, terminate):
    pid_file = tmp_path / "child.pid"
    script = (
        "import os, pathlib, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        " os.setsid()\n"
        " if os.fork() != 0: os._exit(0)\n"
        " pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        " time.sleep(60)\n"
        "else:\n"
        " while not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.01)\n"
        + (" time.sleep(60)\n" if terminate else "")
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            supervise.__file__,
            sys.executable,
            "-c",
            script,
            str(pid_file),
        ],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists()
        child = int(pid_file.read_text())
        if terminate:
            process.terminate()
        assert process.wait(timeout=10) == (143 if terminate else 0)
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
    finally:
        for pid in ([int(pid_file.read_text())] if pid_file.exists() else []) + [
            process.pid
        ]:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
