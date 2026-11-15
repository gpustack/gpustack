"""Unit tests for downloader file selection and ordering.

The first entry of the returned paths feeds the instance resolved_path;
main weights must always precede mmproj projector files.
"""

import os


class _NullLock:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_hf_download_file_keeps_main_file_first(monkeypatch, tmp_path):
    from gpustack.worker import downloaders

    # Simulate a user pattern that matches both the projector and the main
    # weights, with the projector leading (alphabetical sort would keep it).
    matching = [
        "mmproj-vllm-d3-mini-Q8_0.gguf",
        "vllm-d3-mini-BF16.gguf",
    ]
    monkeypatch.setattr(downloaders, 'HeartbeatSoftFileLock', _NullLock)
    monkeypatch.setattr(
        downloaders,
        'match_hugging_face_files',
        lambda repo_id, filename, extra_filename, token: matching,
    )

    def fake_download(repo_id, filename, token, local_dir, tqdm_class):
        path = os.path.join(str(local_dir), filename)
        with open(path, 'wb') as f:
            f.write(b'0' * 16)
        return path

    monkeypatch.setattr(
        downloaders,
        'hf_hub_download',
        fake_download,
    )

    got = downloaders.HfDownloader.download_file(
        repo_id='ggml-org/vllm-d3-mini-GGUF',
        filename='*.gguf',
        local_dir=str(tmp_path),
    )

    assert got == [
        str(tmp_path) + "/vllm-d3-mini-BF16.gguf",
        str(tmp_path) + "/mmproj-vllm-d3-mini-Q8_0.gguf",
    ]
