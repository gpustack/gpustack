"""The GPUStack image ``list-images`` reports for this build.

It has to name an image that was actually published: whatever a worker would
pull, i.e. the repository from ``--image-repo`` at the version
``resolve_version_info()`` reports, which a plugin may override.
"""

import gpustack.cmd.images as images
import gpustack.extension as extension


class _Plugin(extension.Plugin):
    @classmethod
    def get_version_info(cls):
        return "v2.2.3-repack1", "repackcommit"


def _self_image(monkeypatch, **env):
    for key in ("GPUSTACK_IMAGE_REPO", "GPUSTACK_IMAGE_NAME_OVERRIDE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    appended = []
    monkeypatch.setattr(images, "append_images", lambda *imgs: appended.extend(imgs))
    images._append_self_image()
    assert len(appended) == 1
    return appended[0]


def _with_plugin(monkeypatch):
    monkeypatch.setattr(
        extension, "iter_plugin_classes", lambda: iter([("test", _Plugin)])
    )


def test_defaults_to_core_repo_and_version(monkeypatch):
    monkeypatch.setattr(extension, "iter_plugin_classes", lambda: iter([]))
    monkeypatch.setattr("gpustack.__version__", "v2.2.3")

    assert _self_image(monkeypatch) == "gpustack/gpustack:v2.2.3"


def test_unreleased_version_maps_to_dev_tag(monkeypatch):
    monkeypatch.setattr(extension, "iter_plugin_classes", lambda: iter([]))
    monkeypatch.setattr("gpustack.__version__", "v0.0.0")

    assert _self_image(monkeypatch) == "gpustack/gpustack:dev"


def test_takes_version_from_plugin(monkeypatch):
    # A repackaged distribution ships its own repository and version; core's
    # baked-in __version__ names an image that was never published for it.
    _with_plugin(monkeypatch)
    monkeypatch.setattr("gpustack.__version__", "v2.2-dev")

    image = _self_image(monkeypatch, GPUSTACK_IMAGE_REPO="example/gpustack")

    assert image == "example/gpustack:v2.2.3-repack1"


def test_ignores_an_image_name_override(monkeypatch):
    # An override is a full reference that may carry its own registry, which
    # save-images / copy-images would prefix with --source a second time.
    _with_plugin(monkeypatch)

    image = _self_image(
        monkeypatch,
        GPUSTACK_IMAGE_REPO="example/gpustack",
        GPUSTACK_IMAGE_NAME_OVERRIDE="registry.example.com/mine/gpustack:custom",
    )

    assert image == "example/gpustack:v2.2.3-repack1"
