"""What ``resolve_version_info()`` reports, and the surfaces that report it.

A repackaged distribution ships its own build string through a plugin, and
every version an operator sees has to carry it. A module-level ``from gpustack
import __version__`` is bound at import time and cannot, so each surface that
reports a version is covered here.
"""

import pytest

import gpustack.extension as extension
from gpustack.config.config import Config
from gpustack.extension import resolve_version_info

PLUGIN_VERSION = "2.2.3-repack1"
PLUGIN_COMMIT = "repackcommit"


class _Plugin(extension.Plugin):
    @classmethod
    def get_version_info(cls):
        return PLUGIN_VERSION, PLUGIN_COMMIT


class _SilentPlugin(extension.Plugin):
    """A plugin that defers to core, as the base class does."""


class _BrokenPlugin(extension.Plugin):
    @classmethod
    def get_version_info(cls):
        raise RuntimeError("plugin is broken")


@pytest.fixture(autouse=True)
def clear_version_cache():
    resolve_version_info.cache_clear()
    yield
    resolve_version_info.cache_clear()


def _with_plugins(monkeypatch, *plugins):
    monkeypatch.setattr(
        extension,
        "iter_plugin_classes",
        lambda: iter([(plugin.__name__, plugin) for plugin in plugins]),
    )


class TestResolveVersionInfo:
    def test_core_values_without_a_plugin(self, monkeypatch):
        _with_plugins(monkeypatch)
        monkeypatch.setattr("gpustack.__version__", "2.2.3")
        monkeypatch.setattr("gpustack.__git_commit__", "corecommit")

        assert resolve_version_info() == ("2.2.3", "corecommit")

    def test_a_plugin_overrides_both_values(self, monkeypatch):
        _with_plugins(monkeypatch, _Plugin)
        monkeypatch.setattr("gpustack.__version__", "2.2.3")

        assert resolve_version_info() == (PLUGIN_VERSION, PLUGIN_COMMIT)

    def test_a_plugin_that_defers_is_skipped(self, monkeypatch):
        _with_plugins(monkeypatch, _SilentPlugin, _Plugin)

        assert resolve_version_info() == (PLUGIN_VERSION, PLUGIN_COMMIT)

    def test_a_broken_plugin_does_not_break_version_reporting(self, monkeypatch):
        _with_plugins(monkeypatch, _BrokenPlugin, _Plugin)

        assert resolve_version_info() == (PLUGIN_VERSION, PLUGIN_COMMIT)

    def test_the_entry_point_scan_happens_once(self, monkeypatch):
        """The scan reads package metadata off disk and every request-path
        caller would otherwise pay for it.
        """
        scans = []

        def _iter():
            scans.append(1)
            return iter([("test", _Plugin)])

        monkeypatch.setattr(extension, "iter_plugin_classes", _iter)

        assert resolve_version_info() == resolve_version_info()
        assert len(scans) == 1


class TestReportingSurfaces:
    @pytest.mark.asyncio
    async def test_the_version_endpoint(self, monkeypatch):
        from gpustack.routes import probes

        _with_plugins(monkeypatch, _Plugin)

        assert await probes.version() == {
            "version": PLUGIN_VERSION,
            "git_commit": PLUGIN_COMMIT,
        }

    def test_the_deployment_document_header(self, monkeypatch):
        from gpustack.schemas.deployment_document import dump_deployments

        _with_plugins(monkeypatch, _Plugin)

        document = dump_deployments([], set(), {})

        assert document.startswith(f"# Exported from GPUStack v{PLUGIN_VERSION} ")


class TestWorkerVersionCheck:
    """The worker compares the version it reports against the server's.

    Both sides answer with the same helper, so the comparison is between two
    plugin builds. ``is_worker_version_compatible`` short-circuits on the
    unreleased ``0.0.0``, so a build compared against core's own value would
    pass whatever the server reports.
    """

    @staticmethod
    def _manager(monkeypatch, temp_dir, server_version):
        from gpustack.worker import worker_manager as wm

        async def _skip_tls_bootstrap(server_url):
            return None

        async def _server_version(self):
            return {"version": server_version, "git_commit": "servercommit"}

        monkeypatch.setattr(wm, "ensure_server_tls_trust", _skip_tls_bootstrap)
        monkeypatch.setattr(wm.WorkerManager, "_fetch_server_version", _server_version)

        cfg = Config(
            token="test",
            jwt_secret_key="test",
            data_dir=temp_dir,
            server_url="https://gpustack.internal",
        )
        return wm.WorkerManager(cfg=cfg, is_embedded=False, collector=None)

    @pytest.mark.asyncio
    async def test_a_matching_plugin_build_passes(self, monkeypatch, temp_dir, caplog):
        _with_plugins(monkeypatch, _Plugin)
        manager = self._manager(monkeypatch, temp_dir, PLUGIN_VERSION)

        with caplog.at_level("INFO"):
            await manager.check_server_version()

        assert f"worker {PLUGIN_VERSION} matches server {PLUGIN_VERSION}" in caplog.text

    @pytest.mark.asyncio
    async def test_a_differing_server_build_warns(self, monkeypatch, temp_dir, caplog):
        _with_plugins(monkeypatch, _Plugin)
        manager = self._manager(monkeypatch, temp_dir, "2.2.4-repack1")

        with caplog.at_level("WARNING"):
            await manager.check_server_version()

        assert "Version mismatch detected" in caplog.text
        assert f"Worker version: {PLUGIN_VERSION}" in caplog.text
