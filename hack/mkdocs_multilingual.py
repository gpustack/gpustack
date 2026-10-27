"""Keep ordinary MkDocs and mike builds consistent with the multilingual builder."""

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.livereload import LiveReloadServer
from mkdocs.plugins import event_priority


_ROOT_BUILDS = {}


def is_prepared(config):
    extra = config["extra"]
    return extra.get("docs_i18n_prepared") or (extra.get("docs_i18n") or {}).get(
        "prepared"
    )


def on_serve(
    server: LiveReloadServer, *, config: MkDocsConfig, builder: Callable
) -> LiveReloadServer:
    """Watch translated pages and labels with MkDocs' existing rebuild callback."""
    if not is_prepared(config):
        root = Path(__file__).resolve().parents[1]
        server.watch(str(root / "docs-i18n"))
    return server


def on_pre_build(config):
    """Keep the native English pass away from the previously published site."""
    if is_prepared(config):
        return
    root = Path(__file__).resolve().parents[1]
    temporary = TemporaryDirectory(prefix=".docs-root-", dir=root)
    _ROOT_BUILDS[id(config)] = (config, temporary, Path(config["site_dir"]))
    config["site_dir"] = str(Path(temporary.name) / "initial")


def finish_root_build(config):
    active = _ROOT_BUILDS.pop(id(config), None)
    if active:
        _, temporary, destination = active
        config["site_dir"] = str(destination)
        temporary.cleanup()


def on_build_error(error):
    """Restore output configuration and discard an unsuccessful native pass."""
    for config, _, _ in list(_ROOT_BUILDS.values()):
        finish_root_build(config)


@event_priority(-200)
def on_post_build(config):
    """Assemble all languages once, after the root build has resolved its version."""
    extra = config["extra"]
    settings = extra.get("docs_i18n") or {}
    if is_prepared(config):
        return
    root = Path(__file__).resolve().parents[1]
    helpers = str(root / "hack")
    spec = importlib.util.spec_from_file_location(
        "gpustack_documentation_builder", root / "hack/build_docs.py"
    )
    builder = importlib.util.module_from_spec(spec)
    sys.path.insert(0, helpers)
    try:
        spec.loader.exec_module(builder)
    finally:
        sys.path.remove(helpers)
    destination = _ROOT_BUILDS.get(id(config), (None, None, Path(config["site_dir"])))[
        2
    ]
    try:
        builder.build_docs(
            root,
            offline=bool(settings.get("offline", extra.get("offline"))),
            destination=destination,
            site_url=config["site_url"],
            stamp=extra.get("docs_stamp"),
            strict=config["strict"],
        )
    finally:
        finish_root_build(config)
