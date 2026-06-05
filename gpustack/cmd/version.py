import argparse
import logging

from gpustack import __version__, __git_commit__
from gpustack.extension import Plugin, iter_plugin_classes

logger = logging.getLogger(__name__)


def setup_version_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "version",
        help="Print version.",
        description="Print version.",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Print without commit hash.",
        default=False,
    )
    parser.set_defaults(func=run)


def _resolve_version_info() -> tuple[str, str]:
    """Let plugins override the reported version. First non-None wins; on
    any failure, fall back to core's baked-in values."""
    for name, plugin_class in iter_plugin_classes():
        if not (isinstance(plugin_class, type) and issubclass(plugin_class, Plugin)):
            continue
        try:
            info = plugin_class.get_version_info()
        except Exception:
            logger.debug(
                "Failed to read version info from plugin '%s'", name, exc_info=True
            )
            continue
        if info is not None:
            return info
    return __version__, __git_commit__


def run(args):
    version, git_commit = _resolve_version_info()
    if args.short:
        print(version)
    else:
        print(f"{version} ({git_commit})")
