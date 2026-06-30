import argparse
import logging

from gpustack.extension import resolve_version_info

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


def run(args):
    version, git_commit = resolve_version_info()
    if args.short:
        print(version)
    else:
        print(f"{version} ({git_commit})")
