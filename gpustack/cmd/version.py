import argparse
from gpustack import __version__, __git_commit__


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
    if args.short:
        print(__version__)
    else:
        print(f"{__version__} ({__git_commit__})")
