import argparse
import logging
import sys

from gpustack.cmd.start import get_gpustack_env
from gpustack.logging import setup_logging
from gpustack.worker.tools_manager import ToolsManager

logger = logging.getLogger(__name__)


def setup_download_tools_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "download-tools",
        help="Download dependency tools.",
        description="Download dependency tools.",
    )

    parser.add_argument(
        "--tools-download-base-url",
        type=str,
        help="Base URL to download dependency tools.",
        default=get_gpustack_env("TOOLS_DOWNLOAD_BASE_URL"),
    )
    parser.add_argument(
        "--save-archive",
        type=str,
        help="Path to save downloaded tools as a tar archive.",
        default=get_gpustack_env("SAVE_ARCHIVE"),
    )
    parser.add_argument(
        "--load-archive",
        type=str,
        help="Path to load downloaded tools from a tar archive, instead of downloading.",
        default=get_gpustack_env("LOAD_ARCHIVE"),
    )
    parser.add_argument(
        "--system",
        type=str,
        help="Operating system to download tools for. Default is the current OS. (e.g. linux, windows, darwin)",
        default=get_gpustack_env("SYSTEM"),
    )
    parser.add_argument(
        "--arch",
        type=str,
        help="Architecture to download tools for. Default is the current architecture. (e.g. amd64, arm64)",
        default=get_gpustack_env("ARCH"),
    )

    parser.set_defaults(func=run)


def run(args):
    setup_logging(False)
    try:
        verify(args)

        tools_download_base_url = None
        if args.tools_download_base_url:
            tools_download_base_url = args.tools_download_base_url.rstrip("/")
        tools_manager = ToolsManager(
            tools_download_base_url=tools_download_base_url,
            system=args.system,
            arch=args.arch,
        )

        if args.load_archive:
            tools_manager.load_archive(args.load_archive)
            return

        tools_manager.remove_cached_tools()
        tools_manager.prepare_tools()

        if args.save_archive:
            tools_manager.save_archive(args.save_archive)
    except KeyboardInterrupt:
        pass
    except ValueError as e:
        logger.fatal(e)
        sys.exit(1)
    except Exception as e:
        logger.fatal(f"Failed to download tools: {e}")
        sys.exit(1)


def verify(args):
    if args.load_archive and args.tools_download_base_url:
        raise ValueError(
            "Cannot specify both load-archive and tools-download-base-url."
        )

    if args.save_archive and args.load_archive:
        raise ValueError("Cannot specify both save-archive and load-archive.")
