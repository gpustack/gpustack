import argparse
import logging

from gpustack.cmd.start import get_gpustack_env
from gpustack.worker.tools_manager import ToolsManager

logger = logging.getLogger(__name__)


def setup_download_tools_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "download-tools",
    )

    parser.add_argument(
        "--tools-download-base-url",
        type=str,
        help="Base URL to download dependency tools.",
        default=get_gpustack_env("TOOLS_DOWNLOAD_BASE_URL"),
    )

    parser.set_defaults(func=run)


def run(args):
    try:
        tools_download_base_url = None
        if args.tools_download_base_url:
            tools_download_base_url = args.tools_download_base_url.rstrip("/")
        tools_manager = ToolsManager(
            tools_download_base_url=tools_download_base_url,
        )
        tools_manager.prepare_tools()
    except Exception as e:
        logger.fatal(f"Failed to download tools: {e}")
