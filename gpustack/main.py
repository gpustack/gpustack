import argparse
from multiprocessing import freeze_support

from gpustack.cmd import setup_start_cmd
from gpustack.cmd.chat import setup_chat_cmd
from gpustack.cmd.download_tools import setup_download_tools_cmd
from gpustack.cmd.draw import setup_draw_cmd
from gpustack.cmd.reset_admin_password import setup_reset_admin_password_cmd
from gpustack.cmd.version import setup_version_cmd


def main():
    parser = argparse.ArgumentParser(
        description="GPUStack",
        conflict_handler="resolve",
        add_help=True,
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=55, indent_increment=2, width=200
        ),
    )
    subparsers = parser.add_subparsers(
        help="sub-command help", metavar='{start,chat,download-tools,version}'
    )

    setup_start_cmd(subparsers)
    setup_chat_cmd(subparsers)
    setup_draw_cmd(subparsers)
    setup_download_tools_cmd(subparsers)
    setup_version_cmd(subparsers)
    setup_reset_admin_password_cmd(subparsers)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    # When using multiprocessing with 'spawn' mode, freeze_support() must be called in the main module
    # to ensure the main process environment is correctly initialized when child processes are spawned.
    # See: https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
    freeze_support()
    main()
