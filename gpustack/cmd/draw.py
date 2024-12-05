import argparse
import sys
from gpustack.cli.draw import DrawCLIClient, parse_arguments


def setup_draw_cmd(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "draw",
        help="Generate an image with a diffusion model.",
        description="Generate an image with a diffusion model.",
    )

    parser.add_argument(
        "model",
        type=str,
        help="The model to use for image generation. This can be either a GPUStack model name or a Hugging Face GGUF model reference (e.g., 'hf.co/gpustack/stable-diffusion-v3-5-large-turbo-GGUF:stable-diffusion-v3-5-large-turbo-Q4_0.gguf').",
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The text prompt to use for generating the image.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="512x512",
        help="The size of the generated image in 'widthxheight' format. Default is 512x512.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        help="The sampler algorithm for image generation. Options include 'euler_a', 'euler', 'heun', 'dpm2', 'dpm++2s_a', 'dpm++2m', 'dpm++2mv2', 'ipndm', 'ipndm_v', and 'lcm'. Default is 'euler'.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=None,
        help="The number of sampling steps to perform. Higher values may improve image quality at the cost of longer processing time.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="The scale for classifier-free guidance. A higher value increases adherence to the prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not specified, a random seed will be used.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="A negative prompt to specify what the image should avoid.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the generated image file. If not specified, the image will be saved with a generated filename in the current directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated image in the default image viewer",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

    parser.set_defaults(func=run)


def run(args):
    try:
        cfg = parse_arguments(args)
        DrawCLIClient(cfg).run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
        sys.exit(1)
