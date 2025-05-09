import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
import shutil
import stat
import subprocess
import time
from typing import Optional, Dict
import zipfile
import requests

from gpustack.schemas.models import BackendEnum
from gpustack.utils.command import get_versioned_command
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils import platform, envs

logger = logging.getLogger(__name__)


BUILTIN_LLAMA_BOX_VERSION = "v0.0.139"
BUILTIN_GGUF_PARSER_VERSION = "v0.14.1"
BUILTIN_RAY_VERSION = "2.43.0"


class ToolsManager:
    """
    ToolsManager is responsible for managing prebuilt binary tools including the following:
    - `fastfetch`
    - `gguf-parser`
    - `llama-box`

    """

    def __init__(
        self,
        tools_download_base_url: str = None,
        bin_dir: Optional[str] = None,
        pipx_path: Optional[str] = None,
        device: Optional[str] = None,
        system: Optional[str] = None,
        arch: Optional[str] = None,
    ):
        with pkg_resources.path("gpustack.third_party", "bin") as third_party_bin_path:
            self.third_party_bin_path: Path = third_party_bin_path
            self.versions_file = third_party_bin_path.joinpath("versions.json")

            self._current_tools_version = {}
            if os.path.exists(self.versions_file):
                try:
                    with open(self.versions_file, 'r', encoding='utf-8') as file:
                        self._current_tools_version = json.load(file)
                except Exception as e:
                    logger.warning(f"Failed to load versions.json: {e}")

        self._os = system if system else platform.system()
        self._arch = arch if arch else platform.arch()
        self._device = device if device else platform.device()
        if self._device == platform.DeviceTypeEnum.CUDA.value:
            self._llama_box_cuda_version = self._get_llama_box_cuda_version()
        self._download_base_url = tools_download_base_url
        self._bin_dir = bin_dir
        self._pipx_path = pipx_path

    def _check_and_set_download_base_url(self):
        urls = [
            "https://github.com",
            "https://gpustack-1303613262.cos.ap-guangzhou.myqcloud.com",
        ]

        test_path = f"/gpustack/gguf-parser-go/releases/download/{BUILTIN_GGUF_PARSER_VERSION}/gguf-parser-linux-amd64"
        test_size = 512 * 1024  # 512KB
        download_tests = []
        for url in urls:
            test_url = f"{url}{test_path}"
            try:
                start_time = time.time()
                headers = {"Range": f"bytes=0-{test_size - 1}"}
                response = requests.get(
                    test_url, headers=headers, timeout=5, stream=True
                )
                response.raise_for_status()

                if "Content-Range" not in response.headers:
                    continue

                if len(response.content) == 0:
                    continue

                elapsed_time = time.time() - start_time
                download_tests.append((url, elapsed_time))
                logger.debug(f"Tested {url}, elapsed time {elapsed_time:.2f} seconds")
            except requests.exceptions.RequestException as e:
                logger.debug(f"Failed to connect to {url}: {e}")

        if not download_tests:
            raise Exception(
                f"It is required to download dependency tools from the internet, but failed to connect to any of {urls}"
            )

        best_url, _ = min(download_tests, key=lambda x: x[1])
        self._download_base_url = best_url
        logger.debug(
            f"Using {best_url} as the base URL for downloading dependency tools"
        )

    def prepare_tools(self):
        """
        Prepare prebuilt binary tools.
        """
        logger.debug("Preparing dependency tools")
        logger.debug(f"OS: {self._os}, Arch: {self._arch}, Device: {self._device}")
        self.download_llama_box()
        self.download_gguf_parser()
        self.download_fastfetch()

    def remove_cached_tools(self):
        """
        Remove all cached tools.
        """
        if os.path.exists(self.third_party_bin_path):
            shutil.rmtree(self.third_party_bin_path)

    def save_archive(self, archive_path: str):
        """
        Save all downloaded tools as a tar archive.
        """
        # Ensure the directory exists
        target_dir = os.path.dirname(archive_path)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Remove extension from archive_path for make_archive. e.g., .tar.gz
        base_name = os.path.splitext(os.path.splitext(archive_path)[0])[0]

        logger.info(f"Saving dependency tools to {archive_path}")
        shutil.make_archive(base_name, "gztar", self.third_party_bin_path)

    def load_archive(self, archive_path: str):
        """
        Load downloaded tools from a tar archive.
        """
        if not os.path.isfile(archive_path):
            raise FileNotFoundError(f"Archive file not found: {archive_path}")

        if not os.path.exists(self.third_party_bin_path):
            os.makedirs(self.third_party_bin_path)

        logger.info(f"Loading dependency tools from {archive_path}")
        shutil.unpack_archive(archive_path, self.third_party_bin_path)

        # Ensure the rpc server is linked correctly
        self._link_llama_box_rpc_server()

    def prepare_versioned_backend(self, backend: str, version: str):
        if backend == BackendEnum.LLAMA_BOX:
            self.install_versioned_llama_box(version)
        elif backend == BackendEnum.VLLM:
            self.install_versioned_vllm(version)
        elif backend == BackendEnum.VOX_BOX:
            self.install_versioned_package_by_pipx("vox-box", version)
        elif backend == BackendEnum.ASCEND_MINDIE:
            self.install_versioned_ascend_mindie(version)
        else:
            raise NotImplementedError(
                f"Auto-installation for versioned {backend} is not supported. Please install it manually."
            )

    def download_llama_box(self):
        version = BUILTIN_LLAMA_BOX_VERSION
        target_dir = self.third_party_bin_path / "llama-box"
        file_name = "llama-box.exe" if self._os == "windows" else "llama-box"
        target_file = target_dir / file_name

        if (
            target_file.is_file()
            and self._current_tools_version.get(file_name) == version
        ):
            logger.debug(f"{file_name} already exists, skipping download")
            return

        self._download_llama_box(version, target_dir, file_name)

        # Update versions.json
        self._update_versions_file(file_name, version)

    def install_versioned_ascend_mindie(self, version: str):
        if self._os != "linux":
            raise Exception("Only Linux is supported")

        target_dir = next(
            (
                rp
                for rp in envs.get_unix_available_root_paths_of_ascend()
                if rp.joinpath("mindie", version).is_dir()
            ),
            None,
        )
        if target_dir:
            # NB(thxCode): Only check mindie-service here,
            # but MindIE must work with mindie-service, mindie-rt, mindie-torch and mindie-llm.
            # We assume that the mindie-service is installed by ascend run package,
            # so that we check whether the set_env.sh exists to determine the installation.
            version = version if not version.startswith("v") else version[1:]
            target_file = target_dir.joinpath(
                "mindie", version, "mindie-service", "set_env.sh"
            )
            if target_file.exists():
                if target_file.is_file():
                    logger.debug(
                        f"Ascend MindIE {version} already exists, skipping download"
                    )
                    return
                else:
                    raise Exception(
                        f"Ascend MindIE {version} already exists, but not a file"
                    )

        target_dir = next(
            (
                rp
                for rp in envs.get_unix_available_root_paths_of_ascend(writable=True)
                if rp.joinpath("mindie").is_dir()
            ),
            None,
        )
        if target_dir is None:
            # If we cannot find an available path, pick the latest one.
            target_dir = envs.get_unix_available_root_paths_of_ascend(writable=True)[-1]
        self._download_acsend_mindie(version, target_dir)

    def install_versioned_llama_box(self, version: str):
        target_dir = Path(self._bin_dir)
        file_name = get_versioned_command(
            "llama-box.exe" if self._os == "windows" else "llama-box", version
        )

        target_file = target_dir / file_name
        if target_file.is_file():
            logger.debug(f"{file_name} already exists, skipping download")
            return

        self._download_llama_box(version, target_dir, file_name)

    def install_versioned_vllm(self, version: str):
        system = platform.system()
        arch = platform.arch()
        device = platform.device()

        if system != "linux" or arch != "amd64":
            target_path = Path(self._bin_dir) / get_versioned_command("vllm", version)
            raise Exception(
                f"Auto-installation for versioned vLLM is only supported on amd64 Linux. Please install vLLM manually and link it to {target_path}."
            )
        elif device != platform.DeviceTypeEnum.CUDA.value:
            raise Exception(
                f"Auto-installation for versioned vLLM is only supported on CUDA devices. Please install vLLM manually and link it to {target_path}."
            )

        self.install_versioned_package_by_pipx(
            "vllm",
            version,
            extra_packages=[
                "gpustack",  # To apply Ray patch for dist vLLM
                f"ray=={BUILTIN_RAY_VERSION}",  # To avoid version conflict with Ray cluster
            ],
        )

    def install_versioned_package_by_pipx(
        self, package: str, version: str, extra_packages: Optional[list] = None
    ):
        """
        Install a versioned package using pipx.

        :param package: The name of the package to install.
        :param version: The version of the package to install.
        """
        target_path = Path(self._bin_dir) / get_versioned_command(package, version)
        if target_path.exists():
            logger.debug(f"{package} {version} already exists, skipping installation")
            return

        pipx_path = shutil.which("pipx")
        if self._pipx_path:
            pipx_path = self._pipx_path

        if not pipx_path:
            raise Exception(
                f"pipx is required to install versioned {package} but not found in system PATH. "
                "Please install pipx first or provide the path to pipx using the server option `--pipx-path`. "
                f"Alternatively, you can install {package} manually and link it to {target_path}."
            )

        pipx_bin_path = self._get_pipx_bin_dir(pipx_path)
        if not pipx_bin_path:
            raise Exception(
                "Failed to determine pipx binary directory. Ensure pipx is correctly installed."
            )

        suffix = f"_{version}"
        install_command = [
            pipx_path,
            "install",
            "-vv",
            "--force",
            "--suffix",
            suffix,
            f"{package}=={version}",
        ]

        try:
            logger.info(f"Installing {package} {version} using pipx")
            subprocess.run(install_command, check=True, text=True)

            installed_bin_path = pipx_bin_path / f"{package}{suffix}"
            if not installed_bin_path.exists():
                raise Exception(
                    f"Installation succeeded, but executable not found at {installed_bin_path}"
                )

            # Create a symlink to the installed binary
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.symlink_to(installed_bin_path)

            if extra_packages:
                for extra_package in extra_packages:
                    self._pipx_inject_package(
                        pipx_path, f"{package}{suffix}", extra_package
                    )

            logger.info(
                f"{package} {version} successfully installed and linked to {target_path}"
            )
        except Exception as e:
            raise Exception(f"Failed to install {package} {version} using pipx: {e}")

    def _pipx_inject_package(self, pipx_path: str, env_name: str, package: str):
        """
        Use `pipx inject` to add a package to an existing pipx environment.
        """
        try:
            logger.info(f"Injecting {package} into pipx environment '{env_name}'")
            subprocess.run(
                [pipx_path, "inject", env_name, package, "--force"],
                check=True,
                text=True,
            )
        except Exception as e:
            logger.warning(
                f"Failed to inject {package} into pipx environment '{env_name}': {e}"
            )

    def _get_pipx_bin_dir(self, pipx_path: str) -> Path:
        """
        Use `pipx environment --value PIPX_BIN_DIR` to get the directory where pipx installs executables.
        """
        try:
            result = subprocess.run(
                [pipx_path, "environment", "--value", "PIPX_BIN_DIR"],
                capture_output=True,
                text=True,
                check=True,
            )
            pipx_bin_dir = result.stdout.strip()
            if pipx_bin_dir:
                return Path(pipx_bin_dir)
        except subprocess.CalledProcessError as e:
            raise Exception(
                f"Failed to execute 'pipx environment --value PIPX_BIN_DIR': {e}"
            )

    def _download_acsend_mindie(self, version: str, target_dir: Path):
        # Check if the system is supported
        if self._os != "linux" or self._arch not in ["amd64", "arm64"]:
            raise Exception(
                "Auto-installation for Ascend MindIE is only supported on Linux amd64/arm64. Please install MindIE manually."
            )

        target_file_arch = "x86_64" if self._arch == "amd64" else "aarch64"
        target_file_name = f"Ascend-mindie_{version}_linux-{target_file_arch}.run"

        # Construct download url, for example:
        # - https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindIE/MindIE%201.0.0/Ascend-mindie_1.0.0_linux-x86_64.run?response-content-type=application/octet-stream
        # - https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindIE/MindIE%201.0.0/Ascend-mindie_1.0.0_linux-aarch64.run?response-content-type=application/octet-stream
        base_url = "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com"
        url_path = f"MindIE/MindIE%20{version}/{target_file_name}?response-content-type=application/octet-stream"

        # Create system temporary directory for downloading and installing
        tmp_dir = tempfile.mkdtemp(prefix="acsend-mindie-")

        # Download and install the MindIE package
        try:
            target_file = os.path.join(tmp_dir, target_file_name)
            logger.info(
                f"Downloading Ascend MindIE '{version}' from '{base_url}/{url_path}' to '{target_file}'"
            )

            headers = {"Referer": "https://www.hiascend.com/"}
            self._download_file(url_path, target_file, base_url, headers)

            logger.info(f"Installing Ascend MindIE '{version}'")
            target_dir.mkdir(parents=True, exist_ok=True)
            self._install_ascend_mindie_run_pkg(target_file, target_dir, version)

            logger.info(f"Postprocessing Ascend MindIE '{version}' installation")
            # Allow writing MindIE service configuration directory.
            service_path = os.path.join(
                target_dir, "mindie", version, "mindie-service", "conf"
            )
            st = os.stat(service_path)
            os.chmod(service_path, st.st_mode | stat.S_IWRITE)
        finally:
            shutil.rmtree(tmp_dir)

    def _download_llama_box(
        self, version: str, target_dir: Path, target_file_name: str
    ):
        llama_box_tmp_dir = target_dir.joinpath("tmp-llama-box")

        # Clean temporary directory if it exists
        if os.path.exists(llama_box_tmp_dir):
            shutil.rmtree(llama_box_tmp_dir)
        os.makedirs(llama_box_tmp_dir, exist_ok=True)

        platform_name = self._get_llama_box_platform_name()
        tmp_file = llama_box_tmp_dir / f"llama-box-{version}-{platform_name}.zip"
        url_path = f"gpustack/llama-box/releases/download/{version}/llama-box-{platform_name}.zip"

        logger.info(f"Downloading llama-box-{platform_name} '{version}'")
        self._download_file(url_path, tmp_file)
        self._extract_file(tmp_file, llama_box_tmp_dir)

        file_name = "llama-box.exe" if self._os == "windows" else "llama-box"
        target_file = target_dir / target_file_name
        shutil.copy(llama_box_tmp_dir / file_name, target_file)

        # Make the file executable (non-Windows only)
        if self._os != "windows":
            st = os.stat(target_file)
            os.chmod(target_file, st.st_mode | stat.S_IEXEC)

        self._link_llama_box_rpc_server()

        # Clean up temporary directory
        shutil.rmtree(llama_box_tmp_dir)

    def _link_llama_box_rpc_server(self):
        """
        Create a symlink for llama-box-rpc-server in the bin directory.
        This is used to help differentiate between the llama-box and llama-box-rpc-server processes.
        """
        target_dir = self.third_party_bin_path / "llama-box"
        file_name = "llama-box.exe" if self._os == "windows" else "llama-box"
        llama_box_file = target_dir / file_name

        if self._os == "windows":
            target_rpc_server_file = target_dir / "llama-box-rpc-server.exe"
        else:
            target_rpc_server_file = target_dir / "llama-box-rpc-server"

        if os.path.lexists(target_rpc_server_file):
            os.remove(target_rpc_server_file)

        if self._os == "windows":
            os.link(llama_box_file, target_rpc_server_file)
        else:
            os.symlink(llama_box_file, target_rpc_server_file)

        logger.debug(f"Linked llama-box-rpc-server to {target_rpc_server_file}")

    def _get_llama_box_cuda_version(self) -> str:
        """
        Gets the appropriate CUDA version of the llama-box based on the system's CUDA version.
        """

        default_version = "12.4"
        cuda_version = platform.get_cuda_version()
        match = re.match(r"(\d+)\.(\d+)", cuda_version)
        if not match:
            return default_version

        major, minor = map(int, match.groups())
        if major == 11:
            return "11.8"
        elif major == 12 and minor >= 8:
            return "12.8"

        return default_version

    def _get_llama_box_platform_name(self) -> str:  # noqa C901
        platform_name = ""
        if (
            self._os == "darwin"
            and self._arch == "arm64"
            and self._device == platform.DeviceTypeEnum.MPS.value
        ):
            platform_name = "darwin-arm64-metal"
        elif self._os == "darwin":
            platform_name = "darwin-amd64-avx2"
        elif (
            self._os in ["linux", "windows"]
            and self._arch in ["amd64", "arm64"]
            and self._device == platform.DeviceTypeEnum.CUDA.value
        ):
            # Only amd64 for windows
            normalized_arch = "amd64" if self._os == "windows" else self._arch
            platform_name = (
                f"{self._os}-{normalized_arch}-cuda-{self._llama_box_cuda_version}"
            )
        elif (
            self._os == "linux"
            and self._arch == "amd64"
            and self._device == platform.DeviceTypeEnum.MUSA.value
        ):
            platform_name = "linux-amd64-musa-rc3.1"
        elif self._os == "linux" and self._device == platform.DeviceTypeEnum.NPU.value:
            # Available version: 8.0.0(.beta1) [default] / 8.0.rc2(.beta1) / 8.0.rc3(.beta1)
            version = "8.0"
            if ".rc2" in os.getenv("CANN_VERSION", ""):
                version = "8.0.rc2"
            elif ".rc3" in os.getenv("CANN_VERSION", ""):
                version = "8.0.rc3"
            # Available variant: 910b [default] / 310p
            variant = ""
            if os.getenv("CANN_CHIP", "") == "310p":
                variant = "-310p"
            platform_name = f"linux-{self._arch}-cann-{version}{variant}"
        elif (
            self._os == "linux"
            and self._arch == "amd64"
            and self._device == platform.DeviceTypeEnum.ROCM.value
        ):
            platform_name = "linux-amd64-hip-6.2"
        elif (
            self._os == "linux"
            and self._arch == "amd64"
            and self._device == platform.DeviceTypeEnum.DCU.value
        ):
            platform_name = "linux-amd64-dtk-24.04"
        elif self._os == "linux" and self._arch == "amd64":
            platform_name = "linux-amd64-avx2"
        elif self._os == "linux" and self._arch == "arm64":
            platform_name = "linux-arm64-neon"
        elif (
            self._os == "windows"
            and self._arch == "amd64"
            and self._device == platform.DeviceTypeEnum.ROCM.value
        ):
            platform_name = "windows-amd64-hip-6.2"
        elif self._os == "windows" and self._arch == "amd64":
            platform_name = "windows-amd64-avx2"
        elif self._os == "windows" and self._arch == "arm64":
            platform_name = "windows-arm64-neon"
        else:
            raise Exception(
                f"unsupported platform, os: {self._os}, arch: {self._arch}, device: {self._device}"
            )

        return platform_name

    def download_gguf_parser(self):
        version = BUILTIN_GGUF_PARSER_VERSION
        gguf_parser_dir = self.third_party_bin_path.joinpath("gguf-parser")
        os.makedirs(gguf_parser_dir, exist_ok=True)

        file_name = "gguf-parser"
        suffix = ""
        if self._os == "windows":
            suffix = ".exe"
            file_name += suffix
        target_file = gguf_parser_dir.joinpath(file_name)
        if (
            os.path.isfile(target_file)
            and self._current_tools_version.get(file_name) == version
        ):
            logger.debug(f"{file_name} already exists, skipping download")
            return

        platform_name = self._get_gguf_parser_platform_name()
        url_path = f"gpustack/gguf-parser-go/releases/download/{version}/gguf-parser-{platform_name}{suffix}"

        logger.info(f"downloading gguf-parser-{platform_name} '{version}'")
        self._download_file(url_path, target_file)

        if self._os != "windows":
            st = os.stat(target_file)
            os.chmod(target_file, st.st_mode | stat.S_IEXEC)

        # Update versions.json
        self._update_versions_file(file_name, version)

    def _get_gguf_parser_platform_name(self) -> str:
        platform_name = ""
        if self._os == "darwin":
            platform_name = "darwin-universal"
        elif self._os == "linux" and self._arch == "amd64":
            platform_name = "linux-amd64"
        elif self._os == "linux" and self._arch == "arm64":
            platform_name = "linux-arm64"
        elif self._os == "windows" and self._arch == "amd64":
            platform_name = "windows-amd64"
        elif self._os == "windows" and self._arch == "arm64":
            platform_name = "windows-arm64"
        else:
            raise Exception(f"Unsupported platform: {self._os} {self._arch}")

        return platform_name

    def download_fastfetch(self):
        version = "2.25.0.1"
        fastfetch_dir = self.third_party_bin_path.joinpath("fastfetch")
        fastfetch_tmp_dir = fastfetch_dir.joinpath("tmp")

        platform_name = self._get_fastfetch_platform_name()

        file_name = "fastfetch"
        if self._os == "windows":
            file_name += ".exe"
        target_file = os.path.join(fastfetch_dir, file_name)
        if (
            os.path.isfile(target_file)
            and self._current_tools_version.get(file_name) == version
        ):
            logger.debug(f"{file_name} already exists, skipping download")
            return

        logger.info(f"downloading fastfetch-{platform_name} '{version}'")

        tmp_file = os.path.join(fastfetch_tmp_dir, f"fastfetch-{platform_name}.zip")
        if os.path.exists(fastfetch_tmp_dir):
            shutil.rmtree(fastfetch_tmp_dir)
        os.makedirs(fastfetch_tmp_dir, exist_ok=True)

        url_path = f"gpustack/fastfetch/releases/download/{version}/fastfetch-{platform_name}.zip"

        self._download_file(url_path, tmp_file)
        self._extract_file(tmp_file, fastfetch_tmp_dir)

        extracted_fastfetch = fastfetch_tmp_dir.joinpath(
            f"fastfetch-{platform_name}",
            "usr",
            "bin",
            "fastfetch",
        )
        if self._os == "windows":
            extracted_fastfetch = fastfetch_tmp_dir.joinpath(
                "fastfetch.exe",
            )

        if os.path.exists(extracted_fastfetch):
            shutil.copy(extracted_fastfetch, target_file)
        else:
            raise Exception("failed to find fastfetch binary in extracted archive")

        if self._os != "windows":
            st = os.stat(target_file)
            os.chmod(target_file, st.st_mode | stat.S_IEXEC)

        # Clean up.
        if os.path.exists(fastfetch_tmp_dir):
            shutil.rmtree(fastfetch_tmp_dir)

        # Update versions.json
        self._update_versions_file(file_name, version)

    def _update_versions_file(self, tool_name: str, version: str):
        updated_versions = self._current_tools_version.copy()
        updated_versions[tool_name] = version

        try:
            with open(self.versions_file, 'w', encoding='utf-8') as file:
                json.dump(updated_versions, file, indent=4)
                self._current_tools_version[tool_name] = version
        except Exception as e:
            logger.error(f"Failed to update versions.json: {e}")

    def _get_fastfetch_platform_name(self) -> str:
        platform_name = ""
        if self._os == "darwin":
            platform_name = "macos-universal"
        elif self._os == "linux" and self._arch == "amd64":
            platform_name = "linux-amd64"
        elif self._os == "linux" and self._arch == "arm64":
            platform_name = "linux-aarch64"
        elif self._os == "windows":
            platform_name = "windows-amd64"
        else:
            raise Exception(f"unsupported platform: {self._os} {self._arch}")

        return platform_name

    def _download_file(
        self,
        url_path: str,
        target_path: str,
        base_url: str = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Download a file from the URL to the target path."""
        if not self._download_base_url:
            self._check_and_set_download_base_url()

        url = f"{self._download_base_url}/{url_path}"
        if base_url:
            url = f"{base_url}/{url_path}"
        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                with requests.get(
                    url,
                    stream=True,
                    timeout=30,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    with open(target_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                break
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    raise Exception(f"Error downloading from {url}: {e}")
                else:
                    logger.debug(
                        f"Attempt {retries} failed: {e}. Retrying in 2 seconds..."
                    )
                    time.sleep(2)

    def _extract_file(self, file_path, target_dir):
        """Extract a file to the target directory."""
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        except zipfile.BadZipFile as e:
            raise Exception(f"error extracting {file_path}: {e}")
        except Exception as e:
            raise Exception(f"error extracting {file_path}: {e}")

    def _install_ascend_mindie_run_pkg(
        self,
        run_package_path: str,
        target_dir: Path,
        version: str,
    ):
        """Install Ascend MindIE run package to the target directory."""

        # Create a virtual environment to collect the new Python packages.
        venv_parent_dir = Path("/var/lib/gpustack/venvs/mindie")
        venv_parent_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "venv", "--system-site-packages", version],
                cwd=venv_parent_dir,
            )
        except subprocess.CalledProcessError as e:
            raise Exception(
                f"Failed to create a virtual environment for Ascend MindIE installation: {e}"
            )
        venv_dir = venv_parent_dir.joinpath(version)
        venv_path = venv_dir.joinpath("bin", "activate")
        logger.info(
            f"Created virtual environment for Ascend MindIE installation: {venv_dir}"
        )

        # Install
        command = (
            f"source {venv_path} "
            f"&& {run_package_path} --install --install-path={target_dir} --quiet"
        )
        try:
            # Make run package executable.
            st = os.stat(run_package_path)
            os.chmod(run_package_path, st.st_mode | stat.S_IEXEC)

            # Cheat MindIE installer run package with a fake ASCEND_HOME_PATH env.
            env = os.environ.copy()
            env["ASCEND_HOME_PATH"] = str(target_dir)

            # Run
            out = None
            if logger.isEnabledFor(logging.DEBUG):
                out = sys.stdout
            subprocess.check_call(
                command,
                shell=True,
                executable="/bin/bash",
                stdout=out,
                stderr=out,
                env=env,
                cwd=target_dir,
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to install Ascend MindIE {command}: {e}")
