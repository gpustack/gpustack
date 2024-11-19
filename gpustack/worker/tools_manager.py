import json
import logging
import os
from pathlib import Path
import shutil
import stat
import time
from typing import Optional
import zipfile
import requests
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils import platform

logger = logging.getLogger(__name__)


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
        device: Optional[str] = None,
        system: Optional[str] = None,
        arch: Optional[str] = None,
    ):
        with pkg_resources.path("gpustack.third_party", "bin") as bin_path:
            self.bin_path: Path = bin_path
            self.versions_file = bin_path.joinpath("versions.json")

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
        self._download_base_url = tools_download_base_url

    def _check_and_set_download_base_url(self):
        urls = [
            "https://github.com",
            "https://gpustack-1303613262.cos.ap-guangzhou.myqcloud.com",
        ]
        for url in urls:
            try:
                requests.get(url, timeout=3)
                # Set if the URL is accessible.
                logger.debug(
                    f"Using {url} as the base URL for downloading dependency tools"
                )
                self._download_base_url = url
                break
            except requests.exceptions.RequestException as e:
                logger.debug(f"Failed to connect to {url}: {e}")

        if not self._download_base_url:
            raise Exception(
                f"It is required to download dependency tools from the internet, but failed to connect to any of {urls}"
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
        if os.path.exists(self.bin_path):
            shutil.rmtree(self.bin_path)

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
        shutil.make_archive(base_name, "gztar", self.bin_path)

    def load_archive(self, archive_path: str):
        """
        Load downloaded tools from a tar archive.
        """
        if not os.path.isfile(archive_path):
            raise FileNotFoundError(f"Archive file not found: {archive_path}")

        if not os.path.exists(self.bin_path):
            os.makedirs(self.bin_path)

        logger.info(f"Loading dependency tools from {archive_path}")
        shutil.unpack_archive(archive_path, self.bin_path)

    def download_llama_box(self):
        version = "v0.0.79"
        llama_box_dir = self.bin_path.joinpath("llama-box")
        llama_box_tmp_dir = llama_box_dir.joinpath(llama_box_dir, "tmp")

        file_name = "llama-box"
        if self._os == "windows":
            file_name += ".exe"

        target_file = llama_box_dir.joinpath(file_name)
        if (
            os.path.isfile(target_file)
            and self._current_tools_version.get(file_name) == version
        ):
            logger.debug(f"{file_name} already exists, skipping download")
            return

        platform_name = self._get_llama_box_platform_name()
        if os.path.exists(llama_box_tmp_dir):
            shutil.rmtree(llama_box_tmp_dir)
        os.makedirs(llama_box_tmp_dir, exist_ok=True)

        tmp_file = os.path.join(
            llama_box_tmp_dir, f"llama-box-{version}-{platform_name}.zip"
        )
        url_path = f"gpustack/llama-box/releases/download/{version}/llama-box-{platform_name}.zip"

        logger.info(f"downloading llama-box-{platform_name} '{version}'")
        self._download_file(url_path, tmp_file)
        self._extract_file(tmp_file, llama_box_tmp_dir)

        shutil.copy(llama_box_tmp_dir.joinpath(file_name), target_file)

        if self._os != "windows":
            st = os.stat(target_file)
            os.chmod(target_file, st.st_mode | stat.S_IEXEC)

        # Clean up.
        if os.path.exists(llama_box_tmp_dir):
            shutil.rmtree(llama_box_tmp_dir)

        # Update versions.json
        self._update_versions_file(file_name, version)

    def _get_llama_box_platform_name(self) -> str:  # noqa C901
        platform_name = ""
        if self._os == "darwin" and self._arch == "arm64" and self._device == "mps":
            platform_name = "darwin-arm64-metal"
        elif self._os == "darwin":
            platform_name = "darwin-amd64-avx2"
        elif self._os == "linux" and self._arch == "amd64" and self._device == "cuda":
            platform_name = "linux-amd64-cuda-12.4"
        elif self._os == "linux" and self._arch == "amd64" and self._device == "musa":
            platform_name = "linux-amd64-musa-rc3.1"
        elif self._os == "linux" and self._arch == "amd64" and self._device == "npu":
            platform_name = "linux-amd64-cann-8.0"
        elif self._os == "linux" and self._arch == "arm64" and self._device == "npu":
            platform_name = "linux-arm64-cann-8.0"
        elif self._os == "linux" and self._arch == "amd64":
            platform_name = "linux-amd64-avx2"
        elif self._os == "linux" and self._arch == "arm64":
            platform_name = "linux-arm64-neon"
        elif self._os == "windows" and self._arch == "amd64" and self._device == "cuda":
            platform_name = "windows-amd64-cuda-12.4"
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
        version = "v0.12.1"
        gguf_parser_dir = self.bin_path.joinpath("gguf-parser")
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
        fastfetch_dir = self.bin_path.joinpath("fastfetch")
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

    def _download_file(self, url_path: str, target_path: str):
        """Download a file from the URL to the target path."""
        if not self._download_base_url:
            self._check_and_set_download_base_url()

        url = f"{self._download_base_url}/{url_path}"
        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                with requests.get(url, stream=True, timeout=30) as response:
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
