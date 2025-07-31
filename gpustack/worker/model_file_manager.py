import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob
from itertools import chain
import logging
from pathlib import Path
import platform
import time
import threading
from typing import Dict, Tuple
from modelscope.hub.constants import TEMPORARY_FOLDER_NAME
from multiprocessing import Manager, cpu_count
from huggingface_hub._local_folder import get_local_download_paths
from huggingface_hub.file_download import get_hf_file_metadata, hf_hub_url
import huggingface_hub.constants
from huggingface_hub.utils import build_hf_headers

from gpustack.api.exceptions import NotFoundException
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.schemas.model_files import ModelFile, ModelFileUpdate, ModelFileStateEnum
from gpustack.client import ClientSet
from gpustack.schemas.models import SourceEnum
from gpustack.server.bus import Event, EventType
from gpustack.utils import hub
from gpustack.utils.file import delete_path
from gpustack.worker import downloaders


logger = logging.getLogger(__name__)

max_concurrent_downloads = 5


class ModelFileManager:
    def __init__(
        self,
        worker_id: int,
        clientset: ClientSet,
        cfg: Config,
    ):
        self._worker_id = worker_id
        self._config = cfg
        self._clientset = clientset
        self._active_downloads: Dict[int, Tuple] = {}
        self._download_pool = None

    async def watch_model_files(self):
        self._prerun()
        while True:
            try:
                logger.debug("Started watching model files.")
                await self._clientset.model_files.awatch(
                    callback=self._handle_model_file_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Failed to watch model files: {e}")
                await asyncio.sleep(5)

    def _prerun(self):
        self._mp_manager = Manager()
        self._download_pool = ProcessPoolExecutor(
            max_workers=min(max_concurrent_downloads, cpu_count()),
        )

    def _handle_model_file_event(self, event: Event):
        mf = ModelFile.model_validate(event.data)

        if mf.worker_id != self._worker_id:
            # Ignore model files that are not assigned to this worker.
            return

        logger.trace(f"Received model file event: {event.type} {mf.id} {mf.state}")

        if event.type == EventType.DELETED:
            asyncio.create_task(self._handle_deletion(mf))
        elif event.type in {EventType.CREATED, EventType.UPDATED}:
            if mf.state != ModelFileStateEnum.DOWNLOADING:
                return
            self._create_download_task(mf)

    def _update_model_file(self, id: int, **kwargs):
        model_file_public = self._clientset.model_files.get(id=id)

        model_file_update = ModelFileUpdate(**model_file_public.model_dump())
        for key, value in kwargs.items():
            setattr(model_file_update, key, value)

        self._clientset.model_files.update(id=id, model_update=model_file_update)

    async def _handle_deletion(self, model_file: ModelFile):
        entry = self._active_downloads.pop(model_file.id, None)
        if entry:
            future, cancel_flag = entry
            cancel_flag.set()
            future.cancel()
            try:
                await asyncio.wrap_future(future)
            except (asyncio.CancelledError, NotFoundException):
                pass
            except Exception as e:
                logger.error(
                    f"Error while cancelling download for {model_file.readable_source}(id: {model_file.id}): {e}"
                )
            finally:
                logger.info(
                    f"Cancelled download for deleted model: {model_file.readable_source}(id: {model_file.id})"
                )

        if model_file.cleanup_on_delete:
            await self._delete_model_file(model_file)

    async def get_hf_file_metadata(self, model_file: ModelFile, filename: str):
        token = self._config.huggingface_token
        url = hf_hub_url(model_file.huggingface_repo_id, filename)
        headers = build_hf_headers(token=token)

        metadata = await asyncio.to_thread(
            get_hf_file_metadata,
            url=url,
            timeout=huggingface_hub.constants.DEFAULT_ETAG_TIMEOUT,
            headers=headers,
            token=token,
        )
        return metadata

    async def _get_incomplete_model_files(  # noqa: C901
        self, model_file: ModelFile
    ) -> set:
        """
        Finds cached files of models being downloaded.
        1.For models from Hugging Face, their .incomplete filenames are encoded. The process requires:
        [filename_pattern → model_name → etag → incomplete_pattern → .incomplete_filename] to ultimately confirm the file.
        2.For models from ModelScope, the incomplete files are stored in a temporary folder.
        we just need to find them by the filename pattern.
        """
        paths_to_delete = set()

        try:
            if model_file.source == SourceEnum.HUGGING_FACE:
                if not model_file.huggingface_filename:
                    # The resolved_paths in vLLM model points to entire dir of cache, delete it directly
                    paths_to_delete.update(model_file.resolved_paths)
                    return paths_to_delete

                for path in model_file.resolved_paths:
                    path_obj = Path(str(path))
                    filename_pattern = path_obj.name
                    local_dir = path_obj.parent
                    download_paths = get_local_download_paths(
                        local_dir, filename_pattern
                    )
                    cache_dir = download_paths.lock_path.parent
                    filename = ""

                    # Get actual filename by pattern
                    for cache_file in await asyncio.to_thread(
                        glob.glob, str(cache_dir / filename_pattern) + "*"
                    ):
                        # cut off the path and useless extension
                        filename = cache_file.rsplit("/", 1)[-1]
                        filename = filename.rsplit(".", 1)[0]
                        break

                    metadata = await self.get_hf_file_metadata(model_file, filename)

                    # Collect lock files and incomplete files
                    paths_to_delete.add(str(cache_dir / (filename + ".lock")))
                    paths_to_delete.add(str(cache_dir / (filename + ".metadata")))
                    for item_path_str in await asyncio.to_thread(
                        glob.glob, str(cache_dir / f"*.{metadata.etag}.incomplete")
                    ):
                        paths_to_delete.add(item_path_str)

            elif model_file.source == SourceEnum.MODEL_SCOPE:
                if not model_file.model_scope_file_path:
                    # The resolved_paths in vLLM model points to entire dir of cache, delete it directly
                    paths_to_delete.update(model_file.resolved_paths)
                    return paths_to_delete

                for path in model_file.resolved_paths:
                    path_obj = Path(str(path))
                    filename_pattern = path_obj.name
                    local_dir = path_obj.parent
                    for delete_file in await asyncio.to_thread(
                        glob.glob,
                        str(local_dir / f"{TEMPORARY_FOLDER_NAME}/{filename_pattern}"),
                    ):
                        paths_to_delete.add(delete_file)

        except Exception as e:
            logger.error(
                f"Error deleting incomplete Download files for "
                f"file '{filename}': {e}"
            )

        return paths_to_delete

    async def _delete_incomplete_model_files(self, model_file: ModelFile):
        paths_to_delete = await self._get_incomplete_model_files(model_file)

        for delete_file in paths_to_delete:
            logger.info(f"Attempting to delete incomplete file: {delete_file}")
            await asyncio.to_thread(delete_path, delete_file)

    async def _delete_model_file(self, model_file: ModelFile):
        try:
            if model_file.resolved_paths:
                paths = chain.from_iterable(
                    glob.glob(p) if '*' in p else [p] for p in model_file.resolved_paths
                )
                for path in paths:
                    delete_path(path)

            await self._delete_incomplete_model_files(model_file)
            logger.info(
                f"Deleted model file {model_file.readable_source}(id: {model_file.id}) from disk"
            )
        except Exception as e:
            logger.error(
                f"Failed to delete {model_file.readable_source}(id: {model_file.id}: {e}"
            )
            await self._update_model_file(
                model_file.id,
                state=ModelFileStateEnum.ERROR,
                state_message=f"Deletion failed: {str(e)}",
            )

    def _create_download_task(self, model_file: ModelFile):
        if model_file.id in self._active_downloads:
            return

        cancel_flag = self._mp_manager.Event()

        download_task = ModelFileDownloadTask(model_file, self._config, cancel_flag)
        future = self._download_pool.submit(download_task.run)
        self._active_downloads[model_file.id] = (future, cancel_flag)

        logger.debug(f"Created download task for {model_file.readable_source}")

        async def _check_completion():
            try:
                await asyncio.wrap_future(future)
            except NotFoundException:
                logger.info(
                    f"Model file {model_file.readable_source} not found. Maybe it was cancelled."
                )
            except Exception as e:
                logger.error(f"Failed to download model file: {e}")
                await self._update_model_file(
                    model_file.id,
                    state=ModelFileStateEnum.ERROR,
                    state_message=str(e),
                )
            finally:
                self._active_downloads.pop(model_file.id, None)

            logger.debug(f"Download completed for {model_file.readable_source}")

        asyncio.create_task(_check_completion())


class ModelFileDownloadTask:

    def __init__(self, model_file: ModelFile, cfg: Config, cancel_flag):
        self._model_file = model_file
        self._config = cfg
        self._cancel_flag = cancel_flag
        # Store download log file paths for related model instances
        self._instance_download_log_file = None
        self._download_completed = False

    def prerun(self):
        setup_logging(self._config.debug)
        self._clientset = ClientSet(
            base_url=self._config.server_url,
            username=f"system/worker/{self._config.worker_ip}",
            password=self._config.token,
        )
        self._download_start_time = time.time()
        self._ensure_model_file_size_and_paths()

        self._speed_lock = threading.Lock()
        # Lock for _model_downloaded_size/_last_download_update_time/_last_downloaded_size to avoid race condition
        self._model_downloaded_size = 0
        self._last_download_update_time = 0
        self._last_downloaded_size = 0

        self._setup_instance_log_files()

        self._model_downloaded_size = 0
        self._last_download_update_time = 0
        self._last_downloaded_size = 0

        logger.debug(f"Initializing task for {self._model_file.readable_source}")
        self._update_progress_func = partial(
            self._update_model_file_progress, self._model_file.id
        )
        self._model_file_size = self._model_file.size
        self._model_downloaded_size = 0
        self.hijack_tqdm_progress()

    def _setup_instance_log_files(self):
        try:
            log_dir = Path(self._config.log_dir) / "serve"

            # Use model file ID for shared download log across all instances using the same model file
            download_log_file_path = (
                log_dir / f"model_file_{self._model_file.id}.download.log"
            )
            self._instance_download_log_file = str(download_log_file_path)

            logger.debug(f"Setup shared download log file: {download_log_file_path}")

        except Exception as e:
            logger.warning(f"Failed to setup instance download log files: {e}")

    def _write_log_with_windows_lock(self, log_file_path: str, log_message: str):
        """
        Write log message to file using Windows msvcrt file locking
        """
        try:
            import msvcrt
        except ImportError:
            # msvcrt not available, fallback to basic write
            self._write_log_without_lock(log_file_path, log_message)
            return

        with open(log_file_path, 'a', encoding='utf-8') as f:
            try:
                # Acquire exclusive lock on the file
                # Lock a single byte at the beginning of the file for coordination
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                f.seek(0, 2)  # Move to end of file for appending
                f.write(log_message)
                f.flush()  # Ensure immediate write to disk
            except (OSError, IOError):
                # If locking fails, fallback to basic write
                f.seek(0, 2)  # Move to end of file for appending
                f.write(log_message)
                f.flush()
            finally:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass  # Ignore unlock errors

    def _write_log_with_unix_lock(self, log_file_path: str, log_message: str):
        """
        Write log message to file using Unix/Linux fcntl file locking
        """
        try:
            import fcntl
        except ImportError:
            # fcntl not available, fallback to basic write
            self._write_log_without_lock(log_file_path, log_message)
            return

        with open(log_file_path, 'a', encoding='utf-8') as f:
            try:
                # Acquire exclusive lock on the file
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(log_message)
                f.flush()  # Ensure immediate write to disk
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_log_without_lock(self, log_file_path: str, log_message: str):
        """
        Write log message to file without file locking (fallback method)
        """
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_message)
                f.flush()  # Ensure immediate write to disk
        except Exception as e:
            logger.warning(
                f"Failed to write to instance download log {log_file_path}: {e}"
            )

    def _write_to_instance_download_logs(self, message: str, is_error=False):
        """
        Write download log message to all associated model instance download log files
        Skip writing if download is completed to avoid unnecessary logs
        """
        if not self._instance_download_log_file or self._download_completed:
            return

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_level = "ERROR" if is_error else "INFO"
        log_message = f"[{timestamp}] [{log_level}] {message}\n"

        # Determine file locking mechanism based on platform
        is_windows = platform.system() == 'Windows'

        # Ensure log directory exists
        Path(self._instance_download_log_file).parent.mkdir(parents=True, exist_ok=True)

        # Use appropriate locking method based on platform
        if is_windows:
            self._write_log_with_windows_lock(
                self._instance_download_log_file, log_message
            )
        else:
            self._write_log_with_unix_lock(
                self._instance_download_log_file, log_message
            )

    def run(self):
        try:
            self.prerun()
            self._write_to_instance_download_logs(
                f"Model file download task started: {self._model_file.readable_source}"
            )
            self._download_model_file()
            self._write_to_instance_download_logs(
                f"Model file download task completed successfully: {self._model_file.readable_source}"
            )
        except asyncio.CancelledError:
            self._write_to_instance_download_logs(
                f"Download task cancelled: {self._model_file.readable_source}"
            )
        except Exception as e:
            self._write_to_instance_download_logs(
                f"Download task failed: {self._model_file.readable_source} - {str(e)}",
                is_error=True,
            )
            self._update_model_file(
                self._model_file.id,
                state=ModelFileStateEnum.ERROR,
                state_message=str(e),
            )

    def _download_model_file(self):
        self._write_to_instance_download_logs(
            f"Downloading model file: {self._model_file.readable_source}"
        )

        model_paths = downloaders.download_model(
            self._model_file,
            local_dir=self._model_file.local_dir,
            cache_dir=self._config.cache_dir,
            ollama_library_base_url=self._config.ollama_library_base_url,
            huggingface_token=self._config.huggingface_token,
        )
        self._download_completed = True
        self._update_model_file(
            self._model_file.id,
            state=ModelFileStateEnum.READY,
            download_progress=100,
            resolved_paths=model_paths,
        )
        self._write_to_instance_download_logs(
            f"Successfully downloaded {self._model_file.readable_source}"
        )

    def hijack_tqdm_progress(task_self):
        """
        Monkey patch the tqdm progress bar to update the model instance download progress.
        tqdm is used by hf_hub_download under the hood.
        """
        from tqdm import tqdm

        _original_init = (
            tqdm._original_init if hasattr(tqdm, "_original_init") else tqdm.__init__
        )
        _original_update = (
            tqdm._original_update if hasattr(tqdm, "_original_update") else tqdm.update
        )

        def _new_init(self: tqdm, *args, **kwargs):
            task_self._handle_tqdm_init(self, _original_init, *args, **kwargs)

        def _new_update(self: tqdm, n=1):
            task_self._handle_tqdm_update(self, _original_update, n)

        tqdm.__init__ = _new_init
        tqdm.update = _new_update
        tqdm._original_init = _original_init
        tqdm._original_update = _original_update

    def _handle_tqdm_init(self, tqdm_instance, original_init, *args, **kwargs):
        kwargs["disable"] = False  # enable the progress bar anyway
        original_init(tqdm_instance, *args, **kwargs)

        if hasattr(self, '_model_file_size'):
            # Resume downloading
            self._model_downloaded_size += tqdm_instance.n

        # Log download start info to model instance logs
        if hasattr(tqdm_instance, 'desc') and tqdm_instance.desc:
            self._write_to_instance_download_logs(
                f"Starting download: {tqdm_instance.desc}"
            )
        else:
            self._write_to_instance_download_logs(
                f"Starting model file download: {self._model_file.readable_source}"
            )

    def _handle_tqdm_update(self, tqdm_instance, original_update, n=1):
        original_update(tqdm_instance, n)

        if self._cancel_flag.is_set():
            raise asyncio.CancelledError("Download cancelled")

        # Calculate download sizes
        total_size = tqdm_instance.total
        downloaded_size = tqdm_instance.n

        if hasattr(self, '_model_file_size'):
            # This is summary for group downloading
            total_size = self._model_file_size
            with self._speed_lock:
                self._model_downloaded_size += n
                downloaded_size = self._model_downloaded_size

        try:
            self._update_progress_and_log(total_size, downloaded_size)
        except Exception as e:
            error_msg = f"Failed to update model file: {e}"
            self._write_to_instance_download_logs(
                f"Download error: {error_msg}", is_error=True
            )
            raise Exception(error_msg)

    def _update_progress_and_log(self, total_size, downloaded_size):
        if total_size <= 0:
            return

        progress_percent = (
            round((downloaded_size / total_size) * 100, 2) if total_size > 0 else 0
        )
        current_time = time.time()
        # Download speed to show in logs
        speed_mb_per_sec = 0

        with self._speed_lock:
            # Only update after 2-second interval or download is completed.
            if (
                current_time - self._last_download_update_time < 2
                and downloaded_size != total_size
            ):
                return

            # Update progress to DB
            self._update_progress_func(progress_percent)

            # Calculate download speed
            time_elapsed = current_time - self._last_download_update_time
            if time_elapsed > 0 and self._last_download_update_time > 0:
                bytes_downloaded_since_last = (
                    downloaded_size - self._last_downloaded_size
                )
                speed_bytes_per_sec = bytes_downloaded_since_last / time_elapsed
                speed_mb_per_sec = speed_bytes_per_sec / (1024 * 1024)

            self._last_download_update_time = current_time
            self._last_downloaded_size = downloaded_size

        # Log download progress to download log files
        downloaded_mb = downloaded_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)

        if downloaded_size == total_size:
            log_msg = (
                f"Download completed: {self._model_file.readable_source} - "
                f"{downloaded_mb:.2f}MB/{total_mb:.2f}MB (100.00%)"
            )
        else:
            log_msg = (
                f"Download progress: {self._model_file.readable_source} - "
                f"{downloaded_mb:.2f}MB/{total_mb:.2f}MB ({progress_percent:.2f}%) - "
                f"Speed: {speed_mb_per_sec:.2f}MB/s"
            )

        self._write_to_instance_download_logs(log_msg)

    def _ensure_model_file_size_and_paths(self):
        if self._model_file.size is not None:
            return

        repo_file_list = downloaders.get_model_file_info(
            self._model_file,
            huggingface_token=self._config.huggingface_token,
            cache_dir=self._config.cache_dir,
            ollama_library_base_url=self._config.ollama_library_base_url,
        )

        (size, file_paths) = hub.match_file_and_calculate_size(
            files=repo_file_list,
            model=self._model_file,
            cache_dir=self._config.cache_dir,
        )

        self._model_file.size = size
        self._update_model_file(
            self._model_file.id, size=size, resolved_paths=file_paths
        )

    def _update_model_file_progress(self, model_file_id: int, progress: float):
        self._update_model_file(model_file_id, download_progress=progress)

    def _update_model_file(self, id: int, **kwargs):
        model_file_public = self._clientset.model_files.get(id=id)

        model_file_update = ModelFileUpdate(**model_file_public.model_dump())
        for key, value in kwargs.items():
            setattr(model_file_update, key, value)

        self._clientset.model_files.update(id=id, model_update=model_file_update)
