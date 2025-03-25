import asyncio
from concurrent.futures import ProcessPoolExecutor, CancelledError
from functools import partial
import glob
from itertools import chain
import logging
import time
from typing import Dict, Tuple
from multiprocessing import Manager, cpu_count


from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.schemas.model_files import ModelFile, ModelFileUpdate, ModelFileStateEnum
from gpustack.client import ClientSet
from gpustack.server.bus import Event, EventType
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
                logger.error(f"Failed watching model files: {e}")
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
                await future
            except (asyncio.CancelledError, CancelledError):
                pass
            finally:
                logger.info(
                    f"Cancelled download for deleted model: {model_file.readable_source}({model_file.id})"
                )

        await self._delete_model_file(model_file)

    async def _delete_model_file(self, model_file: ModelFile):
        try:
            if model_file.resolved_paths:
                paths = chain.from_iterable(
                    glob.glob(p) if '*' in p else [p] for p in model_file.resolved_paths
                )
                for path in paths:
                    delete_path(path)

            logger.info(
                f"Deleted model file {model_file.readable_source}({model_file.id})"
            )
        except Exception as e:
            logger.error(
                f"Failed to delete {model_file.readable_source}({model_file.id}: {e}"
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
            except (asyncio.CancelledError, Exception) as e:
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

    def prerun(self):
        setup_logging(self._config.debug)
        self._clientset = ClientSet(
            base_url=self._config.server_url,
            username=f"system/worker/{self._config.worker_ip}",
            password=self._config.token,
        )

        self._ensure_model_file_size()

        self._last_download_update_time = 0
        self._model_downloaded_size = 0
        logger.debug(f"Initializing task for {self._model_file.id}")
        self._update_progress_func = partial(
            self._update_model_file_progress, self._model_file.id
        )
        self._model_file_size = self._model_file.size
        self._model_downloaded_size = 0
        self.hijack_tqdm_progress()

    def run(self):
        try:
            self.prerun()
            self._download_model_file()
        except asyncio.CancelledError:
            logger.info(f"Download cancelled for {self._model_file.id}")
            self._update_model_file(
                self._model_file.id,
                state=ModelFileStateEnum.ERROR,
                state_message="Download cancelled",
            )
        except Exception as e:
            logger.error(f"Download failed for {self._model_file.id}: {str(e)}")
            self._update_model_file(
                self._model_file.id,
                state=ModelFileStateEnum.ERROR,
                state_message=str(e),
            )

    def _download_model_file(self):
        logger.info(f"Downloading model file {self._model_file.readable_source}")
        model_paths = downloaders.download_model(
            self._model_file,
            local_dir=self._model_file.local_dir,
            cache_dir=self._config.cache_dir,
            ollama_library_base_url=self._config.ollama_library_base_url,
            huggingface_token=self._config.huggingface_token,
        )
        self._update_model_file(
            self._model_file.id,
            state=ModelFileStateEnum.READY,
            download_progress=100,
            resolved_paths=model_paths,
        )
        logger.info(f"Successfully downloaded {self._model_file.readable_source}")

    def hijack_tqdm_progress(task_self):
        """
        Monkey patch the tqdm progress bar to update the model instance download progress.
        tqdm is used by hf_hub_download under the hood.
        """
        from tqdm import tqdm

        _original_init = tqdm.__init__
        _original_update = tqdm.update

        def _new_init(self: tqdm, *args, **kwargs):
            kwargs["disable"] = False  # enable the progress bar anyway
            _original_init(self, *args, **kwargs)

            if hasattr(task_self, '_model_file_size'):
                # Resume downloading
                task_self._model_downloaded_size += self.n

        def _new_update(self: tqdm, n=1):
            _original_update(self, n)

            if task_self._cancel_flag.is_set():
                raise CancelledError("Download cancelled")

            # This is the default for single tqdm downloader like ollama
            # TODO we may want to unify to always get the size before downloading.
            total_size = self.total
            downloaded_size = self.n
            if hasattr(task_self, '_model_file_size'):
                # This is summary for group downloading
                total_size = task_self._model_file_size
                task_self._model_downloaded_size += n
                downloaded_size = task_self._model_downloaded_size

            try:
                if (
                    time.time() - task_self._last_download_update_time < 2
                    and downloaded_size != total_size
                ):
                    # Only update after 2-second interval or download is completed.
                    return

                task_self._update_progress_func(
                    round((downloaded_size / total_size) * 100, 2)
                )
                task_self._last_download_update_time = time.time()
            except Exception as e:
                raise Exception(f"Failed to update model file: {e}")

        tqdm.__init__ = _new_init
        tqdm.update = _new_update

    def _ensure_model_file_size(self):
        if self._model_file.size is not None:
            return

        size = downloaders.get_model_file_size(
            self._model_file,
            huggingface_token=self._config.huggingface_token,
            cache_dir=self._config.cache_dir,
            ollama_library_base_url=self._config.ollama_library_base_url,
        )
        self._model_file.size = size
        self._update_model_file(self._model_file.id, size=size)

    def _update_model_file_progress(self, model_file_id: int, progress: float):
        self._update_model_file(model_file_id, download_progress=progress)

    def _update_model_file(self, id: int, **kwargs):
        model_file_public = self._clientset.model_files.get(id=id)

        model_file_update = ModelFileUpdate(**model_file_public.model_dump())
        for key, value in kwargs.items():
            setattr(model_file_update, key, value)

        self._clientset.model_files.update(id=id, model_update=model_file_update)
