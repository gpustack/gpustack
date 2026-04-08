from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    ModelInstanceStateEnum,
)
from gpustack.worker.serve_manager import ServeManager
from tests.utils.model import new_model, new_model_instance


def _build_serve_manager(worker_id: int = 1):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    cfg = SimpleNamespace(log_dir="/tmp")
    manager = ServeManager(lambda: worker_id, lambda: clientset, cfg)
    manager._inference_backend_manager = MagicMock()
    return manager, clientset


def test_sync_model_instances_state_marks_main_unreachable_when_subordinate_unreachable():
    manager, clientset = _build_serve_manager()

    model_instance = new_model_instance(
        1,
        "distributed-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
    )
    model_instance.worker_ip = "127.0.0.1"
    model_instance.port = 8000
    model_instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.RUN_FIRST,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                worker_name="worker-2",
                worker_ip="10.0.0.2",
                state=ModelInstanceStateEnum.UNREACHABLE,
                state_message="Worker is unreachable from the server",
            )
        ],
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model.backend = BackendEnum.VLLM
    model.backend_version = "0.8.0"

    with (
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state="running"),
        ),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=model),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once_with(
        model_instance.id,
        state=ModelInstanceStateEnum.UNREACHABLE,
        state_message=(
            "Distributed serving unreachable in subordinate worker "
            "10.0.0.2: Worker is unreachable from the server."
        ),
    )
