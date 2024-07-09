import json
import os
from gpustack.scheduler.calculator import modelResoruceClaim
from gpustack.scheduler.policy import SystemReservedResource
from gpustack.schemas.workers import (
    GPUDeviceInfo,
    MemoryInfo,
    Worker,
    WorkerStatus,
)


def llama3_8b_estimate_claim():
    # gguf-parser-darwin-universal(0.2.1) -ol-crawl -ol-model llama3:8b --offload-layers=-1 --offload-layers-step=1 -ctx-size -1 -flash-attention -skip-tokenizer -skip-architecture -skip-model -json
    return load_model_estimate_claim_from_file("llama3_8b_estimate_claim.json")


def llama3_70b_estimate_claim():
    # gguf-parser-darwin-universal(0.2.1) -ol-crawl -ol-model llama3:70b --offload-layers=-1 --offload-layers-step=1 -ctx-size -1 -flash-attention -skip-tokenizer -skip-architecture -skip-model -json
    return load_model_estimate_claim_from_file("llama3_70b_estimate_claim.json")


def worker_macos_metal():
    return load_worker_from_file("worker_macos_metal.json")


def worker_linux_nvidia_single_gpu():
    return load_worker_from_file("worker_linux_nvidia_single_gpu.json")


def worker_linux_nvidia_multi_gpu():
    return load_worker_from_file("worker_linux_nvidia_multi_gpu.json")


def system_reserved(memory=0, gpu_memory=0):
    return SystemReservedResource(memory=memory, gpu_memory=gpu_memory)


def load_worker_from_file(file_name) -> Worker:
    dir = os.path.dirname(__file__)
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'r') as file:
        worker_dict = json.loads(file.read())
        status_dict = worker_dict.get("status")
        memory = status_dict.get("memory")
        gpu_devices = status_dict.get("gpu_devices")

        status = WorkerStatus(**status_dict)
        status.memory = MemoryInfo(**memory)
        status.gpu_devices = [GPUDeviceInfo(**device) for device in gpu_devices]
        worker = Worker(**worker_dict)
        worker.status = status
    return worker


def load_model_estimate_claim_from_file(file_name) -> modelResoruceClaim:
    dir = os.path.dirname(__file__)
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'r') as file:
        model_estimate_claim = modelResoruceClaim.from_json(file.read())
    return model_estimate_claim
