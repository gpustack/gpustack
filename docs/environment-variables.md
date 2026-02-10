# Environment Variables

GPUStack supports various environment variables for configuration.

Most command line parameters can also be set via environment variables with the `GPUSTACK_` prefix and in uppercase format (e.g., `--data-dir` can be set via `GPUSTACK_DATA_DIR`).

For a complete list of command line parameters that can be set as environment variables, see [CLI Reference](cli-reference/start.md#available-environment-variables). This will not be discussed here.

### Priority Order

Configuration values are applied in the following priority order (highest to lowest):

1. Command line arguments
2. Environment variables
3. Configuration file
4. Default values

This means that command line arguments will always override environment variables, and environment variables will override values in the configuration file.

## GPUStack Core Environment Variables

These environment variables are typically used for third-party service integrations.

The **Applies to** column indicates where the environment variable should be set:

- **Server** - Applies to the GPUStack server.
- **Worker** - Applies to GPUStack workers.
- **Model** - Applies to model deployment configurations.

### Proxy Configuration

| <div style="width:100px">Variable</div> | Description                                                                                                                 | Default | Applies to      |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------- | --------------- |
| `HTTP_PROXY`                            | HTTP proxy URL. e.g., `http://proxy-server:port`                                                                            | (empty) | Server & Worker |
| `HTTPS_PROXY`                           | HTTPS proxy URL. e.g., `https://proxy-server:port`                                                                          | (empty) | Server & Worker |
| `NO_PROXY`                              | Comma-separated list of hosts to exclude. e.g., `127.0.0.1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/16,localhost,cluster.local` | (empty) | Server & Worker |

### Hugging Face Hub

| <div style="width:100px">Variable</div> | Description                                              | Default | Applies to      |
| --------------------------------------- | -------------------------------------------------------- | ------- | --------------- |
| `HF_ENDPOINT`                           | Hugging Face Hub endpoint. e.g., `https://hf-mirror.com` | (empty) | Server & Worker |
| `HF_TOKEN`                              | Hugging Face Hub access token.                           | (empty) | Server & Worker |

### Database Configuration

| Variable                                 | Description                                                                | Default | Applies to |
| ---------------------------------------- | -------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_DB_ECHO`                       | Enable database query logging.                                             | `false` | Server     |
| `GPUSTACK_DB_POOL_SIZE`                  | Database connection pool size.                                             | `10`    | Server     |
| `GPUSTACK_DB_MAX_OVERFLOW`               | Database connection pool max overflow.                                     | `10`    | Server     |
| `GPUSTACK_DB_POOL_TIMEOUT`               | Database connection pool timeout in seconds.                               | `30`    | Server     |
| `GPUSTACK_DB_SUBSCRIBE_INIT_CONCURRENCY` | Maximum concurrent subscriptions that can perform initial DB list queries. | `20`    | Server     |

### Network Configuration

| Variable                         | Description                      | Default | Applies to      |
| -------------------------------- | -------------------------------- | ------- | --------------- |
| `GPUSTACK_PROXY_TIMEOUT_SECONDS` | Proxy timeout in seconds.        | `1800`  | Server          |
| `GPUSTACK_TCP_CONNECTOR_LIMIT`   | HTTP client TCP connector limit. | `1000`  | Server & Worker |

### Authentication & Security

| Variable                            | Description                           | Default | Applies to |
| ----------------------------------- | ------------------------------------- | ------- | ---------- |
| `GPUSTACK_JWT_TOKEN_EXPIRE_MINUTES` | JWT token expiration time in minutes. | `120`   | Server     |

### Gateway Configuration

| Variable                                  | Description                                                                         | Default | Applies to |
| ----------------------------------------- | ----------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS`    | Higress external authentication timeout in milliseconds.                            | `30000` | Server     |
| `GPUSTACK_GATEWAY_PORT_CHECK_INTERVAL`    | The interval in seconds of GPUStack Server checking embedded gateway listening port | `2`     | Server     |
| `GPUSTACK_GATEWAY_PORT_CHECK_RETRY_COUNT` | The retry count of GPUStack Server checking embedded gateway listening port         | `300`   | Server     |
| `GPUSTACK_GATEWAY_EXTERNAL_METRICS_URL`   | The external gateway metrics url. e.g. `http://<gateway-ip>:15020/stats/prometheus` | None    | Server     |

### Cluster Configuration

| Variable                              | Description                                                                                                     | Default | Applies to |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_DEFAULT_CLUSTER_KUBERNETES` | If a default cluster is created automatically, it will use the `Kubernetes` provider when this variable is set. | `false` | Server     |

### Worker and Model Configuration

| Variable                                                         | Description                                                                                                          | Default | Applies to |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_WORKER_HEARTBEAT_INTERVAL`                             | Worker heartbeat interval in seconds.                                                                                | `30`    | Worker     |
| `GPUSTACK_WORKER_STATUS_SYNC_INTERVAL`                           | Worker status synchronization interval in seconds.                                                                   | `30`    | Worker     |
| `GPUSTACK_WORKER_UNREACHABLE_CHECK_MODE`                         | Worker unreachable check mode. Options: `auto`, `enabled`, `disabled`. `auto` disables check when worker count > 50. | `auto`  | Server     |
| `GPUSTACK_WORKER_HEARTBEAT_GRACE_PERIOD`                         | Worker heartbeat grace period in seconds.                                                                            | `150`   | Server     |
| `GPUSTACK_MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD`                | Model instance reschedule grace period in seconds.                                                                   | `300`   | Server     |
| `GPUSTACK_MODEL_EVALUATION_CACHE_MAX_SIZE`                       | Maximum size of model evaluation cache.                                                                              | `1000`  | Server     |
| `GPUSTACK_MODEL_EVALUATION_CACHE_TTL`                            | TTL of model evaluation cache in seconds.                                                                            | `3600`  | Server     |
| `GPUSTACK_WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD`           | Worker orphan workload cleanup grace period in seconds.                                                              | `300`   | Worker     |
| `GPUSTACK_WORKER_ORPHAN_BENCHMARK_WORKLOAD_CLEANUP_GRACE_PERIOD` | Worker orphan benchmark workload cleanup grace period in seconds.                                                    | `300`   | Worker     |
| `GPUSTACK_WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS`             | Add debug log for slow worker status collection if it exceeds this time in seconds.                                  | `180`   | Worker     |
| `GPUSTACK_MODEL_INSTANCE_HEALTH_CHECK_INTERVAL`                  | Model instance health check interval in seconds.                                                                     | `3`     | Worker     |
| `GPUSTACK_DISABLE_OS_FILELOCK`                                   | Disable OS file lock.                                                                                                | `false` | Worker     |
| `GPUSTACK_AUTO_GENERATE_UUID`                                    | If set to true, always generate a new worker UUID at startup instead of reading the UUID from the system.            | `false` | Worker     |

### Benchmark Configuration

| Variable                                   | Description                                                                                                                                      | Default                                                                               | Applies to |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ---------- |
| `GPUSTACK_BENCHMARK_DATASET_SHAREGPT_PATH` | ShareGPT dataset path used by the benchmark container when `dataset_name` is set to `ShareGPT`. The default image already includes this dataset. | `/workspace/benchmark-runner/sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json` | Worker     |
| `GPUSTACK_BENCHMARK_REQUEST_TIMEOUT`       | Timeout for each benchmark request in seconds.                                                                                                   | `3600`                                                                                | Worker     |

### Model Deployment Configuration

!!! note

    These environment variables are **not** set when starting GPUStack. Instead, they should be configured in the **Advanced Options > Environment Variables** section when deploying a model. They are used to customize the model serving behavior.

| <div style="width:180px">Variable</div>          | Description                                                                                                                                                                                    | Default | Applies to |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED` | Disable the automatic serving command script execution. When set to `1` or `true`, the script that handles package installation and other setup tasks will not run.                            | `0`     | Model      |
| `PYPI_PACKAGES_INSTALL`                          | Additional PyPI packages to install in the model serving environment. Multiple packages should be space-separated. The script will use `uv pip install` if available, otherwise `pip install`. | (empty) | Model      |
| `GPUSTACK_MODEL_RAM_CLAIM`                       | User-declared RAM requirement (in Byte) for the model, used by the scheduler for capacity planning.                                                                                            | (empty) | Model      |
| `GPUSTACK_MODEL_VRAM_CLAIM`                      | User-declared VRAM requirement (in Byte) for the model, used by the scheduler for capacity planning.                                                                                           | (empty) | Model      |
| `GPUSTACK_APPLY_QWEN3_RERANKER_TEMPLATES`        | Apply Qwen3 reranker templates to the request body. See instructions in https://huggingface.co/Qwen/Qwen3-Reranker-0.6B.                                                                       | (empty) | Model      |
| `GPUSTACK_SKIP_MODEL_EVALUATION`                 | Skips the model evaluation or validation step during deployment.                                                                                                                               | (empty) | Model      |
| `GPUSTACK_DISABLE_METRICS`                       | Disables metric expose and collection for the model.                                                                                                                                           | (empty) | Model      |
| `GPUSTACK_MODEL_HEALTH_CHECK_PATH`               | Specifies the HTTP health check path exposed by the model.                                                                                                                                     | (empty) | Model      |
| `GPUSTACK_MODEL_RUNTIME_UID`                     | Control the user permissions of processes running inside the container.                                                                                                                        | (empty) | Model      |
| `GPUSTACK_MODEL_RUNTIME_GID`                     | Control the group permissions of processes running inside the container.                                                                                                                       | (empty) | Model      |
| `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB`            | Shared memory size for the container in GiB.                                                                                                                                                   | `10.0`  | Model      |

#### Usage Example

When deploying a model, navigate to **Advanced Options > Environment Variables** and add:

```bash
# Install additional packages before model serving starts
PYPI_PACKAGES_INSTALL=torch-audio==2.0.0 transformers==4.30.0

# Disable the serving command script entirely
GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED=1
```

The serving command script automatically handles:

- Installing additional PyPI packages specified in `PYPI_PACKAGES_INSTALL`
- Supporting both `uv pip` and `pip` for package installation
- Handling custom PyPI indices via `PIP_INDEX_URL` and `PIP_EXTRA_INDEX_URL`

## GPUStack Runtime Environment Variables

These environment variables are used by GPUStack runtime. Commonly used to adjust the behavior of inference backends running in Docker/Kubernetes.

They are only usable within workers. Please set the environment variables in the workers’ containers to ensure they take effect properly.

### Global Variables

| Variable                         | Description                | Default |
| -------------------------------- | -------------------------- | ------- |
| `GPUSTACK_RUNTIME_LOG_LEVEL`     | Log level.                 | `INFO`  |
| `GPUSTACK_RUNTIME_LOG_WARNING`   | Enable logging warnings.   | `0`     |
| `GPUSTACK_RUNTIME_LOG_EXCEPTION` | Enable logging exceptions. | `0`     |

### Detector Variables

| Variable                                           | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Default                                 |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| `GPUSTACK_RUNTIME_DETECT`                          | Detector to use. Options: Auto, AMD, ASCEND, CAMBRICON, HYGON, ILUVATAR, METAX, MTHREADS, NVIDIA.                                                                                                                                                                                                                                                                                                                                                                                                                                          | `Auto`                                  |
| `GPUSTACK_RUNTIME_DETECT_NO_PCI_CHECK`             | Enable no PCI check during detection. Useful for WSL environments.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | (empty)                                 |
| `GPUSTACK_RUNTIME_DETECT_NO_TOOLKIT_CALL`          | Enable only using management libraries calls during detection. Device detection typically involves calling platform-side management libraries and platform-side toolkit to retrieve extra information. For example, during NVIDIA detection, the NVML and CUDA are called, with CUDA used to retrieve GPU cores. However, if certain toolchains are not correctly installed in the environment, such as the Nvidia Fabric Manager being missing, calling the CUDA can cause blocking. Enabling this parameter can prevent blocking events. | `0`                                     |
| `GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY` | The detected backend mapping to resource keys, e.g `{"cuda": "nvidia.com/devices", "rocm": "amd.com/devices"}`. Used to map the gpustack-runner's backend name to the corresponding resource key.                                                                                                                                                                                                                                                                                                                                          | The default values named by each vendor |
| `GPUSTACK_RUNTIME_DETECT_PHYSICAL_INDEX_PRIORITY`  | Use physical index priority at detecting devices.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `1`                                     |

#### ROCm Detector Specific Variables

!!! note

    Also applicable to ROCm-based backends, like Hygon.

| Variable             | Description             | Default     |
| -------------------- | ----------------------- | ----------- |
| `ROCM_SMI_LIB_PATH`  | ROCm SMI library path.  | (empty)     |
| `ROCM_HOME`          | ROCm home directory.    | (empty)     |
| `ROCM_PATH`          | ROCm path.              | `/opt/rocm` |
| `ROCM_CORE_LIB_PATH` | ROCm core library path. | (empty)     |

### Deployer Variables

| Variable                                                           | Description                                                                                                                                                                                                                                                                                                                                               | Default                                         |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `GPUSTACK_RUNTIME_DEPLOY`                                          | Deployer to use. Options: Auto, Docker, Kubernetes, Podman(Experimental).                                                                                                                                                                                                                                                                                 | `Auto`                                          |
| `GPUSTACK_RUNTIME_DEPLOY_API_CALL_ERROR_DETAIL`                    | Enable detailing the API call error during deployment.                                                                                                                                                                                                                                                                                                    | `1`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_PRINT_CONVERSION`                         | Enable printing the conversion during deployment. GPUStack Runtime provides a unified Workload definition API, which will be converted to the specific Container Runtime API calls(e.g., Docker SDK, Kubernetes API, Podman SDK). Enabling this option will print the final converted API calls in INFO log for debugging purposes.                       | `0`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_ASYNC`                                    | Enable asynchronous deployment.                                                                                                                                                                                                                                                                                                                           | `1`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_ASYNC_THREADS`                            | The number of threads in the threadpool.                                                                                                                                                                                                                                                                                                                  | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME`                            | The name of the deployer.                                                                                                                                                                                                                                                                                                                                 | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_ENVIRONMENTS`  | Environment variable names to ignore during mirrored deployment.                                                                                                                                                                                                                                                                                          | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES`       | Volume mount destinations to ignore during mirrored deployment.                                                                                                                                                                                                                                                                                           | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME`      | Username for the default container registry.                                                                                                                                                                                                                                                                                                              | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD`      | Password for the default container registry.                                                                                                                                                                                                                                                                                                              | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE`              | Namespace for default runner container images.                                                                                                                                                                                                                                                                                                            | `gpustack`                                      |
| `GPUSTACK_RUNTIME_DEPLOY_IMAGE_PULL_POLICY`                        | Image pull policy for the deployer (e.g., Always, IfNotPresent, Never).                                                                                                                                                                                                                                                                                   | `IfNotPresent`                                  |
| `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY`                      | During deployment, path of directory containing Container Device Interface (CDI) specifications, or the directory to generate CDI specifications into.                                                                                                                                                                                                    | `/var/run/cdi`                                  |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI`                     | Manual mapping of container device interfaces, which is used to tell the Container Runtime which devices to inject into the container, e.g., `nvidia.com/devices=nvidia.com/gpu;amd.com/devices=amd.com/gpu`. The key is the resource key, and the value is the Container Device Interface(CDI) key.                                                      | The default values named by each vendor         |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES` | Manual mapping of runtime visible devices environment variables, which is used to tell the Container Runtime which devices to inject into the container, e.g., `nvidia.com/devices=NVIDIA_VISIBLE_DEVICES;amd.com/devices=AMD_VISIBLE_DEVICES`. The key is the resource key, and the value is the environment variable name.                              | The default values named by each vendor         |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_BACKEND_VISIBLE_DEVICES` | Manual mapping of backend visible devices environment variables, which is used to tell the Device Runtime (e.g., ROCm, CUDA, OneAPI) which devices to use inside the container, e.g., `nvidia.com/devices=CUDA_VISIBLE_DEVICES;amd.com/devices=ROCR_VISIBLE_DEVICES`. The key is the resource key, and the value is a list of environment variable names. | The default values named by each vendor         |
| `GPUSTACK_RUNTIME_DEPLOY_RUNTIME_VISIBLE_DEVICES_VALUE_UUID`       | Use UUIDs for the given runtime visible devices environment variables.                                                                                                                                                                                                                                                                                    | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_BACKEND_VISIBLE_DEVICES_VALUE_ALIGNMENT`  | Enable value alignment for the given backend visible devices environment variables.                                                                                                                                                                                                                                                                       | `ASCEND_RT_VISIBLE_DEVICES,NPU_VISIBLE_DEVICES` |
| `GPUSTACK_RUNTIME_DEPLOY_CPU_AFFINITY`                             | Enable CPU affinity for deployed workloads.                                                                                                                                                                                                                                                                                                               | `0`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_NUMA_AFFINITY`                            | Enable NUMA affinity for deployed workloads. When enabled, `GPUSTACK_RUNTIME_DEPLOY_CPU_AFFINITY` is also implied.                                                                                                                                                                                                                                        | `0`                                             |

#### Docker Deployer Specific Variables

| Variable                                            | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Default                            |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `GPUSTACK_RUNTIME_DOCKER_HOST`                      | Host for Docker connection. Used to override the default Docker host.                                                                                                                                                                                                                                                                                                                                                                                                                         | `http+unix:///var/run/docker.sock` |
| `GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE`               | Container image used for the pause container in Docker.                                                                                                                                                                                                                                                                                                                                                                                                                                       | `gpustack/runtime:pause`           |
| `GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE`   | Container image used for unhealthy restart container in Docker.                                                                                                                                                                                                                                                                                                                                                                                                                               | `gpustack/runtime:health`          |
| `GPUSTACK_RUNTIME_DOCKER_RESOURCE_INJECTION_POLICY` | Resource injection policy for the Docker deployer (e.g., Env, CDI). <br/>`Env`: Injects resources using standard environment variable, based on `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES`. <br/>`CDI`: Injects resources using CDI, based on `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI`.                                                                                                                                                                         | `Env`                              |
| `GPUSTACK_RUNTIME_DOCKER_CDI_SPECS_GENERATE`        | Generate CDI specifications during deployment when using CDI resource injection policy, requires `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` to exist. Works only when `GPUSTACK_RUNTIME_DOCKER_RESOURCE_INJECTION_POLICY` is set to `CDI`. Using internal knowledge to generate the CDI specifications for deployer, if the output file conflicts with other tools generating CDI specifications(e.g., NVIDIA Container Toolkit), please disable this and remove the output file manually. | `1`                                |

#### Kubernetes Deployer Specific Variables

| Variable                                                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Default         |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `GPUSTACK_RUNTIME_KUBERNETES_NODE_NAME`                      | Name of the Kubernetes Node to deploy workloads to.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | (empty)         |
| `GPUSTACK_RUNTIME_KUBERNETES_NAMESPACE`                      | Namespace of the Kubernetes to deploy workloads to.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `default`       |
| `GPUSTACK_RUNTIME_KUBERNETES_DOMAIN_SUFFIX`                  | Domain suffix for Kubernetes services.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `cluster.local` |
| `GPUSTACK_RUNTIME_KUBERNETES_SERVICE_TYPE`                   | Service type for Kubernetes services. Options: ClusterIP, NodePort, LoadBalancer.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `ClusterIP`     |
| `GPUSTACK_RUNTIME_KUBERNETES_QUORUM_READ`                    | Whether to use quorum read for Kubernetes services.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `0`             |
| `GPUSTACK_RUNTIME_KUBERNETES_DELETE_PROPAGATION_POLICY`      | Deletion propagation policy for Kubernetes resources. Options: Foreground, Background, Orphan.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `Foreground`    |
| `GPUSTACK_RUNTIME_KUBERNETES_RESOURCE_INJECTION_POLICY`      | Resource injection policy for the Kubernetes deployer. Options: Auto, Env, KDP. <br/>`Auto`: Automatically choose the resource injection policy based on the environment. <br/>`Env`: Injects resources using standard environment variable, depends on underlying Container Toolkit, based on `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES`. <br/>`KDP`: Injects resources using Kubernetes Device Plugin.                                                                                                                                                                                                                                                                                                            | `Auto`          |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_PER_DEVICE_MAX_ALLOCATIONS` | Maximum allocations for one device in Kubernetes Device Plugin.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `10`            |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_DEVICE_ALLOCATION_POLICY`   | Device allocation policy for the Kubernetes Device Plugin. Options: Auto, CDI, Env, Opaque. <br/>`Auto`: Automatically choose the device allocation policy based on the environment. <br/>`Env`: Allocates devices using runtime-visible environment variables, based on `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES`; requires Container Toolkit support. <br/>`CDI`: Allocates devices using generated CDI specifications, based on `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI`, making it easy to debug and troubleshoot; requires `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` to exist. <br/>`Opaque`: Uses internal logic for allocation, which is convenient for deployment but difficult to troubleshoot. | `Auto`          |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_CDI_SPECS_GENERATE`         | Generate CDI specifications during deployment, requires `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` to exist. Works only when `GPUSTACK_RUNTIME_KUBERNETES_KDP_DEVICE_ALLOCATION_POLICY` is set to `CDI`. Using internal knowledge to generate the CDI specifications for deployer, if the output file conflicts with other tools generating CDI specifications(e.g., NVIDIA Container Toolkit), please disable this and remove the output file manually.                                                                                                                                                                                                                                                                                 | `1`             |

#### Podman Deployer Specific Variables

!!! note

    Podman deployer is experimental and needs version 4.9 or higher.

| Variable                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                      | Default                               |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `GPUSTACK_RUNTIME_PODMAN_HOST`                    | Host for Podman connection. Used to override the default Podman host.                                                                                                                                                                                                                                                                                                                                                            | `http+unix:///run/podman/podman.sock` |
| `GPUSTACK_RUNTIME_PODMAN_PAUSE_IMAGE`             | Container image used for the pause container in Podman. Default is same as `GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE`.                                                                                                                                                                                                                                                                                                                | `gpustack/runtime:pause`              |
| `GPUSTACK_RUNTIME_PODMAN_UNHEALTHY_RESTART_IMAGE` | Container image used for unhealthy restart container in Podman. Default is same as `GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE`.                                                                                                                                                                                                                                                                                            | `gpustack/runtime:health`             |
| `GPUSTACK_RUNTIME_PODMAN_CDI_SPECS_GENERATE`      | Generate CDI specifications during deployment, requires `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` to exist. Using internal knowledge to generate the CDI specifications for deployer, if the output file conflicts with other tools generating CDI specifications(e.g., NVIDIA Container Toolkit), please disable this and remove the output file manually. Default is same as `GPUSTACK_RUNTIME_DOCKER_CDI_SPECS_GENERATE`. | `1`                                   |
