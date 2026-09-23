import logging
import os
import re
import sys
import shlex
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
from transformers import PretrainedConfig

from gpustack_runner.runner import BackendVersionedRunner
from gpustack_runtime.deployer import (
    ContainerResources,
    ContainerMount,
    ContainerMountModeEnum,
    ContainerPort,
)
from gpustack_runtime.deployer.__utils__ import compare_versions
from gpustack_runtime.detector import (
    ManufacturerEnum,
    available_backends,
)
from gpustack_runtime.detector.ascend import get_ascend_cann_variant
from gpustack_runtime import envs as runtime_envs
from gpustack_runtime.envs import (
    to_bool,
)
from gpustack_runtime.logging import setup_logging as setup_runtime_logging
from gpustack_runtime.deployer.docker import DockerWorkloadPlan
from gpustack_runtime.deployer import WorkloadPlan

from gpustack.client.generated_clientset import ClientSet
from gpustack import envs
from gpustack.config.config import Config, set_global_config
from gpustack.logging import setup_logging
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    ContainerEnvConfig,
    ParameterFormatEnum,
)
from gpustack.schemas.runner_source import (
    RunnerOverrideEntriesPublic,
    RunnerOverrideEntryPublic,
    merged_backend_runners,
)
from gpustack.schemas.models import (
    get_backend,
    role_container_resources,
    role_takes_no_accelerator,
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStateEnum,
    ModelUpdate,
    ModelInstanceDeploymentMetadata,
    role_effective_model,
)
from gpustack.schemas.workers import GPUDevicesStatus
from gpustack.server.bus import Event
from gpustack.utils.command import flatten_to_argv, is_parameter_key
from gpustack.utils.config import apply_registry_override_to_image
from gpustack.utils.envs import filter_env_vars
from gpustack.utils.template import deployment_variables, render, render_values
from gpustack.utils.hub import get_hf_text_config, get_max_model_len
from gpustack.utils.hub import get_pretrained_config, safe_pretrained_config_from_dict
from gpustack.utils.profiling import time_decorator
from gpustack.utils import platform
from gpustack.utils.version import pick_runtime_version
from gpustack.utils.runtime import transform_workload_plan
from gpustack.worker.pd_injection import PDInjection, render_pd_injection

logger = logging.getLogger(__name__)

# Distinguishes "the caller said nothing" from "the caller said None". A
# managed router legitimately has no backend row of its own, and passing None
# has to mean that rather than falling back to this server's.
_INHERIT = object()
lock = threading.Lock()


class ModelInstanceStateError(Exception):
    pass


def _normalize_param_format(
    tokens: List[str], target: ParameterFormatEnum
) -> List[str]:
    """
    Walk an argv-style token stream, regroup each ``--key [value...]`` cluster,
    and emit each cluster in ``target`` format. The key's leading dashes are
    preserved verbatim — ``-n`` / ``-ngl`` / ``-m`` are real llama.cpp short
    options and must not be coerced into ``--n`` / ``--ngl`` / ``--m``.

    Multi-value clusters (``--lora-modules v1 v2``) always stay in space form —
    ``--key=v1 --key=v2`` would change argparse semantics (the later value
    overwrites the earlier one), so equal form is unsafe to use for them.

    Stray positional tokens that do not follow a key (rare; only happens on
    malformed input) pass through verbatim — let the inference server reject
    them rather than guessing.
    """
    result: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not is_parameter_key(tok):
            result.append(tok)
            i += 1
            continue

        if "=" in tok:
            head_key, _, head_val = tok.partition("=")
            values = [head_val]
            i += 1
        else:
            head_key = tok
            values = []
            i += 1
            while i < n and not is_parameter_key(tokens[i]):
                values.append(tokens[i])
                i += 1

        if not values:
            result.append(head_key)
        elif len(values) == 1 and target == ParameterFormatEnum.EQUAL:
            result.append(f"{head_key}={values[0]}")
        elif len(values) == 1:
            result.extend([head_key, values[0]])
        else:
            result.append(head_key)
            result.extend(values)

    return result


# Reference: requirements for `usage.prompt_tokens_details.cached_tokens` to
# appear in OpenAI-compatible responses, per backend.
#
# Tracked in https://github.com/gpustack/gpustack/issues/5189.
#
# vLLM
# ----
#   - Prefix caching must be on. V1 enables it by default; V0 requires
#     `--enable-prefix-caching`. *User responsibility.*
#   - `--enable-prompt-tokens-details` is required to populate the field.
#     Broken in V1 prior to v0.9.0.1
#     (https://github.com/vllm-project/vllm/pull/18149).
#   - GPUStack auto-injects `--enable-prompt-tokens-details` for vLLM
#     >= 0.9.0.1. See `gpustack.worker.backends.vllm.get_cache_report_arguments`.
#
# SGLang
# ------
#   - RadixAttention prefix caching is on by default; no extra flag.
#   - `--enable-cache-report` is required to populate the field (since v0.3.4).
#   - GPUStack auto-injects `--enable-cache-report` for SGLang >= 0.3.4. See
#     `gpustack.worker.backends.sglang.get_cache_report_arguments`.
#
# Ascend MindIE
# -------------
#   - Prefix caching must be on. *User responsibility:* add
#     `--enable-prefix-caching` to the model's backend parameters. Conflicts
#     with `--rope-scaling` and `--data-parallel-size > 1` (validated by
#     `AscendMindIEParameters._validate`).
#   - Cache token details require MindIE >= 2.3.0, GPUStack does not auto-inject any cache-related flag for MindIE.


class InferenceServer(ABC):
    _model_path: Optional[str] = None
    _draft_model_path: Optional[str] = None
    """
    The absolute path to the model files.
    This is set when the model instance state changes to STARTING.
    """

    _pretrained_config: Optional[PretrainedConfig] = None
    """The model configuration, if available."""
    _pretrained_config_initialized: bool = False
    """Whether pretrained config loading has been attempted."""

    _fallback_registry: Optional[str] = None
    """The fallback container registry to use if needed."""

    _runner_overrides: Optional[List[RunnerOverrideEntryPublic]] = None
    """The runner overrides this deploy resolved images against, fetched once."""

    _model_spec = None
    """The model as the server holds it, before the role's overrides are
    projected onto it. `self._model` is the projection and is what everything
    reads; this is only for the one path that writes the model back, which must
    not push a role's values up to the Model-level spec."""

    _pd_injection_cache: Optional[PDInjection] = None
    _pd_injection_resolved: bool = False
    """Rendered once per deploy: three seams read it (env, files, arguments)
    and rendering logs every unresolved placeholder, so doing it three times
    would triple the warnings for one fact."""

    @time_decorator
    def __init__(
        self,
        clientset: ClientSet,
        mi: ModelInstance,
        cfg: Config,
        worker_id: int,
        inference_backend: InferenceBackend,
        fallback_registry: Optional[str] = None,
    ):
        setup_logging(debug=cfg.debug)
        setup_runtime_logging()
        set_global_config(cfg)

        try:
            self._clientset = clientset
            self._model_instance = mi
            self._config = cfg
            self._fallback_registry = fallback_registry
            self._worker = self._clientset.workers.get(worker_id)
            if not self._worker:
                raise KeyError(f"Worker {worker_id} not found")

            self.get_model()
            self.inference_backend = inference_backend

            # A managed router's image and command come from the catalog and
            # its peers' live addresses, so they are materialised rather than
            # stored — the addresses change on every scale.
            #
            # Placed exactly here, between two things that both constrain it.
            # It must come after `inference_backend` is assigned, because
            # resolving the runner image reads it; doing this inside
            # `get_model()` left the image as the literal `{{runner_image}}`,
            # which Kubernetes rejected as an invalid reference. And it must
            # come before the fallback below, because what that fallback needs
            # in order to synthesise a custom backend — an image and a run
            # command — is precisely what this produces.
            self._model = self._apply_managed_router(self._model)

            if (
                not inference_backend
                and self._model.image_name
                and self._model.run_command
            ):
                # Any deployment that directly specifies an image and command is treated as a Custom backend.
                # A basic InferenceBackend object is created to prevent exceptions in subsequent workflows.
                self.inference_backend = InferenceBackend(
                    backend_name=BackendEnum.CUSTOM.value,
                    run_command=self._model.run_command,
                )
            if not self.inference_backend:
                raise KeyError(
                    f"Inference backend {self._model.backend} not specified or not found"
                )

            logger.info("Preparing model files...")

            self._until_model_instance_starting()

            logger.info("Model files are ready.")
        except ModelInstanceStateError:
            sys.exit(1)
        except Exception as e:
            error_message = f"Failed to initialize: {e}"
            logger.error(error_message)
            try:
                patch_dict = {
                    "state_message": error_message,
                    "state": ModelInstanceStateEnum.ERROR,
                }
                self._update_model_instance(mi.id, **patch_dict)
            except Exception as ue:
                logger.error(f"Failed to update model instance: {ue}")
            sys.exit(1)

    def _stop_when_starting(self, event: Event) -> bool:
        if event.data["state"] == ModelInstanceStateEnum.ERROR:
            raise ModelInstanceStateError()
        elif event.data["state"] == ModelInstanceStateEnum.STARTING:
            resolved_path = event.data["resolved_path"]
            if not resolved_path:
                raise ValueError(
                    "Model instance reached STARTING without a resolved model path"
                )
            self._model_path = str(Path(resolved_path).absolute())
            if event.data["draft_model_resolved_path"]:
                self._draft_model_path = str(
                    Path(event.data["draft_model_resolved_path"]).absolute()
                )
            self._model_instance = ModelInstance.model_validate(event.data)
            return True

        return False

    @abstractmethod
    def start(self):
        pass

    def get_model(self):
        model = self._clientset.models.get(id=self._model_instance.model_id)
        # Keep the model as the server holds it, for the one path that writes
        # back: a projection must never be persisted, and a PUT built from one
        # would push a role's overrides up to the Model-level spec.
        self._model_spec = model
        # Apply the role's overrides before anything reads the model. This is
        # the only place the worker does it: everything below reads
        # `self._model.<field>` and knows nothing about roles. Projecting
        # first also means a role's own `backend_parameters` get the
        # `{data_dir}` substitution, which they would miss the other way
        # round.
        model = role_effective_model(model, self._model_instance.role)
        data_dir = self._config.data_dir
        for i, param in enumerate(model.backend_parameters or []):
            model.backend_parameters[i] = param.replace("{data_dir}", data_dir)

        self._model = model

    def _until_model_instance_starting(self):
        self._clientset.model_instances.watch(
            callback=None,
            stop_condition=self._stop_when_starting,
            params={"id": self._model_instance.id},
        )

    def _update_model_instance(self, id: int, **kwargs):
        mi_public = self._clientset.model_instances.get(id=id)

        mi = ModelInstanceUpdate(**mi_public.model_dump())
        for key, value in kwargs.items():
            setattr(mi, key, value)

        self._clientset.model_instances.update(id=id, model_update=mi)

    def _handle_error(self, error: Exception):
        """
        Handle errors during backend server startup in a unified way.
        Updates model instance state and re-raises the original error.
        """
        cause = getattr(error, "__cause__", None)
        cause_text = f": {cause}" if cause else ""
        error_message = f"Failed to run {self._model.backend}: {error}{cause_text}"

        try:
            is_main_worker = self._model_instance.worker_id == self._worker.id
            if is_main_worker:
                patch_dict = {
                    "state_message": error_message,
                    "state": ModelInstanceStateEnum.ERROR,
                }
                self._update_model_instance(self._model_instance.id, **patch_dict)
            else:
                # For subordinate workers, update sw.state instead of mi.state
                # to avoid race conditions with the main worker's state management.
                self._update_subordinate_worker_error(error_message)
        except Exception as ue:
            logger.error(f"Failed to update model instance: {ue}")

        raise error

    def _update_subordinate_worker_error(self, error_message: str):
        """
        Update the subordinate worker's state to ERROR.
        Fetches the latest model instance to get the current subordinate worker state,
        then updates only this worker's entry.
        """
        mi_public = self._clientset.model_instances.get(id=self._model_instance.id)
        mi = ModelInstanceUpdate(**mi_public.model_dump())
        sw_pos = next(
            (
                i
                for i, sw in enumerate(mi.distributed_servers.subordinate_workers)
                if sw.worker_id == self._worker.id
            ),
        )
        mi.distributed_servers.subordinate_workers[sw_pos].state = (
            ModelInstanceStateEnum.ERROR
        )
        mi.distributed_servers.subordinate_workers[sw_pos].state_message = error_message
        self._clientset.model_instances.update(
            id=self._model_instance.id, model_update=mi
        )

    def _get_deployment_metadata(self) -> ModelInstanceDeploymentMetadata:
        """
        Get the deployment metadata for the model instance.

        Returns:
            The deployment metadata.

        Raises:
            RuntimeError:
                If the model instance is not handling by the current worker.
        """
        deployment_metadata = self._model_instance.get_deployment_metadata(
            self._worker.id
        )
        if not deployment_metadata:
            raise RuntimeError(
                "Failed to get deployment metadata: model instance is not handling by the current worker"
            )
        return deployment_metadata

    def _get_pretrained_config(self) -> Optional[PretrainedConfig]:
        """
        Get the pretrained model configuration, if available.

        Returns:
            The pretrained model configuration dictionary, or None if not available.
        """
        if self._pretrained_config_initialized:
            return self._pretrained_config

        auto_config_error: Optional[Exception] = None
        try:
            pretrained_config = get_pretrained_config(self._model)
            if isinstance(pretrained_config, dict):
                # Ensure we have a PretrainedConfig object, not a dict, for consistency.
                pretrained_config = safe_pretrained_config_from_dict(pretrained_config)

            self._pretrained_config = pretrained_config
            self._pretrained_config_initialized = True

            return pretrained_config
        except Exception as e:
            logger.debug(
                f"Failed to get pretrained config via AutoConfig, falling back to local config.json. Error: {e}"
            )
            auto_config_error = e

        try:
            fallback_config = self._load_pretrained_config_from_local_config_json()
            self._pretrained_config = fallback_config
            self._pretrained_config_initialized = True
            return fallback_config
        except Exception as e:
            raise RuntimeError(
                "Failed to load pretrained config. "
                f"AutoConfig error: {auto_config_error}. "
                f"Local config.json fallback error: {e}."
            ) from e

    def _load_pretrained_config_from_local_config_json(
        self,
    ) -> Optional[PretrainedConfig]:
        """
        Load PretrainedConfig from local config.json under resolved model path.
        """
        if not self._model_path:
            return None

        config_path = os.path.join(self._model_path, "config.json")
        if not os.path.isfile(config_path):
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        if isinstance(config_dict, dict):
            return safe_pretrained_config_from_dict(config_dict)

        return None

    def _derive_max_model_len(self, default: Optional[int] = None) -> Optional[int]:
        """
        Derive max model length from model config.
        Returns default value if unavailable.

        Args:
            default:
                The default max model length to return if unable to derive from config.

        Returns:
            The derived max model length, or the default value if derivation fails.
        """
        try:
            pretrained_config = self._get_pretrained_config()
            pretrained_or_hf_text_config = get_hf_text_config(pretrained_config)
            return get_max_model_len(pretrained_or_hf_text_config)
        except Exception as e:
            logger.warning(
                f"Failed to derive max model length: {e}, continuing with default"
            )

        return default

    def _get_model_architecture(self) -> List[str]:
        """
        Get model architecture from model config.

        Returns:
            A list of model architecture strings.
        """
        try:
            pretrained_config = self._get_pretrained_config()
            if pretrained_config and hasattr(pretrained_config, "architectures"):
                return pretrained_config.architectures
        except Exception as e:
            logger.warning(
                f"Failed to derive model architecture: {e}, continuing with empty list"
            )

        return []

    def _template_variables(self, **overrides) -> Dict[str, object]:
        """The `{{name}}` values this instance can resolve.

        One builder for both rendering paths — the run command and the env
        values — so the two cannot drift into resolving different things.

        Every source is read tolerantly, on purpose. Rendering enriches an env
        value; it must not gain the power to end a start. A source that isn't
        resolvable yet — the device list before scheduling, the instance on a
        caller that only set the model — should leave its placeholder
        unresolved and logged, which is a diagnosable outcome, rather than
        raise from underneath `_get_configured_env`.
        """
        try:
            gpu_indexes = sorted(d.index for d in self._get_selected_gpu_devices())
        except Exception:
            gpu_indexes = None

        instance = getattr(self, "_model_instance", None)
        worker = getattr(self, "_worker", None)

        variables = deployment_variables(
            model_path=self._model_path,
            port=getattr(instance, "port", None),
            worker_ip=getattr(worker, "ip", None),
            model_name=getattr(instance, "model_name", None),
            gpu_count=len(gpu_indexes) if gpu_indexes is not None else None,
            gpu_ids=gpu_indexes,
            role=getattr(instance, "role", None),
            group_id=getattr(instance, "group_id", None),
        )
        variables.update(overrides)
        return variables

    def _pd_injection(self) -> Optional[PDInjection]:
        """What this instance's PD role adds to its launch, or None when the
        deployment is not disaggregated — in which case every seam below falls
        through to the path it takes today, byte for byte.

        The *unprojected* model is what goes in. A projection has already
        pushed this role's overrides up to the Model level, so a cross-role
        reference read off one — `{{roles.decode.tensor_parallel_size}}` while
        prefill is starting — would resolve decode's inherited parameters
        against prefill's values and produce a wrong number instead of a
        failure. The injector re-derives the running role's own effective
        values itself.
        """
        if self._pd_injection_resolved:
            return self._pd_injection_cache

        model = self._model_spec or getattr(self, "_model", None)
        # Cheap pre-check on the same conditions the injector returns None
        # for, so a non-PD deploy does not pay for the PD context (an image
        # resolution among other things).
        instance = getattr(self, "_model_instance", None)
        if getattr(model, "disaggregation", None) is None or not getattr(
            instance, "role", None
        ):
            self._pd_injection_resolved = True
            return None

        variables = self._template_variables(**self._pd_template_variables())
        # Marked resolved only once it is: a refused injection raises, and
        # caching "nothing to inject" for the seams that come after would turn
        # that hard failure into the silent aggregated start it exists to stop.
        injection = render_pd_injection(model, instance, variables)
        self._pd_injection_cache = injection
        self._pd_injection_resolved = True
        return injection

    def _apply_managed_router(self, model):
        """Materialise this instance's role if it is a router the catalog
        assembles. A no-op for every other role and every non-PD deployment.

        Peers are read from the instance's own generation, which is what makes
        a cross-generation router impossible rather than unlikely: the
        `group_id` filter cannot resolve a member of another generation even
        if one is running beside it.
        """
        from gpustack.worker.pd_router import (
            apply_managed_router,
            group_peer_addresses,
            is_managed_router,
        )

        instance = self._model_instance
        if not is_managed_router(model, instance.role):
            return model

        siblings = self._clientset.model_instances.list(
            params={"model_id": instance.model_id}
        )
        members = getattr(siblings, "items", siblings) or []
        worker_ips = {}
        for member in members:
            if member.worker_id and member.worker_id not in worker_ips:
                try:
                    worker = self._clientset.workers.get(id=member.worker_id)
                    if worker and worker.ip:
                        worker_ips[member.worker_id] = worker.ip
                except Exception as e:
                    logger.warning(
                        f"Failed to resolve worker {member.worker_id} "
                        f"while rendering the router: {e}"
                    )

        peers = group_peer_addresses(members, instance.group_id, worker_ips)
        variables = self._template_variables(**self._pd_template_variables())
        # A router's own named bands. The engine roles get these from the
        # injector, which builds its own context; the router does not go
        # through the injector at all, so without this its
        # `--prometheus-port {{ports.prometheus}}` reached the container
        # verbatim — the band was allocated and declared as a host port, and
        # the process was told to bind a string.
        for name, band in (instance.named_ports or {}).items():
            variables[f"ports.{name}"] = band.base
            variables[f"ports.{name}.count"] = band.count
        rendered = apply_managed_router(
            model, instance.role, peers=peers, variables=variables
        )
        _refuse_unrendered_router(rendered, instance)
        return rendered

    def _net_device_plane(self):
        """Which plane this deployment's `{{net_device}}` rides, read off the
        recipe.

        The judgement is the catalog's (`PDMode.net_device_plane`), not this
        layer's: `{{net_device}}` lands on `UCX_NET_DEVICES` in one recipe and
        on `HCCL_SOCKET_IFNAME` in another, and only the recipe knows which.
        Deciding it here by mode name would be an if-else the next Ascend-family
        recipe silently falls off.

        Every failure to answer degrades to `data`, the stricter plane: an
        unreadable catalog then costs an operator one `kv_ifname` on a
        multi-NIC host, where the reverse default would put KV bytes on the
        management NIC without saying so.
        """
        from gpustack.schemas.pd_modes import PDNetDevicePlaneEnum

        try:
            from gpustack.server.pd_mode_catalog import get_pd_mode

            model = self._model_spec or getattr(self, "_model", None)
            disaggregation = getattr(model, "disaggregation", None)
            mode_name = getattr(disaggregation, "mode", None)
            mode_name = getattr(mode_name, "value", mode_name)
            mode = get_pd_mode(str(mode_name)) if mode_name else None
            if mode is not None:
                return mode.net_device_plane
        except Exception as e:
            logger.warning(
                f"Failed to read the PD recipe's network-device plane ({e}); "
                "treating it as the data plane."
            )
        return PDNetDevicePlaneEnum.DATA

    def _pd_template_variables(self) -> Dict[str, object]:
        """The two placeholders only this layer can resolve: the KV-plane NIC
        and the runner image.

        Both are read tolerantly. An unresolvable one is left out of the
        context so its placeholder survives into the launch with a warning,
        which is diagnosable; inventing a value is not. `UCX_NET_DEVICES=all`
        in particular makes UCX advertise docker0 addresses the peer cannot
        route, and the failure surfaces as `NIXL_ERR_BACKEND` on the far side.
        """
        variables: Dict[str, object] = {}

        net_device = None
        try:
            # Imported where it is used: only a PD member ever needs a KV-plane
            # NIC, and every other backend start on this worker should be
            # unaffected by whether this module resolves.
            from gpustack.worker.net_device import derive_net_device

            net_device = derive_net_device(
                self._worker, self._config, self._net_device_plane()
            )
        except Exception as e:
            logger.warning(f"Failed to derive the KV-plane network device: {e}")
        if net_device:
            variables["net_device"] = net_device

        try:
            # Resolved against the GROUP's engine rather than this member's
            # backend. A managed router has already been switched to the custom
            # backend by the time this runs, and the custom backend resolves no
            # image by definition — its image is supposed to come from the
            # model. The router binary ships inside the engine's runner image,
            # so that is the one to name.
            #
            # The raw image: no registry override and no version write-back,
            # since this is a template value, not the image being deployed.
            spec = self._model_spec or self._model
            runner_image, _ = self._resolve_image(
                backend=get_backend(spec),
                spec=spec,
                inference_backend=self._engine_inference_backend(spec),
            )
            if runner_image:
                variables["runner_image"] = runner_image
        except Exception as e:
            # Warning, not debug. A router's image in the catalog is
            # `{{runner_image}}` and nothing else can supply it, so losing this
            # value does not degrade the launch — it sends the literal
            # placeholder to the container runtime, which rejects it as an
            # invalid reference several layers from anything that names the
            # cause.
            logger.warning(f"Failed to resolve the runner image for templating: {e}")

        return variables

    def _engine_inference_backend(self, spec) -> Optional[InferenceBackend]:
        """The backend row of the group's engine, for a member that is not one.

        This server was handed the row for its *own* backend, and a managed
        router's own backend is `custom` — which has no row, so what it was
        handed is None. That is correct for launching it and useless for
        answering "which image do the engines run", which is the only thing
        the router needs the row for: a custom backend version's image lives
        nowhere else. The runner catalog cannot stand in, because a
        user-defined version is by definition not in it.
        """
        name = get_backend(spec)
        if self.inference_backend and self.inference_backend.backend_name == name:
            return self.inference_backend

        # Built here rather than passed down: the row is needed by one role of
        # one deployment shape, and threading it through the fork boundary
        # would put it in every backend's constructor.
        from gpustack.worker.inference_backend_manager import InferenceBackendManager

        return InferenceBackendManager(self._clientset).get_backend_by_name(
            name, getattr(spec, "owner_principal_id", None)
        )

    def _pd_arguments(self) -> List[str]:
        """The PD role's engine arguments, or an empty list."""
        injection = self._pd_injection()
        return list(injection.args) if injection else []

    def _get_configured_env(self, **kwargs) -> Dict[str, str]:
        """
        Get the environment variables for the model instance.
        Merge the model's env with the system env.
        If there are conflicts, the model's env takes precedence.

        Returns:
            A dictionary of environment variables for the model instance.
        """

        env = {}
        if not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT:
            env = filter_env_vars(os.environ)

        pd_injection = self._pd_injection()
        if pd_injection and pd_injection.env:
            # Before the model's own env, so a deliberate per-model or
            # per-role override still wins — but never silently: a shadowed
            # side-channel host is a wrong address the group hands its peers,
            # not a setting that fails to apply.
            shadowed = sorted(set(pd_injection.env) & set(self._model.env or {}))
            if shadowed:
                logger.warning(
                    "The model's env overrides PD connection variables: "
                    f"{', '.join(shadowed)}. The overriding values are what the "
                    "engine advertises to its peers."
                )
            env.update(pd_injection.env)

        if self._model.env:
            # Render the *values*. This is the point of gpustack.utils.template
            # existing at all: `replace_command_param` is gated on a version
            # config that has a run_command and no built_in_frameworks, so no
            # built-in backend reaches it, and without this call a connector
            # variable would reach the engine verbatim (`ZMQError: No such
            # device (addr='tcp://{{worker_ip}}:5600')`).
            env.update(render_values(self._model.env, self._template_variables()))

        # Skip the container toolkit's NVIDIA_REQUIRE_CUDA check so a newer-minor
        # image starts on an older host driver. setdefault keeps user overrides.
        if self._should_disable_cuda_compat():
            env.setdefault("NVIDIA_DISABLE_REQUIRE", "1")

        return env

    def _set_cache_env(self, env: Dict[str, str], variable: str, subdirectory: str):
        """
        Point a backend's cache-root variable at a persistent directory under
        gpustack's data dir so compiled kernel caches survive container restarts.

        The inference container only sees that directory under a mirrored
        deployment, where gpustack-runtime replicates the worker's data-dir mount
        onto the container. Without it the backend writes into the container's own
        filesystem and recompiles on every start.
        """
        if variable in env:
            return
        if not self._config or not self._config.cache_dir:
            return
        cache_dir = os.path.join(self._config.cache_dir, subdirectory)
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as e:
            logger.warning(
                f"Failed to create cache dir {cache_dir}: {e}. "
                f"{variable} will not be set and caches will not be persisted."
            )
            return
        env[variable] = cache_dir

    @lru_cache
    def _get_selected_gpu_devices(self) -> GPUDevicesStatus:
        """
        Get the GPU devices assigned to the model instance.

        Returns:
            A list of GPU device information assigned to the model instance.
        """
        minstance = self._model_instance
        dservers = minstance.distributed_servers
        gpu_type = None
        if (
            dservers
            and dservers.subordinate_workers
            and minstance.worker_id != self._worker.id
        ):
            subworker = next(
                (
                    w
                    for w in dservers.subordinate_workers
                    if w.worker_id == self._worker.id
                ),
                None,
            )
            gpu_indexes = sorted(subworker.gpu_indexes or [])
            gpu_type = subworker.gpu_type
        else:
            gpu_indexes = sorted(self._model_instance.gpu_indexes or [])
            gpu_type = self._model_instance.gpu_type

        gpu_devices: GPUDevicesStatus = []
        if gpu_indexes and self._worker.status.gpu_devices:
            for index in gpu_indexes:
                gpu_device = next(
                    (
                        d
                        for d in self._worker.status.gpu_devices
                        if d.index == index and (gpu_type is None or d.type == gpu_type)
                    ),
                    None,
                )
                if gpu_device:
                    gpu_devices.append(gpu_device)
        return gpu_devices

    def _get_device_info(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Get the device information for the serving.
        If not found, retrieve from the first device of the worker.

        Returns:
            A tuple of (type, runtime_version, arch_family).
        """
        gpu_devices = self._get_selected_gpu_devices()
        if gpu_devices:
            gpu_device = gpu_devices[0]
            return (
                gpu_device.type,
                gpu_device.runtime_version,
                gpu_device.arch_family,
            )
        elif self._worker.status.gpu_devices:
            gpu_device = self._worker.status.gpu_devices[0]
            return (
                gpu_device.type,
                gpu_device.runtime_version,
                gpu_device.arch_family,
            )
        return None, None, None

    def _host_ipc_enabled(self) -> bool:
        """Whether the workload joins the host IPC namespace. Only a
        deployment attached to a shared cache service needs it — the
        CUDA-IPC zero-copy transfer path imports KV buffers across the
        engine and cache containers — so it follows the attachment; a
        ``GPUSTACK_HOST_IPC`` env (per-model, or global on the worker)
        overrides the derivation either way. Everything else keeps its
        private /dev/shm: Docker ignores shm_size under host IPC, and
        Kubernetes PodSecurity baseline rejects hostIPC pods."""
        model_env = self._model.env or {}
        if envs.HOST_IPC_ENV in model_env:
            return to_bool(model_env[envs.HOST_IPC_ENV])
        if envs.HOST_IPC is not None:
            return to_bool(envs.HOST_IPC)
        cache_config = getattr(self._model_instance, "cache_config", None)
        derived = bool(cache_config and cache_config.injected)
        if derived:
            self._warn_host_ipc_trade_off()
        return derived

    def _warn_host_ipc_trade_off(self) -> None:
        """Say out loud that a disaggregated member with a shared cache is a
        choice, not a default.

        The two want opposite things and only one can be had. A shared cache
        wants the host IPC namespace, because that is what lets the engine and
        the cache container pass KV buffers by CUDA-IPC handle instead of
        copying. A KV connector wants a private /dev/shm, and joining the host
        namespace replaces the container's with the host's — which drops the
        `shm_size` the workload was given.

        Neither is wrong, and both run, so this does not refuse: measured, the
        connector's actual /dev/shm use was two orders of magnitude under the
        allotment, so the lost guarantee is a risk rather than a failure. What
        would be wrong is deciding it silently, because the person who cares
        about the answer cannot see that the question was asked. Both
        directions are reachable per model with GPUSTACK_HOST_IPC.
        """
        if getattr(self, "_host_ipc_trade_off_warned", False):
            return
        instance = getattr(self, "_model_instance", None)
        model = self._model_spec or getattr(self, "_model", None)
        if not getattr(instance, "role", None) or not getattr(
            model, "disaggregation", None
        ):
            return
        self._host_ipc_trade_off_warned = True
        logger.warning(
            "Role '%s' of %s attaches a shared KV cache, so its workload joins "
            "the host IPC namespace for the cache's zero-copy path — which "
            "replaces its private /dev/shm with the host's and drops the "
            "shm_size it was allocated. The KV connector uses /dev/shm too. "
            "Set %s=false in the model's env to keep the private /dev/shm "
            "instead, at the cost of the cache falling back to host copies.",
            getattr(instance, "role", "?"),
            getattr(model, "name", "?"),
            envs.HOST_IPC_ENV,
        )

    def _cuda_minor_version_compatibility_enabled(self) -> bool:
        """Resolve the switch: a per-model
        ``GPUSTACK_ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY`` in the model's env
        overrides the global default (``gpustack.envs``); off by default."""
        model_env = self._model.env or {}
        if envs.ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY_ENV in model_env:
            return to_bool(model_env[envs.ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY_ENV])
        return envs.ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY

    def _should_disable_cuda_compat(self) -> bool:
        """Whether to disable the image's cuda-compat and rely on the host driver.

        True when enabled and the runner image targets a newer CUDA minor than the
        host driver (same major) -- there cuda-compat would break consumer GPUs.
        """
        if not self._cuda_minor_version_compatibility_enabled():
            return False
        backend, host_runtime_version, _ = self._get_device_info()
        if backend != "cuda" or not host_runtime_version:
            return False
        # Resolve the raw image (no registry override / version write-back side
        # effect); the tag still carries the cudaX.Y we parse.
        resolved_image, _ = self._resolve_image()
        image_cuda_version = _parse_image_cuda_version(resolved_image)
        if not image_cuda_version:
            return False
        return (
            _major_version(image_cuda_version) == _major_version(host_runtime_version)
            and compare_versions(image_cuda_version, host_runtime_version) > 0
        )

    def _get_configured_resources(
        self, mount_all_devices: bool = False
    ) -> ContainerResources:
        """
        Get the resource requests for the model instance.

        Args:
            mount_all_devices:
                Whether to mount all available GPU devices.
                If true, ignores the GPUs assigned to the model instance and try to mount all available GPUs.

        Returns:
            A ContainerResources object representing the resource requests for the model instance.

        Raises:
            If the GPUs assigned to the model instance are of different types.
        """
        resources = ContainerResources()
        # Ahead of both device paths, because this role declares CPU and memory
        # *instead of* any device key and that is true on either one: with
        # `gpu_type_selector` it must not claim a slice, without one it has no
        # devices to mount. Checking it inside the selector branch only left
        # the commoner path — a group with no explicit card type — handing the
        # router an empty request.
        #
        # `gpu_type_selector` is a Model-level field every role inherits by
        # projection, which is why the slice case needs saying rather than
        # catching. Measured on a live cluster: the scheduler correctly placed
        # a managed router with no VRAM claim, the container asked for one
        # anyway, and the device plugin handed it 40% of a card that its own
        # prefill and decode were sharing. Nothing failed — the group ran.
        if role_takes_no_accelerator(
            self._model_spec or self._model,
            getattr(self._model_instance, "role", None),
        ):
            return self._get_accelerator_free_resources(resources)
        if getattr(self._model, "gpu_type_selector", None) is not None:
            return self._get_vgpu_configured_resources(resources)
        gpu_devices = self._get_selected_gpu_devices()
        if gpu_devices:
            gpu_type = gpu_devices[0].type
            for device in gpu_devices[1:]:
                if device.type != gpu_type:
                    raise RuntimeError(
                        "All GPUs assigned to the model instance must be of the same type."
                    )
            key = runtime_envs.GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY.get(
                gpu_type
            )
            if key:
                resources[key] = (
                    ",".join(str(d.index) for d in gpu_devices)
                    if not mount_all_devices
                    else "all"
                )
        return resources

    def _get_accelerator_free_resources(
        self, resources: ContainerResources
    ) -> ContainerResources:
        """CPU and memory for a role that holds no weights — the router.

        The only role whose footprint the platform knows outright: it forwards
        requests and loads nothing, so a fixed floor is a better answer than an
        estimate. Without this it declared *nothing*, which on Kubernetes is a
        Pod with no requests — invisible to kubelet admission and, once the
        ledger grows a CPU dimension, to placement as well.

        Both numbers land in the container's requests and limits alike
        (Guaranteed QoS on Kubernetes; a share rather than a cap on Docker).
        Only `memory` also reaches the scheduler, as the role's RAM claim —
        CPU is not a dimension the allocatable view has.
        """
        declared = role_container_resources(
            self._model_spec or self._model,
            getattr(self._model_instance, "role", None),
        )
        resources["cpu"] = declared.cpu
        resources["memory"] = declared.memory
        return resources

    def _get_vgpu_configured_resources(
        self, resources: ContainerResources
    ) -> ContainerResources:
        """
        Translate the model's ``gpu_type_selector`` into the operator-recognized
        vGPU resource requests, INSTEAD of the whole-card device key:

        - whole card (both slice percentages 0): the bare exclusive resource
          ``<base>: 1`` — the operator webhook rejects percentage budgets
          outside (0,100], and an exclusive request also works on pools
          without slicing capability;
        - soft slice (any percentage > 0): ``<base>.sliced: 1`` plus the
          ``<base>.sliced.memory-percentage`` / ``<base>.sliced.cores-percentage``
          budgets;
        - hard partition (``accelerator_partitioned_profile`` set):
          ``<base>.partitioned: 1`` plus the per-profile key
          ``<base>.partitioned.mig-<profile>: 1``.

        ``<base>`` is the operator resource base of the GPU manufacturer
        (``nvidia.com/gpu``, ``amd.com/gpu``, ...). Key names mirror the
        operator's resource families (gpustack-operator ``pkg/nodefeature``);
        the runtime's Kubernetes deployer lands them verbatim in the
        container limits/requests.

        Raises:
            RuntimeError: if the operator resource base cannot be resolved —
                fail closed rather than deploy without isolation.
        """
        selector = self._model.gpu_type_selector
        base = self._get_vgpu_resource_base()

        profile = selector.accelerator_partitioned_profile
        if profile:
            resources[f"{base}.partitioned"] = "1"
            resources[f"{base}.partitioned.mig-{profile}"] = "1"
            return resources

        memory = selector.accelerator_sliced_memory_percentage or 0
        cores = selector.accelerator_sliced_cores_percentage or 0
        if memory == 0 and cores == 0:
            # Whole-card exclusive: the bare base resource, no slicing keys.
            #
            # The count is the member's own card count, not a hard 1. A slice
            # is a fraction of one card so "1" is the only answer there, but a
            # whole-card claim for a tp=4 member needs four — and the
            # operator's resource model hands out several at once (its
            # `Accelerator` view is documented as "1", "4"). Writing 1 here
            # while the engine was told tp=4 is the shape of the bug: the pod
            # is admitted with one card and the engine then cannot start.
            resources[base] = str(self._whole_card_count())
            return resources
        resources[f"{base}.sliced"] = "1"
        resources[f"{base}.sliced.memory-percentage"] = str(memory)
        resources[f"{base}.sliced.cores-percentage"] = str(cores)
        return resources

    def _whole_card_count(self) -> int:
        """How many whole cards this member was scheduled with.

        Taken from the claim the scheduler already computed rather than
        re-derived from backend parameters: the claim is what the placement
        decision was made against, and a second derivation here could disagree
        with it — which would mean the pod asks for a different number of cards
        than the worker was chosen for.
        """
        claim = getattr(self._model_instance, "computed_resource_claim", None)
        vram = getattr(claim, "vram", None) if claim else None
        if vram:
            return max(len(vram), 1)
        return 1

    def _get_vgpu_resource_base(self) -> str:
        """
        Resolve the operator resource base (e.g. ``nvidia.com/gpu``) for the
        model instance's GPU type, reusing the runtime's existing mapping
        chain: backend type -> detect resource key (``nvidia.com/devices``)
        -> CDI / operator base (``nvidia.com/gpu``).

        Raises:
            RuntimeError: if the base cannot be resolved.
        """
        gpu_type = self._model_instance.gpu_type
        if gpu_type is None:
            gpu_type, _, _ = self._get_device_info()
        device_key = (
            runtime_envs.GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY.get(gpu_type)
            if gpu_type
            else None
        )
        base = (
            runtime_envs.GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI.get(device_key)
            if device_key
            else None
        )
        if not base:
            raise RuntimeError(
                f"Cannot resolve the operator resource base for GPU type "
                f"'{gpu_type}': refusing to deploy with gpu_type_selector "
                f"without resource isolation."
            )
        return base

    def _get_configured_mounts(self) -> List[ContainerMount]:
        """
        Get the volume mounts for the model instance.
        If runtime mirrored deployment is enabled, no mounts will be set up.

        Returns:
            A list of ContainerMount objects for the model instance.
        """
        mounts: List[ContainerMount] = []
        if (
            self._model_path
            and not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT
        ):
            model_dir = os.path.dirname(self._model_path)
            mounts.append(
                ContainerMount(
                    path=model_dir,
                ),
            )

        # The PD recipe's host mounts, unlike the model directory, apply under
        # mirrored deployment too: mirroring copies what the *worker* container
        # happens to have, and a transport's host file is a property of the
        # recipe rather than of how the worker was launched. The runtime merges
        # explicit mounts with mirrored ones and keeps the explicit entry, so
        # naming a path the worker already carries is harmless.
        pd_injection = self._pd_injection()
        if pd_injection and pd_injection.host_mounts:
            declared = {m.path for m in mounts}
            for path in pd_injection.host_mounts:
                if path in declared:
                    continue
                mounts.append(
                    ContainerMount(
                        path=path,
                        mode=ContainerMountModeEnum.ROX,
                    ),
                )
        return mounts

    def _get_configured_ports(self) -> List[ContainerPort]:
        """
        Get the ports for the model instance.

        Returns:
            A list of ContainerPort objects for the model instance.
        """
        return [
            ContainerPort(
                internal=port,
            )
            for port in self._model_instance.ports or []
        ]

    @staticmethod
    def _get_container_env_config(env: Dict[str, str]) -> ContainerEnvConfig:
        """
        Read container configuration from environment variables passed to the container.

        Args:
            env: The environment variables dictionary passed to the container.

        Returns:
            A ContainerEnvConfig containing container configuration:
            - user: Run as specific UID (int)
            - group: Run as specific GID (int)
            - shm_size_gib: Shared memory size in GiB (float, default 10.0)
        """
        config = ContainerEnvConfig()

        # Read user ID
        uid_str = env.get("GPUSTACK_MODEL_RUNTIME_UID")
        if uid_str:
            try:
                config.user = int(uid_str)
            except ValueError:
                logger.warning(
                    f"Invalid GPUSTACK_MODEL_RUNTIME_UID value: {uid_str}, ignoring"
                )

        # Read group ID
        gid_str = env.get("GPUSTACK_MODEL_RUNTIME_GID")
        if gid_str:
            try:
                config.group = int(gid_str)
            except ValueError:
                logger.warning(
                    f"Invalid GPUSTACK_MODEL_RUNTIME_GID value: {gid_str}, ignoring"
                )

        # Read shared memory size in GiB
        shm_str = env.get("GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB", "10")
        try:
            config.shm_size_gib = float(shm_str)
        except ValueError:
            logger.warning(
                f"Invalid GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB value: {shm_str}, using default 10.0"
            )
            config.shm_size_gib = 10.0

        return config

    def _get_serving_port(self) -> int:
        """
        Get the (main) serving port for the model instance.

        Returns:
            The (main) serving port for the model instance.
        """
        return (
            self._model_instance.ports[0]
            if self._model_instance.ports
            else self._model_instance.port
        )

    def _cache_injection_files(self) -> dict[str, str]:
        """Connector config files (container path -> contents) the serving
        script writes before the engine starts — for a connector that reads
        its settings from a path an env var points at (e.g. Mooncake's
        MOONCAKE_CONFIG_PATH JSON).

        Two declarations land here — the shared cache service's injection and
        the PD role's — because a connector that reads its transport config
        only from a file leaves no other way in. A path both declare is a
        genuine collision rather than a merge: the second write would silently
        replace the first, so it is logged as it happens.
        """
        files: dict[str, str] = {}
        cache_config = getattr(self._model_instance, "cache_config", None)
        if cache_config and cache_config.injected:
            files.update(cache_config.files or {})

        pd_injection = self._pd_injection()
        if pd_injection and pd_injection.files:
            collisions = sorted(set(files) & set(pd_injection.files))
            if collisions:
                logger.warning(
                    "The PD role and the shared cache service both declare "
                    f"{', '.join(collisions)}; the PD contents win."
                )
            files.update(pd_injection.files)
        return files

    def _get_serving_command_script(self, env: dict[str, str]) -> Optional[str]:
        """
        Get the serving command script for the model instance.

        Return None if `GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED` is set, or
        no prep step (installing PyPi packages, disabling cuda-compat) is needed.

        Args:
            env:
                The environment variables for the model instance.

        Returns:
            The serving command script for the model instance, or None if not needed.

        """

        # Skip if explicitly disabled.
        if env and to_bool(
            env.get("GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED", "0")
        ):
            return None

        disable_cuda_compat = self._should_disable_cuda_compat()
        cache_files = self._cache_injection_files()

        # Skip if no prep step is needed.
        if (
            not disable_cuda_compat
            and not cache_files
            and not (env and "PYPI_PACKAGES_INSTALL" in env)
        ):
            return None

        cache_files_step = ""
        for path, content in sorted(cache_files.items()):
            # A quoted heredoc keeps the rendered content verbatim (no
            # shell expansion of $ or backticks inside e.g. JSON).
            cache_files_step += (
                f'echo "Writing connector config {path}"\n'
                f"mkdir -p \"$(dirname '{path}')\"\n"
                f"cat > '{path}' <<'GPUSTACK_CACHE_FILE_EOF'\n"
                f"{content}\n"
                "GPUSTACK_CACHE_FILE_EOF\n\n"
            )

        cuda_compat_step = ""
        if disable_cuda_compat:
            cuda_compat_step = """if [ -d /usr/local/cuda/compat ]; then
    echo "Disabling bundled cuda-compat to run with the host driver (CUDA minor version compatibility)"
    rm -rf /usr/local/cuda/compat 2>/dev/null || true
    ldconfig 2>/dev/null || true
    if [ -d /usr/local/cuda/compat ]; then
        echo "Warning: failed to remove /usr/local/cuda/compat (needs a root container); the bundled cuda-compat may still be used"
    fi
fi

"""

        return (
            """#!/bin/sh

#
# Prepare
#

"""
            + cache_files_step
            + cuda_compat_step
            + """if [ -n "${PYPI_PACKAGES_INSTALL:-}" ]; then
    if command -v uv >/dev/null 2>&1; then
        echo "Installing additional PyPi packages: ${PYPI_PACKAGES_INSTALL}"
        export UV_HTTP_TIMEOUT=500
        export UV_NO_CACHE=1
        if [ -n "${PIP_INDEX_URL:-}" ]; then
            export UV_DEFAULT_INDEX="${PIP_INDEX_URL}"
            export UV_INDEX_URL="${PIP_INDEX_URL}"
        fi
        if [ -n "${PIP_EXTRA_INDEX_URL:-}" ]; then
            export UV_INDEX="${PIP_EXTRA_INDEX_URL}"
            export UV_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL}"
        fi
        uv pip install --system ${PYPI_PACKAGES_INSTALL}
        uv pip tree --system
    elif command -v pip >/dev/null 2>&1; then
        echo "Installing additional PyPi packages: ${PYPI_PACKAGES_INSTALL}"
        export PIP_DISABLE_PIP_VERSION_CHECK=1
        export PIP_ROOT_USER_ACTION=ignore
        export PIP_TIMEOUT=500
        export PIP_NO_CACHE_DIR=1
        pip install ${PYPI_PACKAGES_INSTALL}
        pip freeze
    fi
    unset PYPI_PACKAGES_INSTALL
fi

#
# Execute
#

exec "$@"
"""
        )

    def build_versioned_command_args(
        self,
        default_args: List[str],
        model_path: Optional[str] = None,
        port: Optional[int] = None,
    ) -> List[str]:
        """
        Override default startup arguments based on version configuration
        when the version uses non-built-in version and defines a custom run_command

        Args:
        - default_args: The default command argument list.
        - model_path: Path used to replace {{model_path}}; if None, fall back to self._model_path.
        - port: Port used to replace {{port}}; if None, fall back to self._model_instance.port.

        Returns:
            The final command argument list used for container execution.
        """

        # if no version or inference backend is available, return default_args
        version = self._model.backend_version
        if not version or not self.inference_backend:
            return default_args

        # Load version configuration
        version_config = None
        try:
            version_config, version = self.inference_backend.get_version_config(version)
        except Exception:
            version_config = self.inference_backend.version_configs.root.get(version)

        # Only perform replacement when the version uses non-built-in version and defines run_command
        if (
            version_config
            and version_config.built_in_frameworks is None
            and version_config.run_command
        ):
            resolved_model_path = (
                model_path if model_path is not None else self._model_path
            )
            resolved_port = port if port is not None else self._model_instance.port
            resolved_model_name = self._model_instance.model_name
            selected_gpu_indexes = sorted(
                d.index for d in self._get_selected_gpu_devices()
            )

            command = self.inference_backend.replace_command_param(
                version=version,
                model_path=resolved_model_path,
                port=resolved_port,
                worker_ip=self._worker.ip,
                model_name=resolved_model_name,
                gpu_count=len(selected_gpu_indexes),
                gpu_ids=selected_gpu_indexes,
                command=version_config.run_command,
                env=self._model.env,
            )
            if command:
                return shlex.split(command)

        # Return original default_args by default
        return default_args

    @staticmethod
    def _override_entrypoint(
        command: Optional[List[str]],
        command_args: List[str],
        command_script: Optional[str],
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """
        Map command / command_args / command_script onto the container's
        ENTRYPOINT/CMD override semantics.

        - command_script becomes the entrypoint, so the original command is
          prepended to its args.
        - Otherwise merge command and args into command to override the image
          ENTRYPOINT (args alone would be appended to ENTRYPOINT, not replace it).
        - When there is no command to override the ENTRYPOINT, leave the args
          appended as-is.
        """
        if command_script:
            return None, (command or []) + command_args
        if not command:
            return None, command_args
        return command + command_args, None

    @staticmethod
    def _get_backend_parameter_start_index(
        arguments: List[str],
        entrypoint: Optional[List[str]] = None,
    ) -> int:
        """
        Return where backend parameters start in `arguments`, skipping any
        command prefix or model path that may precede the first option.
        """
        if not arguments:
            return 0

        if not entrypoint:
            command = os.path.basename(arguments[0])
            if (
                len(arguments) >= 3
                and command.startswith("python")
                and arguments[1] == "-m"
            ):
                return 3

        for index, argument in enumerate(arguments):
            if argument.startswith("-"):
                return index

        return len(arguments)

    def _get_injected_backend_parameters(
        self,
        arguments: List[str],
        user_backend_parameters: List[str],
        entrypoint: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Derive injected backend parameters from the final command line.

        The final command is the source of truth: remove the command prefix (or
        separate container entrypoint) and the user-specified backend
        parameters, and the remaining backend parameters are injected by
        GPUStack.
        """
        start_index = self._get_backend_parameter_start_index(arguments, entrypoint)
        candidates = arguments[start_index:]

        # `_flatten_backend_param` prepends the PD role's arguments to what the
        # caller then hands back as "the user's parameters". Strip that prefix
        # off first, or connector state GPUStack injected is reported as the
        # user's own — and the deployment view is where a user goes to find out
        # what GPUStack added.
        pd_arguments = self._pd_arguments()
        if (
            pd_arguments
            and user_backend_parameters[: len(pd_arguments)] == pd_arguments
        ):
            user_backend_parameters = user_backend_parameters[len(pd_arguments) :]

        if not user_backend_parameters:
            return candidates

        user_param_len = len(user_backend_parameters)
        for start in range(len(candidates) - user_param_len, -1, -1):
            end = start + user_param_len
            if candidates[start:end] == user_backend_parameters:
                return candidates[:start] + candidates[end:]

        return candidates

    def _get_configured_image(
        self,
        backend: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolve the container image to use for the current backend, then apply
        registry override once if needed.

        See _resolve_image for resolution details.
        """
        image_name, target_version = self._resolve_image(backend)
        if image_name is None:
            return None
        # Update model backend service version at upper layer if we detected it
        if target_version:
            self._update_model_backend_service_version(target_version)
        return apply_registry_override_to_image(
            self._config, image_name, self._fallback_registry
        )

    def _fetch_runner_overrides(self) -> List[RunnerOverrideEntryPublic]:
        """Fetch runner overrides fresh from the server at deploy time.

        Read directly (unpaginated) so the worker resolves images against the
        same DB-fresh override set the scheduler saw — no stale cache, so a
        version the scheduler allowed can't fail to resolve here.

        Held for the rest of this deploy: ``_resolve_image`` is reached three times
        (image, environment, serving command) and the whole set is one unpaginated
        response.

        A failure raises rather than degrading to an empty list: empty means "no
        custom source is configured, layer over the packaged catalog", and a
        custom source *replaces* that catalog — so degrading would deploy the
        very images an admin replaced, silently and with the wrong registry.
        Failing the deploy is recoverable; a running model on the wrong image is
        not.
        """
        if self._runner_overrides is not None:
            return self._runner_overrides
        try:
            resp = self._clientset.http_client.get_httpx_client().get(
                "/runner-override-entries", params={"page": -1}
            )
            resp.raise_for_status()
            self._runner_overrides = RunnerOverrideEntriesPublic.model_validate(
                resp.json()
            ).items
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch runner overrides from the server: {e}. "
                f"The image cannot be resolved without them"
            ) from e
        return self._runner_overrides

    def _resolve_image(  # noqa: C901
        self,
        backend: Optional[str] = None,
        spec: Optional[Model] = None,
        inference_backend: Optional["InferenceBackend"] = _INHERIT,
    ) -> (Optional[str], Optional[str]):
        """
        Resolve the container image to use for the current backend.

        This method returns the raw image name without applying any registry
        override. Callers should apply overrides as needed.

        `spec` and `inference_backend` answer the question for a model other
        than the one this server is starting. Exactly one caller needs that: a
        managed router asking which image its *engines* run, because that is
        what `{{runner_image}}` names. The router's own projected model says
        `backend=custom` — deliberately, so it launches a command rather than
        an engine — and custom is neither a runner service nor a backend row,
        so resolving against it returns nothing twice over.

        Precedence:
        1) Explicitly configured image on the model (model.image_name)
        2) Prefer image name from the user's config when using custom backend or built-in backend with a custom version
        3) Auto-detected image from gpustack-runner based on device vendor/arch and backend

        Return:
            image_name, backend_version

        """
        model = spec if spec is not None else self._model
        if inference_backend is _INHERIT:
            inference_backend = self.inference_backend

        # 1) Return directly if explicitly provided.
        if model.image_name:
            return model.image_name, None

        # 2) Configuration takes priority when backend_version is set
        if model and inference_backend:
            image_name, target_version = inference_backend.get_image_name(
                model.backend_version
            )
            if image_name and target_version:
                return image_name, target_version

        """
        Prepare queries for retrieving runners.
        """

        def get_docker_image(bvr: BackendVersionedRunner) -> str:
            return bvr.variants[0].services[0].versions[0].platforms[0].docker_image

        backend, runtime_version, arch_family = self._get_device_info()
        if not backend:
            # Return directly if there is not a valid device.
            # GPUStack-Runner does not provide CPU-only platform images.
            # To use a CPU-only version, user must configure in `Inference Backend` page.
            return None, None

        if backend not in available_backends():
            # Return directly if found backend is not within the available backends.
            return None, None

        """
        Retrieve runners by queries.

        For example, the queries of runners is as below.

        - backend: cuda
          backend_variant: None
          service: vllm
          service_version: 0.10.0
          platform: linux/amd64
        - backend: cann
          backend_variant: 910b
          service: vllm
          service_version: 0.10.0
          platform: linux/arm64
        """

        backend_variant = None
        service = model.backend.lower()
        # A blank backend version means "Auto", same as None. Legacy/migrated data
        # and API clients can store "", which would otherwise be used as an exact
        # version filter and match no runner at all.
        model_service_version = model.backend_version or None
        service_version = model_service_version

        # Default variant for some backends.
        if backend == "cann":
            if arch_family:
                backend_variant = get_ascend_cann_variant(arch_family)
            if not backend_variant:
                backend_variant = "910b"

        logger.info(
            f"_resolve_image query: backend={backend}, backend_variant={backend_variant}, service={service}, service_version={service_version}, platform={platform.system_arch()}, runtime_version={runtime_version}"
        )

        overrides = self._fetch_runner_overrides()
        runners = merged_backend_runners(
            overrides,
            backend=backend,
            backend_variant=backend_variant,
            service=service,
            service_version=model_service_version,
            platform=platform.system_arch(),
            with_deprecated=model_service_version is not None,
        )
        if not runners:
            # Return directly if there is not a valid runner.
            return None, None

        """
        Pick the appropriate backend version from among the multiple versions.

        For example, the content of runners is as below.

        [
            {
                "backend": "cuda",
                "versions": [
                    {
                        "version": "12.8",
                        ...
                    },
                    {
                        "version": "12.6",
                        ...
                    },
                    {
                        "version": "12.4",
                        ...
                    }
                ]
            }
        ]
        """

        backend_versioned_runners = runners[0].versions

        # Try to update backend version for server model.
        if backend_versioned_runners and len(backend_versioned_runners) > 0:
            service_version = _get_service_version_from_versioned_runner(
                backend_versioned_runners[0]
            )

        # Return directly if there is only one versioned backend.
        if len(backend_versioned_runners) == 1:
            return get_docker_image(backend_versioned_runners[0]), service_version

        # The runtime-match rule (newest <= host, same-major then oldest
        # fallbacks) is shared with cache-provider runtime_images.
        picked_version = pick_runtime_version(
            [candidate.version for candidate in backend_versioned_runners],
            runtime_version,
        )
        picked_runner = next(
            candidate
            for candidate in backend_versioned_runners
            if candidate.version == picked_version
        )
        service_version = _get_service_version_from_versioned_runner(picked_runner)
        return get_docker_image(picked_runner), service_version

    def _update_model_backend_service_version(
        self, service_version: Optional[str]
    ) -> None:
        """
        Update model backend (service) version back to server if not already set.

        This method is extracted from image resolution flow to be called from the upper
        layer after the version is detected.
        """
        if not service_version:
            return
        try:
            if not self._model.backend_version:
                # Write the *unprojected* model back. `self._model` carries the
                # role's overrides merged in, so sending it as a ModelUpdate
                # would persist one role's engine parameters, env and image as
                # the Model-level spec — silently, and for every other role to
                # then inherit. The non-table projection class cannot prevent
                # this one: the write goes out over HTTP, not through a
                # session. A role that overrides `backend_version` never
                # reaches here anyway, since the value is then already set.
                spec = self._model_spec or self._model
                spec.backend_version = service_version
                self._model.backend_version = service_version
                self._clientset.models.update(spec.id, ModelUpdate(**spec.model_dump()))
            if not self._model_instance.backend_version:
                self._update_model_instance(
                    self._model_instance.id, backend_version=service_version
                )
        except Exception as e:
            logger.error(
                f"Failed to update model service version {service_version}: {e}"
            )

    def _flatten_backend_param(self) -> List[str]:
        """
        Reduce ``backend_parameters`` to a flat argv-style token list.

        ``backend_parameters`` is semantically a concatenated argv: each element
        may be one token (``"--host"``, ``"0.0.0.0"``), one full
        ``--key value`` / ``--key=value`` string, or a whole pasted command line
        (``"--a 1 --b=2 --flag"``). ``flatten_to_argv`` recovers the underlying
        token stream uniformly.

        If the backend's ``parameter_format`` is set, each ``--key value(s)``
        cluster is normalized to that form (multi-value clusters always stay in
        space form — equal form can't safely express them).
        """
        tokens = flatten_to_argv(self._model.backend_parameters or [])

        # Rendered, exactly as `env` values are (`_get_configured_env`). The
        # two halves of one deployment's configuration had different rules
        # until now: an env value could say `{{worker_ip}}` and get the
        # address, while the same placeholder in a parameter reached the engine
        # verbatim. Nothing justified the split — it was simply that `env` grew
        # the feature first.
        #
        # What makes it matter is the PD form. It seeds the recipe's own rows
        # into the role's parameter list so they can be edited like any other
        # row, and those rows are written in placeholders: a prefill's
        # connector carries `{{ports.kv_port}}`, a router's command carries
        # `{{worker_ip}}`. Submitting one unrendered puts
        # `{kv_connector:NixlConnector,...,{{kv_lease_duration}}}` on the
        # command line, and the launch fails to parse its own JSON.
        #
        # Token by token rather than over the joined string: `flatten_to_argv`
        # has already decided what a token is, and re-rendering the joined form
        # would give a value containing a space a second chance to be split.
        #
        # An unknown name survives verbatim with a warning (see `render`),
        # which is the right posture here: a user parameter may legitimately
        # contain `{{...}}` that is not ours, and blanking it would turn text
        # the user chose into a plausible-looking wrong value.
        variables = self._template_variables()
        tokens = [
            render(token, variables, context="backend parameter") or token
            for token in tokens
        ]

        # The PD role's connector arguments ride in front of the user's. This
        # is the one seam in this file that every backend's command builder
        # goes through — each of them extends its argument list with this
        # result — so it is where a declaration in `pd-modes.yaml` reaches an
        # actual command line. In front, because argparse lets the later of
        # two spellings win and a user's parameter is the one that should:
        # the parameters PD cannot share at all (`--kv-transfer-config`) are
        # refused outright when they are rendered, not resolved by position.
        # They are appended already tokenized: `flatten_to_argv` would re-split
        # a rendered JSON document on its spaces.
        pd_arguments = self._pd_arguments()
        if pd_arguments:
            tokens = pd_arguments + tokens

        parameter_format = (
            getattr(self.inference_backend, "parameter_format", None)
            if self.inference_backend
            else None
        )
        if parameter_format is None or not tokens:
            return tokens

        return _normalize_param_format(tokens, parameter_format)

    def _transform_workload_plan(
        self, workload: WorkloadPlan
    ) -> Union[DockerWorkloadPlan, WorkloadPlan]:
        """
        If the deployer is docker, transform the generic WorkloadPlan to DockerWorkloadPlan,
        and fill the pause image and restart image with registry override.
        """
        selector = getattr(self._model, "gpu_type_selector", None)
        if selector is not None:
            # Hand the operator InstanceType name to the runtime; the
            # runtime's Kubernetes deployer owns queue admission from here.
            workload.instance_type = selector.type
        return transform_workload_plan(self._config, workload, self._fallback_registry)


def _get_service_version_from_versioned_runner(
    backend_versioned_runner: BackendVersionedRunner,
) -> Optional[str]:
    """
    Get the service version from the backend versioned runner.

    Args:
        backend_versioned_runner:
            The backend versioned runner.
    Returns:
        The service version string, or None if not found.
    """
    try:
        return backend_versioned_runner.variants[0].services[0].versions[0].version
    except Exception as e:
        logger.error(
            f"Failed to get service version from backend versioned runner: {e}"
        )
        return None


def _major_version(version: Optional[str]) -> Optional[str]:
    """Extract the major version segment, e.g. '12' from 'v12.8'."""
    if not version:
        return None
    return version.removeprefix("v").split(".", 1)[0]


_CUDA_IMAGE_VERSION_PATTERN = re.compile(r"cuda(\d+)\.(\d+)")


def _parse_image_cuda_version(image: Optional[str]) -> Optional[str]:
    """Extract the CUDA ``major.minor`` from a runner image tag
    (``gpustack/runner:cuda12.9-...`` -> ``"12.9"``); None if the tag has none."""
    if not image:
        return None
    # Isolate the tag first so a registry/namespace segment carrying a cudaX.Y
    # (e.g. a mirror path) can't shadow the version in the actual tag.
    reference = image.split("/")[-1]
    if ":" not in reference:
        return None
    tag = reference.split(":", 1)[1]
    match = _CUDA_IMAGE_VERSION_PATTERN.search(tag)
    if not match:
        return None
    return f"{match.group(1)}.{match.group(2)}"


def is_ascend_310p(devices: GPUDevicesStatus) -> bool:
    """
    Check if the model instance is running on VLLM Ascend 310P.

    An empty device list is not a match: callers gate Ascend 310P specifics on
    this, and ``all()`` is vacuously true over nothing, so a model instance
    that named no device -- one scheduled by GPU type rather than by device
    index, for instance -- would be served as if it ran on 310P.
    """

    return bool(devices) and all(
        gpu.vendor == ManufacturerEnum.ASCEND.value
        and get_ascend_cann_variant(gpu.arch_family) == "310p"
        for gpu in devices
    )


def is_ascend(devices: GPUDevicesStatus) -> bool:
    """
    Check if all devices are Ascend.

    An empty device list is not a match, for the same reason as
    :func:`is_ascend_310p`.
    """

    return bool(devices) and all(
        gpu.vendor == ManufacturerEnum.ASCEND.value for gpu in devices
    )


def cal_distributed_parallelism_arguments(
    model_instance: ModelInstance,
) -> tuple[int, int]:
    pp = len(model_instance.distributed_servers.subordinate_workers) + 1
    tp = len(model_instance.gpu_indexes) if model_instance.gpu_indexes else 1
    uneven_pp = tp
    uneven = False
    for subordinate_worker in model_instance.distributed_servers.subordinate_workers:
        num_gpus = len(subordinate_worker.gpu_indexes)
        uneven_pp += num_gpus
        if num_gpus != tp:
            uneven = True

    if uneven:
        tp = 1
        pp = uneven_pp
        logger.warning(
            f"The number of GPUs selected for each worker is not equal: {num_gpus} != {tp}, fallback to using pipeline parallelism."
        )
    return tp, pp


def read_lora_max_rank(paths: List[str]) -> Optional[int]:
    """
    Read the max LoRA rank across adapter dirs' adapter_config.json.

    Returns the largest of each adapter's `r` and any `rank_pattern` values.
    Returns None when no readable rank is found, so callers can skip injecting
    --max-lora-rank and fall back to the engine default / user-provided value.
    """
    max_rank: Optional[int] = None
    for path in paths:
        if not path:
            continue
        config_path = os.path.join(path, "adapter_config.json")
        try:
            data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Skip reading LoRA rank from {config_path}: {e}")
            continue
        if not isinstance(data, dict):
            logger.warning(
                f"Skip reading LoRA rank from {config_path}: not a JSON object"
            )
            continue
        ranks = []
        if isinstance(data.get("r"), int):
            ranks.append(data["r"])
        # PEFT allows per-module overrides in rank_pattern that may exceed `r`.
        rank_pattern = data.get("rank_pattern")
        if isinstance(rank_pattern, dict):
            ranks.extend(
                value for value in rank_pattern.values() if isinstance(value, int)
            )
        if ranks:
            max_rank = max([max_rank, *ranks]) if max_rank is not None else max(ranks)
    return max_rank


def _refuse_unrendered_router(model, instance) -> None:
    """The same rule as `_refuse_unrendered`, at the second place it can break.

    A router does not go through the injector, so the injector's check never
    sees it. Its three rendered fields fail in three different ways, and only
    one of them is legible on its own:

      image_name   the container runtime rejects `{{runner_image}}` as an
                   invalid reference — an error that names the placeholder but
                   not why it has no value, several layers from the cause
      run_command  the router starts and forwards to the literal string as if
                   it were a host
      env          silent, exactly as on the engine side

    A custom backend version can resolve the engines' image correctly and still
    leave the router's unresolved, because `_resolve_image` reads
    `model.image_name` first and the catalog puts the placeholder there — the
    launch then fails at docker with `invalid reference format`.
    """
    from gpustack.worker.pd_injection import _ANY_PLACEHOLDER, PDInjectionError

    unrendered = []
    for field in ("image_name", "run_command"):
        for match in _ANY_PLACEHOLDER.finditer(str(getattr(model, field, "") or "")):
            unrendered.append(f"{field}={match.group(0)}")
    for name, value in sorted((getattr(model, "env", None) or {}).items()):
        for match in _ANY_PLACEHOLDER.finditer(str(value)):
            unrendered.append(f"{name}={match.group(0)}")

    if not unrendered:
        return

    raise PDInjectionError(
        f"The managed router for {instance.name} would start with "
        f"{len(unrendered)} unrendered value(s) — {', '.join(unrendered)}. "
        "`{{runner_image}}` in particular means the backend version in use "
        "has no image on this worker: check that the model's backend version "
        "exists and, if it is a custom one, that its image is pullable here."
    )
