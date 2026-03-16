import jinja2
import base64
import yaml
from typing import List, Optional
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.schemas.clusters import ClusterRegistrationTokenPublic
from gpustack.schemas.clusters import K8sVolumeMount
from gpustack_runtime.detector import ManufacturerEnum


class TemplateConfig(ClusterRegistrationTokenPublic):
    namespace: Optional[str] = None
    cluster_suffix: Optional[str] = None
    runtime_enum: Optional[ManufacturerEnum] = None
    runtime: Optional[str] = None
    k8s_volume_mounts: Optional[List[K8sVolumeMount]] = None

    def render(self) -> str:
        def b64encode(value):
            return base64.b64encode(value.encode("utf-8")).decode("utf-8")

        def to_yaml(value, indent=0):
            if hasattr(value, "model_dump"):
                value = value.model_dump(by_alias=True, exclude_none=True)
            elif isinstance(value, list):
                value = [
                    (
                        v.model_dump(by_alias=True, exclude_none=True)
                        if hasattr(v, "model_dump")
                        else v
                    )
                    for v in value
                ]

            dumped = yaml.dump(value, default_flow_style=False)
            if dumped.endswith("...\n"):
                dumped = dumped[:-4]

            return dumped.replace("\n", "\n" + " " * indent).strip()

        with pkg_resources.path("gpustack.k8s", "manifests.jinja") as manifest_path:
            with manifest_path.open(encoding="utf-8") as f:
                template_data = f.read()
        with pkg_resources.path("gpustack.k8s", "daemonset.jinja") as daemon_set_path:
            with daemon_set_path.open(encoding="utf-8") as f:
                daemon_set_data = f.read()
        env = jinja2.Environment()
        env.filters["b64encode"] = b64encode
        env.filters["to_yaml"] = to_yaml
        template = env.from_string(template_data)
        rendered = template.render(config=self)
        template_daemonset = env.from_string(daemon_set_data)
        daemon_set = template_daemonset.render(config=self)
        return "\n".join([rendered, daemon_set])

    def __init__(
        self, registration: Optional[ClusterRegistrationTokenPublic] = None, **data
    ):
        if registration is not None:
            base_data = registration.model_dump()
            base_data.update(data)
            super().__init__(**base_data)
        else:
            super().__init__(**data)
        if self.namespace is None and self.cluster_suffix is not None:
            self.namespace = f"gpustack-system-{self.cluster_suffix}"
        elif self.namespace is None:
            self.namespace = "gpustack-system"
        self.runtime = (
            self.runtime_enum.value if self.runtime_enum is not None else None
        )
