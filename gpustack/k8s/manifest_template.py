import jinja2
import base64
from typing import Optional
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.schemas.clusters import ClusterRegistrationTokenPublic
from gpustack_runtime.detector import ManufacturerEnum


class TemplateConfig(ClusterRegistrationTokenPublic):
    namespace: Optional[str] = None
    cluster_suffix: Optional[str] = None
    runtime_enum: Optional[ManufacturerEnum] = None
    runtime: Optional[str] = None

    def render(self) -> str:
        def b64encode(value):
            return base64.b64encode(value.encode("utf-8")).decode("utf-8")

        with pkg_resources.path("gpustack.k8s", "manifests.jinja") as manifest_path:
            with manifest_path.open(encoding="utf-8") as f:
                template_data = f.read()
        with pkg_resources.path("gpustack.k8s", "daemonset.jinja") as daemon_set_path:
            with daemon_set_path.open(encoding="utf-8") as f:
                daemon_set_data = f.read()
        env = jinja2.Environment()
        env.filters["b64encode"] = b64encode
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
