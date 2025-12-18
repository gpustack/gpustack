import jinja2
import base64
from typing import Optional
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.schemas.clusters import ClusterRegistrationTokenPublic


class TemplateConfig(ClusterRegistrationTokenPublic):
    namespace: Optional[str] = None
    cluster_suffix: Optional[str] = None

    def render(self) -> str:
        def b64encode(value):
            return base64.b64encode(value.encode("utf-8")).decode("utf-8")

        with pkg_resources.path("gpustack.k8s", "manifests.jinja") as manifest_path:
            with manifest_path.open(encoding="utf-8") as f:
                template_data = f.read()
        env = jinja2.Environment()
        env.filters["b64encode"] = b64encode
        template = env.from_string(template_data)
        return template.render(config=self)

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
