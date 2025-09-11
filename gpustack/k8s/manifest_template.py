import jinja2
import base64
from dataclasses import dataclass
from gpustack.utils.compat_importlib import pkg_resources


@dataclass
class TemplateConfig:
    cluster_suffix: str
    token: str
    server_url: str
    image: str

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
