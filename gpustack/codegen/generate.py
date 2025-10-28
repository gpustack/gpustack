from dataclasses import dataclass
import os
import shutil
from typing import List
from jinja2 import Environment, FileSystemLoader

from .filters import to_dash_plural, to_snake_case, to_plural, to_underscore_plural


def main():
    cfg = Config(
        class_names=[
            "Worker",
            "Model",
            "ModelInstance",
            "ModelFile",
            "User",
            "InferenceBackend",
        ]
    )

    env = Environment(loader=FileSystemLoader(cfg.template_dir), auto_reload=True)
    env.filters["to_snake_case"] = to_snake_case
    env.filters["to_plural"] = to_plural
    env.filters["to_underscore_plural"] = to_underscore_plural
    env.filters["to_dash_plural"] = to_dash_plural

    reset(cfg)
    gen_http_clients(env, cfg)
    gen_clients(env, cfg)
    gen_clientset(env, cfg)
    write_init(cfg)

    print("Code gen succeeded!")


@dataclass
class Config:
    template_dir: str = os.path.join(os.path.dirname(__file__), "templates")
    output_dir: str = "gpustack/client"
    class_names: List[str] = None


def gen_clients(env: Environment, cfg: Config):
    template = env.get_template("client.py.jinja")

    for class_name in cfg.class_names:
        data = {
            "class_name": class_name,
        }
        client_code = template.render(data)

        with open(
            f"{cfg.output_dir}/generated_{to_snake_case(class_name)}_client.py", "w"
        ) as f:
            f.write(client_code)


def gen_clientset(env: Environment, cfg: Config):
    template = env.get_template("clientset.py.jinja")

    data = {
        "class_names": cfg.class_names,
    }
    client_code = template.render(data)

    with open(f"{cfg.output_dir}/generated_clientset.py", "w") as f:
        f.write(client_code)


def gen_http_clients(env: Environment, cfg: Config):
    shutil.copyfile(
        f"{cfg.template_dir}/http_client.py.jinja",
        f"{cfg.output_dir}/generated_http_client.py",
    )


def write_init(cfg: Config):
    with open(f"{cfg.output_dir}/__init__.py", "w") as f:
        f.write(
            """from .generated_clientset import ClientSet


__all__ = ["ClientSet"]
"""
        )


def reset(cfg: Config):
    output_dir = cfg.output_dir
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file == "__init__.py" or file.startswith("generated_"):
                os.remove(os.path.join(output_dir, file))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


if __name__ == "__main__":
    main()
