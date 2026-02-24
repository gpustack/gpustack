from gpustack.cloud_providers.user_data import UserDataTemplate, ManufacturerEnum
import yaml


def test_userdata_template_basic():
    template = UserDataTemplate(
        server_url="http://test-server",
        token="test-token",
        image_name="gpustack/test:latest",
        worker_name="test-worker",
    )
    template.distribution = "ubuntu"
    template.install_driver = ManufacturerEnum.NVIDIA
    template.setup_driver = ManufacturerEnum.NVIDIA
    result = template.format()
    assert result.startswith("#cloud-config\n")
    data = yaml.safe_load(result)
    assert data["write_files"]
    assert any(
        f["path"] == "/var/lib/gpustack/config.yaml" for f in data["write_files"]
    )
    assert any(f["path"] == "/opt/gpustack-run-worker.sh" for f in data["write_files"])
    assert "runcmd" in data
    assert "systemctl enable post-reboot.service" in data["runcmd"]
    assert "packages" in data
    assert "docker.io" in data["packages"]
    assert "power_state" in data
    assert data["power_state"]["mode"] == "reboot"


def test_userdata_template_no_driver():
    template = UserDataTemplate(
        server_url="http://test-server",
        token="test-token",
        image_name="gpustack/test:latest",
        worker_name="test-worker",
    )
    template.distribution = "ubuntu"
    template.install_driver = None
    template.setup_driver = None
    result = template.format()
    data = yaml.safe_load(result)
    assert "power_state" not in data
    assert "/opt/gpustack-run-worker.sh" in data["runcmd"]


def test_userdata_template_debian():
    template = UserDataTemplate(
        server_url="http://test-server",
        token="test-token",
        image_name="gpustack/test:latest",
        worker_name="test-worker",
    )
    template.distribution = "debian"
    template.install_driver = ManufacturerEnum.NVIDIA
    template.setup_driver = ManufacturerEnum.NVIDIA
    result = template.format()
    data = yaml.safe_load(result)
    assert "docker.io" in data["packages"]
    assert any("dkms" in pkg for pkg in data["packages"])
    assert any("build-essential" in pkg for pkg in data["packages"])
    assert "runcmd" in data
    assert any("nvidia-ctk runtime configure" in cmd for cmd in data["runcmd"])


def test_userdata_template_env_in_worker_script():
    template = UserDataTemplate(
        server_url="http://test-server",
        token="test-token",
        image_name="gpustack/test:latest",
        worker_name="test-worker",
    )
    template.distribution = "ubuntu"
    result = template.format()
    data = yaml.safe_load(result)
    worker_script_file = next(
        f for f in data["write_files"] if f["path"] == "/opt/gpustack-run-worker.sh"
    )
    assert (
        "--config-file=/var/lib/gpustack/config.yaml" in worker_script_file["content"]
    )


def test_userdata_template_setup_driver():
    template = UserDataTemplate(
        server_url="http://test-server",
        token="test-token",
        image_name="gpustack/test:latest",
        worker_name="test-worker",
    )
    template.distribution = "ubuntu"
    template.setup_driver = ManufacturerEnum.NVIDIA
    result = template.format()
    data = yaml.safe_load(result)
    assert any("nvidia-ctk runtime configure" in cmd for cmd in data["runcmd"])


def test_userdata_template_secret_configs():
    template = UserDataTemplate(
        server_url="http://test-server",
        token="test-token",
        image_name="gpustack/test:latest",
        worker_name="test-worker",
        secret_configs={
            "SECRET_KEY": "mysecret",
            "OPTIONAL_KEY": None,
            "ANOTHER_KEY": 123,
        },
    )
    template.distribution = "ubuntu"
    result = template.format()
    data = yaml.safe_load(result)
    config_file = next(
        f for f in data["write_files"] if f["path"] == "/var/lib/gpustack/config.yaml"
    )
    content = config_file["content"]
    # SECRET_KEY and ANOTHER_KEY should appear, OPTIONAL_KEY should not
    assert "SECRET_KEY: mysecret" in content
    assert "ANOTHER_KEY: 123" in content
    assert "OPTIONAL_KEY" not in content
