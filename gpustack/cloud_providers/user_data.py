# flake8: noqa: W605
import jinja2
from typing import Dict, Optional, Any, List
import yaml
from gpustack_runtime.detector import ManufacturerEnum

# default_user_data_template is assuming the NVIDIA drivers and container toolkit
# are pre-installed on the base image
default_user_data_template_jinja = """#cloud-config
write_files:
  - path: /var/lib/gpustack/config.yaml
    permissions: '0600'
    content: |
      server_url: {{ server_url }}
      token: {{ token }}
      {%- for k, v in secret_configs.items() %}
      {{ k }}: {{ v }}
      {%- endfor %}
        

  - path: /opt/gpustack-run-worker.sh
    permissions: '0755'
    content: |-
      #!/bin/bash
      set -e
      echo "$(date): trying to bring up gpustack worker container..." >> /var/log/post-reboot.log

      docker run -d --name gpustack-worker \\
      -e "GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME=gpustack-worker" \\
      --restart=unless-stopped \\
      --privileged \\
      --network=host \\
      -v /var/lib/gpustack:/var/lib/gpustack \\
      -v /var/run/docker.sock:/var/run/docker.sock \\
      {{ image_name }} \\
      --config-file=/var/lib/gpustack/config.yaml

      echo "$(date): gpustack worker container started" >> /var/log/post-reboot.log
"""

post_boot_service = """[Unit]
Description=bootstrap gpustack worker container
After=network.target docker.service
Wants=network.target docker.service

[Service]
Type=oneshot
RemainAfterExit=no
ExecStart=/bin/bash -c "/opt/gpustack-run-worker.sh && systemctl disable post-reboot.service"
StandardOutput=journal

[Install]
WantedBy=default.target
"""

dkms_service = """[Unit]
Description=DKMS Autoinstall
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/dkms autoinstall
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
"""


debian_driver_map = {"debian": "nvidia-open", "ubuntu": "nvidia-driver-570"}


class UserDataTemplate:
    """
    Template for user data. Only supports Debian/Ubuntu and nvidia drivers for now.
    """

    server_url: str
    token: str
    image_name: str
    distribution: Optional[str]
    install_driver: Optional[ManufacturerEnum]
    setup_driver: Optional[ManufacturerEnum]
    _data: Optional[Dict[str, Any]]
    secret_configs: Dict[str, Any]

    def __init__(
        self,
        server_url: str,
        token: str,
        image_name: str,
        secret_configs: Dict[str, Any] = {},
    ):
        self.server_url = server_url
        self.token = token
        self.image_name = image_name
        self.install_driver = None
        self.setup_driver = None
        self.distribution = None
        self.secret_configs = secret_configs
        template = jinja2.Environment().from_string(default_user_data_template_jinja)
        self._data = yaml.safe_load(
            template.render(
                server_url=self.server_url,
                token=self.token,
                image_name=self.image_name,
                secret_configs=self.secret_configs,
            )
        )
        self.distribution = "ubuntu"
        self._data.setdefault('packages', [])
        self._data.setdefault('runcmd', [])
        self._data.setdefault('write_files', [])

    def insert_runcmd(self, *commands: str):
        command_list: List[str] = self._data.setdefault('runcmd', [])
        for idx, command in enumerate(commands):
            command_list.insert(idx, command)

    def _process_install_driver(self) -> bool:
        """
        process_install_driver handles the installation of the GPU drivers.
        Returns True if a reboot is required after installation.
        """
        self._data['package_update'] = True
        self._data['package_upgrade'] = True
        packages: List[str] = self._data.get('packages')
        write_files: List[Dict[str, Any]] = self._data.get('write_files')
        # only supports nvidia and debian/ubuntu for now
        if self.install_driver != ManufacturerEnum.NVIDIA or self.distribution not in [
            "debian",
            "ubuntu",
        ]:
            return False
        driver_name = debian_driver_map.get(self.distribution, "nvidia-driver-570")
        nvidia_toolkit_version = "1.17.8-1"
        packages.extend(
            [
                "build-essential",
                "dkms",
                "linux-headers-generic",
                "curl",
            ]
        )
        write_files.append(
            {
                "path": "/etc/systemd/system/dkms-autoinstall.service",
                "content": dkms_service,
            }
        )
        self.insert_runcmd(
            'echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf',
            'echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist.conf',
            'update-initramfs -u',
            r"distribution=$(. /etc/os-release; echo $ID$(echo $VERSION_ID | sed 's/\.//g'))",
            'wget "https://developer.download.nvidia.com/compute/cuda/repos/$distribution/$(uname -m)/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring_1.1-1_all.deb',
            'dpkg -i /tmp/cuda-keyring_1.1-1_all.deb',
            'rm /tmp/cuda-keyring_1.1-1_all.deb',
            'curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg',
            "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
            "apt-get update",
            f"""DEBIAN_FRONTEND=noninteractive \
apt-get install -y \
  {driver_name} \
  cuda-toolkit-12-8 \
  nvidia-container-toolkit={nvidia_toolkit_version} \
  nvidia-container-toolkit-base={nvidia_toolkit_version} \
  libnvidia-container-tools={nvidia_toolkit_version} \
  libnvidia-container1={nvidia_toolkit_version}""",
            "echo 'export PATH=/usr/local/cuda/bin:$PATH' | tee /etc/profile.d/cuda.sh",
            "echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' | tee /etc/profile.d/cuda_ld_library_path.sh",
            "systemctl enable dkms-autoinstall.service",
        )
        return True

    def _process_setup_driver(self) -> None:
        """
        process_setup_driver handles the setup of the GPU drivers.
        """
        # only supports nvidia and debian/ubuntu for now
        if self.setup_driver != ManufacturerEnum.NVIDIA or self.distribution not in [
            "debian",
            "ubuntu",
        ]:
            return
        runcmd: List[str] = self._data.get('runcmd')
        runcmd.extend(
            [
                "nvidia-ctk runtime configure --runtime=docker --set-as-default",
                "systemctl restart docker",
            ]
        )

    def format(self) -> str:
        # hand packages
        packages: List[str] = self._data.get('packages', [])
        runcmds: List[str] = self._data.get('runcmd', [])
        write_files: List[Dict[str, Any]] = self._data.get('write_files', [])
        should_restart = False
        if self.distribution in ["debian", "ubuntu"]:
            packages.append("docker.io")
        # handle driver installation
        if self._process_install_driver():
            should_restart = True
        # handle driver setup
        self._process_setup_driver()
        # handle start on first boot
        if not should_restart:
            runcmds.append("/opt/gpustack-run-worker.sh")
        else:
            write_files.append(
                {
                    "content": post_boot_service,
                    "path": "/etc/systemd/system/post-reboot.service",
                }
            )
            runcmds.append("systemctl enable post-reboot.service")
            self._data["power_state"] = {
                "mode": "reboot",
                "timeout": 30,
                "message": "Rebooting after initial setup.",
            }
        return "#cloud-config\n" + yaml.dump(self._data, default_style='')
