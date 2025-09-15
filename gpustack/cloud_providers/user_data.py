# flake8: noqa: W605
from typing import Dict, Optional, Any
import yaml

default_user_data = """#cloud-config
package_update: true
package_upgrade: true
packages:
  - docker.io
  - docker-engine
runcmd:
  - systemctl enable docker
  - systemctl start docker
  - /opt/gpustack-run-worker.sh
"""

debian_user_data = """#cloud-config
package_update: true
package_upgrade: true
packages:
  - build-essential
  - dkms
  - linux-headers-generic
  - curl
  - docker.io
write_files:
  - content: |
      [Unit]
      Description=DKMS Autoinstall
      After=network.target

      [Service]
      Type=oneshot
      ExecStart=/usr/sbin/dkms autoinstall
      RemainAfterExit=true

      [Install]
      WantedBy=multi-user.target
    path: /etc/systemd/system/dkms-autoinstall.service
  - path: /etc/systemd/system/post-reboot.service
    content: |
      [Unit]
      Description=bootstrap gpustack worker container
      After=network.target docker.service
      Wants=network.target docker.service

      [Service]
      Type=oneshot
      RemainAfterExit=no
      ExecStart=/bin/bash -c '/opt/gpustack-run-worker.sh && systemctl disable post-reboot.service'
      StandardOutput=journal

      [Install]
      WantedBy=default.target
runcmd:
  - |
    echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf
    echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist.conf
  - update-initramfs -u
  - |
    distribution=$(. /etc/os-release; echo $ID$(echo $VERSION_ID | sed 's/\.//g'))
    wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/$(uname -m)/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
  - |
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  - apt-get update
  - |
      export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
      DEBIAN_FRONTEND=noninteractive \
      apt-get install -y \
        {driver_name} \
        cuda-toolkit-12-8 \
        nvidia-container-toolkit=${{NVIDIA_CONTAINER_TOOLKIT_VERSION}} \
        nvidia-container-toolkit-base=${{NVIDIA_CONTAINER_TOOLKIT_VERSION}} \
        libnvidia-container-tools=${{NVIDIA_CONTAINER_TOOLKIT_VERSION}} \
        libnvidia-container1=${{NVIDIA_CONTAINER_TOOLKIT_VERSION}}
  - |
    echo 'export PATH=/usr/local/cuda/bin:$PATH' | tee /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' | tee /etc/profile.d/cuda_ld_library_path.sh
  - systemctl enable dkms-autoinstall.service
  - systemctl enable post-reboot.service
  - nvidia-ctk runtime configure --runtime=docker --set-as-default 
power_state:
  mode: reboot
  timeout: 30
  message: "Rebooting after NVIDIA driver and CUDA installation."
"""

debian_driver_map = {"debian": "nvidia-open", "ubuntu": "nvidia-driver-570"}


def user_data_distribution(distribution: Optional[str]) -> Dict[str, Any]:
    if distribution in ["ubuntu", "debian"]:
        return yaml.safe_load(
            debian_user_data.format(driver_name=debian_driver_map[distribution])
        )
    else:
        return yaml.safe_load(default_user_data)
