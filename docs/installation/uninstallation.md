# Uninstallation

## Docker

If you install GPUStack using Docker, the followings are example commands to uninstall GPUStack. You can modify according to your setup:

```bash
# Remove the container
docker rm -f gpustack
# Remove the data volume
docker volume rm gpustack-data
```

## pip

If you install GPUStack using pip, the followings are example commands to uninstall GPUStack. You can modify according to your setup:

```bash
# Stop and remove the service
systemctl stop gpustack.service
rm -f /etc/systemd/system/gpustack.service
systemctl daemon-reload
# Uninstall the CLI
pip uninstall gpustack
# Remove the data directory
rm -rf /var/lib/gpustack
```

## Script

!!! warning

    Uninstallation script deletes the data in local datastore, configuration, model cache, and all of the scripts and CLI tools. It does not remove any data from external datastores.

If you installed GPUStack using the installation script, a script to uninstall GPUStack was generated during installation.

Run the following command to uninstall GPUStack:

=== "Linux"

    ```bash
    sudo /var/lib/gpustack/uninstall.sh
    ```

=== "macOS"

    ```bash
    sudo /var/lib/gpustack/uninstall.sh
    ```

=== "Windows"

    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; & "$env:APPDATA\gpustack\uninstall.ps1"
    ```
