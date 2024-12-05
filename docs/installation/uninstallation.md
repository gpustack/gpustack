# Uninstallation

## Uninstallation Script

!!! warning

    Uninstallation script deletes the data in local datastore(sqlite), configuration, model cache, and all of the scripts and CLI tools. It does not remove any data from external datastores.

If you installed GPUStack using the installation script, a script to uninstall GPUStack was generated during installation.

### Linux or macOS

Run the following command to uninstall GPUStack:

```bash
sudo /var/lib/gpustack/uninstall.sh
```

### Windows

Run the following command in PowerShell to uninstall GPUStack:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; & "$env:APPDATA\gpustack\uninstall.ps1"
```

## Manual Uninstallation

If you install GPUStack manually, the followings are example commands to uninstall GPUStack. You can modify according to your setup:

```bash
# Stop and remove the service.
systemctl stop gpustack.service
rm /etc/systemd/system/gpustack.service
systemctl daemon-reload
# Uninstall the CLI.
pip uninstall gpustack
# Remove the data directory.
rm -rf /var/lib/gpustack
```
