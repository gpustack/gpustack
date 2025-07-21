# Upgrade

You can upgrade GPUStack using the installation script or by manually installing the desired version of the GPUStack Python package.

!!! note

    1. When upgrading, upgrade the GPUStack server first, then upgrade the workers.
    2. Please **DO NOT** upgrade from/to the main(dev) version or a release candidate(rc) version, as they may contain breaking changes. Use a fresh installation if you want to try the main or rc versions.

!!! note

    Before proceeding with an upgrade, it’s strongly recommended to back up your database. For default installations, stop the GPUStack server and create a backup of the file located at `/var/lib/gpustack/database.db`.

## Upgrade GPUStack Using the Installation Script(Deprecated)

!!! note

    The installation script method is deprecated as of version 0.7.

To upgrade GPUStack from an older version, re-run the installation script using the same configuration options you originally used.

Running the installation script will:

1. Install the latest version of the GPUStack Python package.
2. Update the system service (systemd, launchd, or Windows) init script to reflect the arguments passed to the installation script.
3. Restart the GPUStack service.

### Linux or macOS

For example, to upgrade GPUStack to the latest version on a Linux system and macOS:

```bash
curl -sfL https://get.gpustack.ai | <EXISTING_INSTALL_ENV> sh -s - <EXISTING_GPUSTACK_ARGS>
```

!!! Note

    `<EXISTING_INSTALL_ENV>` are the environment variables you set during the initial installation, and `<EXISTING_GPUSTACK_ARGS>` are the startup parameters you configured back then.

    **Simply execute the same installation command again, and the system will automatically perform an upgrade.**

To upgrade to a specific version, specify the `INSTALL_PACKAGE_SPEC` environment variable similar to the `pip install` command:

```bash
curl -sfL https://get.gpustack.ai | INSTALL_PACKAGE_SPEC=gpustack==x.y.z <EXISTING_INSTALL_ENV> sh -s - <EXISTING_GPUSTACK_ARGS>
```

### Windows

To upgrade GPUStack to the latest version on a Windows system:

```powershell
$env:<EXISTING_INSTALL_ENV> = <EXISTING_INSTALL_ENV_VALUE>
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content
```

!!! Note

    `<EXISTING_INSTALL_ENV>` are the environment variables you set during the initial installation, and `<EXISTING_GPUSTACK_ARGS>` are the startup parameters you configured back then.

    **Simply execute the same installation command again, and the system will automatically perform an upgrade.**

To upgrade to a specific version:

```powershell
$env:INSTALL_PACKAGE_SPEC = gpustack==x.y.z
$env:<EXISTING_INSTALL_ENV> = <EXISTING_INSTALL_ENV_VALUE>
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } <EXISTING_GPUSTACK_ARGS>"
```

## Docker Upgrade

If you installed GPUStack using Docker, upgrade to the a new version by pulling the Docker image with the desired version tag.

For example:

```bash
docker pull gpustack/gpustack:vX.Y.Z
```

Then restart the GPUStack service with the new image.

## pip Upgrade

If you install GPUStack manually using pip, upgrade using the common `pip` workflow.

For example, to upgrade GPUStack to the latest version:

```bash
pip install --upgrade gpustack
```

Then restart the GPUStack service according to your setup.

## Installer Upgrade

On macOS and Windows, the installation script is deprecated in versions after 0.7. You should migrate to the GPUStack installer in version 0.7.0 or later. You can download and install the new version of GPUStack via [Download Installer](./installation/desktop-installer.md#download-installer).

!!!note

    The Installer Upgrade has only been tested upgrading GPUStack from v0.6.2 to v0.7.0. It should be possible to upgrade from versions prior to v0.6.2 to Installer v0.7.0, but it is recommended to upgrade to v0.6.2 first and then use the Installer for migration upgrade.

1. Start GPUStack and a system tray icon will appear. It will show the `To Upgrade` state if an old version of GPUStack is installed.

   ![darwin-to-upgrade-state](./assets/desktop-installer/to-upgrade-darwin.png)

1. To upgrade and migrate to the new GPUStack version, you can click `Start` in the submenu of `Status`.
1. The original configuration will be migrated to the corresponding location according to the running operating system. Detailed configuration can be reviewed in [desktop configuration](./user-guide/desktop-setup.md#configuration)

   - macOS
     - Configuration via arguments will be migrated into a single configuration file `~/Library/Application Support/GPUStackHelper/config.yaml`.
     - Configuration from environment variables will be migrated into the `launchd` plist configuration `~/Library/Application Support/GPUStackHelper/ai.gpustack.plist`.
     - GPUStack data will be moved to the new data location `/Library/Application Support/GPUStack`.
   - Windows
     - Configuration via arguments will be migrated into a single configuration file `C:\Users\<Name>\AppData\Roaming\GPUStackHelper\config.yaml`.
     - Service configuration such as environment variables won't be merged as the system service `GPUStack` will be reused.
     - GPUStack data will be moved to the new data location `C:\ProgramData\GPUStack`.
