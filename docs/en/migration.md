# Migration from Legacy Script Installation

If you previously installed GPUStack using the legacy installation script, follow the instructions below to migrate to a supported method.

!!! note

    Before proceeding with a migration, it’s strongly recommended to back up your database. For default installations, stop the GPUStack server and create a backup of the file located at `/var/lib/gpustack/database.db`.

## Linux Migration

### Step 1: Locate Your Existing Data Directory

Find the path to your existing data directory used by the legacy installation. The default path is:

```bash
/var/lib/gpustack
```

We'll refer to this as `${your-data-dir}` in the next step.

### Step 2: Reinstall GPUStack via Docker

If you are using Nvidia GPUs, run the following Docker command to migrate your GPUStack server, replacing the volume mount path with your data directory location.

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v ${your-data-dir}:/var/lib/gpustack \
    gpustack/gpustack
```

This will launch GPUStack using Docker, preserving your existing data.

For workers and other GPU hardware platforms, please refer to the commands in the [Installation Documentation](installation/installation-requirements.md).

## macOS / Windows Migration

Download and install the new version of GPUStack via [Desktop Installer](./installation/desktop-installer.md#download-installer).

!!!note

    The Installer Upgrade has only been tested upgrading GPUStack from v0.6.2 to v0.7.0. It should be possible to upgrade from versions prior to v0.6.2 to Installer v0.7.0, but it is recommended to upgrade to v0.6.2 first and then use the Installer for migration upgrade.

1. Start GPUStack and a system tray icon will appear. It will show the `To Upgrade` state if an old version of GPUStack is installed.

   ![darwin-to-upgrade-state](../assets/desktop-installer/to-upgrade-darwin.png)

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
