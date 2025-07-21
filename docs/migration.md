# Migration from Legacy Script Installation

!!! note

      The installation script method is deprecated as of version 0.7. We recommend using **Docker** on Linux, and the [desktop installer](https://gpustack.ai/) on macOS or Windows.

If you previously installed GPUStack using the legacy installation script, follow the instructions below to migrate to a supported method.

## Linux Migration

### Step 1: Locate Your Existing Database

Find the path to your existing database directory (used in the legacy installation). For example:

```bash
/path/to/your/legacy/gpustack/data
```

We'll refer to this as `${your-database-file-location}` in the next step.

### Step 2: Reinstall GPUStack via Docker

Make sure your hardware platform is supported. Then run the following Docker command, replacing the volume mount path with your database location.

**Example: For NVIDIA GPUs**

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v ${your-database-file-location}:/var/lib/gpustack \
    gpustack/gpustack
```

This will launch GPUStack using Docker, preserving your existing data.

For other hardware platforms, please refer to the commands in the [Installation Documentation](installation/installation-requirements.md).

## macOS / Windows Migration

Download and install the new version of GPUStack via [Download Installer](./installation/desktop-installer.md#download-installer).

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
