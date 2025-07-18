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

- Download and Run the Desktop Installer.

- After installation,  the application will show status as `To Upgrade`.

- Click `Start` to automatically migrate your existing data and configuration.
