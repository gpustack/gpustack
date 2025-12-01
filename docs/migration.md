# Migrating from v0.7 and Earlier Versions to v2

!!! note

      Since v2.0.0, GPUStack supports Linux only. For other OS, move the data directory to a Linux system and run the migration.

## Embedded Database Migration (SQLite → Embedded PostgreSQL)

In v0.7 and earlier, GPUStack used an embedded SQLite database by default to store management data. Starting from v2.0.0, GPUStack dropped SQLite support and now uses an embedded PostgreSQL database by default for improved performance and scalability.

If you previously deployed GPUStack with the embedded SQLite database, follow the steps below to migrate your data to the new PostgreSQL-based format.

!!! warning

      **Backup First:** Before starting the migration, it’s strongly recommended to back up your database.

      For default installations on v0.7 or earlier, stop the GPUStack server and create a backup of data dir located inside the container at:

      ```
      /var/lib/gpustack
      ```

### Migration Steps

#### Identify Your Legacy Data Directory

Locate the data directory used by your previous GPUStack installation. The default path is:

```
/var/lib/gpustack
```

For other installation methods, refer to this [link](faq.md/#where-are-gpustacks-data-stored) to locate the data directory.

In the following steps, this path is referenced as `${your-data-dir}`.

#### Migrate Using Docker

- **Server Migration (NVIDIA GPUs)**

    If you are using NVIDIA GPUs, run the following Docker command to start the migration. Replace `${your-data-dir}` with your legacy data directory containing the original SQLite database and related files.

    By mounting `${your-data-dir}` to `/var/lib/gpustack` and setting the environment variable `GPUSTACK_MIGRATION_DATA_DIR`, GPUStack to automatically migrate the SQLite data to the new embedded PostgreSQL database during startup.

    ```diff
    sudo docker run -d --name gpustack-server \
        --restart=unless-stopped \
        --privileged \
        --network=host \
        --volume /var/run/docker.sock:/var/run/docker.sock \
    +   --env GPUSTACK_MIGRATION_DATA_DIR=/var/lib/gpustack \
    +   --volume ${your-data-dir}:/var/lib/gpustack \
        --runtime nvidia \
        gpustack/gpustack
    ```

    This command will launch the GPUStack server in Docker, preserving and migrating your existing data.

- **Worker Migration (NVIDIA GPUs)**

    For worker nodes, replace `${your-data-dir}` with the legacy worker data directory path. Use the following command:

    ```diff
    sudo docker run -d --name gpustack-worker \
        --restart=unless-stopped \
        --privileged \
        --network=host \
        --volume /var/run/docker.sock:/var/run/docker.sock \
    +   --volume ${your-data-dir}:/var/lib/gpustack \
        --runtime nvidia \
        gpustack/gpustack \
        --server-url ${server-url} \
        --token ${token}
    ```

    This will launch the GPUStack worker using your existing data and connect it to the specified server.

- **Other GPU Architectures**

    For architectures other than NVIDIA (e.g., AMD, Ascend), the migration process remains the same. To migrate on these platforms:

    1. Get the installation commands, please refer to the commands in the [Installation Documentation](installation/requirements.md).

    2. Mount the legacy data directory to `/var/lib/gpustack`.

    3. (Server Only) Add the environment variable `GPUSTACK_MIGRATION_DATA_DIR` as shown in the NVIDIA examples, only the server needs to add this environment variable.

    The Server and Worker migration commands can be used directly after applying these changes.

#### Recreating Model Instances

After the upgrade is complete, existing Model Instances may remain stuck in the `Starting` state. If this happens, recreating the Model Instance will allow the model to run normally.

## External Database Migration

GPUStack supports using an external database to store the management data. If you previously deployed GPUStack with an external database, follow the steps below to migrate your data.

!!! warning

      **Backup Required.** Before proceeding, back up the following:

      - The data directory used by your previous GPUStack installation, typically located at:

        ```
        /var/lib/gpustack
        ```

      - Your external database, using the backup procedure recommended by your database system.

### Migration Steps

#### Identify Your Legacy Data Directory

Locate the data directory used by your previous GPUStack installation. The default path is:

```
/var/lib/gpustack
```

For other installation methods, refer to this [link](faq.md/#where-are-gpustacks-data-stored) to locate the data directory.

In the following steps, this path is referenced as `${your-data-dir}`.

#### Migrate Using Docker

- **Server Migration (NVIDIA GPUs)**

    Keep the `--database-url` flag, `GPUSTACK_DATABASE_URL` environment variable, or `database_url` in your configuration file exactly as before. It must continue pointing to your existing external database.

    If you are using NVIDIA GPUs, run the following Docker command to start the migration. Replace `${your-data-dir}` with your legacy data directory. By mounting `${your-data-dir}` to `/var/lib/gpustack`.

    ```diff
    sudo docker run -d --name gpustack-server \
        --restart=unless-stopped \
        --privileged \
        --network=host \
        --volume /var/run/docker.sock:/var/run/docker.sock \
    +   --volume ${your-data-dir}:/var/lib/gpustack \
        --runtime nvidia \
        gpustack/gpustack \
    +     --database-url ${your-database-url}
    ```

- **Worker Migration (NVIDIA GPUs)**

    For worker nodes, replace `${your-data-dir}` with the legacy worker data directory path. Use the following command:

    ```diff
    sudo docker run -d --name gpustack-worker \
        --restart=unless-stopped \
        --privileged \
        --network=host \
        --volume /var/run/docker.sock:/var/run/docker.sock \
    +   --volume ${your-data-dir}:/var/lib/gpustack \
        --runtime nvidia \
        gpustack/gpustack \
        --server-url ${server-url} \
        --token ${token}
    ```

- **Other GPU Architectures**

    For architectures other than NVIDIA (e.g., AMD, Ascend), the migration process remains the same. To migrate on these platforms:

    1. Get the installation commands, please refer to the commands in the [Installation Documentation](installation/requirements.md).

    2. Mount the legacy data directory to `/var/lib/gpustack`.

    3. (Server Only) Continue use gpustack start flag `--database-url` or `GPUSTACK_DATABASE_URL` environment variable or config file with `database_url` as before.

#### Recreating Model Instances

After the upgrade is complete, existing Model Instances may remain stuck in the `Starting` state. If this happens, recreating the Model Instance will allow the model to run normally.

#### Migration from llama-box
If you were using llama-box as the inference backend in previous versions, please note that llama-box is no longer supported after v2.0.0. Use llama.cpp via the custom inference backend instead.

1. Configure llama.cpp on Inference Backend page. For llama.cpp configuration, refer to this [document](./tutorials/using-custom-backends.md/#deploy-gguf-models-with-llamacpp).
2. Go to the Deployment page, modify the model originally launched with llama-box, change the backend to llama.cpp
3. Recreate the model instance after saving.

> Note: As of v2.0.0, llama-box is no longer supported, and the system now uses a custom inference backend for GGUF model deployments. Distributed inference across multiple workers is currently not supported with custom inference backends.
