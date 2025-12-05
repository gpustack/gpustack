# Quickstart

Use one of the following guides to quickly get started with GPUStack.

- [Using Self-Hosted GPUs](#using-self-hosted-gpus)
- [Using DigitalOcean GPUs](#using-digitalocean-gpus)

## Using Self-Hosted GPUs

This guide will help you set up GPUStack using your own self-hosted GPU servers.

!!! info "Prerequisites"

    1. A node with at least one NVIDIA GPU. For other GPU types, please check the guidelines in the GPUStack UI when adding a worker, or refer to the [Installation documentation](./installation/requirements.md) for more details.
    2. Ensure the NVIDIA driver, [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) are installed on the worker node.
    3. **(Optional)** A CPU node for hosting the GPUStack server. The GPUStack server does not require a GPU and can run on a CPU-only machine. [Docker](https://docs.docker.com/engine/install/) must be installed. Docker Desktop (for Windows and macOS) is also supported. If no dedicated CPU node is available, the GPUStack server can be installed on the same machine as a GPU worker node.
    4. Only Linux is supported for GPUStack worker nodes. If you use Windows, consider using WSL2 and avoid using Docker Desktop. macOS is not supported for GPUStack worker nodes.

### Install GPUStack Server

Run the following command to install and start the GPUStack server using [Docker](https://docs.docker.com/engine/install/):

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

??? Note "Alternative: Use Quay Container Registry Mirror"

    If you cannot pull images from Docker Hub or the download is very slow, you can use our Quay Container Registry mirror by pointing your registry to `quay.io`:

    ```bash
    sudo docker run -d --name gpustack \
        --restart unless-stopped \
        -p 80:80 \
        --volume gpustack-data:/var/lib/gpustack \
        quay.io/gpustack/gpustack \
        --system-default-container-registry quay.io
    ```

Check the GPUStack startup logs:

```bash
sudo docker logs -f gpustack
```

After GPUStack starts, run the following command to get the default admin password:

```bash
sudo docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

Open your browser and navigate to `http://your_host_ip` to access the GPUStack UI. Use the default username `admin` and the password you retrieved above to log in.

### Set Up a GPU Cluster Using Docker

1. On the GPUStack UI, navigate to the `Clusters` page.

2. Click the `Add Cluster` button.

3. Select `Docker` as the cluster provider.

4. Fill in the `Name` and `Description` fields for the new cluster, then click the `Save` button.

5. Follow the UI guidelines to configure the new worker node. You will need to run a Docker command on the worker node to connect it to the GPUStack server. The command will look similar to the following:

```bash
sudo docker run -d --name gpustack-worker \
      --restart=unless-stopped \
      --privileged \
      --network=host \
      --volume /var/run/docker.sock:/var/run/docker.sock \
      --volume gpustack-data:/var/lib/gpustack \
      --runtime nvidia \
      gpustack/gpustack \
      --server-url http://your_gpustack_server_url \
      --token your_worker_token \
      --advertise-address 192.168.1.2
```

4. Execute the command on the worker node to connect it to the GPUStack server.

5. After the worker node connects successfully, it will appear on the `Workers` page in the GPUStack UI.

Now you have set up a self-hosted GPU cluster. You can proceed to deploy models as described in the [Deploy a Model](#deploy-a-model) section below.

## Using DigitalOcean GPUs

This guide will help you set up GPUStack using [DigitalOcean](https://www.digitalocean.com/) GPU droplets.

!!! info "Prerequisites"

    1. Ensure you have a DigitalOcean account with sufficient credits to create GPU droplets. [Create a DigitalOcean API token](https://cloud.digitalocean.com/account/api/tokens) with read and write permissions.
    2. A CPU node for hosting the GPUStack server. The GPUStack server does not require a GPU and can run on a CPU-only machine. [Docker](https://docs.docker.com/engine/install/) must be installed. Docker Desktop (for Windows and macOS) is also supported.
    3. Your server URL must be accessible from the DigitalOcean droplet. For example, you can run the GPUStack server on a DigitalOcean droplet to get started quickly. If you run the GPUStack server on your local machine, consider using a tunneling service like [ngrok](https://ngrok.com/) to expose your server to the internet.

### Install GPUStack Server

Run the following command to install and start the GPUStack server using [Docker](https://docs.docker.com/engine/install/):

```bash
# Replace `http://your_server_external_url` with your actual server URL, which should be accessible from the DigitalOcean droplet.
# For example, http://server_droplet_public_ip or https://your_id.ngrok-free.app
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    -e GPUSTACK_SERVER_EXTERNAL_URL=http://your_server_external_url \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

Check the GPUStack startup logs:

```bash
sudo docker logs -f gpustack
```

After GPUStack starts, run the following command to get the default admin password:

```bash
sudo docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

Open your browser and navigate to `http://your_host_ip` to access the GPUStack UI. Use the default username `admin` and the password you retrieved above to log in.

### Set Up a DigitalOcean GPU Cluster

1. On the GPUStack UI, navigate to the `Cloud Credentials` page.

2. Click the `Add Cloud Credential` button and select `DigitalOcean` as the cloud provider.

3. Fill in the `Name` and `Access Token` fields, then click the `Save` button.

![create DigitalOcean cloud credential](assets/quick-start/create-do-credential.png)

4. Navigate to the `Clusters` page and click the `Add Cluster` button.

5. Select `DigitalOcean` as the cluster provider.

6. Fill in the `Name` field, select the DigitalOcean cloud credential you created earlier, then select the `Region` for the new cluster. Click the `Next` button to configure the worker pool.

![create DigitalOcean cluster basic](assets/quick-start/create-do-cluster-basic.png)

8. Select the `Instance Type` as `NVIDIA H100 1X`, or any other GPU instance type available in your selected region. For other fields, you can keep the default values or adjust them according to your needs. Click the `Save` button to create the cluster.

![create DigitalOcean cluster worker pool](assets/quick-start/create-do-cluster-worker-pool.png)

9. GPUStack will start creating the GPU droplet on DigitalOcean. You can monitor the status on the `Workers` page. Once the droplet is created and connected successfully, it will appear as `Ready` on the `Workers` page.

![DigitalOcean worker is ready](assets/quick-start/do-worker-ready.png)

Now you have set up a DigitalOcean GPU cluster. You can proceed to deploy models as described in the [Deploy a Model](#deploy-a-model) section below.

## Deploy a Model

1. Navigate to the `Catalog` page in the GPUStack UI.

2. Select the `Qwen3 0.6B` model from the list of available models.

3. After the deployment compatibility checks pass, click the `Save` button to deploy the model.

![deploy qwen3 from catalog](assets/quick-start/quick-start-qwen3.png)

4. GPUStack will start downloading the model files and deploying the model. When the deployment status shows `Running`, the model has been deployed successfully.

!!! note

    GPUStack uses containers to run models. The first-time model deployment may take some time to download the model files and container images. You can click `View Logs` in the UI to monitor the deployment progress.

![model is running](assets/quick-start/model-running.png)

5. Click `Playground - Chat` in the navigation menu, check that the model `qwen3-0.6b` is selected from the top-right `Model` dropdown. Now you can chat with the model in the UI playground.

![quick chat](assets/quick-start/quick-chat.png)

## Use the model via API

1. Hover over the user avatar and navigate to the `API Keys` page, then click the `New API Key` button.

2. Fill in the `Name` and click the `Save` button.

3. Copy the generated API key and save it somewhere safe. Please note that you can only see it once on creation.

4. You can now use the API key to access the OpenAI-compatible API endpoints provided by GPUStack. For example, use curl as the following:

```bash
# Replace `your_api_key` and `your_gpustack_server_url`
# with your actual API key and GPUStack server URL.
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Tell me a joke."
      }
    ],
    "stream": true
  }'
```

## Cleanup

After you complete using the deployed model, you can go to the `Deployments` page in the GPUStack UI and delete the model to free up resources.
