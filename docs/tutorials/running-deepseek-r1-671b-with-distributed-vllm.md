# Running DeepSeek R1 671B with Distributed vLLM

This tutorial guides you through the process of configuring and running the original **DeepSeek R1 671B** using **Distributed vLLM** on a GPUStack cluster. Due to the extremely large size of the model, distributed inference across multiple workers is usually required.

GPUStack enables easy setup and orchestration of distributed inference using vLLM, making it possible to run massive models like DeepSeek R1 with minimal manual configuration.

## Prerequisites

Before you begin, make sure the following requirements are met:

- You have access to a sufficient number of Linux nodes, each equipped with the required GPUs. For example:

<div class="center-table" markdown>

| **GPU**          | **Number of Nodes** |
| ---------------- | ------------------- |
| H100/H800:8      | 2                   |
| A100/A800-80GB:8 | 4                   |
| A100/A800:8      | 8                   |

</div>
- High-speed interconnects such as NVLink or InfiniBand are recommended for optimal performance.
- Model files should be downloaded to the same path on each node. While GPUStack supports on-the-fly model downloading, pre-downloading is recommended as it can be time consuming depending on the network speed.

!!! note

    - In this tutorial, we assume a setup of 4 nodes, each equipped with 8 A800-80GB GPUs and connected via 200G InfiniBand.
    - A100/A800 GPUs do not support the FP8 precision originally used by DeepSeek R1. Hence, we use the BF16 version from [Unsloth](https://huggingface.co/unsloth/DeepSeek-R1-BF16).

## Step 1: Install GPUStack Server

According to the [NVIDIA Installation](../installation/nvidia/installation.md), you can use the following command to start the GPUStack server **with the built-in worker**:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --volume /path/to/your/model:/path/to/your/model \
    --runtime nvidia \
    gpustack/gpustack

```

!!! note

    - Replace `/path/to/your/model` with the actual path.

After GPUStack server is up and running, run the following commands to get the initial admin password:

```bash
sudo docker exec gpustack \
    cat /var/lib/gpustack/initial_admin_password

```

## Step 2: Access GPUStack UI

Login to the GPUStack UI using the `admin` user and the obtained password.

```
http://your_gpustack_server_ip_or_hostname
```

## Step 3: Install GPUStack Workers

Navigate to the `Workers` page in the GPUStack UI, click `Add Worker` button to get the command for adding workers.

And then on **each worker node**, run the worker adding command to start a GPUStack worker:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --volume /path/to/your/model:/path/to/your/model \
    --runtime nvidia \
    gpustack/gpustack \
    --server-url http://your_gpustack_server_ip_or_hostname \
	--token your_gpustack_cluster_token

```

!!! note

    - Replace the placeholder paths, IP address/hostname, and cluster token accordingly.
    - Replace `/path/to/your/model` with the actual path on your system where the DeepSeek R1 model files are stored.

After all workers are added, return to the GPUStack UI.

Navigate to the `Workers` page to verify that all workers are in the Ready state and their GPUs are listed.

![initial-resources](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/initial-resources.png)

## Step 4: Deploy the DeepSeek R1 Model

1. Go to the `Deployments` page.
2. Click `Deploy Model`.
3. Select `Local Path` as your source.
4. Enter a name (e.g., `DeepSeek-R1`) in the `Name` field.
5. Specify the `Model Path` as the directory that contains the DeepSeek R1 model files on each worker node.
6. Ensure the `Backend` is set to `vLLM`.
7. After passing the compatibility check, click `Save` to deploy.

![deploy-model](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/deploy-model.png)

## Step 5: Monitor Deployment

You can monitor the deployment status on the `Deployments` page. Hover over `distributed across workers` to view GPU and worker usage. Click `View Logs` to see real-time logs showing model loading progress. It may take a few minutes to load the model.

![model-info](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/model-info.png)

After the model is running, navigate to the `Workers` page to check GPU utilization. By default, vLLM uses 90% of GPU memory. You may adjust this in the model configuration settings.

![resources-loaded](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/resources-loaded.png)

## Step 6: Run Inference via Playground

Once the model is deployed and running, you can test it using the GPUStack Playground.

1. Navigate to the `Playground` -> `Chat`.
2. If only one model is deployed, it will be selected by default. Otherwise, use the dropdown menu to choose `DeepSeek-R1`.
3. Enter prompts and interact with the model.

![playground-chat](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/playground-chat.png)

You can also use the `Compare` tab to test concurrent inference scenarios.

![playground-compare](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/playground-compare.png)

You have now successfully deployed and run DeepSeek R1 671B using Distributed vLLM on a GPUStack cluster. Explore the model’s performance and capabilities in your own applications.

For further assistance, feel free to reach out to the GPUStack community or support team.
