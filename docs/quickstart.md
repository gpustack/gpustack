# Quickstart

## Install GPUStack

=== "Linux"

    We provide two installation methods on Linux:

    - **Docker installation**(Recommended)
    - **pip installation**

    We strongly recommend using [Docker](https://docs.docker.com/engine/install/) for installing the GPUStack on Linux. 
    
    **For Example: NVIDIA CUDA**

    Run the following command to start the GPUStack server.

    ```bash
    docker run -d --name gpustack \
          --restart=unless-stopped \
          --gpus all \
          --network=host \
          --ipc=host \
          -v gpustack-data:/var/lib/gpustack \
          gpustack/gpustack
    ```

    If you need to change the default server port 80, please use the --port parameter:

    ```bash
    docker run -d --name gpustack \
          --restart=unless-stopped \
          --gpus all \
          --network=host \
          --ipc=host \
          -v gpustack-data:/var/lib/gpustack \
          gpustack/gpustack
          --port 9090
    ```

    !!! tip
        If the quick installation fails, please refer to the [Docker Installation](installation/nvidia-cuda/online-installation/#docker-installation) guide for more detailed instructions.


    
    For on other hardware platforms Docker installation or pip installation details please refer to the [Installation Documentation](installation/installation-requirements.md).

    !!! note
        If you're a beginner, you don't need to run the `Add worker` action — it's not required for the Quick Start experience.  
        You can also skip it if you're not sure what `Add worker` does for now.

=== "macOS"

    **Supported platforms:** Apple Silicon (M series), macOS 14 or later

    1. [Download the installer](https://gpustack.ai)

    2. Run the installer
    
    ![mac installer](assets/quick-start/mac-installer.png)

    3. Installation Successful
    
    After successful installation, the GPUStack icon appears in the status bar.

     ![mac installer](assets/quick-start/mac-done.png)


=== "Windows"

    **Supported platforms:** Windows 10, Windows 11
    
    1. [Download the installer](https://gpustack.ai)

    2. Run the installer
    
    ![windows installer](assets/quick-start/windows-installer.png)

    3. Installation Successful
    
    After successful installation, the GPUStack icon will appear in the system tray.
    
     ![windows done](assets/quick-start/windows-done.png)


## Open GPUStack UI

Open `http://your_host_ip` in the browser to access the GPUStack UI. Log in to GPUStack with username `admin` and the default password. You can run the following command to get the password for the default setup:

=== "Linux"

    ```bash
    cat /var/lib/gpustack/initial_admin_password
    ```

=== "macOS"

    ```bash
    cat /var/lib/gpustack/initial_admin_password
    ```

=== "Windows"

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\initial_admin_password" -Raw
    ```

![login](assets/quick-start/quick-start-login.png)

### Deploy a Model
1. Navigate to the `Catalog` page in the GPUStack UI.

2. In the catalog list page, use the search bar in the top left to search for the model keyword `qwen3`.

3. In the search results, select `Qwen3`. If the **Compatibility Check Passed** message appears, click the `Save` button to deploy the model. You will be automatically redirected to the `Models` page once the deployment starts successfully.

![deploy qwen3 from catalog](assets/quick-start/quick-start-qwen3.png)

4. When the status shows `Running`, the model has been deployed successfully.

![deploy qwen3 from catalog](assets/quick-start/model-running.png)

5. Click `Playground - Chat` in the navigation menu, then select the  model from the top-right corner `Model` dropdown. Now you can chat with the LLM in the UI playground.

![deploy qwen3 from catalog](assets/quick-start/quick-chat.png)

### Try the Model with curl

1. Hover over the user avatar and navigate to the `API Keys` page, then click the `New API Key` button.

2. Fill in the `Name` and click the `Save` button.

3. Copy the generated API key and save it somewhere safe. Please note that you can only see it once on creation.

4. Now you can use the API key to access the OpenAI-compatible API. For example, use curl as the following:

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1-openai/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
```

## Cleanup

After you complete using the deployed models, you can go to the `Models` page in the GPUStack UI and delete the models to free up resources.
