# CPU Inference

If GPU resources are limited, some layers will be offloaded to the CPU, with full CPU inference used only when no GPU is available.

![VRAM Lack](../assets/tutorials/vram-lack.png)

Deploy a model and enable `Allow CPU Offloading`.

![Allow CPU Offload](../assets/tutorials/allow-cpu-offload.png)

After the deployment is successful, you will be able to see the number of layers offloaded to the CPU.

![CPU Offload](../assets/tutorials/cpu-offload.png)

Then you can test its inference metrics in the `Playground`.

![CPU Inference](../assets/tutorials/cpu-inference.png)