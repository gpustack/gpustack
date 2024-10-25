# Running Inference With Ascend Device

GPUStack supports running inference on Ascend NPUs. This tutorial will guide you through the configuration steps.

## Environment Preparation

### System and Hardware

| OS    | Status  | Verified     |
| ----- | ------- | ------------ |
| Linux | Support | Ubuntu 20.04 |

| Device           | Status  | Verified      |
| ---------------- | ------- | ------------- |
| Training Server  | Support | Atlas 800T A2 |
| Inference Server | Support | /             |

### Ascend Components

1. Install Ascend driver, firmware, toolkit and kernel.

    Based on your device model, refer to the corresponding [documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0019.html) to install the Ascend driver and firmware. 

    Next, install the CANN toolkit and kernel by following the provided [instructions](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/softwareinst/instg/instg_0038.html). We currently support CANN version 8.x. 

    The Ascend component packages can be downloaded from the [resources download center](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.alpha003).

2. Verify Installation.

    After a successful installation, run the `npu-smi info` command to check if the driver was installed correctly.

    ```bash
    $npu-smi info
    +------------------------------------------------------------------------------------------------+
    | npu-smi 23.0.1                   Version: 23.0.1                                               |
    +---------------------------+---------------+----------------------------------------------------+
    | NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
    | Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
    +===========================+===============+====================================================+
    | 4     910B3               | OK            | 93.6        40                0    / 0             |
    | 0                         | 0000:01:00.0  | 0           0    / 0          3161 / 65536         |
    +===========================+===============+====================================================+
    +---------------------------+---------------+----------------------------------------------------+
    | NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
    +===========================+===============+====================================================+
    | No running processes found in NPU 4                                                            |
    +===========================+===============+====================================================+
    ```

## Installing GPUStack

Once your environment is ready, you can install GPUStack following the [installation guide](../installation/installation-script.md).

## Running Inference

After installation, you can deploy models and run inference. Refer to the [model management](../user-guide/model-management.md) for usage details. 

The Ascend NPU supports inference through the llama-box (llama.cpp) backend. For supported models, see the [llama.cpp Ascend NPU model supports](https://github.com/ggerganov/llama.cpp/blob/958367bf530d943a902afa1ce1c342476098576b/docs/backend/CANN.md).
