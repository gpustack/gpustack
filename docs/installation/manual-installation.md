# Manual Installation

## Prerequites:

Install python3.10 or above with pip.

## Install GPUStack CLI

Run the following to install GPUStack:

```shell
# You can add extra dependencies, options are "vllm", "audio" and "all".
# e.g., gpustack[all]
pip install gpustack
```

To verify, run:

```shell
gpustack version
```

## Run GPUStack

Run the following command to start the GPUStack server:

```shell
gpustack start
```

By default, GPUStack uses `/var/lib/gpustack` as the data directory so you need `sudo` or proper permission for that. You can also set a custom data directory by running:

```
gpustack start --data-dir mypath
```

### Run GPUStack as a System Service

A recommended way is to run GPUStack as a startup service. For example, using systemd:

Create a service file in `/etc/systemd/system/gpustack.service`:

```
[Unit]
Description=GPUStack Service
Wants=network-online.target
After=network-online.target

[Service]
EnvironmentFile=-/etc/default/%N
ExecStart=gpustack start
Restart=always
RestartSec=3
StandardOutput=append:/var/log/gpustack.log
StandardError=append:/var/log/gpustack.log

[Install]
WantedBy=multi-user.target
```

Then start GPUStack:

```shell
systemctl daemon-reload
systemctl enable gpustack
```
