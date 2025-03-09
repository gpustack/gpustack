# Manual Installation

## Prerequites:

Install Python version 3.10 to 3.12.

## Install GPUStack CLI

Run the following to install GPUStack:

```shell
pip install gpustack
```

You can add extra dependencies, options are "vllm", "audio" and "all", such as:

```shell
# vllm is currently only available for Linux on AMD64
pip install gpustack[all]
```

To verify, run:

```shell
gpustack version
```

## Run GPUStack

### Run Server

Run the following command to start the GPUStack server:

```shell
gpustack start
```

By default, GPUStack uses `/var/lib/gpustack` as the data directory so you need `sudo` or proper permission for that. You can also set a custom data directory by running:

```
gpustack start --data-dir mypath
```

### (Optional) Add Worker

To add a worker to the GPUStack cluster, you need to specify the server URL and the authentication token.

To retrieve the token, run the following command on the GPUStack server host:

```shell
cat /var/lib/gpustack/token
```

To start a GPUStack worker and **register it with the GPUStack server**, run the following command on the worker host. Replace your specific URL, token, and IP address accordingly:

```shell
gpustack start --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_worker_host_ip
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
