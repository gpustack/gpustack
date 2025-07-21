# Apple Metal Installation

## Supported Devices

- [x] Apple Metal (M-series chips)

## Supported Platforms

| OS    | Version                 | Arch  | Supported methods                                                                                                                                                         |
| ----- | ----------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| macOS | 14 Sonoma<br>15 Sequoia | ARM64 | [Desktop Installer](./desktop-installer.md) (Recommended)<br>[Installation Script](#installation-scriptdeprecated) (Deprecated) <br>[pip Installation](#pip-installation) |

## Supported backends

- [x] llama-box
- [x] vox-box (CPU backend)

## Prerequisites

- Python 3.10 ~ 3.12

Check the Python version:

```bash
python -V
```

## Installation Script(Deprecated)

!!! note

    The installation script method is deprecated as of version 0.7.

GPUStack provides a script to install it as a service with default port 80.

### Run GPUStack

```bash
curl -sfL https://get.gpustack.ai | sh -s -
```

If you need support for audio models, run:

```bash
curl -sfL https://get.gpustack.ai | INSTALL_SKIP_BUILD_DEPENDENCIES=0 sh -s -
```

To configure additional environment variables and startup flags when running the script, refer to the [Installation Script](./installation-script.md).

After installed, ensure that the GPUStack startup logs are normal:

```bash
tail -200f /var/log/gpustack.log
```

If the startup logs are normal, open `http://your_host_ip` in the browser to access the GPUStack UI. Log in to GPUStack with username `admin` and the default password. You can run the following command to get the password for the default setup:

```bash
cat /var/lib/gpustack/initial_admin_password
```

If you specify the `--data-dir` parameter to set the data directory, the `initial_admin_password` file will be located in the specified directory.

### (Optional) Add Worker

To add workers to the GPUStack cluster, you need to specify the server URL and authentication token when installing GPUStack on the workers.

To get the token used for adding workers, run the following command on the GPUStack **server node**:

```bash
cat /var/lib/gpustack/token
```

If you specify the `--data-dir` parameter to set the data directory, the `token` file will be located in the specified directory.

To install GPUStack and start it as a worker, and **register it with the GPUStack server**, run the following command on the **worker node**. Be sure to replace the URL and token with your specific values:

```bash
curl -sfL https://get.gpustack.ai | sh -s - --server-url http://your_gpustack_url --token your_gpustack_token
```

If you need support for audio models, run:

```bash
curl -sfL https://get.gpustack.ai | INSTALL_SKIP_BUILD_DEPENDENCIES=0 sh -s - --server-url http://your_gpustack_url --token your_gpustack_token
```

After installed, ensure that the GPUStack startup logs are normal:

```bash
tail -200f /var/log/gpustack.log
```

## pip Installation

### Install GPUStack

Run the following to install GPUStack:

```bash
pip install gpustack
```

If you need support for audio models, run:

```bash
pip install "gpustack[audio]"
```

To verify, run:

```bash
gpustack version
```

### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker**:

```bash
sudo gpustack start
```

If the startup logs are normal, open `http://your_host_ip` in the browser to access the GPUStack UI. Log in to GPUStack with username `admin` and the default password. You can run the following command to get the password for the default setup:

```bash
cat /var/lib/gpustack/initial_admin_password
```

If you specify the `--data-dir` parameter to set the data directory, the `initial_admin_password` file will be located in the specified directory.

By default, GPUStack uses `/var/lib/gpustack` as the data directory so you need `sudo` or proper permission for that. You can also set a custom data directory by running:

```
gpustack start --data-dir mypath
```

You can refer to the [CLI Reference](../cli-reference/start.md) for available CLI Flags.

### (Optional) Add Worker

To add a worker to the GPUStack cluster, you need to specify the server URL and the authentication token.

To get the token used for adding workers, run the following command on the GPUStack **server node**:

```bash
cat /var/lib/gpustack/token
```

If you specify the `--data-dir` parameter to set the data directory, the `token` file will be located in the specified directory.

To start a GPUStack worker and **register it with the GPUStack server**, run the following command on the **worker node**. Be sure to replace the URL and token with your specific values:

```bash
sudo gpustack start --server-url http://your_gpustack_url --token your_gpustack_token
```

### Run GPUStack as a Launchd Service

A recommended way is to run GPUStack as a startup service. For example, using launchd:

Create a service file in `/Library/LaunchDaemons/ai.gpustack.plist`:

```bash
sudo tee /Library/LaunchDaemons/ai.gpustack.plist > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.gpustack</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(command -v gpustack)</string>
    <string>start</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>EnableTransactions</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/var/log/gpustack.log</string>
  <key>StandardErrorPath</key>
  <string>/var/log/gpustack.log</string>
</dict>
</plist>
EOF
```

Then start GPUStack:

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

Check the service status:

```bash
sudo launchctl print system/ai.gpustack
```

And ensure that the GPUStack startup logs are normal:

```bash
tail -200f /var/log/gpustack.log
```
