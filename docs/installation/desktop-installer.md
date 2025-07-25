# Desktop Installer

## Download Installer

=== "macOS"

    Download the [installer](https://gpustack.ai/download/gpustack.pkg).

    !!! note

        **Supported platforms:** Apple Silicon (M series), macOS 14 or later

=== "Windows"

    Download the [installer](https://gpustack.ai/download/GPUStackInstaller.msi).

    !!! note

        **Supported platforms:** Windows 10 and Windows 11

## Run GPUStack

Double-click the downloaded installer and follow the on-screen instructions to complete the installation.

## Open GPUStack UI

Click the `GPUStack icon` in the menu bar (on macOS) or system tray (on Windows), then select the `Web Console` menu.

![web console](../assets/desktop-installer/open-web-console.png)

No password verification is required when using the method above. If you need to access via IP, use the following command to check the initial password.

- macOS

```bash
cat /Library/Application\ Support/GPUStack/initial_admin_password
```

- Windows

```powershell
Get-Content "C:\ProgramData\GPUStack\initial_admin_password" -Tail 200 -Wait
```

## (Optional) Add Worker

### 1. Get Token

Click the `GPUStack icon` in the menu bar (macOS) or system tray (Windows), then select the `Copy Token` menu on the server machine.

### 2. Register Worker

On the worker machine:

1. Click the `GPUStack icon`, then select the `Quick Config` menu.

2. Select `Worker` as the service role.

3. Enter the `Server URL`.

4. Paste the `Token` copied from the server.

5. Click `Restart` to apply the settings.

6. Refresh the worker list on the server to confirm the new worker has been added.

![web console](../assets/desktop-installer/add-worker.png)
