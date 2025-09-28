<a id="desktop-installer"></a>

# 桌面安装程序

<a id="download-installer"></a>

## 下载安装程序

=== "macOS"

    下载[安装程序](https://gpustack.ai/download/gpustack.pkg)。

    !!! note

        **支持平台：** Apple Silicon（M 系列），macOS 14 或更高版本

=== "Windows"

    下载[安装程序](https://gpustack.ai/download/GPUStackInstaller.msi)。

    !!! note

        **支持平台：** Windows 10 和 Windows 11

## 运行 GPUStack

双击下载的安装程序，并按照屏幕提示完成安装。

## 打开 GPUStack 界面

点击菜单栏（macOS）或系统托盘（Windows）中的 `GPUStack icon`，然后选择 `Web Console` 菜单。

![网页控制台](../../assets/desktop-installer/open-web-console.png)

使用上述方法无需密码验证。如需通过 IP 访问，使用以下命令查看初始密码。

- macOS

```bash
cat /Library/Application\ Support/GPUStack/initial_admin_password
```

- Windows

```powershell
Get-Content "C:\ProgramData\GPUStack\initial_admin_password" -Tail 200 -Wait
```

## （可选）添加 Worker

### 1. 获取 Token

在服务器机器上，点击菜单栏（macOS）或系统托盘（Windows）中的 `GPUStack icon`，然后选择 `Copy Token` 菜单。

### 2. 注册 Worker

在 Worker 机器上：

1. 点击 `GPUStack icon`，选择 `Quick Config` 菜单。
2. 将服务角色选择为 `Worker`。
3. 输入 `Server URL`。
4. 粘贴从服务器复制的 `Token`。
5. 点击 `Restart` 应用设置。
6. 在服务器端刷新 Worker 列表，确认新的 Worker 已添加。

![添加 Worker](../../assets/desktop-installer/add-worker.png)