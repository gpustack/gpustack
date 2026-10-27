# 卸载 { #uninstallation }

GPUStack 通常通过容器部署，因此卸载主要涉及移除容器及关联的数据卷。

例如，GPUStack 运行在名为 `gpustack` 的 Docker 容器中时，执行：

```bash
docker rm -f gpustack

```

如需同时移除关联的数据卷，可以执行：

```bash
docker volume rm <data_volume_name>

```
