# Multiple Node Cluster

If you want to start a `worker` on a remote host and add it to the current service to form a cluster, follow these steps:

1. Execute the following command to obtain the current service `token`.

**Linux or MacOS**
```bash
cat /var/lib/gpustack/token
```
**Windows**
```bash
Get-Content -Path (Join-Path -Path $env:APPDATA -ChildPath "gpustack\token") -Raw
```

![Get Token](../assets/tutorials/get-token.png)

2. Current service address: `http://myserver`.

3. Execute the following command on the **remote host**, and add the token obtained in Step 1, such as `mytoken`, into the command.
```bash
curl -sfL https://get.gpustack.ai | sh -s - --server-url http://myserver --token mytoken
```

![Add New Worker](../assets/tutorials/add-new-worker.png)

4. Refresh the `worker` list in `http://myserver`, and you will see the new worker.

![New Woker](../assets/tutorials/new-worker.png)