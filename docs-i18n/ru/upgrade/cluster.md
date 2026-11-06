---
translation_status: translated
---

# Обновление развертывания кластера { #upgrade-a-cluster-deployment }

GPUStack организует рабочие узлы в [кластеры](../user-guide/cluster-management.md). На этой странице описано обновление рабочего развертывания кластера после обновления сервера GPUStack.

!!! note

    1. Сначала обновите GPUStack **сервер** (см. [Обновление через Docker](docker.md) или [Обновление через Helm](helm.md)), затем обновите развертывания кластера.

    2. Обновите работника до **той же версии**, что и сервер. Не смешивайте рабочий образ `main`(dev) или релиз-кандидат(rc) со стабильным сервером.

Процедура обновления отражает исходное развертывание кластера (см. [Управление кластером](../user-guide/cluster-management.md)): повторно запустите развертывание для кластера Docker или повторно примените манифесты для кластера Kubernetes, используя новый образ.

## Докер-кластер { #docker-cluster }

Воркеры в кластере Docker запускаются как контейнеры, запущенные командой `docker run` с шага `Add Worker` кластера. Обновите каждого рабочего, вытащив новый образ, удалив старый контейнер и воссоздав его с **теми же аргументами и томами**, что и раньше:

```bash
docker pull gpustack/gpustack:vx.y.z   # match the server version

docker stop gpustack-worker
docker rm gpustack-worker

docker run -d --name gpustack-worker \
  ... \
  gpustack/gpustack:vx.y.z \
  --server-url http://your_gpustack_server_url \
  --token your_worker_token \
  --advertise-address your_worker_ip
```

!!! tip

    Вы можете получить актуальную команду `docker run` для целевой версии из пользовательского интерфейса GPUStack: на странице `Clusters` откройте шаг `Add Worker` кластера и скопируйте сгенерированный `Run command`.

## Кластер Kubernetes { #kubernetes-cluster }

Рабочие процессы в кластере Kubernetes развертываются как наборы DaemonSet, управляемые оператором GPUStack, из манифестов, применяемых во время регистрации кластера. Для обновления повторно примените манифесты, созданные для новой версии:

1. Перейдите на страницу `Clusters` и найдите кластер Kubernetes, который вы хотите обновить.
2. Нажмите кнопку с многоточием в столбце операций, затем выберите `Register Cluster`.
3. Выполните действия до шага `Run command` и скопируйте сгенерированную команду.
4. Запустите команду в среде, где установлен `kubectl` и `kubeconfig` настроен для целевого кластера. При этом обновленные манифесты будут повторно применены, а оператор переведет рабочие DaemonSets на новую версию.

Как только рабочие модули достигают `Ready`, в кластере запускается новая версия.

## Кластер облачных провайдеров { #cloud-provider-cluster }

Для облачных кластеров (например, DigitalOcean) каждый рабочий узел загружается с помощью Cloud-init, который записывает рабочую конфигурацию в `/var/lib/gpustack/config.yaml` и запускает контейнер `gpustack-worker` через `/opt/gpustack-run-worker.sh`. Чтобы обновить существующий узел, подключитесь к нему по SSH и заново создайте контейнер с новым образом, повторно используя ту же конфигурацию и тома:

```bash
# SSH into the worker node, then:
NEW_IMAGE=gpustack/gpustack:vx.y.z   # match the server version

docker pull "$NEW_IMAGE"

docker rm -f gpustack-worker

docker run -d --name gpustack-worker \
  -e "GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME=gpustack-worker" \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  -v /var/lib/gpustack:/var/lib/gpustack \
  -v /var/run/docker.sock:/var/run/docker.sock \
  "$NEW_IMAGE" \
  --config-file=/var/lib/gpustack/config.yaml
```

!!! note

    Чтобы **новые подготовленные** узлы (например, в результате масштабирования рабочего пула) появлялись и в новой версии, также обновите конфигурацию образа кластера в пользовательском интерфейсе. См. [Управление кластером](../user-guide/cluster-management.md#operating-worker-pools).
