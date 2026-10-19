---
translation_status: translated
---

# Установка через Helm { #installation-via-helm }

Начиная с версии 2.2.0, GPUStack можно развертывать в Kubernetes с помощью внутрикластерного API-шлюза [Higress](https://higress.io/) с использованием официального чарта Helm.

!!! note

    Развертывание GPUStack в Kubernetes с помощью Higress в настоящее время считается **экспериментальным**. Прежде чем продолжить, ознакомьтесь с [ограничениями](#limitations).

## Предварительные условия { #prerequisites }

| Компонент | Версия |
| ---------- | ---------- |
| Helm       | >= v3.18.4 |
| Kubernetes | >= v1.30.0 |
| GPUStack   | >= v2.2.0  |

Кроме того:

- В кластере необходимо настроить значение по умолчанию `StorageClass` для тома данных сервера (в k3s значение по умолчанию — `local-path`). Альтернативно, установите `server.dataVolume.hostPath`, чтобы использовать том пути хоста вместо PVC.
- Для работников графического процессора убедитесь, что на узлах установлены соответствующие драйверы графического процессора и наборы инструментов для контейнеров (см. [Требования к установке](./requirements.md)), а также установлено [Обнаружение функций узла (NFD)](https://kubernetes-sigs.github.io/node-feature-discovery/), чтобы узлы графического процессора были помечены.

## Ограничения { #limitations }

- Сервер GPUStack развернут как `StatefulSet` и в настоящее время **не** поддерживает более одной реплики.
- По умолчанию используется встроенная база данных `PostgreSQL`. Для производства рекомендуется указать внешнюю базу данных через `server.externalDatabaseURL`.
- Плагины Higress обслуживаются специальным развертыванием `gpustack/higress-plugins`, установленным вместе с GPUStack. Шлюз Higress загружает плагины из этого сервиса при перезапуске; если сервис недоступен, запуск шлюза блокируется до тех пор, пока плагины не станут доступны.
- Входящий в комплект субчарт `higress-core` развертывает Higress в качестве входного контроллера кластера. Если другой входящий контроллер уже запущен, установите `higress-core.enabled=false` и укажите `gateway.ingressClassname` на существующий экземпляр Higress.

## Установите k3s (необязательно) { #install-k3s-optional }

В следующих шагах в качестве примера дистрибутива Kubernetes используется k3s. Другие CNCF-совместимые дистрибутивы (RKE2, кластеры на базе kubeadm или управляемые облачные Kubernetes) также работают, если они соответствуют указанным выше требованиям к версии.

Установите k3s с отключенным Traefik, поскольку в качестве входящего контроллера используется Higress. Информацию о кластерах k3s высокой доступности см. в документации [k3s](https://docs.k3s.io/datastore/ha).

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.30.11+k3s1 INSTALL_K3S_EXEC="--disable=traefik" sh -
```

Проверьте настройку:

```bash
kubectl version
```

## Установите GPUStack с помощью Helm. { #install-gpustack-with-helm }

Чарт публикуется как артефакт OCI в Docker Hub по адресу `oci://registry-1.docker.io/gpustack/gpustack-chart`. Версия чарта отслеживает выпуск GPUStack (например, чарт `2.2.0` поставляется с GPUStack `v2.2.0`).

Установите последнюю стабильную версию. Если `--version` опущено, Helm автоматически обрабатывает новейшую стабильную версию чарта и пропускает сборки для разработчиков и предварительные версии:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace
```

Чтобы закрепить конкретную версию, добавьте `--version <chart-version>` (версия чарта соответствует выпуску GPUStack, например `2.2.0`):

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --version <chart-version>
```

По умолчанию субчарт `higress-core` включён и развертывается вместе с GPUStack. Если в вашем кластере уже установлен Higress, отключите встроенный Higress и укажите GPUStack на существующий экземпляр:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set higress-core.enabled=false \
  --set gateway.ingressClassname=<your-higress-ingressclass>
```

Чтобы настроить параметры, используйте `--set key=value` или `-f your-values.yaml` во время установки. См. [Параметры чарта](#chart-parameters) ниже.

### Установка из клонированного репозитория. { #installing-from-a-cloned-repository }

Альтернативно клонируйте репозиторий и установите его из локального каталога чарта (полезно для изолированных сред или при изменении чарта):

```bash
git clone https://github.com/gpustack/gpustack.git
cd gpustack/charts
helm dependency update ./gpustack-chart
helm install gpustack ./gpustack-chart \
  --namespace gpustack-system --create-namespace
```

### Установка Higress отдельно { #installing-higress-separately }

Если вы установили `higress-core.enabled=false`, установите совместимый экземпляр Higress перед развертыванием GPUStack:

```bash
# Add the Higress Helm repository
helm repo add higress.io https://higress.io/helm-charts

# Install higress-core (match the version pinned by the GPUStack chart)
helm install higress higress.io/higress-core \
  --namespace higress-system --create-namespace \
  --version 2.1.9
```

Убедитесь, что IngressClass доступен и что его имя соответствует `gateway.ingressClassname`:

```bash
kubectl get ingressclass higress
# NAME      CONTROLLER                      PARAMETERS   AGE
# higress   higress.io/higress-controller   <none>       3m46s
```

Для настройки Higress обратитесь к [документации Higress](https://higress.cn/en/docs/latest/ops/deploy-by-helm).

## Доступ к GPUStack { #accessing-gpustack }

Подождите, пока серверный модуль будет готов:

```bash
kubectl get pods -n gpustack-system -w
```

Получите первоначальный пароль администратора:

```bash
kubectl exec -it -n gpustack-system gpustack-server-0 -- \
  cat /var/lib/gpustack/initial_admin_password
```

Если вы не установили `server.ingress.hostname`, получите адрес пользовательского интерфейса GPUStack из входа:

```bash
kubectl get ingress -n gpustack-system gpustack \
  -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
```

Откройте адрес в браузере и войдите в систему, используя имя пользователя `admin` и пароль, полученные выше.

## Общая конфигурация { #common-configuration }

### Включение рабочих демонсетов { #enabling-worker-daemonsets }

Worker DaemonSet по умолчанию отключены (`worker.enabled=false`). При включении чарт создаёт DaemonSet для CPU-воркеров (`<release>-worker`) и по одному DaemonSet для каждого производителя GPU из `worker.gpuVendors` (с именем `<release>-worker-<vendor>`).

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set 'worker.gpuVendors={nvidia}'
```

Поддерживаемые значения `worker.gpuVendors`: `nvidia`, `mthreads`, `amd`, `ascend`, `hygon`, `metax`, `iluvatar`, `cambricon`, `thead`.

!!! note

    Каждый набор демонов графического процессора получает автоматическую метку присутствия PCI `nodeSelector` (например, `feature.node.kubernetes.io/pci-10de.present: "true"` для NVIDIA), рекламируемую Node Feature Discovery. **NFD должен быть установлен** в кластере, иначе ни один узел не будет нести необходимые метки и все рабочие модули останутся `Pending`.

Всякий раз, когда в списке указан хотя бы один поставщик графического процессора (т. е. `worker.gpuVendors` не пусто), каждый рабочий модуль дополнительно получает обязательный `podAntiAffinity` (topologyKey=hostname), поэтому два рабочих не могут совместно использовать узел — это защищает порты `hostNetwork: true` от коллизий.

Рабочий DaemonSet ЦП охватывает узлы, на которые не претендует ни один поставщик графического процессора, через `nodeSelector` `feature.gpustack.ai/acceleratable: "false"`. Установите `worker.cpuEnabled=false`, чтобы оставить эти узлы в покое — обычно, когда плоскость управления разделяет кластер с узлами графического процессора и не должна привлекать рабочих:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set worker.cpuEnabled=false \
  --set 'worker.gpuVendors={nvidia}'
```

`worker.gpuVendors` должен содержать хотя бы одного поддерживаемого производителя; иначе чарт не создаст ни одного DaemonSet для воркеров и установка будет отклонена. GPU DaemonSet в любом случае сохраняют суффикс `-<vendor>`, поэтому отключение CPU DaemonSet никогда не переименовывает GPU DaemonSet в `<release>-worker`.

Альтернативно, добавьте кластеры графических процессоров и рабочие узлы через пользовательский интерфейс на страницах **Кластеры** и **Рабочие** после установки.

### Использование внешней базы данных { #using-an-external-database }

По умолчанию GPUStack использует встроенную базу данных PostgreSQL. Чтобы использовать внешнюю базу данных PostgreSQL или MySQL, установите `server.externalDatabaseURL`:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set server.externalDatabaseURL="postgresql://user:password@host:port/dbname"
```

### Включение HTTPS с помощью специального сертификата. { #enabling-https-with-a-custom-certificate }

Предоставьте сертификат и содержимое ключа (PEM) через `server.ingress.tls`. Когда оба установлены, входная схема становится HTTPS:

```yaml
# values.yaml
server:
  ingress:
    hostname: gpustack.example.com
    tls:
      cert: |-
        -----BEGIN CERTIFICATE-----
        MIID...
        -----END CERTIFICATE-----
      key: |-
        -----BEGIN PRIVATE KEY-----
        MIIE...
        -----END PRIVATE KEY-----
```

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  -f values.yaml
```

### Получение образов из частного реестра { #pulling-images-from-a-private-registry }

Чтобы загружать все образы (сервера/воркера GPUStack, higress-plugins и входящих в комплект компонентов higress-core: gateway/controller/pilot) из зеркального частного реестра, переопределите `global.hub`. Благодаря распространению глобальных значений Helm одна настройка применяется ко всем образам:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set global.hub=myregistry.example.com
```

`global.hub` также передается на сервер как `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY`, гарантируя, что образы механизма вывода (например, vLLM, llama.cpp) извлекаются из одного и того же реестра.

Чтобы предоставить учетные данные извлечения, чарт может создать секрет `docker-registry` с именем `gpustack-image-pull-secret` и подключить его ко всем модулям:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set imagePullSecret.credentials.registry=registry.example.com \
  --set imagePullSecret.credentials.username=myuser \
  --set imagePullSecret.credentials.password=mypassword
```

Чтобы вместо этого ссылаться на свои собственные ранее существовавшие секреты, замените `global.imagePullSecrets`:

```yaml
global:
  imagePullSecrets:
    - name: my-existing-secret
```

## Параметры чарта { #chart-parameters }

Ниже перечислены наиболее часто используемые параметры. Полный и авторитетный список см. в [`values.yaml`](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/values.yaml) и [README](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/README.md) таблицы.

| Параметр | По умолчанию | Описание |
| -------------------------------------- | ------------------------ | -------------------------------------------------------------------------- |
| `debug`                                | `false`                  | Включить режим отладки. |
| `registrationToken`                    | `null`                   | Регистрационный жетон работника; случайный генерируется и используется повторно, если `null`. |
| `clusterDomain`                        | `cluster.local`          | Kubernetes суффикс домена службы кластера. |
| `global.hub`                           | `docker.io`              | Хост реестра контейнеров; переопределить для частного реестра. |
| `global.imagePullSecrets`              | `[gpustack-image-pull-secret]` | Секреты извлечения образов, прикрепленные ко всем модулям и распространяемые на субчарты. |
| `global.nodeSelector`                  | `{}`                     | NodeSelector по умолчанию для каждого компонента; заменено значением на уровне компонента. |
| `image.repository`                     | `gpustack/gpustack`      | Репозиторий образов с пространством имен; итоговая ссылка — `{global.hub}/{repository}:{tag}`. |
| `image.tag`                            | `null`                   | Тег образа; по умолчанию используется значение `appVersion` чарта. |
| `image.pullPolicy`                     | `IfNotPresent`           | Политика получения образов. |
| `imagePullSecret.credentials.registry` | `docker.io`              | Хост реестра, используемый, когда чарт создает секрет извлечения. |
| `imagePullSecret.credentials.username` | `null`                   | Имя пользователя реестра; создает секрет при установке пароля. |
| `imagePullSecret.credentials.password` | `null`                   | Пароль реестра; создает секрет, если он установлен с именем пользователя. |
| `server.ingress.hostname`              | `null`                   | Входное имя хоста для сервера. |
| `server.ingress.tls.cert`              | `null`                   | Входной сертификат TLS (PEM); включает HTTPS при установке с помощью ключа. |
| `server.ingress.tls.key`               | `null`                   | Входной закрытый ключ TLS (PEM). |
| `server.externalDatabaseURL`           | `null`                   | Строка подключения к внешней базе данных (PostgreSQL или MySQL). |
| `server.dataVolume.hostPath`           | `null`                   | Путь к хосту для тома данных сервера; использует HostPath вместо PVC. |
| `server.dataVolume.size`               | `100Gi`                  | Размер тома данных сервера (PVC). |
| `server.apiPort`                       | `30080`                  | Порт службы API. |
| `server.metricsPort`                   | `10161`                  | Порт метрик сервера. |
| `server.environmentConfig`             | `{}`                     | Дополнительные переменные среды для сервера GPUStack. |
| `server.nodeSelector`                  | `{}`                     | Серверный модуль nodeSelector; заменяет `global.nodeSelector`, если не пусто. |
| `gateway.ingressClassname`             | `higress`                | Higress Имя IngressClass; при обнаружении включает режим шлюза внутри кластера. |
| `higress-core.enabled`                 | `true`                   | Разверните входящий в комплект шлюз Higress; отключить, если он уже установлен. |
| `worker.enabled`                       | `false`                  | Создавать манифесты DaemonSet для воркеров. |
| `worker.gpuVendors`                    | `[nvidia]`               | поставщики графических процессоров; один набор DaemonSet от каждого поставщика плюс набор демонов ЦП. |
| `worker.cpuEnabled`                    | `true`                   | Создавать манифест DaemonSet для CPU-воркеров; `false` требуется поставщик графического процессора. |
| `worker.nodeSelector`                  | `{}`                     | Базовый nodeSelector воркеров; заменяет `global.nodeSelector`, если не пусто. |
| `worker.port`                          | `10150`                  | Рабочий порт обслуживания. |
| `worker.metricsPort`                   | `10151`                  | Порт рабочих метрик. |
| `worker.dataDir`                       | `/var/lib/gpustack`      | Путь к хосту установлен в точке `/var/lib/gpustack` внутри каждого рабочего модуля. |
| `worker.environmentConfig`             | `{}`                     | Дополнительные переменные среды для работника GPUStack. |

## Удаление { #uninstallation }

В выпуске развертывается больше, чем просто сервер: субчарт `gpustack-operator` содержит контроллеры Kueue и Node Feature Discovery, а оператор создает пользовательские ресурсы, содержащие финализаторы. `helm uninstall` удаляет контроллеры вместе со всем остальным, поэтому пользовательские ресурсы должны быть удалены **сначала** — удалены, пока контроллеры еще работают — иначе объекты останутся в `Terminating` навсегда, а пространство имен `gpustack-system` никогда не покинет его. По порядку:

1. Сбросьте рабочие нагрузки, пока **каждый** контроллер все еще работает. Оператор очищает финализаторы своих объектов `Instance`, а Kueue отказывается освобождать `ClusterQueue`, пока допущенные `Workload` все еще занимают его — поэтому они должны удалиться, прежде чем что-либо будет отключено. `Instance` находятся в пространстве имен и живут в пространствах имен арендаторов, поэтому перечислите их все с помощью `-A`. Удалить модели с сервера (или их экземпляры напрямую); модуль каждого экземпляра завершается вместе с ним, и рабочая нагрузка отправляется вместе с экземпляром. Не удаляйте поды пространства имен целиком — это уничтожит оператора, контроллеры Kueue и NFD, и именно они завершат очистку:

    ```bash
    kubectl delete instances.worker.gpustack.ai --all -A
    kubectl get instances.worker.gpustack.ai -A; kubectl get workloads.kueue.x-k8s.io -A   # both empty before moving on
    ```

    Если Kueue используется совместно с другими приложениями, не используйте `--all` здесь или на шаге 3 — удалите только объекты GPUStack. Все, что оператор создает в Kueue, имеет префикс `gpustack` (`clusterqueue/gpustack--…`, `admissioncheck/gpustack-node-devices`). Виды в области кластера выбираются только по имени; для пространств имен необходимо, чтобы пространство имен сохранялось, потому что `-o name` его опускает:

    ```bash
    kubectl get clusterqueues.kueue.x-k8s.io -o name | grep '/gpustack' \
      | xargs -I {} kubectl delete {}                # clusterqueues, resourceflavors, admissionchecks
    kubectl get localqueues.kueue.x-k8s.io -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | awk '$2 ~ /^gpustack/' | while read ns name; do \
        kubectl -n "$ns" delete localqueues.kueue.x-k8s.io "$name"; done
    ```

2. Остановите собственные контроллеры оператора, но оставьте версию установленной, чтобы контроллеры Kueue продолжали работать — именно они завершают удаление на следующем этапе. Пока `gpustack-operator-worker` и `gpustack-operator-device-manager-*` DaemonSets работают, они воссоздают `LocalQueue`, `ClusterQueue` и `gpustack-node-devices` AdmissionCheck, как только вы их удаляете, поэтому счетчики никогда не достигают нуля:

    ```bash
    kubectl -n gpustack-system scale deploy/gpustack-operator-worker --replicas=0
    kubectl -n gpustack-system get ds -l app.kubernetes.io/name=gpustack-operator-device-manager -o name | xargs -I {} \
      kubectl -n gpustack-system patch {} -p '{"spec":{"template":{"spec":{"nodeSelector":{"gpustack.ai/uninstall-pause":"true"}}}}}'
    ```

3. Удалите ресурсы Kueue в порядке зависимости — `LocalQueue`s, затем `ClusterQueue`s, затем `AdmissionCheck`s, затем `ResourceFlavor`s. Kueue удерживает финализатор `kueue.x-k8s.io/resource-in-use` для объекта, пользователи которого все еще присутствуют: AdmissionCheck `gpustack-node-devices` остается в состоянии удаления до тех пор, пока не исчезнут все ClusterQueue, и каждый ClusterQueue до тех пор, пока не исчезнут его LocalQueues и допущенные рабочие нагрузки. Подождите, пока каждый вид опустеет, прежде чем переходить к следующему. В общем кластере примените к каждой команде префикс `gpustack` из шага 1 вместо `--all`:

    ```bash
    kubectl delete localqueues.kueue.x-k8s.io --all -A
    kubectl delete clusterqueues.kueue.x-k8s.io --all
    kubectl delete admissionchecks.kueue.x-k8s.io --all
    kubectl delete resourceflavors.kueue.x-k8s.io --all
    ```

4. Удалите релиз. Не передавайте `--wait`: хук очистки Node Feature Discovery после удаления блокирует ожидание завершения дольше, чем задано в `--timeout`. Если принудительно завершить Helm, созданные для очистки ресурсы RBAC останутся без владельца (на шаге 5 они удаляются вручную — выполните шаг 5 и в случае, когда `--wait` не использовался):

    ```bash
    helm uninstall gpustack -n gpustack-system
    ```

5. Удалите то, что не удаляет `helm uninstall`. Helm никогда не удаляет CRD, а оператор регистрирует admission webhooks и APIServices, не принадлежащие ни одному релизу. После исчезновения Service оператора оба механизма блокируют запросы при ошибке обращения к нему (fail closed), а webhooks не позволяют снять финализаторы приведённой ниже командой, пока сами не будут удалены:

    ```bash
    kubectl delete mutatingwebhookconfiguration gpustack-worker-mutation --ignore-not-found
    kubectl delete validatingwebhookconfiguration gpustack-worker-validation --ignore-not-found
    kubectl delete apiservice v1.gpustack.ai v1.worker.gpustack.ai --ignore-not-found
    kubectl get instances.worker.gpustack.ai -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | while read ns name; do [ -n "$name" ] && kubectl -n "$ns" patch instances.worker.gpustack.ai "$name" \
        --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null; done
    kubectl get instancetypes.worker.gpustack.ai -o name | xargs -I {} \
      kubectl patch {} --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null
    kubectl delete crd devices.worker.gpustack.ai instances.worker.gpustack.ai \
      instancetypes.worker.gpustack.ai --ignore-not-found
    kubectl delete clusterrole node-feature-discovery-prune --ignore-not-found
    kubectl delete clusterrolebinding node-feature-discovery-prune --ignore-not-found
    ```

    Группа `worker.gpustack.ai` всегда принадлежит GPUStack — ни один другой компонент ее не использует — поэтому ее CRD можно безопасно удалить без каких-либо условий.

6. Удалите CRD, оставшиеся после выпуска — **только те, которые установлены в этом выпуске**. При удалении CRD удаляются все объекты такого типа во всем кластере, поэтому, если в кластере уже выполнялись Kueue, Node Feature Discovery или Higress до GPUStack (см. `higress-core.enabled` выше) и что-то еще все еще использует их, оставьте их CRD на месте и пропустите этот шаг для этих групп. Узнайте, что есть в этом выпуске, а не гадайте:

    ```bash
    kubectl get crd -o json | jq -r '.items[]
      | select(((.metadata.annotations // {})["meta.helm.sh/release-name"] // "") == "gpustack")
      | .metadata.name'
    ```

    В стандартной установке, в которой указаны CRD Kueue, а также NFD и Higress, если они установлены в таблице:

    ```bash
    kubectl get crd -o name | grep 'kueue\.x-k8s\.io' | xargs -I {} kubectl delete {} --ignore-not-found
    kubectl delete crd nodefeatures.nfd.k8s-sigs.io nodefeaturerules.nfd.k8s-sigs.io \
      nodefeaturegroups.nfd.k8s-sigs.io --ignore-not-found
    kubectl delete crd envoyfilters.networking.istio.io http2rpcs.networking.higress.io \
      mcpbridges.networking.higress.io wasmplugins.extensions.higress.io --ignore-not-found
    ```

    Собственные CRD провайдера кластера никогда не являются объектами GPUStack для удаления — например, `*.k3s.cattle.io` и `helm.cattle.io` в кластере k3s принадлежат кластеру.

7. *(Необязательно)* Удалите пространство имен, которое включает в себя объекты Higress, загрузочную карту ConfigMap, ServiceAccount и регистрационный токен Secret. Знайте, что с этим связано: пространство имен содержит PVC сервера (`gpustack-data-dir-gpustack-server-0`, `<volumeClaimTemplate>-<statefulset>-<ordinal>`), в котором хранятся постоянные данные сервера — встроенная база данных PostgreSQL с ее пользователями, моделями и кластерами. В политике возврата `Delete` (по умолчанию `local-path`) базовый том также удаляется. Сохраните пространство имен, если вам нужны данные для переустановки; в противном случае удалите пространство имен (и сначала PVC, если политика восстановления еще не удалила его):

    ```bash
    kubectl delete pvc gpustack-data-dir-gpustack-server-0 -n gpustack-system
    kubectl delete namespace gpustack-system
    ```
