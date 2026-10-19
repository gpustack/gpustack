---
translation_status: translated
---

# Добавление кластера графических процессоров с помощью DigitalOcean. { #adding-a-gpu-cluster-using-digitalocean }

При создании кластера GPUStack может использовать DigitalOcean для создания рабочих процессов и добавления их в кластер GPUStack.

## Подготовка { #preparation }

Вам необходимо зарегистрировать учетную запись DigitalOcean и создать токен личного доступа на странице API.

![цифровой-океан-пат](../assets/tutorials/adding-gpucluster-using-digitalocean/digital-ocean-pat.png)

> Примечание. Область токена должна быть установлена ​​на «Полный доступ». Если вы выбираете разрешения с помощью пользовательских областей, вы можете столкнуться с проблемами при удалении Droplet.

При запуске сервера GPUStack необходимо указать параметр `--server-external-url`. Этот параметр используется для настройки `--server-url` работника после создания Droplet и запуска работника. Если ваш сервер работает через прокси-сервер, установите адрес прокси-сервера, чтобы гарантировать, что Droplet, работающие в общедоступной сети, смогут получить доступ к API сервера GPUStack, используя этот адрес после запуска.

## Создать кластер DigitalOcean { #create-digitalocean-cluster }

См. шаги в разделе [Создание кластера DigitalOcean](../user-guide/cluster-management.md#creating-digitalocean-cluster).

### Создать облачные учетные данные { #create-cloud-credential }

Создайте облачные учетные данные DigitalOcean на странице `Cloud Credentials`.

![создать облако](../assets/tutorials/adding-gpucluster-using-digitalocean/create-cloud-credential.png)

### Создание кластера с использованием облачных учетных данных. { #create-cluster-with-cloud-credential }

На странице создания кластера выберите `DigitalOcean` в качестве `Cloud Provider`:

![облачный-провайдер](../assets/tutorials/adding-gpucluster-using-digitalocean/select-cloud-provider.png)

Введите имя, выберите только что созданные облачные учетные данные и выберите регион, поддерживающий Droplet графического процессора.

![выбрать-регион](../assets/tutorials/adding-gpucluster-using-digitalocean/select-region.png)

Нажмите `Next`, чтобы создать рабочий пул для кластера.

### Создать пул рабочих { #create-worker-pool }

Заполните `Name`, `Replicas` и `Batch Size` по мере необходимости. Затем выберите `Instance Type` (размер Droplet в терминах DigitalOcean) для рабочего пула.

![выберите тип-экземпляра](../assets/tutorials/adding-gpucluster-using-digitalocean/select-instance-type.png)

Далее выберите `OS Image`.

![select-os-image](../assets/tutorials/adding-gpucluster-using-digitalocean/select-os-image.png)

В настоящее время поддерживаются только Droplet графического процессора серии Nvidia, поскольку Droplet графического процессора AMD часто недоступны. Поддерживаются только операционные системы на базе Debian.

- Для Nvidia AI/ML Ready (**Рекомендуется**), которая основана на Ubuntu 22.04 и включает в себя предустановленные драйверы, CUDA и набор инструментов для контейнеров, дополнительные пакеты не требуются.
- Для дистрибутивов Debian Droplet загружается с помощью драйвера `nvidia-open`, CUDA 12.8 и nvidia-container-toolkit 1.17.8-1 через Cloud-init.
- Для дистрибутивов Ubuntu Droplet загружается с помощью драйвера `nvidia-driver-570`, CUDA 12.8 и nvidia-container-toolkit 1.17.8-1 через Cloud-init.

Поддерживаются метки и тома, которые можно установить в пуле рабочих для подготовки рабочего. Тома, созданные и подключенные для Droplet, не будут автоматически монтироваться в рабочий контейнер. Вы можете изменить рабочий сценарий запуска в `/opt/gpustack-run-worker.sh`, чтобы при необходимости монтировать тома.

Нажмите `Save`, если все настроено.

## Ожидание предоставления рабочих { #waiting-for-workers-to-be-provisioned }

После сохранения кластера перейдите на страницу `Workers`, чтобы просмотреть ход подготовки рабочих процессов DigitalOcean.

![провизор](../assets/tutorials/adding-gpucluster-using-digitalocean/provisioning-workers.png)

Процесс подготовки включает в себя несколько этапов:

1. Создайте ключ SSH.
2. Создайте Droplet с помощью ключа SSH.
3. Подождите, пока Droplet запустится.
4. Подождите, пока будет присвоен общедоступный IP-адрес.
5. Создайте объёмы и прикрепите их к Droplet.
6. Рабочий переходит в статус `Initialized` и ждет, пока рабочий контейнер запустится и подключится к серверу.

Как только рабочий достигнет статуса `Ready`, на нем можно будет развернуть модели.

![предоставленный работником](../assets/tutorials/adding-gpucluster-using-digitalocean/worker-provisioned.png)

## Безопасное уменьшение числа воркеров DigitalOcean { #safely-scale-down-digitalocean-workers }

Если работник DigitalOcean больше не нужен, выполните следующие действия, чтобы безопасно уничтожить Droplet:

- Отрегулируйте количество реплик рабочего пула, чтобы оно соответствовало количеству рабочих, которое вы хотите удалить. Обратите внимание, что работники не удаляются автоматически на этом этапе.
- Убедитесь, что ни один экземпляр модели не развернут на рабочих процессах, которые вы собираетесь удалить.
- Удалите рабочих при необходимости. Соответствующие Droplet будут соответственно удалены.
