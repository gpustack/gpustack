---
translation_status: translated
---

# Запуск DeepSeek R1 671B с распределенным Ascend MindIE { #running-deepseek-r1-671b-with-distributed-ascend-mindie }

В этом руководстве описан процесс настройки и запуска исходной версии **DeepSeek R1 671B** с использованием **Distributed Ascend MindIE** в кластере GPUStack.

Из-за чрезвычайно большого размера модели обычно требуется распределенный вывод по нескольким исполнителям.

GPUStack обеспечивает простую настройку и оркестрацию распределенного вывода с помощью Ascend MindIE, что позволяет запускать такие массивные модели, как DeepSeek R1, с минимальной ручной настройкой.

## Предварительные условия { #prerequisites }

Прежде чем начать, убедитесь, что выполнены следующие требования:

- У вас есть доступ к достаточному количеству узлов Linux, каждый из которых оснащен необходимыми процессорами искусственного интеллекта Ascend Atlas. Например:

<div class="center-table" markdown>

| **Сервер(NPU)** | **Количество узлов** |
| ------------------------- | ------------------- |
| Atlas 800T A2 (910B3 x 8) | 4                   |

</div>

- Все узлы должны поддерживать адаптивный протокол коллективной связи Huawei (HCCP), иметь установленную библиотеку коллективной связи Huawei (HCCL) и включенную систему когерентности кэша Huawei (HCCS).
- Файлы модели следует загружать по одному и тому же пути на каждом узле. Хотя GPUStack поддерживает загрузку модели «на лету», рекомендуется выполнить предварительную загрузку, поскольку это может занять много времени в зависимости от скорости сети.

!!! note

    - В этом руководстве мы предполагаем настройку из 4 узлов, каждый из которых оснащен 8 NPU 910B3 и подключен к сети 200G Huawei Cache Conherence Network (HCCN).
    - NPU Atlas не поддерживают точность FP8, первоначально использовавшуюся в DeepSeek R1. Следовательно, мы используем версию BF16 из [Unsloth](https://huggingface.co/unsloth/DeepSeek-R1-BF16).

## Шаг 1. Установите сервер GPUStack. { #step-1-install-gpustack-server }

Согласно [Установка](../installation/installation.md), вы можете использовать следующую команду для запуска сервера GPUStack:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    --volume /path/to/your/model:/path/to/your/model \
    gpustack/gpustack
```

!!! note

    - Замените `/path/to/your/model` фактическим путем в вашей системе, где хранятся файлы модели DeepSeek R1.

После запуска сервера GPUStack выполните следующие команды, чтобы получить первоначальный пароль администратора:

```bash
sudo docker exec gpustack \
    cat /var/lib/gpustack/initial_admin_password

```

## Шаг 2. Доступ к пользовательскому интерфейсу GPUStack. { #step-2-access-gpustack-ui }

Войдите в пользовательский интерфейс GPUStack, используя пользователя `admin` и полученный пароль.

```
http://your_gpustack_server_ip_or_hostname
```

## Шаг 3. Установите работники GPUStack. { #step-3-install-gpustack-workers }

Перейдите на страницу `Workers` в пользовательском интерфейсе GPUStack, нажмите кнопку `Add Worker`, чтобы получить команду для добавления рабочих процессов.

А затем на **каждом рабочем узле** запустите команду добавления рабочего, чтобы запустить рабочий GPUStack:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --env "ASCEND_VISIBLE_DEVICES=$(sudo ls /dev/davinci* | head -1 | grep -o '[0-9]\+' || echo "0")" \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --volume /path/to/your/model:/path/to/your/model \
    --runtime ascend \
    gpustack/gpustack \
    --server-url http://your_gpustack_server_ip_or_hostname \
    --token your_gpustack_cluster_token

```

!!! note

    - Замените соответственно пути-заполнители, IP-адрес/имя хоста и токен кластера.
    - Замените `/path/to/your/model` фактическим путем в вашей системе, где хранятся файлы модели DeepSeek R1.
    - Убедитесь, что `hccn_tool` правильно установлен и настроен в вашей системе. Это необходимо для обнаружения сетевой связи HCCN.

После добавления всех рабочих процессов вернитесь в пользовательский интерфейс GPUStack.

Перейдите на страницу `Workers` и убедитесь, что все рабочие процессы находятся в состоянии готовности и их графические процессоры указаны в списке.

![начальные ресурсы](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/initial-resources.png)

## Шаг 4. Развертывание модели DeepSeek R1. { #step-4-deploy-the-deepseek-r1-model }

1. Перейдите на страницу `Deployments`.
2. Нажмите `Deploy Model`.
3. Выберите `Local Path` в качестве источника.
4. Введите имя (например, `DeepSeek-R1`) в поле `Name`.
5. Укажите `Model Path` в качестве каталога, содержащего файлы модели DeepSeek R1 на каждом рабочем узле.
6. Убедитесь, что для `Backend` установлено значение `Ascend MindIE`.
7. Разверните `Advanced`, добавьте следующие параметры к `Backend Parameters`:
   - `--data-parallel-size=4`
   - `--tensor-parallel-size=8`
   - `--moe-tensor-parallel-size=1`
   - `--moe-expert-parallel-size=32`
   - `--npu-memory-fraction=0.95`, поскольку мы используем параллелизм данных, долю памяти следует установить равной 0,95, чтобы обеспечить эффективное использование памяти во всех NPU.
8. После прохождения проверки совместимости нажмите `Save` для развертывания.

![deploy-model-1](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/deploy-model-1.png)
![deploy-model-2](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/deploy-model-2.png)

## Шаг 5. Мониторинг развертывания { #step-5-monitor-deployment }

Вы можете отслеживать статус развертывания на странице `Deployments`. Наведите указатель мыши на `distributed across workers`, чтобы просмотреть использование графического процессора и рабочих процессов. Нажмите `View Logs`, чтобы просмотреть журналы в реальном времени, показывающие ход загрузки модели. Загрузка модели может занять несколько минут.

![информация о модели](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/model-info.png)

После запуска модели перейдите на страницу `Workers`, чтобы проверить загрузку графического процессора. Ascend MindIE использует 95% памяти NPU с указанными выше настройками.

![загружено ресурсами](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/resources-loaded.png)

## Шаг 6: Запустите вывод через Песочницу { #step-6-run-inference-via-playground }

После того как модель развернута и запущена, вы можете протестировать ее с помощью GPUStack Playground.

1. Перейдите к `Playground` -> `Chat`.
2. Если развернута только одна модель, она будет выбрана по умолчанию. В противном случае используйте раскрывающееся меню, чтобы выбрать `DeepSeek-R1`.
3. Вводите подсказки и взаимодействуйте с моделью.

![Песочница-чат](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/playground-chat.png)

Вы также можете использовать вкладку `Compare` для тестирования сценариев параллельного вывода.

![Песочница-сравнить](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/playground-compare.png)

Теперь вы успешно развернули и запустили DeepSeek R1 671B с помощью Distributed Ascend MindIE в кластере GPUStack. Исследуйте производительность и возможности модели в своих собственных приложениях.

Для получения дополнительной помощи обращайтесь в сообщество GPUStack или в службу поддержки.
