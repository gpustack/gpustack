---
translation_status: translated
---

# Устранение неполадок { #troubleshooting }

## Просмотр журналов GPUStack { #view-gpustack-logs }

Вы можете просмотреть журналы GPUStack с помощью следующих команд для настроек по умолчанию:

```bash
docker logs -f gpustack
```

## Включить режим отладки. { #enable-debug-mode }

Вы можете включить режим `DEBUG`, установив флаг `--debug` при запуске GPUStack:

```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --debug
    ...
```

Вы также можете включить режим отладки GPUStack во время выполнения, выполнив следующую команду внутри контейнера **server**:

```bash
gpustack reload-config --set debug=true
```

## Настройка уровня журнала. { #configure-log-level }

Конечные точки уровня журнала требуют аутентификации.

Вы можете настроить уровень журнала сервера GPUStack во время выполнения, выполнив следующую команду внутри контейнера **server**. Аутентификация с помощью ключа API администратора:

```bash
curl -X PUT http://localhost/debug/log_level \
    -H "Authorization: Bearer <YOUR_API_KEY>" \
    -d "debug"
```

То же самое относится и к работникам GPUStack. Аутентификация с помощью локального токена работника (по умолчанию `/var/lib/gpustack/worker_token`):

```bash
curl -X PUT http://localhost:10150/debug/log_level \
    -H "Authorization: Bearer $(cat /var/lib/gpustack/worker_token)" \
    -d "debug"
```

Доступные уровни журнала: `trace`, `debug`, `info`, `warning`, `error`, `critical`.

## Сбросить пароль администратора { #reset-admin-password }

Если вы забыли пароль администратора, сбросьте его, выполнив следующую команду внутри **серверного контейнера**:

```bash
gpustack reset-admin-password
```

Если вы изменили порт по умолчанию, используя `--port` при запуске GPUStack, укажите URL-адрес GPUStack, используя параметр `--server-url`. Он должен быть запущен локально на сервере и доступен через `localhost`:

```bash
gpustack reset-admin-password --server-url http://localhost:9090
```

## Помощь в диагностике обнаружения ускорителей { #assist-in-accelerators-detection-diagnosis }

После успешного развертывания GPUStack Worker, как описано в [руководстве по установке](./installation/requirements.md),
если Работнику не удается обнаружить какие-либо устройства,
, войдите в соответствующий рабочий контейнер, выполните следующую команду и сообщите о результатах в [GPUStack](https://github.com/gpustack/gpustack/issues).

```bash
time GPUSTACK_RUNTIME_LOG_LEVEL=debug GPUSTACK_RUNTIME_LOG_EXCEPTION=1 gpustack-runtime detect --format json
```

## Помощь в диагностике развертывания модели { #assist-in-model-deployment-diagnosis }

Если у вас возникли проблемы после развертывания модели, войдите в соответствующий рабочий контейнер, выполните следующую команду и сообщите о результатах в [GPUStack](https://github.com/gpustack/gpustack/issues).

```bash
gpustack-runtime inspect <model instance name>
```
