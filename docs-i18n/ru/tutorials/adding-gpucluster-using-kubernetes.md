---
translation_status: translated
---

# Добавление кластера графических процессоров с помощью Kubernetes. { #adding-a-gpu-cluster-using-kubernetes }

GPUStack поддерживает добавление кластера Kubernetes в качестве кластера графического процессора.

## Подготовка { #preparation }

Кластер Kubernetes должен быть построен с установленным и настроенным классом времени выполнения на всех узлах. В этом уроке в качестве примера мы будем использовать кластер k3s.

## Создание кластера Kubernetes. { #create-kubernetes-cluster }

См. шаги в разделе [Регистрация кластера Kubernetes](../user-guide/cluster-management.md#register-kubernetes-cluster).

При создании кластера Kubernetes вы можете настроить следующие параметры:

![kubernetes-cluster-options](../assets/tutorials/adding-gpucluster-using-kubernetes/k8s-options.png)

Нажмите `Save`, чтобы создать кластер.

После создания кластера поставщиков `Kubernetes` выберите нужный вам `GPU Vendors`. Вы можете выбрать нескольких поставщиков или не выбрать ни одного.

![nvidia-производитель графического процессора](../assets/tutorials/adding-gpucluster-using-kubernetes/nvidia-gpu-vendor.png)

Нажмите `Next`, чтобы получить сценарий проверки среды, который проверяет готовность кластера к регистрации. Запустите скрипт на хосте, где установлен k3s.

![проверить среду](../assets/tutorials/adding-gpucluster-using-kubernetes/check-environment.png)

Имейте в виду, что ресурс `runtimeclass` в k8s — это просто запись данных и не означает, что среда выполнения контейнера правильно настроена на каждом узле. Убедитесь, что среда выполнения контейнера для указанного поставщика графического процессора настроена, иначе рабочий процесс не сможет запуститься.

Скопируйте сценарий из шага `Run Command` и запустите его на установленном хосте k3s.

![команда запуска](../assets/tutorials/adding-gpucluster-using-kubernetes/run-command.png)

![применить-манифесты](../assets/tutorials/adding-gpucluster-using-kubernetes/apply-manifests.png)

## Ожидание предоставления рабочих { #waiting-for-workers-to-be-provisioned }

После применения манифестов в кластере Kubernetes мы можем дождаться заполнения рабочих узлов на странице `Workers`.

![обеспечивающие работники](../assets/tutorials/adding-gpucluster-using-kubernetes/provisioned-workers.png)

Как только рабочий достигнет статуса `Ready`, на нем можно будет развернуть модели.
