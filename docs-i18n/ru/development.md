---
translation_status: translated
---

# Руководство по разработке { #development-guide }

## Предварительные условия { #prerequisites }

1. Установите Python (версии с 3.11 по 3.12).
2. Запустите базу данных (PostgreSQL или MySQL).

Например, чтобы запустить базу данных PostgreSQL с помощью Docker:
```bash
docker run --name my-postgres -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d --restart=always postgres
```

## Настройка среды { #set-up-environment }

```bash
make install
```

## Запуск { #run }

Используйте режим шлюза `disabled` и подключитесь к своей базе данных для разработки:

```bash
uv run gpustack start --database-url postgresql://postgres:mysecretpassword@localhost:5432/postgres --gateway-mode disabled --api-port 80
```

## Сборка { #build }

```bash
make build
```

И проверьте артефакты в `dist`.

## Тест { #test }

```bash
make test
```

## Пакет { #package }

Для создания образа контейнера требуется Docker с плагином [Buildx](https://docs.docker.com/build/install-buildx/):

```bash
make package
```

Если `docker buildx version` завершается ошибкой, установите плагин через менеджер пакетов (`apt-get install docker-buildx-plugin` или `yum install docker-buildx-plugin`) либо поместите исполняемый файл выпуска в нужный каталог:

```bash
mkdir -p /usr/local/lib/docker/cli-plugins
VER=$(curl -fsSL https://api.github.com/repos/docker/buildx/releases/latest | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')
ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
curl -fsSL "https://github.com/docker/buildx/releases/download/${VER}/buildx-${VER}.linux-${ARCH}" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx
docker buildx version
```

API GitHub без аутентификации допускает 60 запросов в час на каждый IP-адрес. Если скорость ограничивает вас, `VER` оказывается пустым, а загрузка 404 — выберите версию на странице [релизов](https://github.com/docker/buildx/releases) и установите ее вручную.

Установите плагин под `/usr/local/lib/docker/cli-plugins`, а не под `~/.docker/cli-plugins` при упаковке с помощью `sudo`, иначе измененный `HOME` скроет его от Docker CLI.

## Обновление зависимостей { #update-dependencies }

```bash
uv add <something>
```

Или

```bash
uv add --dev <something>
```

Для зависимостей разработки/тестирования.
