---
translation_status: translated
---

# Geliştirme Rehberi { #development-guide }

## Önkoşullar { #prerequisites }

1. Python'u yükleyin (sürüm 3.11 - 3.12).
2. Bir veritabanı çalıştırın (PostgreSQL veya MySQL).

Örneğin, Docker'ı kullanarak PostgreSQL veritabanını çalıştırmak için:
```bash
docker run --name my-postgres -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d --restart=always postgres
```

## Ortamı Ayarlama { #set-up-environment }

```bash
make install
```

## Çalıştırma { #run }

`disabled` ağ geçidi modunu kullanın ve geliştirme için veritabanınıza bağlanın:

```bash
uv run gpustack start --database-url postgresql://postgres:mysecretpassword@localhost:5432/postgres --gateway-mode disabled --api-port 80
```

## Derleme { #build }

```bash
make build
```

Ve `dist`'deki eserleri kontrol edin.

## Testi { #test }

```bash
make test
```

## Paket { #package }

Konteyner imajını oluşturmak için [Buildx](https://docs.docker.com/build/install-buildx/) eklentisine sahip Docker gerekir:

```bash
make package
```

`docker buildx version` başarısız olursa, eklentiyi paket yöneticisiyle yükleyin (`apt-get install docker-buildx-plugin` veya `yum install docker-buildx-plugin`) veya sürüm ikili dosyasını yerine bırakın:

```bash
mkdir -p /usr/local/lib/docker/cli-plugins
VER=$(curl -fsSL https://api.github.com/repos/docker/buildx/releases/latest | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')
ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
curl -fsSL "https://github.com/docker/buildx/releases/download/${VER}/buildx-${VER}.linux-${ARCH}" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx
docker buildx version
```

Kimliği doğrulanmamış GitHub API, IP başına saatte 60 isteğe izin verir. Hız sizi sınırlıyorsa, `VER` boş kalır ve indirme 404'ler — [sürümler sayfasından](https://github.com/docker/buildx/releases) bir sürüm seçin ve elle ayarlayın.

Eklentiyi `sudo` ile paketleme yaparken `~/.docker/cli-plugins` yerine `/usr/local/lib/docker/cli-plugins` altına yükleyin, aksi takdirde değişen `HOME` onu Docker CLI'den gizler.

## Bağımlılıkları Güncelle { #update-dependencies }

```bash
uv add <something>
```

Veya

```bash
uv add --dev <something>
```

Geliştirme/test bağımlılıkları için.
