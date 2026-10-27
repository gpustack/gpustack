---
translation_status: translated
---

# Docker aracılığıyla yükseltme { #upgrade-via-docker }

Docker tabanlı GPUStack kurulumunu, daha yeni bir imaj çekip kapları yeniden oluşturarak yükseltebilirsiniz.

Aşağıdaki yükseltme talimatları yalnızca GPUStack v2.0 ve üzeri için geçerlidir.

V0.7'den önceki kurulumlar için lütfen [geçiş kılavuzuna](../migration.md) bakın.

!!! note

    1. Yükseltme yaparken önce GPUStack sunucusunu yükseltin, ardından çalışanları yükseltin.

    2. main(dev) sürümünden veya sürüm adayı (rc) sürümünden kararlı sürüme ya da bu sürümlere **yükseltme yapmayın**; geriye dönük uyumluluğu bozan değişiklikler içerebilirler. main veya rc sürümlerini denemek için temiz bir kurulum kullanın.

!!! warning

    **Önce Yedekleme:** Yükseltme işlemine devam etmeden önce veritabanınızı yedeklemeniz önemle tavsiye edilir.

    Gömülü PostgreSQL veritabanını kullanan varsayılan kurulumlar için, GPUStack sunucusunu durdurun ve kapsayıcının içinde bulunan PostgreSQL veritabanı dizininin bir yedeğini şu adreste oluşturun:

    ```
    /var/lib/gpustack/postgresql/data
    ```

    Harici bir veritabanı kullanıyorsanız, bunun yerine veritabanı sağlayıcınızın yedekleme prosedürünü izleyin.

Önce **sunucuyu**, ardından **worker'ları** yükseltin. Sunucuyu yükseltmek için yeni imajı (belirli bir sürüm etiketi veya `latest` etiketi) çekin, eski konteyneri kaldırın ve güncellenmiş imajdan **aynı argümanlar ve birimlerle** yeni bir konteyner başlatın.

## Sunucuyu Yükseltme { #upgrade-the-server }

```bash
docker pull gpustack/gpustack:latest  # or: docker pull gpustack/gpustack:vx.y.z

docker stop gpustack
docker rm gpustack

docker run -d --name gpustack \
  ... \
  gpustack/gpustack:latest
  ...
```

## İşçileri Yükselt { #upgrade-the-workers }

Sunucu yeni sürümle çalışır hale geldikten sonra her kümedeki işçileri yükseltin. Docker, Kubernetes ve bulut kümesi prosedürleri için [Küme Dağıtımını Yükseltme](cluster.md) konusuna bakın.

## Docker Compose aracılığıyla yükseltme { #upgrade-via-docker-compose }

GPUStack klonlanmış bir depodan Docker Compose ile dağıtıldıysa yeni sürüm etiketine geçin ve konteynerleri güncellenmiş imajla yeniden oluşturun:

```bash
cd gpustack

# Fetch the tags and check out the target stable release
git fetch --tags
git checkout <new-tag>   # e.g. vx.y.z

cd docker-compose

# Pull the new image and recreate the containers
sudo docker compose -f docker-compose.server.yaml pull
sudo docker compose -f docker-compose.server.yaml up -d
```
