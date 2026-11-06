---
translation_status: translated
---

# GPUStack'e katkıda bulunmak { #contributing-to-gpustack }

GPUStack'e katkıda bulunmaya zaman ayırdığınız için teşekkür ederiz!

Lütfen [Davranış Kuralları](./code-of-conduct.md)'nı inceleyin ve uygulayın.

## Sorun Bildirme { #filing-issues }

Herhangi bir hata bulursanız veya herhangi bir sorunla karşılaşırsanız, lütfen bildirilen sorunu arayın, çünkü birisi aynı sorunu yaşamış olabilir veya aktif olarak bir çözüm üzerinde çalışıyoruz.

Sorununuzla ilgili herhangi bir şey bulamazsanız sorun bildirerek bizimle iletişime geçin. Tanılamamıza ve çözümlememize yardımcı olmak için lütfen aşağıdakiler de dahil olmak üzere mümkün olduğunca fazla bilgi ekleyin:

- Yazılım: GPUStack sürümü, kurulum yöntemi, işletim sistemi bilgisi vb.
- Donanım: Düğüm bilgisi, GPU bilgisi vb.
- Yeniden oluşturma adımları: Bildirilen duruma nasıl girdiğiniz konusunda mümkün olduğunca fazla ayrıntı verin.
- Günlükler: Lütfen sunucu günlükleri, çalışan günlükleri vb. gibi ilgili günlükleri ekleyin.

## Kod Katkısı { #contributing-code }

Geliştirme ortamını ayarlamak için lütfen [Geliştirme Kılavuzu](./development.md)'e bakın.

Küçük bir sorunu çözüyorsanız PR gönderebilirsiniz. Ancak, yeni bir özelliği uygulamak veya nispeten karmaşık bir hatayı düzeltmek için daha büyük bir PR göndermeyi planlıyorsanız lütfen değişikliği ve bunun nedenini açıklayan bir konu açın. Bir hatayı ele alıyorsanız, lütfen onu nasıl yeniden oluşturacağınızı açıklayın.

## Belgeleri Güncelleme { #updating-documentation }

Belgelerimizde herhangi bir güncelleme varsa, `documentation` etiketiyle bir sorun bildirmekten veya çekme talebinde bulunmaktan çekinmeyin.

`docs/`'deki İngilizce belgeler gerçeğin kaynağıdır. `docs-i18n/zh/`, `docs-i18n/ja/`, `docs-i18n/ru/` ve `docs-i18n/tr/` altındaki eşleşen yollarda Basitleştirilmiş Çince, Japonca, Rusça ve Türkçe çevirileri koruyun. Resimler ve diğer paylaşılan varlıklar `docs/`'den devralınmıştır. Bağlantıların ve dil değiştirmenin tutarlı kalması için çevrilmiş başlıklardaki İngilizce başlık kimliklerini koruyun (örneğin, `## 安装 { #installation }`).

İlgili `docs-i18n/mkdocs/zh.yml`, `docs-i18n/mkdocs/ja.yml`, `docs-i18n/mkdocs/ru.yml` ve `docs-i18n/mkdocs/tr.yml` dosyalarındaki gezinme ve tema etiketlerini çevirin. İngilizce'den çeviri yaparken komutları, yapılandırma anahtarlarını, kod örneklerini, bağlantıları ve model adlarını koruyun. Çevirisi olmayan veya ön kısmında `translation_status: pending` bulunan sayfalar İngilizce kaynağı bir bildirimle gösterir.

Beş dilin tümünü oluşturmak için `make build-docs` komutunu çalıştırın. Beş dili de canlı yeniden yüklemeyle önizlemek için `mkdocs serve` veya `make serve-docs` kullanın; `docs-i18n/` altındaki çeviriler ve dil yapılandırmaları değiştiğinde önizleme otomatik olarak yeniden oluşturulur. İngilizce `/`'de servis edilir; çevrilmiş siteler `/zh/`, `/ja/`, `/ru/` ve `/tr/`'yi kullanır. `bash hack/build-docs.sh --offline`, `gpustack/help/`'de GPUStack tarafından `/help/`'de ve karşılık gelen `/help/<language>/` yollarda sunulan paket siteyi oluşturur. Her dilin kendi yerel arama dizini vardır. Japonca arama, dokümantasyon temasıyla birlikte verilen belirteçleri kullanır ve çevrimdışı çalışır.

Sıradan `mkdocs build` ve `mike deploy` ayrıca tüm dilleri yapılandırılmış belge kancaları aracılığıyla oluşturur. Sürümlendirilmiş dağıtımlar, çevirileri aynı sürüm dizininde tutar; örneğin `/latest/ja/` veya `/2.3/ja/`. Oluşturulan sitenin tamamını yayınlamak için mevcut yayın sürecini kullanın; tek başına bir yapı belgeleri yayınlamaz.
