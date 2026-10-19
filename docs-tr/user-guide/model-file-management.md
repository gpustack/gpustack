---
translation_status: translated
---

# Model Dosya Yönetimi { #model-file-management }

GPUStack, yöneticilerin model dosyalarını indirmesine ve yönetmesine olanak tanır.

## Model Dosyası Ekle { #add-model-file }

GPUStack şu anda [Hugging Face](https://huggingface.co), [ModelScope](https://modelscope.cn) ve yerel yollardan modelleri desteklemektedir. Model dosyaları eklemek için `Model Files` sayfasına gidin.

### Hugging Face Modeli Ekleme { #add-a-hugging-face-model }

1. `Add Model File` düğmesine tıklayın ve açılır menüden `Hugging Face`'i seçin.
2. Bir modeli ada göre bulmak için sol üstteki arama çubuğunu kullanın, örneğin `Qwen/Qwen3-0.6B`.
3. _(İsteğe bağlı)_ GGUF modelleri için, `Available Files` arasından istediğiniz niceleme formatını seçin.
4. Model dosyasını indirmek için hedef çalışanı seçin.
5. _(İsteğe bağlı)_ Modeli GPUStack önbellek dizini yerine özel bir yola indirmek için bir `Local Directory` belirtin.
6. `Save` düğmesine tıklayın.

### ModelScope Modeli Ekleme { #add-a-modelscope-model }

1. `Add Model File` düğmesine tıklayın ve açılır menüden `ModelScope`'i seçin.
2. Bir modeli ada göre bulmak için sol üstteki arama çubuğunu kullanın, örneğin `Qwen/Qwen3-0.6B`.
3. _(İsteğe bağlı)_ GGUF modelleri için, `Available Files` arasından istediğiniz niceleme formatını seçin.
4. Model dosyasını indirmek için hedef çalışanı seçin.
5. _(İsteğe bağlı)_ Modeli GPUStack önbellek dizini yerine özel bir yola indirmek için bir `Local Directory` belirtin.
6. `Save` düğmesine tıklayın.

### Yerel Yol Modeli Ekleme { #add-a-local-path-model }

Yerel yoldan model ekleyebilirsiniz. Yol, çalışan üzerinde bulunan bir dizin (örneğin, Hugging Face modeli klasörü) veya bir dosya (örneğin, bir GGUF modeli) olabilir.

1. `Add Model File` düğmesine tıklayın ve açılır menüden `Local Path`'i seçin.
2. `Model Path`'ı girin.
3. Hedef çalışanı seçin.
4. `Save` düğmesine tıklayın.

## İndirmeyi Tekrar Dene { #retry-download }

Bir model dosyası indirme işlemi başarısız olursa veya çok düşük indirme hızında takılıp kalırsa, yeniden deneyebilirsiniz:

1. `Model Files` sayfasına gidin.
2. Model dosyasını bulun.
3. `Operations` sütunundaki üç nokta düğmesine tıklayın ve `Retry Download`'i seçin.
4. GPUStack, model dosyasını belirtilen kaynaktan tekrar indirmeyi deneyecektir.

## Modeli Dağıt { #deploy-model }

Modeller, model dosyalarından dağıtılabilir. Model belirli bir çalışan üzerinde depolandığından GPUStack, uygun planlamayı sağlamak için `worker-name` tuşunu kullanarak bir çalışan seçici ekleyecektir.

!!! tip

    Bir modelin düğümler arasında yük devretmesini istiyorsanız, kümedeki tüm düğümlerin model dosyalarına aynı yoldan erişebildiğinden emin olun ve `worker-name` etiketini çalışan seçiciden manuel olarak kaldırın.

1. `Model Files` sayfasına gidin.
2. Dağıtmak istediğiniz model dosyasını bulun.
3. `Operations` sütunundaki `Deploy` düğmesine tıklayın.
4. `Name`, `Backend`, `Backend Version`, `Replicas` ve diğer dağıtım parametrelerini inceleyin veya ayarlayın.
5. `Save` düğmesine tıklayın.

## Model Dosyasını Sil { #delete-model-file }

1. `Model Files` sayfasına gidin.
2. Silmek istediğiniz model dosyasını bulun.
3. `Operations` sütunundaki üç nokta düğmesine tıklayın ve `Delete`'i seçin.
4. _(İsteğe bağlı)_ `Also delete the file from disk` seçeneğini işaretleyin.
5. Onaylamak için `Delete` düğmesine tıklayın.
