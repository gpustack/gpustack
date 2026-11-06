---
translation_status: translated
---

# CPU'larda Çıkarım Çalıştırmak { #running-inference-on-cpus }

GPUStack, CPU üzerinde çıkarımı destekler. GPU kaynakları sınırlı olduğunda veya model boyutu ayrılabilir GPU belleğini aştığında bu esneklik sağlar. Şu CPU çıkarım modları kullanılabilir:

- **Hibrit CPU+GPU Çıkarımı**: VRAM kapasitesi yetersiz olduğunda büyük modellerin bazı kısımlarını CPU'ya aktararak kısmi hızlandırmayı etkinleştirir.
- **Tam CPU Çıkarımı**: Hiçbir GPU kaynağı olmadığında tamamen CPU üzerinde çalışır.

!!! note

    Yalnızca özel arka uçlar için kullanılabilir.

    CPU boşaltma etkinleştirildiğinde, GPU kaynakları yetersizse GPUStack CPU belleğini tahsis edecektir. Çıkarım arka ucunu hibrit CPU+GPU veya tam CPU çıkarımını kullanacak şekilde doğru şekilde yapılandırmanız gerekir.

    CPU çıkarımının yalnızca CPU çalışanları üzerinde kullanılması şiddetle tavsiye edilir.

Örneğin, [Metin Gömme Çıkarımı](https://huggingface.co/docs/text-embeddings-inference/index) ile CPU çıkarımına sahip bir modeli dağıtmak için aşağıdaki yapılandırmayı izleyin:

Kaynak: `HuggingFace`

Repo Kimliği: `BAAI/bge-large-en-v1.5`

Arka uç: `Custom`

İmaj Adı: `ghcr.io/huggingface/text-embeddings-inference:cpu-1.8`

Yürütme Komutu: `--model-id BAAI/bge-large-en-v1.5 --huggingface-hub-cache /var/lib/gpustack/cache/huggingface --port {{port}}`

!!! note

    `TEI (Text Embeddings Inference)` yalnızca `HuggingFace`'den itibaren modellerin dağıtılmasını destekler.

    `ghcr.io/huggingface/text-embeddings-inference:cpu-1.8` TEI için CPU çıkarım imajıdır. Bakınız: [TEI Destekli Donanım](https://huggingface.co/docs/text-embeddings-inference/supported_models#supported-hardware).

    `--huggingface-hub-cache /var/lib/gpustack/cache/huggingface` TEI için HuggingFace Hub önbelleğinin konumunu GPUStack'in indirilen HuggingFace modellerini depoladığı yola ayarlar. Varsayılan yol `/var/lib/gpustack/cache/huggingface`'dir. Bakınız: [TEI CLI Argümanları](https://huggingface.co/docs/text-embeddings-inference/cli_arguments).

    `{{port}}` GPUStack tarafından otomatik olarak atanan bağlantı noktasını temsil eden bir yer tutucudur.

![TEI CPU Çıkarımı](../assets/tutorials/inference-on-cpus/tei-cpu-inference.png)

Ve Gelişmiş Ayarlar'da `Allow CPU Offloading`'ı kontrol edin:

![CPU Boşaltmasına İzin Ver](../assets/tutorials/inference-on-cpus/allow-cpu-offloading.png)

OpenAI uyumlu olmayan API'lere erişmeniz gerekiyorsa `Enable Generic Proxy`'ı da işaretleyebilirsiniz.

Daha fazla ayrıntı için bkz. [Genel Proxy'yi Etkinleştirme](../user-guide/model-deployment-management.md#enable-generic-proxy).
