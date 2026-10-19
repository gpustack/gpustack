---
translation_status: translated
---

# Görüntü Oluşturma Modellerini Kullanma { #using-image-generation-models }

GPUStack, **Görüntü Oluşturma Modellerinin** dağıtılmasını ve çalıştırılmasını destekler. Bu kılavuz, **Z-Image-Turbo** metinden görüntüye modelini dağıtma ve GPUStack kullanıcı arayüzündeki metin istemlerinden görüntüler oluşturma konusunda size yol gösterir.

## Önkoşullar { #prerequisites }

Başlamadan önce aşağıdakilere sahip olduğunuzdan emin olun:

- En az 24 GB VRAM'e sahip bir GPU, örneğin NVIDIA RTX 4090.
- GPUStack kurulu ve çalışıyor. Değilse, [Hızlı Başlangıç ​​Kılavuzu](../quickstart.md)'na bakın.
- Model dosyalarını indirmek için Hugging Face veya ModelScope'a erişim.

## Adım 1: Z-Image-Turbo Modelini Dağıtın { #step-1-deploy-the-z-image-turbo-model }

Modeli dağıtmak için şu adımları izleyin:

1. GPUStack kullanıcı arayüzünde `Model Catalog` sayfasına gidin.
2. Kategori filtresinde `Image` seçeneğini seçin.
3. Dağıtım sayfasını açmak için `Z-Image-Turbo` model kartına tıklayın.
4. Varsayılan ayarları koruyun ve dağıtmak için `Save` düğmesine tıklayın.

![Modeli Dağıt](../assets/using-models/using-image-generation-models/deploy-model.png)

Dağıtımdan sonra durumu `Deployments` sayfasından takip edebilirsiniz. Dağıtım başarılı olduktan sonra dağıtımın üç nokta simgesine tıklayın ve modeli Oyun Alanında kullanmaya başlamak için `Open in Playground` öğesini seçin.

![Model Listesi](../assets/using-models/using-image-generation-models/model-list.png)

## Adım 2: Modeli Görüntü Oluşturma için Kullanın { #step-2-use-the-model-for-image-generation }

`Image` oyun alanında:

1. `Size` açılır menüsünü seçin ve istediğiniz görüntü çözünürlüğünü seçin.
2. Oluşturmak istediğiniz görüntüyü açıklayan bir bilgi istemi girin. İlham almak için rastgele bir istem almak üzere `Random Prompt` düğmesine tıklayabilirsiniz.
3. Görüntüyü oluşturmak için `Submit` düğmesine tıklayın.

Oluşturulan görüntü kullanıcı arayüzünde gösterilir. Sonuçlar rastgeleliğe, rastgelelik tohumuna ve istemin ayrıntılarına göre değişir.

![Oluşturuldu](../assets/using-models/using-image-generation-models/image-playground.png)

## Sonuç { #conclusion }

Bu kurulumla Z-Image-Turbo modelini kullanarak metinsel istemlerden benzersiz ve görsel olarak ilgi çekici görüntüler oluşturabilirsiniz. Çok çeşitli görüntüler oluşturmak için farklı komut istemleri ve ayarlarla denemeler yapın!
