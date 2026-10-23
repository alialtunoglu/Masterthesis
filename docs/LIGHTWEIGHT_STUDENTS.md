# Hafif student kapsamı

21 Ekim 2026: MobileNetV3-Small referansına `shufflenet_v2_x0_5` ve `micronet_m0` eklendi. Yeni eğitim veya kuyruğa ekleme yapılmadı; eski veriler, sonuçlar ve checkpointler değiştirilmedi.

## Ölçülen boyutlar

Aşağıdaki değerler ImageNet'in 1000 sınıflı başlığından değil, bu projenin **6 sınıflı** modellerinden alındı. Girdi `1×3×224×224`, FP32; pretrained modeller CUDA üzerinde yalnızca forward ile kontrol edildi.

| Student | Parametre | Yaklaşık model belleği (MiB) | torchinfo MACs | Tahmini FLOPs (2 × MACs) | KD feature |
|---|---:|---:|---:|---:|---|
| `mobilenet_v3_small` | 1.524.006 | 5,860 | 55.508.574 | 111.017.148 | `features.12`: `[B,576,7,7]` |
| `shufflenet_v2_x0_5` | 347.942 | 1,358 | 39.466.550 | 78.933.100 | `conv5`: `[B,1024,7,7]` |
| `micronet_m0` | 380.436 | 1,485 | 2.824.150 | 5.648.300 | `features`: `[B,384,7,7]` |

Bellek, mevcut `get_model_summary_dict` parametre/buffer tahminidir; cihazda ölçülmüş tepe RAM veya checkpoint dosya boyutu değildir. `torchinfo` fonksiyonel aktivasyon, indeksleme ve kanal karıştırma maliyetlerinin tamamını saymaz; özellikle MicroNet'in Dynamic Shift-Max işlemleri nedeniyle MACs/FLOPs değerleri tam donanım maliyeti değildir. Bu tablo gecikme veya enerji üstünlüğünü kanıtlamaz.

ShuffleNet'in parametre sayısı MicroNet-M0'dan daha azdır; MicroNet'in temel avantaj adayı daha düşük hesaplama bütçesidir. Plant disease doğruluğu henüz ölçülmedi. İki model de ana referanstan daha az parametrelidir.

## Kaynak ve pretrained bilgisi

- ShuffleNet: torchvision `shufflenet_v2_x0_5`, `ShuffleNet_V2_X0_5_Weights.DEFAULT`; ImageNet pretrained ağırlıkları yüklendikten sonra `fc` altı sınıflı Linear ile değiştirilir.
- MicroNet: yazar deposu <https://github.com/liyunsheng13/micronet>, commit `ed89b2615e96e945f3dba64645ce8a68b0366222`, ICCV 2021 MicroNet-M0. `backbone/micronet.py`, `activation.py`, M0 konfigürasyonu MIT lisansıyla `src/models/_micronet/` altında tutulur. Lisans hem kaynakta hem kurulu pakette bulunur.
- M0 mimari değerleri yazarın `scripts/eval_micronet_m0.sh` dosyasından alınmıştır. Yeni yacs bağımlılığı eklenmedi.
- Yazarın ağırlık adresi: `http://www.svcl.ucsd.edu/projects/micronet/assets/micronet-m0.pth`.
- İndirilen dosyanın SHA256 değeri: `987159bcdd92ef2792da9dc4e8e698e8065a6aa0899e31d9762080c621d81b61`. Hem indirme hem cache okuma tam digest ile kontrol edilir; `torch.load(..., weights_only=True)` ve strict state loading kullanılır. Uyuşmazlık/indirme hatası rastgele ağırlığa sessiz dönüş yapmaz.
- **Sınır:** Bu digest yazarın bağımsız yayımladığı bir checksum değil, bu entegrasyonda erişilen dosyadan hesaplanmıştır. HTTP aktarımı ilk dosyanın kimliğini kriptografik olarak doğrulamaz. HTTPS denemeleri bu ortamda zaman aşımına uğradı. Sabit digest sonraki dosya değişikliklerini reddeder; cache hazırdır.
- Varsayılan cache: eğitim giriş noktalarının ayarladığı `checkpoints/pretrained/hub/checkpoints/`. Binary dosyalar Git'e dahil edilmez.

## MicroNet transfer uyarlamaları

- ImageNet modelinin tüm ağırlıkları orijinal 1000-class yapı üzerinde strict yüklenir. Ardından son `SwishLinear` (Linear + BatchNorm + activation), projenin CE/KL için beklediği **ham logit üreten Linear** başlıkla değiştirilir. Bu, orijinal ImageNet başlığının aynen kullanıldığı bir deney değildir.
- Ara classifier BatchNorm1d korunur. Eğitim batch'i yalnızca bir örnekse bu katman running statistics kullanır; örnek düşürülmez. Batch en az iki olduğunda normal BatchNorm eğitimi sürer. Temiz veride batch32 son batch3 olduğundan bu özel durum olağan deneyde tetiklenmez.
- Dynamic Shift-Max indeksleri non-persistent buffer olarak device ile taşınır; orijinal pretrained state anahtarlarına yeni kalıcı anahtar eklenmez. Upstream constructor debug printleri ve eval ile blok seçimi kaldırılmıştır.
- MicroNet-M0 giriş boyutu bu sürümde **224×224** olarak sabittir. Baseline ve KD arayüzü bunu gösterir; config ve forward yanlış boyutu reddeder.
- Ortak ImageNet mean/std ve mevcut train/eval dönüşümleri korunur; yazarın ImageNet eğitim reçetesini yeniden üretme iddiası yoktur.

## Arayüz ve deney sırası

`Yeni deney → Baseline → Student → CNN` altında iki yeni model bulunur. Single/multi KD seçimleri aynı ortak kataloğu kullanır. Logit, spatial feature ve relation loss için kısa sentetik backward kontrolleri yapılmıştır; bu bir gerçek veri eğitimi değildir.

Yeni temiz baseline configleri:

- `configs/baseline/shufflenet_v2_x0_5_plantpathology2021_clean.json`
- `configs/baseline/micronet_m0_plantpathology2021_clean.json`

Referansla aynı `dataset_ref`, seed42, image224, batch32, 30 epoch, LR0.0003, AdamW, weight decay0.0001, cosine, patience7, validation macro-F1 kullanılır. Üç student pretrained başlar; başlıkları yeni başlatılır. Ortak ayarlar kontrollü başlangıç protokolüdür, her modelin optimum eğitim reçetesi olduğu iddia edilmez.

İlk aşama artık **3 student + 3 teacher = 6 baseline**. Yeni teacher'lar temiz veri üzerinde tamamlanınca KD koşuları bağlanır. Önce ana student KD matrisi, ardından validation ile seçilen yöntemlerin küçük student'larda doğrulanması önerilir. Hedef cihazda gecikme/RAM/enerji ölçümü ayrıca gerekir.

Genel CNN preset'i sayesinde iki model diğer kayıtlı veri setlerinde de seçilebilir; veri setine özgü optimum ayarlar sunulmuş sayılmaz.

## Notebook ve sonuç aktarımı

Notebook seçim/config snapshot'ı yeni model kimliğini korur; dış sonuç içe aktarma allowlist'i ortak student kataloğunu kullanır. Ancak notebook **yayımlanmış Git commit'ini** çalıştırır. Bu yerel değişiklikler henüz commit/push edilmediği için Colab/Kaggle'da yeni modelleri çalıştırmadan önce kodun yayımlanması ve notebook'un yeniden üretilmesi gerekir. Yerel dosyalar notebook'a otomatik gömülmez.

## Doğrulama

- Önce: 212 test başarılı.
- Yeni model testleri: altı sınıflı çıktı, daha az parametre, singleton backward, strict pretrained yükleme, bozuk cache reddi, state roundtrip ve sabit girdi kontrolü.
- Entegrasyon: iki model × üç KD türü × single/multi sentetik backward; gerçek Streamlit seçim/config/komut kontrolü, eğitim düğmesine basılmadan.
- Üç student için gerçek pretrained ağırlıklarla CUDA no_grad forward ve torchinfo ölçümleri başarılı.
- Son tam test turu: **222/222 başarılı**, 62,191 saniye, masterthesis ortamı; log `reports/preflight_2026-10-21/lightweight_full_suite.log`. Bağımsız incelemede kritik/önemli sorun bulunmadı. İzole feature/relation-only gradient testi küçük bir ek kontrol olarak ertelendi; mevcut testler CE+KD toplam loss üzerinden gradient kontrol eder.
