# Knowledge Distillation Sonuçları: Karşılaştırmalı Analiz

**Analiz tarihi:** 24 Temmuz 2026  
**Veri seti:** Plant Pathology 2021  
**Student:** MobileNetV3-Small  
**Seed:** 42  
**KD yöntemi:** Logit-based knowledge distillation

## 1. Kısa cevap: Student iyileşti mi?

**Evet, doğru teacher seçildiğinde student belirgin biçimde iyileşti.**

MobileNetV3-Small baseline sonucu:

- Test accuracy: **%94,02**
- Test Macro-F1: **%92,56**

En iyi KD sonucu RegNet-Y-8GF teacher ile elde edildi:

- Test accuracy: **%95,29**
- Test Macro-F1: **%94,37**
- Accuracy artışı: **+1,27 yüzde puan**
- Macro-F1 artışı: **+1,81 yüzde puan**
- Test setinde baseline'a göre **33 ek doğru tahmin**

Student'ın inference mimarisi değişmedi: **1,52 milyon parametre**, **5,86 MB**, **55,51 milyon MACs** ve yaklaşık **111,02 milyon FLOPs**. İyileşme deployment maliyetini artırmadan elde edildi.

Ancak sonuç teacher'a bağlıdır:

- RegNet-Y-8GF ve ResNet50 güçlü iyileşme sağlamıştır.
- DenseNet201 Macro-F1'ı çok az artırmış, accuracy'yi düşürmüştür.
- Multi-teacher KD baseline'dan iyi sonuçlar üretmiş, fakat en iyi single-teacher KD'yi geçememiştir.

## 2. Baseline student ve teacher sonuçları

### Baseline student

| Model | Best epoch | Val accuracy | Val Macro-F1 | Test accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 26 | %93,44 | %92,02 | %94,02 | %92,56 |

Test seti 2.592 örnek içerir. Baseline student yaklaşık **2.437 doğru tahmin** üretmiştir.

### Teacher sonuçları

| Teacher | Best epoch | Val accuracy | Val Macro-F1 | Test accuracy | Test Macro-F1 | Parametre | Boyut |
|---|---:|---:|---:|---:|---:|---:|---:|
| DenseNet201 | 19 | %95,56 | %94,04 | **%96,60** | **%95,79** | 18,10 M | 69,94 MB |
| ResNet50 | 20 | %95,52 | **%94,13** | %96,06 | %95,21 | 23,52 M | 89,93 MB |
| RegNet-Y-8GF | 5 | %95,10 | %93,50 | %96,03 | %94,97 | 37,38 M | 142,91 MB |

Teacher'ların baseline student'a göre test Macro-F1 üstünlüğü:

| Teacher | Teacher Macro-F1 | Baseline farkı |
|---|---:|---:|
| DenseNet201 | %95,79 | +3,23 yp |
| ResNet50 | %95,21 | +2,65 yp |
| RegNet-Y-8GF | %94,97 | +2,41 yp |

**yp:** yüzde puan.

## 3. Single-teacher KD sonuçları

Tüm koşullarda temperature 4, alpha 0,5, AdamW, cosine scheduler, en fazla 30 epoch ve validation Macro-F1'a göre 7 epoch patience kullanılmıştır.

| Teacher | KD best epoch | KD val accuracy | KD val Macro-F1 | KD test accuracy | KD test Macro-F1 |
|---|---:|---:|---:|---:|---:|
| DenseNet201 | 10 | %94,05 | %92,70 | %93,83 | %92,87 |
| ResNet50 | 15 | %94,56 | %92,83 | **%95,29** | %94,32 |
| RegNet-Y-8GF | 19 | **%94,94** | **%93,54** | **%95,29** | **%94,37** |

### Baseline student'a göre değişim

| Teacher | Accuracy değişimi | Macro-F1 değişimi | Doğru tahmin değişimi | Sonuç |
|---|---:|---:|---:|---|
| DenseNet201 | **−0,19 yp** | +0,30 yp | −5 | Karışık/zayıf katkı |
| ResNet50 | **+1,27 yp** | **+1,75 yp** | +33 | Güçlü iyileşme |
| RegNet-Y-8GF | **+1,27 yp** | **+1,81 yp** | +33 | En iyi sonuç |

### Student teacher seviyesine ne kadar yaklaştı?

| Teacher | Teacher F1 | KD student F1 | Teacher–student farkı | Teacher avantajının aktarılan kısmı |
|---|---:|---:|---:|---:|
| DenseNet201 | %95,79 | %92,87 | 2,93 yp | yaklaşık %9 |
| ResNet50 | %95,21 | %94,32 | 0,89 yp | yaklaşık %66 |
| RegNet-Y-8GF | %94,97 | %94,37 | **0,60 yp** | yaklaşık **%75** |

Aktarılan kısım, KD kazancının teacher ile baseline arasındaki performans boşluğuna oranıdır. RegNet-Y-8GF kendi baseline üstünlüğünün yaklaşık dörtte üçünü student'a aktarabilmiştir. DenseNet201 ise daha yüksek teacher skoruna rağmen avantajının yalnızca küçük bir bölümünü aktarmıştır.

### Teacher gücü ile KD başarısı aynı mı?

Teacher sıralaması:

1. DenseNet201 — %95,79
2. ResNet50 — %95,21
3. RegNet-Y-8GF — %94,97

KD student sıralaması:

1. RegNet-Y-8GF KD — %94,37
2. ResNet50 KD — %94,32
3. DenseNet201 KD — %92,87

Sıralamalar neredeyse tersine dönmüştür. Bu sonuç, **en yüksek test performansına sahip teacher'ın otomatik olarak en iyi distillation teacher'ı olmadığını** gösterir. Student–teacher mimari uyumu, soft-target yapısı, teacher calibration ve hata çeşitliliği teacher'ın ham doğruluğundan daha önemli olabilir. Bunlar mevcut metriklerden çıkan hipotezlerdir; ayrıca test edilmelidir.

## 4. Single-teacher sınıf bazlı analiz

### F1 skorları

| Sınıf | Baseline | DenseNet201 KD | ResNet50 KD | RegNet-Y-8GF KD |
|---|---:|---:|---:|---:|
| complex | %79,82 | %81,96 | %84,53 | **%85,65** |
| frog_eye_leaf_spot | %94,34 | %94,34 | %94,12 | **%94,98** |
| healthy | %96,75 | %95,82 | **%97,71** | %97,49 |
| powdery_mildew | %94,38 | %96,92 | **%97,77** | %96,94 |
| rust | %95,17 | %93,63 | **%95,22** | %94,95 |
| scab | %94,93 | %94,53 | **%96,56** | %96,21 |

### Baseline'a göre değişimler

| Sınıf | DenseNet201 KD | ResNet50 KD | RegNet-Y-8GF KD |
|---|---:|---:|---:|
| complex | +2,14 yp | +4,71 yp | **+5,84 yp** |
| frog_eye_leaf_spot | 0,00 yp | −0,22 yp | +0,64 yp |
| healthy | −0,93 yp | **+0,96 yp** | +0,74 yp |
| powdery_mildew | +2,54 yp | **+3,38 yp** | +2,55 yp |
| rust | −1,54 yp | +0,05 yp | −0,22 yp |
| scab | −0,40 yp | **+1,63 yp** | +1,28 yp |

RegNet-Y-8GF KD'nin genel Macro-F1 artışındaki en büyük unsur en zor sınıf olan `complex` sınıfıdır: **%79,82'den %85,65'e**, yani **+5,84 yüzde puan**. KD'nin özellikle zayıf sınıfı geliştirmesi, yalnızca accuracy artışından daha değerli bir bulgudur.

ResNet50 KD daha dengeli davranmıştır; dört sınıfı belirgin geliştirmiş, rust sınıfını korumuş ve yalnızca frog-eye-leaf-spot sınıfında küçük düşüş göstermiştir. DenseNet201 KD ise complex ve powdery-mildew sınıflarını iyileştirirken healthy, rust ve scab sınıflarını geriletmiştir.

## 5. Multi-teacher KD sonuçları

Mevcut bütün multi-teacher koşuları uniform aggregation kullanmıştır. Üç ikili koşuda ağırlıklar 0,5/0,5; üçlü koşuda her teacher için 1/3 olarak çözülmüştür.

| Teacher çifti | Best epoch | Val Macro-F1 | Test accuracy | Test Macro-F1 | Baseline F1 farkı |
|---|---:|---:|---:|---:|---:|
| DenseNet201 + ResNet50 | 19 | %93,06 | %94,33 | %93,13 | +0,57 yp |
| DenseNet201 + RegNet-Y-8GF | 18 | %93,21 | **%95,22** | **%94,02** | **+1,46 yp** |
| RegNet-Y-8GF + ResNet50 | 16 | %93,37 | %94,44 | %93,22 | +0,65 yp |
| DenseNet201 + RegNet-Y-8GF + ResNet50 | 22 | **%93,58** | **%95,14** | %93,96 | +1,40 yp |

Dört multi-teacher student da baseline Macro-F1 skorunu geçmiştir. Üçlü teacher student, %93,96 Macro-F1 ile ikili DenseNet201 + RegNet-Y-8GF sonucunun yalnızca 0,06 yüzde puan gerisinde kalmış; ek teacher mevcut uniform ayarda student performansını artırmamıştır.

### En iyi single teacher ile karşılaştırma

| Multi-teacher çifti | Multi-KD F1 | Çifte ait en iyi single-KD F1 | Multi farkı |
|---|---:|---:|---:|
| DenseNet201 + ResNet50 | %93,13 | %94,32 | **−1,19 yp** |
| DenseNet201 + RegNet-Y-8GF | %94,02 | %94,37 | **−0,35 yp** |
| RegNet-Y-8GF + ResNet50 | %93,22 | %94,37 | **−1,15 yp** |
| DenseNet201 + RegNet-Y-8GF + ResNet50 | %93,96 | %94,37 | **−0,41 yp** |

Uniform aggregation altında hiçbir multi-teacher student, içerdiği teacher'lar arasındaki en iyi single-teacher student'ı geçememiştir. Üçlü koşu da en iyi ikili koşuldan 0,06 yüzde puan ve en iyi single-teacher koşudan 0,41 yüzde puan düşüktür. DenseNet201'in zayıf distillation davranışı ve eşit ağırlık, diğer teacher'ların yararlı sinyalini seyreltiyor olabilir.

## 6. Ensemble ile distile student arasındaki fark

| Teacher çifti | Ensemble accuracy | Ensemble Macro-F1 | KD student Macro-F1 | Aktarım boşluğu |
|---|---:|---:|---:|---:|
| DenseNet201 + ResNet50 | %96,72 | **%95,89** | %93,13 | **2,76 yp** |
| DenseNet201 + RegNet-Y-8GF | **%96,76** | %95,82 | **%94,02** | **1,80 yp** |
| RegNet-Y-8GF + ResNet50 | %96,60 | %95,80 | %93,22 | **2,59 yp** |
| DenseNet201 + RegNet-Y-8GF + ResNet50 | **%96,88** | **%96,09** | %93,96 | **2,12 yp** |

Ensemble değerleri student değerleriyle karıştırılmamalıdır. Ensemble inference sırasında iki veya üç büyük teacher'ı birlikte çalıştırır; KD student ise yalnızca MobileNetV3-Small'dır. Üçlü teacher ensemble en yüksek ensemble skorunu (%96,09 Macro-F1) üretmiştir, ancak bu üstünlük student'a tam aktarılamamıştır. DenseNet201 + RegNet-Y-8GF çifti hâlâ en iyi multi-teacher student'ı üretmektedir. Bu çift ve üçlü koşu, validation-weighted aggregation karşılaştırması için en güçlü takip adaylarıdır.


## 6.1 Üç-teacher KD sonucu

### Koşu bilgileri

| Alan | Değer |
|---|---|
| Job ID | `c66cb4e6f1fd` |
| Durum | **finished** |
| Dry-run | Hayır |
| Dataset | Plant Pathology 2021 |
| Student | MobileNetV3-Small |
| Teacher sayısı | 3 |
| Teacher'lar | DenseNet201 + RegNet-Y-8GF + ResNet50 |
| KD türü | Logit-based |
| Aggregation | Uniform |
| Teacher ağırlıkları | 0,3333 + 0,3333 + 0,3333 |
| Seed | 42 |
| En iyi epoch | 22 |
| Toplam çalışan epoch | 29 |
| Sonlanma | Early stopping; kopma yok |

### Üç-teacher student ve ensemble metrikleri

| Model/çıktı | Val accuracy | Val Macro-F1 | Test accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|
| Üç-teacher KD student | **%95,02** | **%93,58** | %95,14 | %93,96 |
| Üç-teacher ensemble | — | — | **%96,88** | **%96,09** |

### Karşılaştırmalı performans

| Referans | Test accuracy | Test Macro-F1 | Üç-teacher student'ın accuracy farkı | Üç-teacher student'ın F1 farkı |
|---|---:|---:|---:|---:|
| MobileNetV3-Small baseline | %94,02 | %92,56 | **+1,12 yp** | **+1,40 yp** |
| En iyi ikili Multi-KD: DenseNet201 + RegNet-Y-8GF | %95,22 | %94,02 | −0,08 yp | −0,06 yp |
| En iyi single-KD: RegNet-Y-8GF | %95,29 | %94,37 | −0,15 yp | −0,41 yp |
| Üç-teacher ensemble | %96,88 | %96,09 | −1,74 yp | −2,12 yp |

Üçüncü teacher ensemble performansını yükseltmiştir; fakat bu ek bilgi uniform aggregation altında student'a tam aktarılamamıştır. Üç-teacher student baseline'dan açıkça iyi, en iyi ikili Multi-KD'ye çok yakın, fakat en iyi single-teacher KD'nin gerisindedir.

### Artifact bütünlüğü

| Artifact | Durum |
|---|---|
| Config snapshot | Mevcut |
| History CSV | Mevcut |
| Best checkpoint | Mevcut |
| Per-class metrics | Mevcut |
| Confusion matrix | Mevcut |
| Learning curves | Mevcut |

History 29 epoch içermektedir. En iyi model 22. epoch'ta seçilmiş ve patience 7 nedeniyle eğitim 29. epoch'ta sonlanmıştır. Bu koşuda kopma veya yarım kalma belirtisi yoktur.

## 7. Multi-teacher sınıf bazlı sonuçlar

| Sınıf | Baseline | DN201 + RN50 | DN201 + RegNet | RegNet + RN50 | Üç teacher |
|---|---:|---:|---:|---:|---:|
| complex | %79,82 | %81,43 | %83,15 | %81,48 | **%83,37** |
| frog_eye_leaf_spot | %94,34 | %93,09 | **%94,66** | %93,21 | %94,58 |
| healthy | %96,75 | %97,58 | **%98,13** | %97,48 | %97,92 |
| powdery_mildew | %94,38 | %96,61 | **%97,48** | %96,95 | %97,74 |
| rust | %95,17 | %94,53 | %94,35 | %94,29 | %93,81 |
| scab | %94,93 | %95,54 | **%96,37** | %95,88 | %96,37 |

En iyi çift DenseNet201 + RegNet-Y-8GF beş sınıfı geliştirmiştir; fakat rust sınıfını **−0,82 yüzde puan** düşürmüştür. Üçlü teacher da beş sınıfı geliştirirken rust sınıfını **−1,36 yüzde puan** düşürmüştür. Bütün multi-teacher koşularında rust gerilemektedir. Bu ortak desen confusion matrix ve örnek bazlı tahminlerle araştırılmalıdır.

## 8. KD student'ın diğer baseline modellerle konumu

| Model | Test accuracy | Test Macro-F1 | Parametre | Boyut |
|---|---:|---:|---:|---:|
| EfficientNet-B0 baseline | **%95,95** | **%95,04** | 4,02 M | 15,48 MB |
| RegNet-Y-8GF KD MobileNetV3-Small | %95,29 | %94,37 | **1,52 M** | **5,86 MB** |
| ResNet18 baseline | %94,37 | %93,22 | 11,18 M | 42,68 MB |
| MobileNetV3-Large baseline | %94,48 | %93,17 | 4,21 M | 16,15 MB |
| MobileNetV3-Small baseline | %94,02 | %92,56 | 1,52 M | 5,86 MB |

En iyi KD student:

- ResNet18 baseline'ı Macro-F1'da **1,15 yp** geçmiştir.
- MobileNetV3-Large baseline'ı **1,20 yp** geçmiştir.
- EfficientNet-B0'ın yalnızca **0,67 yp** gerisindedir.
- EfficientNet-B0'a göre yaklaşık **2,64 kat daha az parametre** taşır.

Bu nedenle performans–model büyüklüğü dengesi açısından güçlü bir Pareto noktasıdır.

## 9. Sıkıştırma başarısı

| Teacher | Teacher parametresi | Student parametresi | Student kaç kat küçük? | Teacher–student F1 farkı |
|---|---:|---:|---:|---:|
| DenseNet201 | 18,10 M | 1,52 M | 11,9× | 2,93 yp |
| ResNet50 | 23,52 M | 1,52 M | 15,4× | 0,89 yp |
| RegNet-Y-8GF | 37,38 M | 1,52 M | **24,5×** | **0,60 yp** |

En başarılı sıkıştırma RegNet-Y-8GF → MobileNetV3-Small koşusudur. Student yaklaşık 24,5 kat daha az parametreli, yaklaşık 24,4 kat daha küçük ve teacher Macro-F1 skorunun yalnızca 0,60 yüzde puan gerisindedir.

## 10. Bilimsel yorum ve sınırlamalar

### Kesin olarak gözlenenler

1. Logit KD, teacher'a bağlı olarak MobileNetV3-Small'ı iyileştirmektedir.
2. RegNet-Y-8GF ve ResNet50 pratik açıdan güçlü kazanç üretmiştir.
3. DenseNet201 en iyi teacher olmasına rağmen en zayıf KD student'ı üretmiştir.
4. İkili ve üçlü uniform multi-teacher KD baseline'ı geliştirmiştir.
5. Üç teacher ensemble skorunu yükseltmiş, fakat üç-teacher student en iyi ikili veya single-teacher student'ı geçememiştir.
6. En güçlü sınıf bazlı kazanç complex ve powdery-mildew sınıflarındadır.
7. Student'ın inference parametresi ve MACs değeri KD sonrasında artmamıştır.

### Henüz kesin söylenemeyecekler

- Farkların istatistiksel olarak anlamlı olduğu söylenemez.
- RegNet-Y-8GF'nin genel olarak en iyi teacher olduğu söylenemez.
- Multi-teacher KD'nin genel olarak başarısız olduğu söylenemez; yalnızca mevcut uniform ayarlar en iyi single teacher'ı geçmemiştir.
- Sonuçların başka seed, dataset veya student mimarisine genelleneceği söylenemez.
- Feature-based veya relation-based KD ile kıyas yapılamaz; nihai sonuçlarda bu koşular yoktur.

### Temel sınırlamalar

- Bütün sonuçlar yalnızca **seed 42** ile elde edilmiştir.
- KD sonuçları yalnızca **logit-based** yöntemi içerir.
- Multi-teacher sonuçlarında yalnızca **uniform aggregation** denenmiştir.
- Üçlü teacher sonucu vardır; ancak yalnızca uniform aggregation ve tek seed ile çalıştırılmıştır.
- Temperature ve alpha ablation'ı yapılmamıştır.
- Örnek bazlı tahminler olmadan paired bootstrap veya McNemar testi yapılamaz.
- Latency ölçümleri koşular arasında değişkendir; standart benchmark olmadan kesin hız kıyası yapılmamalıdır.

## 11. Önerilen takip deneyleri

1. Baseline, ResNet50 KD, RegNet-Y-8GF KD ve en iyi multi-KD koşusunu en az beş seed ile çalıştır.
2. Ortalama, standart sapma ve %95 güven aralığı raporla.
3. Test tahminlerini örnek bazında saklayıp paired bootstrap/McNemar analizi yap.
4. DenseNet201 + RegNet-Y-8GF çiftinde uniform, validation-weighted ve manual aggregation'ı karşılaştır.
5. Aynı teacher'larda feature-based ve relation-based KD çalıştır.
6. Temperature için 2/4/8, alpha için 0,3/0,5/0,7 ablation'ı yap.
7. Teacher calibration, student–teacher agreement ve hata örtüşmesini ölç.
8. Rust sınıfındaki ortak multi-teacher gerilemesini örnek bazlı incele.

## 12. Nihai değerlendirme

KD deneyleri genel olarak başarılıdır, fakat başarı teacher seçimine güçlü biçimde bağlıdır.

> **RegNet-Y-8GF → MobileNetV3-Small logit KD**, test Macro-F1 skorunu %92,56'dan %94,37'ye çıkararak **+1,81 yüzde puan** iyileşme sağlamıştır.

Bu student:

- kendi baseline'ından açıkça daha iyi,
- MobileNetV3-Large ve ResNet18 baseline'larından daha iyi,
- EfficientNet-B0'a yakın,
- RegNet-Y-8GF teacher'dan 24,5 kat daha az parametreli,
- teacher Macro-F1 skorunun yalnızca 0,60 yüzde puan gerisindedir.

Multi-teacher deneyleri baseline'ı geliştirmiş, fakat uniform aggregation altında en iyi single-teacher sonucu geçememiştir. Mevcut bulgular teacher sayısını artırmaktan çok doğru teacher ve doğru aggregation politikasını seçmenin önemli olduğunu göstermektedir.

---

## Kaynak sonuç dosyaları

- results/baseline/baseline_results.csv
- results/teachers/cnn/teacher_results.csv
- results/knowledge_distillation/kd_results.csv
- results/multi_teacher_knowledge_distillation/multi_kd_results.csv
- İlgili run klasörlerindeki history.csv ve per_class_metrics.csv artifact'ları

Kaynak sonuç dosyalarında değişiklik yapılmamıştır.
