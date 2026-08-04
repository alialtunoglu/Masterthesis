# Deney Sonuçları Analizi ve Bildiri Potansiyeli

**Analiz tarihi:** 24 Temmuz 2026  
**Kapsam:** Baseline modeller, CNN teacher modelleri, single-teacher logit KD ve multi-teacher uniform logit KD  
**Ana KD veri seti:** Plant Pathology 2021  
**Ana student:** MobileNetV3-Small  

## 1. Yönetici özeti

Bu deneylerin en güçlü ve savunulabilir sonucu şudur:

> MobileNetV3-Small, RegNet-Y-8GF öğretmeninden yapılan logit tabanlı knowledge distillation ile Plant Pathology 2021 test Macro-F1 skorunu **%92,56'dan %94,37'ye** çıkarmıştır. Bu, **+1,81 yüzde puan** artış anlamına gelir. Modelin parametre sayısı ve çıkarım maliyeti değişmemiştir.

Öne çıkan diğer sonuçlar:

- En iyi Plant Pathology 2021 teacher modeli DenseNet201'dir: **%96,60 accuracy**, **%95,79 Macro-F1**.
- En iyi doğrudan eğitilmiş hafif/orta ölçekli baseline EfficientNet-B0'dır: **%95,95 accuracy**, **%95,04 Macro-F1**.
- En iyi KD student, RegNet-Y-8GF → MobileNetV3-Small koşusudur: **%95,29 accuracy**, **%94,37 Macro-F1**.
- Bu KD student yalnızca **1,52 milyon parametre** ve **5,86 MB** boyutla çalışır.
- RegNet-Y-8GF teacher ile karşılaştırıldığında student yaklaşık **24,5 kat daha az parametreye** ve **24,4 kat daha küçük model boyutuna** sahiptir.
- DenseNet201 + RegNet-Y-8GF multi-teacher koşusu, multi-teacher student'lar içinde en iyi sonucu vermiştir: **%95,22 accuracy**, **%94,02 Macro-F1**.
- Ancak incelenen hiçbir multi-teacher student, en iyi single-teacher KD student'ı geçememiştir.
- Multi-teacher ensemble'ların Macro-F1 skorları yaklaşık **%95,80–%95,89** düzeyindedir; fakat bu değer iki büyük teacher'ın birlikte inference yapmasına aittir ve hafif student ile aynı maliyet sınıfında değildir.

Mevcut sonuçlar **uygulamalı bir bildiri için ümit vericidir**, ancak tek seed kullanımı, feature/relation KD sonuçlarının bulunmaması ve aggregation ablation'larının henüz yapılmamış olması nedeniyle mevcut haliyle güçlü genellenebilirlik veya istatistiksel üstünlük iddiası kurulamaz.

## 2. İncelenen deney envanteri

Nihai sonuç tablolarında bulunan deneyler:

| Deney grubu | Nihai sonuç sayısı | Seed'ler | Açıklama |
|---|---:|---|---|
| Baseline | 12 | 42 | 3 veri seti × 4 model |
| CNN teacher | 28 | 42 | AppleLeaf9: 10, PlantVillage: 10, Plant Pathology 2021: 8 |
| Single-teacher KD | 3 | 42 | Üç teacher, yalnızca logit-based |
| Multi-teacher KD | 3 | 42 | Üç ikili teacher kombinasyonu, yalnızca uniform logit-based |
| **Toplam nihai sonuç** | **46** | **yalnızca 42** | Dry-run/failed job'lar hariç |

Job kayıtlarında ayrıca:

- 1 başarısız single-teacher KD job'ı,
- sonuç tablosuna yazılmamış 1 tamamlanmış single-teacher KD job'ı — önceki dry-run ile uyumlu,
- sonuç tablosuna yazılmamış 1 tamamlanmış teacher job'ı

bulunmaktadır. Dolayısıyla **job sayısı ile bilimsel sonuç satırı sayısı aynı değildir**. Analizde yalnızca nihai özet CSV'lerine yazılmış gerçek deney sonuçları kullanılmıştır.

Nihai tablolardaki checkpoint, config, history, sınıf bazlı metrik, confusion matrix ve learning-curve yolları kontrol edilmiş; incelenen satırlarda eksik artifact bulunmamıştır.

### Veri seti bölünmeleri

| Veri seti | Sınıf | Train | Validation | Test |
|---|---:|---:|---:|---:|
| AppleLeaf9 | 9 | 10.208 | 2.184 | 2.190 |
| PlantVillage | 38 | 38.013 | 8.138 | 8.154 |
| Plant Pathology 2021 | 6 | 12.095 | 2.590 | 2.592 |

## 3. Baseline sonuçları

### 3.1 AppleLeaf9

| Sıra | Model | Accuracy | Macro-F1 | Parametre | Boyut |
|---:|---|---:|---:|---:|---:|
| 1 | MobileNetV3-Large | %96,99 | **%95,92** | 4,21 M | 16,17 MB |
| 2 | EfficientNet-B0 | %96,44 | %94,85 | 4,02 M | 15,49 MB |
| 3 | ResNet18 | %96,12 | %94,14 | 11,18 M | 42,69 MB |
| 4 | MobileNetV3-Small | %95,39 | %94,01 | 1,53 M | 5,87 MB |

MobileNetV3-Large hem accuracy hem Macro-F1 açısından en iyi baseline'dır. MobileNetV3-Small ise en düşük sonuçlu model olmasına rağmen yaklaşık üçte bir parametreyle Macro-F1'de yalnızca 1,91 yüzde puan geridedir.

### 3.2 PlantVillage

| Sıra | Model | Accuracy | Macro-F1 | Parametre | Boyut |
|---:|---|---:|---:|---:|---:|
| 1 | ResNet18 | %99,48 | **%99,26** | 11,20 M | 42,75 MB |
| 2 | MobileNetV3-Large | %99,46 | %99,24 | 4,25 M | 16,31 MB |
| 3 | MobileNetV3-Small | %99,42 | %99,23 | 1,56 M | 5,99 MB |
| 4 | EfficientNet-B0 | **%99,53** | %99,03 | 4,06 M | 15,63 MB |

PlantVillage sonuçları doygunluk bölgesindedir. Modeller arasındaki farklar çok küçüktür; bu veri seti KD yöntemlerini ayırt etmek için Plant Pathology 2021 kadar zorlayıcı görünmemektedir.

### 3.3 Plant Pathology 2021

| Sıra | Model | Accuracy | Macro-F1 | Parametre | Boyut |
|---:|---|---:|---:|---:|---:|
| 1 | EfficientNet-B0 | **%95,95** | **%95,04** | 4,02 M | 15,48 MB |
| 2 | ResNet18 | %94,37 | %93,22 | 11,18 M | 42,68 MB |
| 3 | MobileNetV3-Large | %94,48 | %93,17 | 4,21 M | 16,15 MB |
| 4 | MobileNetV3-Small | %94,02 | %92,56 | 1,52 M | 5,86 MB |

Plant Pathology 2021, KD için doğru seçimdir: baseline'lar arasında anlamlı performans boşluğu vardır ve MobileNetV3-Small'ın geliştirilmesi için alan bırakmaktadır.

## 4. Teacher sonuçları

### 4.1 Veri seti başına en iyi teacher'lar

| Veri seti | En iyi teacher | Accuracy | Macro-F1 |
|---|---|---:|---:|
| AppleLeaf9 | ResNet50 | %98,58 | **%97,98** |
| PlantVillage | ResNet101 | %99,89 | **%99,86** |
| Plant Pathology 2021 | DenseNet201 | %96,60 | **%95,79** |

### 4.2 Plant Pathology 2021 teacher sıralaması

| Sıra | Teacher | Accuracy | Macro-F1 | Parametre | Boyut |
|---:|---|---:|---:|---:|---:|
| 1 | DenseNet201 | **%96,60** | **%95,79** | 18,10 M | 69,94 MB |
| 2 | ResNet50 | %96,06 | %95,21 | 23,52 M | 89,93 MB |
| 3 | ResNet101 | %95,95 | %95,08 | 42,51 M | 162,57 MB |
| 4 | RegNet-Y-8GF | %96,03 | %94,97 | 37,38 M | 142,91 MB |
| 5 | EfficientNet-B3 | %95,76 | %94,68 | 10,71 M | 41,17 MB |
| 6 | EfficientNet-B4 | %94,98 | %93,94 | 17,56 M | 67,46 MB |
| 7 | DenseNet121 | %94,60 | %93,61 | 6,96 M | 26,87 MB |
| 8 | VGG19-BN | %94,41 | %93,05 | 139,61 M | 532,60 MB |

En büyük modelin en iyi teacher olmadığı açıkça görülmektedir. DenseNet201, ResNet101 ve RegNet-Y-8GF'den daha az parametreyle daha yüksek Macro-F1 üretmiştir. VGG19-BN ise çok yüksek boyutuna rağmen en zayıf teacher'dır.

## 5. Single-teacher logit KD sonuçları

Tüm koşularda:

- Student: MobileNetV3-Small
- Temperature: 4
- Alpha: 0,5
- Optimizer: AdamW
- Scheduler: cosine
- Maksimum epoch: 30
- Early-stopping patience: 7
- İzleme metriği: validation Macro-F1

| Teacher | En iyi epoch | Test accuracy | Test Macro-F1 | Baseline'a göre accuracy | Baseline'a göre Macro-F1 |
|---|---:|---:|---:|---:|---:|
| DenseNet201 | 10 | %93,83 | %92,87 | **−0,19 yp** | +0,30 yp |
| ResNet50 | 15 | **%95,29** | %94,32 | **+1,27 yp** | +1,75 yp |
| RegNet-Y-8GF | 19 | **%95,29** | **%94,37** | **+1,27 yp** | **+1,81 yp** |

`yp`: yüzde puan.

Test setindeki doğru tahmin sayısına çevrildiğinde:

| Koşul | Doğru tahmin | Baseline'a göre |
|---|---:|---:|
| Baseline MobileNetV3-Small | 2.437 / 2.592 | — |
| DenseNet201 KD | 2.432 / 2.592 | −5 |
| ResNet50 KD | 2.470 / 2.592 | +33 |
| RegNet-Y-8GF KD | 2.470 / 2.592 | +33 |

### Yorum

Teacher'ın tek başına en yüksek test skoruna sahip olması, onun en iyi distillation teacher'ı olacağını garanti etmemiştir:

- En güçlü teacher DenseNet201 olmasına rağmen en zayıf KD sonucunu üretmiştir.
- RegNet-Y-8GF'nin teacher Macro-F1'ı DenseNet201'den düşük olmasına rağmen en iyi student'ı üretmiştir.

Bu bulgu, **teacher doğruluğunun yanında student–teacher uyumluluğu, hata çeşitliliği ve soft-target yapısının** önemli olabileceğini düşündürür. Ancak bu açıklama mevcut kayıtlardan doğrudan kanıtlanmış değildir; agreement, calibration ve örnek bazlı hata örtüşmesiyle ayrıca test edilmelidir.

## 6. Multi-teacher logit KD sonuçları

Mevcut nihai sonuçların tamamı:

- iki teacher,
- uniform aggregation,
- `w₁ = w₂ = 0,5`,
- temperature 4,
- alpha 0,5

kullanmıştır.

| Teacher çifti | Student accuracy | Student Macro-F1 | Baseline'a göre F1 | Ensemble Macro-F1 |
|---|---:|---:|---:|---:|
| DenseNet201 + ResNet50 | %94,33 | %93,13 | +0,57 yp | **%95,89** |
| DenseNet201 + RegNet-Y-8GF | **%95,22** | **%94,02** | **+1,46 yp** | %95,82 |
| RegNet-Y-8GF + ResNet50 | %94,44 | %93,22 | +0,65 yp | %95,80 |

### Ana sonuç

Multi-teacher KD baseline'ı üç koşulda da geliştirmiştir; fakat:

- en iyi multi-teacher student: **%94,02 Macro-F1**,
- en iyi single-teacher student: **%94,37 Macro-F1**.

Dolayısıyla mevcut ayarlarla multi-teacher yaklaşım, en iyi single-teacher KD'ye göre **0,35 yüzde puan daha düşüktür**.

Bu olumsuz değil, bilimsel olarak değerlidir: uniform ağırlıklandırma, yararlı teacher bilgisini otomatik olarak güçlendirmemekte; bazı teacher çiftleri student'a çelişkili veya gereksiz soft targets verebilmektedir.

Ensemble sonuçları student sonuçlarıyla karıştırılmamalıdır. Örneğin DenseNet201 + ResNet50 ensemble Macro-F1'ı %95,89'dur; ancak bu inference sırasında iki teacher'ın birlikte çalışmasını gerektirir. Distile student'ın Macro-F1'ı %93,13'tür. Aradaki **2,76 yüzde puanlık bilgi aktarım boşluğu**, multi-teacher distillation yönteminin iyileştirme alanını gösterir.

## 7. Sınıf bazlı analiz

Plant Pathology 2021 sınıf dağılımı eşit değildir. Test desteği:

| Sınıf | Test örneği |
|---|---:|
| complex | 240 |
| frog_eye_leaf_spot | 477 |
| healthy | 694 |
| powdery_mildew | 178 |
| rust | 279 |
| scab | 724 |

### En iyi single-teacher KD: RegNet-Y-8GF

| Sınıf | Baseline F1 | KD F1 | Değişim |
|---|---:|---:|---:|
| complex | %79,82 | **%85,65** | **+5,84 yp** |
| frog_eye_leaf_spot | %94,34 | %94,98 | +0,64 yp |
| healthy | %96,75 | %97,49 | +0,74 yp |
| powdery_mildew | %94,38 | **%96,94** | **+2,55 yp** |
| rust | %95,17 | %94,95 | −0,22 yp |
| scab | %94,93 | %96,21 | +1,28 yp |

Macro-F1 artışının önemli kısmı en zor sınıf olan `complex` sınıfındaki **+5,84 yüzde puanlık** iyileşmeden gelmektedir. Bu, KD'nin yalnızca çoğunluk sınıflarını iyileştirmediğini gösteren olumlu bir bulgudur.

### En iyi multi-teacher KD: DenseNet201 + RegNet-Y-8GF

| Sınıf | Baseline F1 | Multi-KD F1 | Değişim |
|---|---:|---:|---:|
| complex | %79,82 | %83,15 | +3,33 yp |
| frog_eye_leaf_spot | %94,34 | %94,66 | +0,32 yp |
| healthy | %96,75 | **%98,13** | +1,38 yp |
| powdery_mildew | %94,38 | **%97,48** | +3,10 yp |
| rust | %95,17 | %94,35 | −0,82 yp |
| scab | %94,93 | %96,37 | +1,44 yp |

Multi-teacher KD beş sınıfı iyileştirirken `rust` sınıfını geriletmiştir. Teacher çiftlerinin hata örtüşmesini sınıf bazında incelemek, uniform aggregation'ın neden en iyi single teacher'ı geçemediğini açıklayabilir.

## 8. Verimlilik ve sıkıştırma

En iyi KD student'ın deployment profili:

| Model | Parametre | Boyut | MACs | Tahmini FLOPs |
|---|---:|---:|---:|---:|
| MobileNetV3-Small KD student | 1,52 M | 5,86 MB | 55,51 M | 111,02 M |
| DenseNet201 teacher | 18,10 M | 69,94 MB | 4,29 B | 8,58 B |
| ResNet50 teacher | 23,52 M | 89,93 MB | 4,09 B | 8,17 B |
| RegNet-Y-8GF teacher | 37,38 M | 142,91 MB | 8,47 B | 16,94 B |

Student:

- DenseNet201'e göre yaklaşık **11,9× daha az parametreli**,
- ResNet50'ye göre yaklaşık **15,4× daha az parametreli**,
- RegNet-Y-8GF'ye göre yaklaşık **24,5× daha az parametreli**,
- teacher'a ihtiyaç duymadan tek başına inference yapmaktadır.

Bu, bildirinin en güçlü pratik mesajıdır: RegNet-Y-8GF KD student, EfficientNet-B0 baseline'ın Macro-F1 skoruna 0,67 yüzde puan yaklaşırken yaklaşık **2,64× daha az parametre** taşır.

Kaydedilen latency değerleri aynı model için farklı koşul gruplarında yaklaşık 4–11 ms arasında değişmektedir. Warm-up, tekrar sayısı, senkronizasyon ve donanım ölçüm protokolü açıkça standardize edilmeden bu değerler bilimsel karşılaştırmada kullanılmamalıdır. Parametre, boyut ve MACs değerleri daha güvenilir karşılaştırma ölçütleridir.

## 9. Eğitim davranışı

Early stopping beklenen şekilde çalışmıştır:

| Koşul | En iyi epoch | Çalışan epoch |
|---|---:|---:|
| Single DenseNet201 | 10 | 17 |
| Single ResNet50 | 15 | 22 |
| Single RegNet-Y-8GF | 19 | 26 |
| Multi DenseNet201 + ResNet50 | 19 | 26 |
| Multi DenseNet201 + RegNet-Y-8GF | 18 | 25 |
| Multi RegNet-Y-8GF + ResNet50 | 16 | 23 |

Patience değeri 7 olduğu için koşuların en iyi epoch'tan yaklaşık yedi epoch sonra durması kayıtlarla uyumludur.

## 10. Sonuçların güvenilirliği ve sınırlamalar

### Kritik sınırlamalar

1. **Bütün deneyler yalnızca seed 42 ile çalıştırılmıştır.**  
   Standart sapma, güven aralığı veya seed kararlılığı bilinmemektedir. 0,3–0,7 yüzde puanlık farklar rastlantısal eğitim varyansından kaynaklanabilir.

2. **KD sonuçları yalnızca logit-based yöntem içermektedir.**  
   Feature-based ve relation-based altyapısı olsa da nihai sonuç tablosunda bu türlere ait koşu yoktur.

3. **Multi-teacher sonuçları yalnızca uniform aggregation içerir.**  
   Validation-weighted ve manual aggregation çalıştırılmadan aggregation yaklaşımı karşılaştırılamaz.

4. **Üçlü teacher sonucu yoktur.**  
   İki teacher ile üç teacher arasındaki marjinal katkı bilinmemektedir.

5. **İstatistiksel test için örnek bazlı tahminler saklanmamış veya bu analizde mevcut değildir.**  
   Accuracy/Macro-F1 özetleri tek başına paired bootstrap veya McNemar testi yapmaya yetmez.

6. **KD yalnızca tek dataset/student üzerinde denenmiştir.**  
   Bulguların başka veri seti veya öğrenci mimarisine genellenebilirliği gösterilmemiştir.

7. **Teacher seçimi için yalnızca teacher test skoru yeterli açıklama değildir.**  
   Teacher calibration, student–teacher agreement ve hata çeşitliliği analiz edilmelidir.

### Olumlu yönler

- Aynı dataset, split, seed ve student mimarisi üzerinde doğrudan karşılaştırma yapılmıştır.
- Accuracy yanında sınıf dengesizliğine daha duyarlı Macro-F1 raporlanmıştır.
- Sınıf bazlı metrikler, confusion matrix, history, config ve checkpoint artifact'ları korunmuştur.
- Student ve teacher hesaplama maliyetleri raporlanmıştır.
- KD kazancı özellikle zor `complex` sınıfında belirgindir.
- Single ve multi-teacher KD aynı deney altyapısında karşılaştırılabilmektedir.

## 11. Bu sonuçlardan bildiri olur mu?

### Kısa yanıt

**Evet, olur; fakat mevcut haliyle daha çok uygulamalı bir ulusal konferans, workshop veya kısa bildiri düzeyindedir.** Güçlü bir uluslararası konferans/dergi makalesi için ek deneyler gereklidir.

### Savunulabilir araştırma hikâyesi

Önerilen ana hikâye:

> Bitki hastalığı sınıflandırmasında büyük CNN teacher'lardan MobileNetV3-Small'a bilgi aktarımı, hafif student'ın Macro-F1 skorunu artırırken model boyutunu ve inference maliyetini düşük tutmaktadır. Bununla birlikte, uniform multi-teacher aggregation her zaman en iyi single teacher'dan üstün değildir; teacher doğruluğundan çok teacher–student uyumluluğu ve teacher çeşitliliği belirleyici olabilir.

Bu hikâyede iki ilginç bulgu vardır:

1. En yüksek skorlu teacher olan DenseNet201, en iyi KD student'ı üretmemiştir.
2. İki teacher kullanmak, en iyi single teacher'ı otomatik olarak geçmemiştir.

Bu iki sonuç, yalnızca “KD accuracy artırdı” söyleminden daha bilimsel ve tartışılabilir bir katkı sunar.

### Mevcut verilerle kurulabilecek iddialar

- Logit KD, incelenen iki güçlü teacher ile MobileNetV3-Small baseline'ını geliştirmiştir.
- En iyi koşuda Macro-F1 artışı **+1,81 yüzde puandır**.
- Kazanç, modelin inference parametresi veya MACs değerini artırmadan elde edilmiştir.
- En büyük sınıf bazlı katkı zor `complex` sınıfındadır.
- Uniform multi-teacher KD baseline'dan iyi, fakat en iyi single-teacher KD'den düşüktür.

### Mevcut verilerle kurulmaması gereken iddialar

- “Multi-teacher KD, single-teacher KD'den üstündür.”
- “RegNet-Y-8GF kesin olarak en iyi teacher'dır.”
- “Farklar istatistiksel olarak anlamlıdır.”
- “Yöntem farklı veri setlerine ve student mimarilerine genellenir.”
- “Feature veya relation KD logit KD'den iyi/kötüdür.”

## 12. Gönderim öncesi gerekli deneyler

### Asgari paket

1. En önemli koşuları en az **5 seed** ile tekrarla: 42, 123, 3407, 2025, 2026 gibi.
2. Her koşul için ortalama ± standart sapma ve tercihen %95 güven aralığı raporla.
3. Örnek bazlı tahminleri sakla; paired bootstrap ve/veya McNemar testi uygula.
4. Üç single-teacher koşu için aynı hiperparametre protokolünü koru.
5. Üç multi-teacher çifti için:
   - uniform,
   - validation Macro-F1 weighted,
   - manual veya açıkça tanımlanmış sabit ağırlık
   
   ablation'larını çalıştır.
6. Üçlü teacher kombinasyonunu ekle.
7. Latency ölçümünü warm-up, CUDA synchronization ve çoklu tekrarlarla standardize et.

### Güçlü makale için ek paket

1. Logit, feature ve relation KD'yi aynı teacher/student çiftlerinde karşılaştır.
2. En az bir ek student mimarisi kullan.
3. En az bir ek veri setinde KD doğrulaması yap.
4. Teacher calibration, agreement ve hata çeşitliliği analizi ekle.
5. Temperature ve alpha için küçük bir ablation yap.
6. Teacher ensemble ile distile student arasındaki aktarım boşluğunu analiz et.
7. Eğitim maliyeti ve GPU bellek kullanımını ayrıca raporla.

## 13. Önerilen bildiri çerçevesi

### Başlık önerileri

1. **Efficient Plant Disease Classification via Single- and Multi-Teacher Knowledge Distillation**
2. **When More Teachers Do Not Guarantee a Better Student: Knowledge Distillation for Plant Disease Recognition**
3. **Compressing CNN Ensembles into MobileNetV3-Small for Plant Pathology Classification**

İkinci başlık mevcut sonuçların en ayırt edici bulgusunu daha iyi yansıtmaktadır.

### Araştırma soruları

- **RQ1:** Logit KD, MobileNetV3-Small'ın Plant Pathology 2021 performansını ne ölçüde artırır?
- **RQ2:** En yüksek doğruluğa sahip teacher, en iyi student'ı da üretir mi?
- **RQ3:** Multi-teacher KD, single-teacher KD'den daha başarılı mıdır?
- **RQ4:** KD kazancı hangi hastalık sınıflarında yoğunlaşmaktadır?
- **RQ5:** Elde edilen performans artışının parametre/MACs maliyeti nedir?

### Önerilen ana tablolar ve şekiller

1. Baseline ve teacher performans tablosu.
2. Single/multi KD karşılaştırma tablosu.
3. Accuracy–parametre veya Macro-F1–MACs Pareto grafiği.
4. Sınıf bazlı F1 değişim grafiği.
5. Teacher ensemble → student aktarım boşluğu grafiği.
6. Çoklu seed boxplot veya confidence-interval grafiği.
7. Aggregation ve KD türü ablation tablosu.

## 14. Nihai değerlendirme

Mevcut sonuçlar, **hafif bir modelin KD ile anlamlı biçimde geliştirilebildiğini** ve **daha fazla teacher kullanımının otomatik üstünlük sağlamadığını** gösteren tutarlı bir ön çalışma oluşturmaktadır.

En güçlü pratik sonuç:

- MobileNetV3-Small,
- 1,52 M parametre,
- 5,86 MB,
- 55,51 M MACs,
- RegNet-Y-8GF logit KD ile **%95,29 accuracy ve %94,37 Macro-F1**.

En güçlü bilimsel tartışma ise şudur:

> Teacher performansı ile distillation etkinliği aynı şey değildir; uniform multi-teacher birleşimi, güçlü teacher'ları bir araya getirse bile en iyi single-teacher student'ı geçmeyebilir.

Bu nedenle sonuçlar **bildiriye dönüşebilir**, fakat gönderimden önce çoklu seed ve aggregation ablation'ı yapılması zorunlu kabul edilmelidir. Feature/relation KD ve ek dataset/student deneyleri eklenirse çalışma kısa bildiriden daha güçlü bir tam makale seviyesine taşınabilir.

---

## Kaynak veri dosyaları

Bu rapor aşağıdaki proje çıktılarından üretilmiştir:

- `results/baseline/baseline_results.csv`
- `results/teachers/cnn/teacher_results.csv`
- `results/knowledge_distillation/kd_results.csv`
- `results/multi_teacher_knowledge_distillation/multi_kd_results.csv`
- Yukarıdaki tablolarda işaret edilen run history ve per-class metrics artifact'ları

Orijinal sonuç dosyalarında değişiklik yapılmamıştır.
