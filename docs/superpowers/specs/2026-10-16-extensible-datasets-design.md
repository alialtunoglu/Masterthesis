# Genişletilebilir Veri Seti ve Deney Platformu Tasarımı

Durum: Kullanıcının istediği plan için taslak; uygulama onayı bekleniyor.
Tarih: 2026-10-16.

## Amaç ve kabul edilen kapsam

Yeni bir klasör tabanlı, single-label görüntü sınıflandırma veri seti eklendiğinde
Python koduna veri seti adı yazmadan Student/Teacher baseline, Single-teacher KD
ve Multi-teacher KD hazırlanabilsin. Veri seti ismi kök klasörden, sınıflar alt
klasörlerden önerilsin. Bu adlar semantik bitki/hastalık tanıma sonucu değildir.
Keşif, doğrulama ve açık kayıt adımı olmadan eğitim başlamaz.

Mevcut Student/Teacher ve CNN/ViT ayrımı korunur. Mevcut model kataloğundaki
modeller için kabiliyetler doğrulanır; yeni bir Python modeli, yalnızca klasör
ekleyerek desteklenmiş sayılmaz. İlk sürüm yeni multi-label CSV, detection,
segmentation, rastgele kullanıcı kodu/model eklentisi, grid search veya yeni KD
loss ailesi içermez. Mevcut PlantPathology CSV işleme protokolü aynen korunur.

## Kaynak kodda doğrulanan başlangıç durumu

| Konu | Mevcut kaynak ve bulgu |
|---|---|
| Dataset kataloğu | `src/datasets/dataset_registry.py`: üç statik kayıt |
| Klasörden sınıf | `imagefolder_utils.collect_imagefolder_samples`: mevcut yardımcı yeniden kullanılabilir |
| Split üretimi | `create_splits.stratified_split`: sınıf bazında deterministik; oranlar global sabit |
| Eğitim girdisi | `torch_datasets.SplitJsonImageDataset`: JSON split ve dinamik `classes` kullanıyor |
| KD kısıtları | `kd_config.py`, `multi_kd_config.py`: PlantPathology, MobileNetV3-S ve üç teacher |
| KD ekranları | sayfa8/9: dataset, split, teacher CSV ve config kökü sabit |
| Model desteği | `model_catalog.py`/`model_factory.py`: CNN/ViT ve iki DeiT student mevcut |
| Feature desteği | `kd_features.py`: dört doğrulanmış katman; adapter yalnızca NCHW CNN |
| Logit engeli | `train_kd.main`: logit koşusunda dahi varsayılan feature katmanı çözülüyor |
| Provenance | `run_metadata.dataset_fingerprint`: görüntü byte'ları değil yol/etiket hash'i |
| Aktarım | `portable_notebook`, `external_bundle`, `external_run_import`: veri/model/stage sınırları |
| Kurtarma | `recovery_metadata`/`validate_resume`: mevcut protokol ve teacher hash kontrolleri korunmalı |

## 1. Veri seti şablonu ve keşif

Temel şablon:

```text
datasets/Potato/
  images/
    Healthy/*.jpg
    Early_Blight/*.jpg
    Late_Blight/*.jpg
```

İkinci kabul edilen şablon, hazır bölmeleri olan aynı single-label yapıdır:

```text
datasets/Potato/
  train/<class>/*
  val/<class>/*
  test/<class>/*
```

İki şablonu desteklemek tasarım önerisidir: hazır test bölmesi yeniden rastgele
bölünmez. `val` adı kanoniktir; `validation` dizini UI'da açık eşleme ile kabul
edilebilir, sessiz tahmin edilmez. Karışık/iki anlamlı yerleşim kayıt olamaz.

`Veri Setleri` sayfasındaki `Klasörleri tara` yalnızca `datasets/` altındaki
adayları listeler; gizli `.registry`, mevcut kayıtlar ve görüntü dışı klasörler
veri seti sayılmaz. Kullanıcı bir adayı seçip `Doğrula` der. Kök klasör adı
görünen ad olarak önerilir, düzenlenebilir. Kalıcı `dataset_id` güvenli ASCII
slug'dır; çakışma doğrulanır, klasör adı değişince yeni kimlik otomatik üretilmez.

## 2. Doğrulama ve kalıcı kayıt

Yeni verilerde en az iki dolu sınıf, desteklenen görüntü uzantısı ve gerçekten
okunabilir görüntü gerekir. RGB dönüşümü mevcut loader'da korunur. Sınıf isimleri
birebir saklanır; çevrilmez veya birleştirilmez. Sıra `(casefold, original)` ile
deterministiktir; büyük/küçük harf çakışan sınıflar platformlar arası belirsizlik
nedeniyle reddedilir. Kök dışına çıkan symlink/junction ve path traversal reddedilir.

Doğrulama sınıf sayımları, dosya boyutu, SHA-256 ve okuma hatalarını raporlar.
Byte bazında aynı görüntü aday revision içindeki birden fazla sample'da veya
etikette bulunursa kayıt engellenir; kullanıcıya dosyalar gösterilir. Bu kontrol
diğer veri setlerini veya önceki revision'ları kapsamaz; değişmeyen görüntülerin
yeni revision'da yeniden kullanılması normaldir. Hiçbir görüntü sessizce silinmez veya
hariç tutulmaz. Farklı JPEG kodlamalı/görsel olarak benzer kopyalar için perceptual
duplicate detection bu sürümde yoktur. Uzun taramada ilerleme ve iptal gösterilir;
yarım doğrulama `ready` kaydı üretemez.

Küçük dosya tabanlı kayıt sistemi yeterlidir; yeni veritabanı gerekmez:

- `configs/datasets/<dataset_id>.json`: schema_version, id, display_name,
  root_relative_path, layout, task_type, default_revision/default_split. İlk
  içerik kaydında default referansları null olabilir; `ready` ancak geçerli bir
  default revision+split'e işaret edildiğinde true'dur.
- `datasets/.registry/<id>/<revision>/manifest.json`: göreli sample path,
  sınıf/indeks, boyut, content_sha256, sınıf eşlemesi ve rapor özeti.
- Aynı revision altında `splits/<split_id>.json`: mevcut loader ile uyumlu split
  payload ve ek dataset kimliği alanları.

Revision, kanonik göreli sample path + label + content hash listesinden türetilir;
mutlak makine yolu/görünen ad/zaman damgası kimliğe girmez. `class_mapping_hash`
sınıf sırasını kapsar. Split kimliği revision, assignment, yöntem, oran ve split
seed'ini kapsar. Yeniden kayıt aynı girdide idempotenttir; içerik/sınıf değişince
yeni revision gerekir. Kayıt dosyaları geçici dosya + atomic replace ile
yayımlanır; mevcut immutable revision üzerine yazılmaz. Aynı dataset_id için
register/default-publication işlemleri işletim sistemi dosya kilidiyle
serileştirilir; iki UI oturumu aynı kimliği farklı köklere sessizce bağlayamaz.
Mevcut result store'un standart kütüphane lock/atomic-write yaklaşımı örnek alınır.
İçerik kaydı
`DatasetRevisionRef(dataset_id, revision)` üretir, henüz eğitim açmaz. Split
başarıyla yazıldıktan sonra descriptor'ın default referansları tek atomic
güncellemeyle yayımlanır ve `DatasetRef(dataset_id, revision, split_id)` oluşur.
UI'daki son `Kaydet ve kullan` bu iki adımı koordine eder; split hatasında içerik
kaydı `validated/not-ready` kalır ve eğitim seçeneklerine girmez. Commit anında taranan
dosyaların değişmediği yeniden doğrulanır. Görüntüler taşınmaz/kopyalanmaz.

## 3. Split protokolü

Varsayılan oranlar 70/15/15, split seed42; kullanıcı kayıt öncesi değiştirebilir.
Oranlar pozitif ve toplam1 olmalı. Mevcut largest-remainder yaklaşımı oranları
parametre olarak alacak şekilde genişletilir; legacy çağrıların sayımları değişmez.
Her sınıf üç split'te bulunamıyorsa kayıt engellenir ve yetersiz sınıf açıklanır.
Hazır bölmelerde assignment aynen korunur, sınıflar ve hash disjointness kontrol
edilir. Augmentation yalnızca train yüklenirken, split sonrasında uygulanır.

Experiment training seed ile split seed ayrıdır. Bir veri revision'ının birden
çok split protokolü olabilir; deney hangi split'i kullandığını açıkça pin'ler.
Baseline/teacher/KD karşılaştırmaları aynı revision + split + class mapping ve
değerlendirme protokolü üzerinden yapılır. Eğitim başlamadan önce içerik hash'leri
doğrulanır; verinin daha sonra değiştirilmemesi gerektiği kullanıcıya belirtilir.
Kuyruğa alındıktan sonra registry'deki varsayılan split değişse bile job değişmez.

## 4. Mevcut veriler ve geçmiş deneyler

Üç mevcut dataset adapter'ı, split dosyaları, numaraları, sınıf sıraları ve mevcut
özel PlantPathology single-label işlemi korunur. Önceki CSV, checkpoint veya
MLflow kayıtları topluca yeniden yazılmaz. Statik `DATASETS` public import'u
uyumluluk için kalır; aktif akışlar root parametreli registry çözümleyicisini
kullanır. Böylece Streamlit'i yeniden başlatmadan kayıt yenilenebilir.

Mevcut yol/etiket fingerprint'i geriye dönük içerik doğrulaması gibi sunulmaz.
Legacy kayıtların provenance seviyesi açıkça `legacy` olur. Mevcut teacher
seçimi eski veri/split sınırları içinde class order ve checkpoint kontrolleriyle
çalışmaya devam eder; legacy teacher yeni revision'a otomatik taşınmaz.
İçerik provenance'ı belirsiz geçmiş kayıtlarla yeni strict kayıtlar aynı deney
grubuna otomatik birleştirilmez. Legacy UI geçmişi görünür kalır.

## 5. Model ve KD kabiliyetleri

Model kataloğu rol, family, giriş boyutu kısıtı ve feature tanımı taşıyan tek
kaynak olur; mevcut public model setleri/factory API'leri korunur. Bu çalışmada
yeni model ailesi eklenmez. Kayıtlı tüm student/teacher modelleri dinamik sınıf
sayısı ile baseline ve logit KD kullanabilmelidir. Teacher rolü, model family'den
ayrıdır; rol bazındaki mevcut seçenekler korunur.

Feature/relation desteği mevcut dört CNN ile sınırlı bırakılmaması hedefiyle tüm
kayıtlı mimariler için doğrulanır. Registry her modelin çıktısını NCHW, NHWC,
tokens veya vector olarak açıkça tanımlar; boyutlardan tahmin yürütmez. Yeni
katman/pooling tanımları gerçek `pretrained=False` forward testleriyle doğrulanır.

- Mevcut CNN feature KD: aynı katmanlar, Conv1x1 ve spatial alignment korunur.
- Yeni ViT/cross-family feature KD: doğrulanmış pre-classifier/pooled `[B,D]`
  temsil ve train-only Linear adapter kullanılır; CLS/mean pooling model
  tanımında açıktır. Sessizce CNN haritası gibi yorumlanmaz.
- Relation KD: doğrulanmış pooled vektörler üzerinden ilişkiler.
- Representation ve adapter türü ayrı metadata'dır; farklı hedefler aynı feature
  protokolü gibi raporlanmaz. Her multi-teacher için ayrı adapter vardır.
- Logit KD feature katmanına bağımlı olmaz; yalnızca ihtiyaç varsa feature probe.
- Geçerli ortak input/preprocessing yoksa kombinasyon açıklama ile engellenir;
  ilk varsayılan mevcut224 girişidir, ViT'nin gerçek boyut kısıtı korunur.

Teacher frozen/eval/no_grad, student tek forward ve adapter gradientleri korunur.
Multi-teacher en az2, üst sınır mevcut uygun farklı model sayısıdır; sabit3 yok.
Mevcut farklı mimari şartı ve uniform/validation/manual aggregation korunur.

## 6. Teacher koşusu uyumluluğu

Ortak servis CNN ve ViT teacher sonuçlarını birlikte inceler. Sadece tamamlanmış,
dry-run/kısmi batch olmayan, checkpoint'i bulunan teacher baseline koşuları adaydır.
Yeni strict koşularda dataset_id/revision, split_id, class_mapping_hash/sıra,
logit boyutu, checkpoint hash'i ve gerekli preprocessing/feature kabiliyeti
eşleşmelidir. Aynı görüntü sayısı veya benzer dataset adı yeterli değildir.
Teacher'ın training seed'i farklı olabilir; split kimliği eşleşmelidir.

Yeni veri setinde teacher yoksa KD ekranı seçimi saklamaz: uyumluluk nedeni ve
`Teacher baseline hazırla` bağlantısı görünür; Başlat/Dry run pasif olur. En az
iki uygun farklı teacher olmadan multi-KD başlamaz. UI ve CLI aynı doğrulayıcıyı
kullanır; istemci seçimi güven sınırı değildir.

## 7. Config ve runtime kimliği

Her dataset×model×teacher kombinasyonu için yeni JSON üretmek yerine yeniden
kullanılabilir model/eğitim/KD presetleri ile dataset referansı birleştirilir.
Eski dataset-specific presetler yüklenmeye devam eder. Dinamik preset örneği
dataset veya checkpoint uydurmaz; teacher run seçimi runtime snapshot'a yazılır.

Ortak ek referans `dataset_ref={dataset_id, revision, split_id}`. Legacy
`dataset_name` korunur. Preset → UI → doğrulanmış immutable runtime config sırası
korunur. Job metadata, MLflow, run_metadata, checkpoint, recovery ve sonuç satırı
bu kimliği taşır. Yeni fingerprint şema sürümü eklenir; v1 fonksiyonunun anlamı
geçmiş kayıtlar için değiştirilmez. Resume farklı revision/split/representation
ve teacher checkpoint'ini reddeder.

## 8. Arayüz

Mevcut sade native Streamlit dili korunur; yeni tasarım dünyası/customCSS yok.
`Veri Setleri` beşinci ana menü öğesi olur. Bu bir tasarım önerisidir; veri ekleme
günlük temel iş akışı olduğu için Ayarlar içine gömülmez.

- Veri Setleri: adaylar → doğrulama/dağılım → split önizleme → açık kayıt.
- Yeni deney: dataset/revision/split → eğitim türü → mevcut role/family/model
  akışı veya student/teacher koşuları → uyumlu preset → doğrulama → başlat/notebook.
- Tüm launcher'lar registry'yi kullanır; yeni ad eklemek için sayfa kodu düzenlenmez.
- Deneyler filtreleri kayıtlı job'lardan, Yeni deney seçenekleri ise registry ve
  kabiliyetlerden beslenir. Mevcut dependent-filter davranışı korunur.
- Sonuçlar aynı isimli farklı revision/split'leri ayırır; provenance yetersizse
  otomatik kazanım/baseline eşlemesi yapılmaz.

## 9. Notebook, aktarım ve güvenlik

Mevcut üç veri setinin indirme sağlayıcıları korunur. Özel verinin internette
kaynağı tahmin edilmez ve veri Git'e otomatik yüklenmez. Yeni veri için notebook
Drive/mounted-folder veya Kaggle Add Input yolu ister; bir upload/ZIP sunucusu
bu sürümde yoktur. Manifest ve split notebook'a taşınır, dataset kökü yeniden
eşlenir ve içerik doğrulanır; bölmeler yeniden rastgele üretilmez.

Single/multi KD notebook'ları teacher checkpoint'lerini Git'te varsaymaz.
Her teacher için uzak mount yolu ve beklenen SHA-256 gerekir. Yerel run ID'leri
taşınabilir kaynak kimliği olarak korunur; uzak ortamda aynısının MLflow'da
bulunması gerekmez. Eksik veri/checkpoint eğitimden önce açıklanır. Uzak kodun
gereken dataset/bundle şema kabiliyetini taşıdığı başlangıç hücresinde kontrol
edilir; eski commit'te anlaşılır güncelleme hatası verilir. Bu kontrol yerel dirty
Git durumunu notebook indirme yasağına dönüştürmez. Kimlik
bilgileri notebook'a gömülmez; mevcut Drive recovery ve canlı loglar korunur.

Bundle şeması v2 dataset manifest/split ve tüm beş stage'i kapsar. v1 legacy
baseline/teacher import'u korunur. Import dataset/model doğrulamasını kaldırmaz:
schema, hash, class mapping, stage path, ZIP boyut/member/symlink kontrolleri
korunur; dataset_id/run_name path traversal için kullanılmaz. Import sırasında
checkpoint unpickle veya kullanıcı kodu çalıştırılmaz.

Yerel görüntüler olmadan gelen v2 dış koşu Sonuçlar'da metadata üzerinden
gösterilebilir; eğitim yapılabilir dataset olarak otomatik kaydedilmez. Yerel
registry'de aynı id farklı içerikse overwrite yoktur. Eşleşen veri kaydedilmeden
o dış run yerel KD öğretmeni olamaz. Teacher transferi için eşleşen checkpoint
ve tam dataset/split doğrulaması gerekir.

## 10. Veri koruma ve sınırlar

Run silme/arşivleme raw dataset, registry manifest ve ortak split dosyalarını
kapsamaz. Reset dataset tanımlarını ve registry'yi korur. Bu çalışmada dataset
silme, otomatik dosya taşıma, internetten özel veri arama veya yeni bağımlılık
yoktur. Kaydedilmiş manifest immutable'dır; görüntüler yerinde kaldığı için
eğitim sırasında elle değiştirilmemelidir. Yeni sürüm üretmeden eski kimlikle
değişmiş içerik çalıştırılamaz.

## Kabul senaryosu

Geçici üç sınıflı Potato fixture'ı kodda adı tanımlanmadan keşfedilir ve kaydedilir.
Student CNN/ViT, Teacher CNN/ViT baseline; logit/feature/relation single ve multi
KD aynı pin'lenmiş split ile kısa CPU fixture testlerinde çalışır. Teacher yok,
tek teacher var, class order farklı, içerik değişmiş veya eski revision kullanılmış
senaryoları anlaşılır hatayla durur. Değişen varsayılanlar queued job'ı değiştirmez.
Notebook farklı köke taşınınca aynı hash/split doğrulanır; v1/v2 aktarım, recovery,
karşılaştırma ve güvenli kaldırma regresyonları geçer. Gerçek araştırma eğitimi
kullanıcının ayrı seçimi olmadan başlatılmaz.
