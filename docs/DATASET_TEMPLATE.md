# Yeni veri seti ekleme

Yeni **single-label classification** verileri Python kodu yazmadan kaydedilir.
Klasör adı başlangıç adını, sınıf klasörleri gerçek sınıf adlarını verir. En az
iki sınıf gerekir. Yeni CSV/multi-label görevleri kapsam dışındadır; mevcut
PlantPathology2021 CSV/tek-etiket protokolü değiştirilmemiştir.

## Şablon

```text
datasets/Potato/images/Healthy/*.jpg
datasets/Potato/images/Early_Blight/*.jpg
datasets/Potato/images/Late_Blight/*.jpg
```

Alternatif: aynı sınıfları içeren `train/<class>/`, `val/<class>/`,
`test/<class>/` klasörleri. Hazır bölmeler yeniden karıştırılmaz. İki düzeni
aynı kökte birleştirmeyin. Alt klasörlerdeki görüntüler üst sınıfa aittir.

## Arayüz

1. **Veri Setleri** → klasörü ve düzeni seçin.
2. **Görüntüleri kontrol et** → tam görüntü okuma ve SHA-256 taraması.
3. Kimlik, görünen ad ve bölmeyi kontrol edin. Varsayılan seed42, %70/%15/%15.
4. **Kaydet ve sabit bölmeyi yayımla** → veri Yeni deney ekranında kullanılabilir.

Bozuk görüntüler, aday sürümün içindeki byte kopyaları, boş sınıflar,
case çakışmaları, symlink/junction ve root dışına yönlenen yollar reddedilir.
Her sınıf her split'te bulunmalıdır. Eğitim seed'i split seed'inden bağımsızdır.
Ham görüntüler taşınmaz, yeniden adlandırılmaz veya silinmez. Tarama disk hızına
bağlıdır ve yalnızca düğmeyle başlar; kayıt ve eğitim öncesi güncellik için
tekrarlanabilir. İçerik değiştiğinde yeni revision gerekir. Eski manifest ham
görüntü yedeği değildir; eski sürümü eğitmek için onun içeriği de bulunmalıdır.

## Değişmez kimlik

- Tanım: `configs/datasets/<id>.json`
- Manifest: `datasets/.registry/<id>/<revision>/manifest.json`
- Bölme: aynı dizindeki `splits/<split_id>.json`
- Deney referansı: `{dataset_id, revision, split_id}`
- Son runtime snapshot: `runs/configs/<job_id>.json`

Revision içerik/yol/etiketleri, split_id atamaları/oranları/split seed'ini
sabitler. Sınıf sırası deterministik, sınıf adları özgündür. Varsayılan sonradan
değişse de kuyruktaki iş aynı referansı kullanır. Worker byte hash'lerini
eğitimden önce doğrular. Eski path/label fingerprint'leri byte hash değildir;
`provenance_level=legacy` olarak ayrılır. Eski teacher yeni strict revision'a
otomatik atanmaz. Split olmadan kayıt eğitim-ready sayılmaz.

## Modeller ve config

Kayıtlı altı student ve on dört teacher modeli, CNN/ViT baseline ve single/multi
KD yollarında kullanılabilir. Yeni model eklentisi bu özelliğin kapsamında
değildir. ViT giriş kısıtları korunur. Generic JSONlar `configs/presets/`
altındadır; eski dataset/model configleri de yüklenir. Öncelik:
şablon → UI → doğrulanmış snapshot.

KD aynı revision/split, class order, preprocessing ve checkpoint hash'ine sahip
tamamlanmış teacher ister; dry/kısmi/failed kayıtlar seçilemez. Multi-KD en az
iki farklı teacher ister; çekirdekte üç üst sınırı yoktur. Uniform,
validation-weighted ve manual ağırlıklar korunur. **Teacher ve veri uyumluluğunu
doğrula** kontrolü pretrained ağırlık indirmeden CPU'da yapılır.

Eski CNN spatial feature + 1×1 Conv hedefi korunur. Diğer eşleşmeler classifier
öncesi vektör + Linear adapter kullanır; representation ayrıca kaydedilir.
Relation KD pooled vektörlerle çalışır. Adapter inference modeline eklenmez.

## Colab / Kaggle

Notebook runtime config, manifest/split ve teacher kaynak kimliklerini taşır;
ham görüntü veya model ağırlığı içermez, özel veri için indirme URL'si tahmin
edilmez. Drive/Kaggle klasörünü bağlayıp input hücresindeki `DATASET_MOUNT` ve
KD için `TEACHER_CHECKPOINTS` yollarını girin. Doğrulanan dosyalar uzak çalışma
dizinine kopyalanır; yeterli geçici disk alanı gerekir. SHA-256 uyuşmazlığı
eğitimden önce durdurur. Kaynak teacher ID'si korunur; uzak MLflow'da aranmaz.

Notebook yayımlanmış Git commit'ini kullanır. Yerel değişiklikler otomatik
taşınmaz; indirme engellenmez, fakat uzak kod yeni API'yi içermiyorsa açık hata
verilir. Bu geliştirmeyi uzakta kullanmak için ayrıca onaylanmış commit/push
ve ardından yeni notebook üretimi gerekir.

Sonuç ZIP'i ve ayrı checkpoint mevcut import akışıyla geri alınır. v2 beş
aşamayı destekler; v1 korunur. Import pickle açmaz ve yerel dataset registry'si
oluşturmaz: dış sonuç gösterilebilir, ama tek başına yerel eğitim açmaz.
Hash'ler bütünlüğü denetler; kaynak kişinin güvenilirliğini kanıtlamaz.

Sonuçlar revision/split'e göre ayrılır; kontrollü baseline kıyası student, seed
ve değerlendirme/eğitim protokolünü de denetler. Run kaldırma ve reset ham
veriyi, descriptor'ları ve paylaşılan split'leri korur. Yeni araştırma sonucu
üretilmemiştir; geçici görüntü/küçük ağ testleri ve `pretrained=False` model
forward kontrolleri gerçek GPU bellek/süre ve bilimsel başarım ölçümü değildir.
