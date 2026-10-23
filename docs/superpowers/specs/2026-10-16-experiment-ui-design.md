# Deney platformu arayüz tasarımı

## Amaç ve onaylı yön
Kullanıcı ilk bakışta deney oluşturmayı, çalışan deneyi bulmayı ve sonuçları karşılaştırmayı anlayabilmeli. Kullanıcı mevcut critique önerisinin önce planlanmasını sonra uygulanmasını istedi. Eğitim motoru, veri protokolü ve kalıcı deney kayıtları değişmeyecek.

## Bilgi mimarisi
- Dört ana giriş: Genel bakış, Deneyler, Sonuçlar, Ayarlar.
- Belirgin Yeni deney eylemi; baseline, CNN, ViT, single KD ve multi KD aynı girişten seçilir.
- Önce veri seti ve model(ler), sonra uyumlu config. Sık kullanılan ayarlar görünür; teknik ve yönteme özgü ayrıntılar açılır bölümlerde.
- Kuyruğa ekledikten sonra seçilen deneyin takibine geçilir.
- Deney kimliği ve durum, sistem ayrıntılarından önce gelir. Yalnızca geçerli eylemler ve bulunan metrikler görünür.
- Sonuç tabloları veri seti kapsamında, model ve metrikler önce olacak şekilde sunulur. Teknik kolonlar isteğe bağlıdır. Farklı veri setleri ortalanmaz.
- Worker yönetimi, GPU, split ve MLflow bilgileri Ayarlar altında toplanır.

## Görsel yön
Native Streamlit araştırma çalışma alanı; düz yüzeyler, normal tipografi, ölçülü boşluk, tek birincil eylem. Dekoratif hero, gradient, emoji navigasyon, pazarlama kartları ve animasyon eklenmez. Kontroller Türkçe, model/metric terimleri özgün biçimde kalır.

## Korunacak sözleşmeler
Uzun eğitim onayı, dry run, recovery, config doğrulaması/snapshot, notebook export, import, karşılaştırma ve güvenli kaldırma. Eski sayfa yolları erişilebilir kalır. Eğitim/worker otomatik başlatılmaz. Mevcut kullanıcı değişiklikleri korunur.

## Kabul ölçütleri
Ana menü dört öğe; Yeni deney bulunabilir. Baseline/CNN config listesi veri seti/model seçimini takip eder. KD yönü Teacher → Student olarak görünür. Tamamlanmış deneyde durdurma düğmeleri yoktur. Eksik KD metrikleri N/A kartları oluşturmaz. Sonuç karşılaştırmasında teacher-only mimari student varsayılanı olamaz. Masaüstü ve dar ekranda kontroller okunabilir.
