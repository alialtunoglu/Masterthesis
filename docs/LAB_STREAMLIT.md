# KTÜ laboratuvarında deney yönetimi

Bu akış RustDesk üzerinden kullanılır. Yeni VPN, Git sonuç senkronizasyonu veya
merkezi sunucu gerektirmez. Kod, veri ve teacher checkpointleri önceden kurulmuş
`~/ali_altunoglu/MasterThesis` projesinde kalır.

## Başlatma

Ubuntu terminalinde:

```bash
source "$HOME/ali_altunoglu/miniforge3/etc/profile.d/conda.sh"
conda activate masterthesis
cd "$HOME/ali_altunoglu/MasterThesis"
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

RustDesk ile bağlandığınız Windows bilgisayarının tarayıcısında
`http://localhost:8501` açın. Uygulamayı internete açmayın. Mevcut ortamı
yeniden kurmak veya veri setini tekrar çıkarmak gerekmez.

### Küçük kod güncellemesini ZIP ile uygulama

`lab_streamlit_update_2026-10-24.zip` yalnızca değişen `app/` dosyalarını,
bu yönergeyi ve dosya hash listesini içerir. Veri/checkpoint/sonuç/ortam içermez.
ZIP'i `~/ali_altunoglu/` klasörüne taşıdıktan sonra, Streamlit açıksa yalnızca
Streamlit sunucusunu durdurun; eğitim runner'larını veya WSL'yi kapatmayın.

```bash
cd "$HOME/ali_altunoglu/MasterThesis"
cp -a app "app_before_lab_ui_$(date +%Y%m%d_%H%M%S)"
python -m zipfile -e "$HOME/ali_altunoglu/lab_streamlit_update_2026-10-24.zip" .
```

Bu güncelleme yayımlanmış `0f7bd236` kod tabanı içindir. KTÜ'de ayrıca değiştirdiğiniz
`app/` dosyaları varsa otomatik olarak üzerlerine çıkarmayın; önce yedekle farkları
karşılaştırın. Paket yalnızca uygulama kodunu günceller, çalışan native eğitim
scriptlerini değiştirmez. Güncellemeden sonra yeni koşuları notebook yerine
Streamlit'ten oluşturun; eski notebook runner'ın commit/değişiklik kontrolü ayrı kalır.

## İlk kontrol: eğitim başlatmadan

1. **Genel bakış** sayfasında iki GPU'yu ve mevcut notebook koşularını kontrol edin.
2. **Deneyler** sayfasında her koşunun GPU, dry-run/gerçek eğitim, kaynağı,
   epoch, son log güncellemesi ve metriklerini inceleyin.
3. Notebook kaydı salt okunur adapte edilir. `state.json` henüz yoksa veya PID
   kimliği doğrulanamıyorsa durum belirsiz/kesintili gösterilir; başarı varsayılmaz.
4. Notebook koşusunu durdurmak istiyorsanız kurtarma kaydı hazır olduktan sonra
   **Epoch Sonunda Kaydet ve Durdur** kullanın. Hücreyi kesmek veya kilit dosyasını
   silmek güvenli durdurma değildir. Notebook koşularında bu UI zorla kapatma,
   native silme veya native devam ettirme sunmaz.

## Kuyruk ve GPU seçimi

- Yeni deneyde **Çalıştırılacak GPU**: `Otomatik`, `GPU 0` veya `GPU 1` seçilir.
- Otomatik seçim uygun fiziksel GPU'yu ayırır. Belirli GPU doluysa deney başka
  GPU'ya taşınmaz; kuyrukta bekler. Diğer uygun GPU'ya atanmış işler ilerleyebilir.
- Her GPU'da en fazla bir koşu çalışır. Notebook runner ile aynı OS kilidi paylaşılır.
- **Ayarlar → Toplam paralel deney sınırı** değerini iki GPU'da birer koşu için
  `2` yapıp onaylayarak kaydedin. Bu global sınır, GPU başına bir koşu sınırından ayrıdır.
- **Queue Worker Başlat**, bekleyen deneyleri gerçekten başlatabilir; yalnızca
  hazır olduğunuzda tıklayın. Sayfa açılması eğitim veya worker başlatmaz.
- GPU tablosundaki uygunluk yalnızca projenin kayıt/kilit sahipliğini anlatır.
  Diğer kullanıcıların GPU işlerini bellek/kullanım değerlerinden ayrıca kontrol edin.
- GPU bulunamazsa otomatik hedef CPU'dur; config'te `device=cuda` zorunluysa
  bunu CPU/auto ile uyumlu hale getirmeden başlatmayın. GPU envanteri okunamıyorsa
  GPU ataması tahmin edilmez; hata gösterilir.

Laboratuvara taşınmış `lab_manifest.json` teacher kimlikleri mevcut KD formuna
aday olarak eklenir; checkpoint hash, sınıf sırası ve veri sürümü kontrolü
eğitim öncesinde aynen uygulanır. Yerel MLflow'da sahte teacher koşusu oluşturulmaz.

## Sonuç ve modeli kendi bilgisayarına taşıma

1. Deneyin **Tamamlandı** olmasını bekleyin; dry-run final model paketi değildir.
2. **Deneyler → Kendi bilgisayarına taşı → Taşınacak klasörü hazırla ve göster**.
3. Uygulama tam klasör yolunu gösterir. İçinde bir sonuç ZIP'i ve bir `.pt`
   checkpoint bulunur; ikisini birlikte taşıyın. Notebook runner'ın hazır
   `lab_exports/<deney>/<attempt>` klasörü varsa yeniden paketlemek yerine doğrulanır.
4. Bu klasörü kendi Windows bilgisayarınızdaki proje içinde örneğin:

   ```text
   C:\Users\alial\Desktop\Projeler\MasterThesis\runs\import_inbox\<gösterilen_klasör_adı>\
   ├── <bundle_id>.zip
   └── <bundle_id>.pt
   ```

5. Yerel Streamlit'te **Sonuçlar → İçe Aktar** sekmesinden paketi seçip onaylayın.
   Hashler/kimlikler doğrulanır, sonuç tablosuna ve yerel MLflow'a kayıt eklenir.
   Aynı bundle tekrar seçildiğinde kayıt çoğaltılmaz; kaynak run kimliği korunur.

`mlflow.db`, `lab_tracking`, aktif job/PID/kilit dosyaları ve veri seti taşınmaz.
Yerel MLflow run ID'si farklı olabilir; kaynak run ID'si ayrıca saklanır.

## Sınırlar

- İki GPU × birer koşu desteklenir; GPU başına iki deney bu sürümde yoktur.
- Tarayıcı kapanınca eğitim sürer. Windows/WSL kapanması veya elektrik kesintisi
  süreci durdurur; recovery kaydı varsa uyumlu ortamda devam ettirme gerekir.
- Notebook durum adaptörü ve GPU kilitleri aynı KTÜ proje klasöründe kullanılmalıdır.
- Eski aktif native job'ın GPU ataması bilinmiyorsa yeni GPU işleri temkinli biçimde bekler.
- Yerel diskteki recovery, ayrı bir fiziksel yedek değildir. Transfer klasörünü ikinci
  bilgisayara almak ayrıca bir kopya sağlar.
- İlerleme ilk epoch tamamlanana kadar “Hazırlanıyor / ilk epoch” olarak gösterilir;
  bu aşamada içerik kontrolü CPU/disk üzerinde sürebilir. Son log zamanı heartbeat
  değildir ve tek başına takılma kanıtı sayılmaz.
