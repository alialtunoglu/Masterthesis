# Eğitim Kurtarma ve Deney Karşılaştırma Uygulama Planı

> Uygulama sırasında `superpowers:executing-plans` ile görev görev ilerle. Bu belge planlama çıktısıdır; eğitim, kod uygulaması, commit, merge veya push yetkisi vermez.

**Amaç:** Kesilen deneyleri son tamamlanan epoch'tan sürdürebilmek, kurtarma verilerini kalıcı konuma aktarmak ve doğrulanabilir deney karşılaştırmalarından makale tabloları üretmek.

**Mimari:** Mevcut üç trainer korunur; ortak kurtarma dosyası işlemleri küçük bir modülde paylaşılır. Karşılaştırma hesapları Streamlit'ten bağımsız tutulur ve mevcut Results Explorer'a bir sekme olarak bağlanır. Yeni servis, bulut API'si veya veritabanı kurulmaz.

**Teknoloji:** Mevcut Python/PyTorch, pandas, Streamlit, MLflow ve Python standart kütüphanesi. Test ortamı `conda activate masterthesis`.

**Tasarım kaynağı:** Kullanıcının önceki önerilerdeki 1 ve 3 numaralı maddeleri seçmesi; bu belgenin aşağıdaki kararları uygulama için önerilen tasarımı oluşturur.

**Uygulama durumu (2026-09-13):** Kod ve izole otomatik testler tamamlandı (132/132). Gerçek Google Drive/Colab/Kaggle uçtan uca koşusu yapılmadı; eğitim veya kullanıcı sonuçları üzerinde değişiklik yapılmadı.

## 1. Mevcut koddan doğrulanan başlangıç

- Baseline ve CNN/ViT teacher aynı `src/training/trainer.py:fit()` fonksiyonunu kullanır. Single-KD ve multi-KD ayrı trainer'lardır.
- Mevcut best checkpoint formatları farklıdır: standart model `model_state_dict`, KD `student_state_dict` ve `strategy_state_dict` taşır. Hiçbiri tam optimizer/scheduler/RNG kurtarma durumu sağlamaz.
- `src/datasets/dataloaders.py` train shuffle kullanır; worker sayısı pozitifken `persistent_workers=True` yapar. Worker rastgeleliği ayrıca ele alınmadan birebir devam iddiası yapılamaz.
- `app/experiment_runner.py:stop_job()` süreçleri terminate/kill eder. Güvenli epoch sonu durdurma mevcut değildir.
- History CSV'leri eğitim çağrısından sonra eğitim giriş scriptlerinde yazılır.
- `app/portable_notebook.py` Drive'a eğitim sonu sonuç paketini yazar; eğitim sürerken kurtarma yedeği almaz.
- Harici paketleme/import yalnızca baseline, CNN teacher ve ViT teacher kabul eder.
- Results Explorer beş ana sonuç kaynağını zaten biliyor. Yeni bağımsız dashboard yerine buraya karşılaştırma sekmesi eklenebilir.
- Mevcut multi-KD özeti ensemble test accuracy/F1 ve teacher kimliklerini taşır. Eksik ensemble sınıf bazlı sonuçlar varmış gibi gösterilmeyecek.

## 2. Kapsam ve temel kararlar

### İş paketi A: Kurtarma

- Yerel baseline, CNN, ViT, single-KD ve multi-KD desteklenir.
- Mevcut baseline/CNN/ViT notebook export'una kurtarma eklenir. KD notebook export'unu sıfırdan geliştirmek bu planın kapsamında değildir.
- Varsayılan kayıt sıklığı her tamamlanan epoch; kayıt tam validation, best seçimi, scheduler adımı ve early-stopping güncellemesinden sonra yapılır.
- Epoch ortasındaki çalışma kaybolabilir. İlk sürüm batch ortasından devam etmez.
- `best.pt` değerlendirme/inference içindir; kurtarma kaydı son eğitim durumudur. Best dosya tam resume kaydı gibi yüklenmez.
- Toplam epoch hedefi başlangıç config'inde sabittir. Resume sırasında LR, optimizer, scheduler, batch, KD ağırlıkları veya hedef epoch değiştirilmez. Bunları değiştirmek yeni deneydir.
- Eski best-only checkpoint'ler görüntülenmeye devam eder; tam resume seçeneği kapalı olur. Eğitim sırasında çalışan eski süreçlere canlı özellik eklenmez.

### İş paketi B: Karşılaştırma

- Results Explorer'a `Karşılaştırma ve Rapor` sekmesi eklenir.
- Başlangıçta beş aşamanın tamamlanmış, tam değerlendirmeli sonuçları ve kayıtlı ensemble metrikleri karşılaştırılır.
- Dry-run, batch-limitli smoke test, stopped/failed ve silinmiş kayıtlar ana sonuç tablosundan dışlanır; nedenleri ayrıca gösterilir.
- Yeni multi-seed eğitim, güven aralığı, anlamlılık testi, grid search, Grad-CAM ve yeni KD yöntemleri kapsam dışıdır.
- Çıktılar Markdown, CSV, IEEE tablosuna uygun LaTeX ve kaynak manifesti JSON'dur. Tam makale yazımı veya otomatik bilimsel başarı iddiası yoktur.

## 3. Kurtarma durumu sözleşmesi

`schema_version=1` kurtarma snapshot'ı şunları saklar:

| Grup | İçerik |
|---|---|
| Kimlik | experiment_id, attempt_id, parent_attempt_id, stage, run_name, kaynak commit, MLflow run ID |
| İlerleme | tamamlanan epoch, toplam epoch, phase: training/finalizing/completed |
| Model | son student/model state; varsa strategy/adapter state; en iyi modelin snapshot içindeki ayrı best dosyası |
| Eğitim | optimizer, scheduler, varsa AMP scaler state; best metric/epoch; patience sayacı; history |
| Rastgelelik | Python, NumPy, Torch CPU, varsa CUDA RNG ve DataLoader generator durumları |
| Protokol | immutable runtime config, sınıf sırası, split fingerprint, transform tanımı, dataset kimliği |
| Teacher | model/run kimlikleri, checkpoint SHA-256, feature katmanları ve normalized ağırlıklar |
| Süre | tamamlanan çalışmanın süreleri, ölçüm donanımı ve precision bilgisi |

Teacher ağırlıkları her epoch kurtarma dosyasına kopyalanmaz. Resume öncesi gerekli teacher checkpoint'leri aynı hash ile bulunmalıdır; aksi halde başlatma engellenir. Taşınan kurtarma dosyasının teacher bağımlılıkları UI'da listelenir.

### Güvenli kayıt ve yedek

- Her snapshot bir generation dizinidir: `recovery/<attempt_id>/epoch-000012/` içinde state, best, history, config ve checksum manifesti.
- Önce geçici dosyalar yazılır ve kapatılır; checksum'lar doğrulanır; en son manifest yayımlanır. Geçersiz/yarım generation yüklenmez.
- Yerelde aynı filesystem üzerinde atomic replace kullanılır. Bağlı Drive üzerinde rename'in uzak sunucuya atomik/kalıcı ulaşım garantisi olduğu iddia edilmez.
- Yedek hedefi isteğe bağlı bağlı klasördür. Colab'da `MyDrive/MasterThesis/recovery/<experiment_id>/`; final bundle mevcut bundles dizininde kalır.
- Yerel ve yedek tarafta son iki doğrulanmış generation saklanır. Yeni yedek doğrulanmadan eskisi temizlenmez. Temizleme yalnızca ilgili attempt'in yönetilen dizininde yapılır.
- UI yerel checkpoint epoch'unu ve doğrulanmış yedek epoch'unu ayrı gösterir. Hash kontrolü geçmiş olmak uzak depolama hizmetinin kalıcılık garantisi olarak sunulmaz.
- Varsayılan yedekleme senkron ve epoch sonundadır; ek süre raporlanır. İlk sürüm arka plan kopyalama kuyruğu içermez.
- Yedekleme hatası yerel checkpoint'i silmez. Eğitim yerel kayıtla sürebilir, hata loga/UI'a yazılır ve sonraki epoch'ta yeniden denenir. Colab'da yedek gerekli seçilmişse ilk bağlantı/yazma kontrolü geçmeden eğitim başlamaz.
- Canlı `mlflow.db` dosyası Drive'a kopyalanmaz. Kurtarma manifesti/history esas alınır; tamamlanmış sonuç paketi mevcut import akışını kullanır.

### Resume ve run kimliği

- Her devam başlatışı yeni job, attempt, run dizini ve MLflow run oluşturur. Hepsi değişmeyen `experiment_id` ile bağlanır; eski FAILED/KILLED kaydı yeniden FINISHED yapılmaz.
- Resume kaynağı snapshot ve best state yeni attempt'e alınır; kaynak attempt sonradan arşivlendiğinde child kurtarma durumu bozulmaz. Teacher dosya bağımlılıkları ayrıca doğrulanır.
- Yeni attempt metrik adımları global epoch numarasıyla sürer; history eski ve yeni epoch'ları tekilleştirir. Kaynak MLflow erişimi yeni ortamda zorunlu değildir.
- Aynı experiment için iki aktif resume kuyruğa alınmaz; enqueue ve worker başlangıcında kontrol edilir.
- Snapshot'ın epoch'u `k` ise `k+1` ile başlanır. Early stopping zaten sağlanmışsa veya epoch hedefi bitmişse yalnızca eksik final değerlendirme/artifact adımı tamamlanır.
- Finalizasyon yeniden çalışabilir: sonuç CSV'si attempt/run anahtarıyla idempotent yazılır; aynı run için çift sonuç satırı oluşturulmaz.
- Donanım değişiminde cihaz state'leri taşınır ve değişiklik raporlanır; farklı CUDA/donanımda bit düzeyi eşitlik vaat edilmez.
- Model/dataset/class order/split/transform/config/teacher hash farkı engelleyicidir. Commit veya bağımlılık sürümü farkında varsayılan resume engellenir; ilk sürüm uyumsuzluğu aşma seçeneği içermez.

### RNG ve worker kararı

Kurtarma etkin yeni deneylerde epoch bazlı seed ve açık DataLoader generator kullanılır. Worker init seed'leri tanımlanır; worker'ların gizli RNG durumunu taşımak yerine bu modda persistent worker kapatılır. Kesintisiz referans koşu da aynı modu kullanır. Eski deneylerin varsayılan akışı sessizce değiştirilmez; kurtarma modu/protokol sürümü config'e kaydedilir.

## 4. Kullanıcı arayüzü

### Eğitim sayfalarında `Kayıt ve Kurtarma`

- `Epoch sonunda kurtarma kaydı`: yeni deneylerde açık.
- `Kalıcı yedek klasörü`: isteğe bağlı; Colab'da Drive bağlantısından çözülür.
- `Yedek olmadan başlatma`: Colab'da varsayılan kapalı, yerelde açık.
- `Yeni eğitim / Kaydedilmiş eğitimden devam`: resume seçiminde config kilitlenir ve doğrulama paneli açılır.
- Kurtarma preview checkpoint boyutu, epoch, best epoch, dataset/student/teacher, gerekli dosyalar ve config eşleşmesini gösterir.
- CLI standart girişlerde `--resume-from` ve `--backup-dir`; KD şemalarında aynı anlamda `recovery` bölümü. Nihai runtime snapshot her aşamada bu bilgileri taşır.

### Job Monitor

- Son yerel kayıt, son yedek epoch/zamanı, yedekleme hatası, resume parent ve attempt sayısı.
- `Epoch sonunda kaydet ve durdur`: stop-request dosyası; trainer epoch sonunda snapshot/yedek işleminden sonra çıkış yapar. UI isteği bekleyen durum olarak gösterir, uzun süren epoch'ta başarı iddiasında bulunmaz.
- `Zorla durdur`: mevcut terminate/kill akışı korunur; tamamlanmamış epoch ve devam eden yedeğin kaybolabileceği açıkça belirtilir.
- stopped/failed + geçerli snapshot için `Devam et`; cancelled + başlamamış job için bu seçenek yoktur.
- Normal sonuç, kontrollü durdurma ve hata MLflow'da sırasıyla FINISHED, KILLED, FAILED ile ayrılır. Zorla sonlandırılan süreç için yerel izleyici erişebildiği MLflow kaydını uzlaştırır; uzaktaki ani runtime kaybında anlık güncelleme garanti edilmez.
- Silme/arşivleme yerel kurtarma dosyalarını kapsar; aktif resume kaynağı korunur. Drive yedekleri varsayılan silinmez, bu durum kaldırma önizlemesinde belirtilir.

## 5. Karşılaştırma veri sözleşmesi

Yeni kayıtlar ortak metadata taşır; eski CSV'ler değiştirilmeden okunur. Kaynak sırası: run'a ait immutable metadata/config/artifact, sonra mevcut summary alanları. Çelişki varsa sessizce bir değer seçilmez.

Ortak alanlar: stage, dataset, class mapping, canonical split hash, train/eval transform tanımı, eğitim seed'i, model/student, teacher run listesi, KD türü/ağırlıklar, tamamlanma/dry-run/limit bilgisi, test ve validation metrikleri, params/MACs/FLOPs/model boyutu/latency, hardware/precision, experiment_id/attempt_id, kaynak dosya ve commit.

- Split hash mutlak Windows/Colab kökünden bağımsız hesaplanır: göreli örnek kimliği, split üyeliği, etiket ve sınıf sırasının canonical gösterimi kullanılır. Ham split JSON hash'i ayrıca saklanabilir; tek başına portable eşleşme ölçütü değildir.
- Dataset adı eşitliği içerik eşitliği sayılmaz. Dataset fingerprint/manifest kanıtı yoksa veri sürümü doğrulanamadı etiketi eklenir; tüm görüntüleri her UI yenilemesinde hash'lemek yoktur.
- Model adı teacher ve student arasında normalize edilir fakat kaynak isimler korunur. Seed, split ve protokol farkları model adına göre groupby ile gizlenmez.
- Eksik veriler `N/A`; eksik süre/FLOPs/ensemble per-class metrikleri üretilmez veya sıfırla doldurulmaz.
- Yeni kurtarma denemeleri tek seed tekrarı gibi sayılmaz. Aynı experiment'in nihai tamamlanmış attempt'i seçilir; birden fazla tamamlanmış dal varsa kullanıcıya seçim yaptırılır.

### Uyumluluk seviyeleri

1. **Değerlendirme uyumlu:** aynı veri sürümü, test split, class mapping ve eval preprocessing. Genel model performans tablosuna alınabilir.
2. **Kontrollü baseline/KD eşleşmesi:** yukarıdakilere ek aynı student, seed ve ortak eğitim protokolü (augmentation, initialization, epoch bütçesi, optimizer/scheduler, batch, monitor). KD'ye özgü teacher/loss farkları beklenen müdahaledir.
3. **Kısmen doğrulanmış/eski kayıt:** eksik metadata açık uyarıyla gösterilir; doğrulanmış deneylerle otomatik birleştirilmez. Kullanıcı keşif tablosuna alabilir; raporda sınırlılık notu zorunludur.

Genel performans farkı hesaplanabilir; fakat ortak eğitim ayarları farklıysa fark yalnızca KD'nin etkisi diye etiketlenmez.

### Seçim ve metrik kuralları

- Kullanıcı dataset, student, seed, KD türü, teacher kombinasyonu ve run/job ID ile seçim yapar.
- Otomatik baseline seçimi yalnızca uyumlu havuzdan en yüksek validation macro-F1 ile yapılır; eşitlikte sabit run ID sırası kullanılır. Seçilen run kimliği ve aday sayısı kaydedilir. Test skoru otomatik seçim ölçütü değildir.
- Single-KD ve multi-KD karşılaştırmasında teacher checkpoint kimliği ve KD türü eşleşmesi aranır; farklı aggregation/hiperparametreler ayrı satırlarda belirtilir.
- Teacher satırları seçilmiş KD run'ın gerçek teacher run ID'lerinden gelir; aynı mimarinin başka run'ıyla sessizce değiştirilmez.
- Ensemble sonucu kaydedilmiş ensemble değerlendirmesinden gelir; teacher accuracy/F1 ortalaması alınmaz. Aynı teacher ve ağırlıklar farklı kayıtlarda çelişen ensemble sonucu taşıyorsa kaynaklarıyla gösterilir.
- Delta sütunları yüzde puandır: `100 * (KD_F1 - baseline_F1)`. Yüzde artışla karıştırılmaz.
- Compression oranı açık pay/payda ile tanımlanır: teacher inference parametresi / student inference parametresi. Multi-teacher'da teacher toplamı ayrıca etiketlenir.
- `FLOPs = 2 × MACs` tahmini etiketi korunur. Training adapter parametreleri student inference parametrelerine katılmaz.
- Latency yalnızca aynı donanım/input/batch/precision/ölçüm protokolünde doğrudan kıyaslanır. Ensemble latency teacher latency toplamından gerçek ölçümmüş gibi üretilmez.
- Eğitim süresi attempt'ler arasında birikir; yeniden hesaplanan kayıp epoch süreleri tam ölçülemiyorsa toplamın alt sınır/eksik olduğu belirtilir. Yedekleme süresi ayrılır.

## 6. Rapor ekranı ve çıktılar

Ana tablo: Rol, Model, Teacher(lar), KD türü, Aggregation, Seed, Val Macro-F1, Test Accuracy, Test Macro-F1, baseline farkı (pp), Params, MACs, tahmini FLOPs, MB, latency, eğitim süresi, kaynak run ID.

İkinci tablo: kodun class mapping sırasıyla sınıf başına precision/recall/F1/support ve baseline'a göre F1 farkı. Eksik per-class artifact varsa ilgili hücre `N/A` kalır.

Grafikler: seçilen uyumlu run'ların Macro-F1 karşılaştırması ve Params–Macro-F1 dağılımı. Farklı seed'leri otomatik ortalamak yoktur.

Butonlar: `Markdown indir`, `CSV indir`, `LaTeX indir`, `Kaynak manifesti indir`. Dosya yazma kullanıcı ayrıca kaydet seçmedikçe download belleğinde yapılır.

Markdown: seçim kapsamı, genel tablo, sınıf bazlı tablo, sayısal iyileşme/gerileme özeti ve sınırlılıklar. Yalnızca deterministik metin; LLM veya harici API gerekmez.

LaTeX: İngilizce başlıklar, güvenli karakter escaping, iki sütunlu IEEE belgeye uygun kompakt ana tablo; çok geniş alanlar maliyet tablosuna ayrılır. Kaynak değerler tam hassasiyetle manifest/CSV'de, görüntüde metrikler yüzde ve iki ondalıkla tutulur. Eşitlik/best vurgusu yuvarlanmamış değerlere göre yapılır.

Manifest: seçilen run/attempt kimlikleri, kaynak dosya SHA-256'ları, split/protokol fingerprint'leri, filtre/seçim politikası, üretilme zamanı ve rapor şema sürümü. Sonradan kaynak silinirse eski rapor snapshot olarak korunur; canlı görünüm yeniden taramada silinen kaydı göstermez.

## 7. Dosya ve sorumluluk planı

| Dosya | İş |
|---|---|
| Yeni `src/training/recovery.py` | Snapshot yükleme/yazma, generation doğrulama, yedekleme ve resume uygunluğu |
| Yeni `src/tracking/run_metadata.py` | Portable protokol fingerprint'leri ve ortak kimlik metadata'sı |
| `src/training/trainer.py`, `kd_trainer.py`, `multi_kd_trainer.py` | State restore, epoch sonu snapshot, stop-request ve finalizasyon |
| `src/training/train_baseline.py`, `train_teacher.py`, `train_kd.py`, `train_multi_kd.py` | CLI/config, attempt kimliği, metadata, idempotent final kayıt |
| `src/training/kd_config.py`, `multi_kd_config.py` | Geriye uyumlu optional recovery alanları |
| `src/datasets/dataloaders.py`, `src/utils/seed.py` | Kurtarma modunda generator/worker/epoch seed yönetimi |
| `app/experiment_runner.py`, `app/queue_worker.py` | Resume kuyruğu, stop-request ve metadata/status uzlaştırma |
| Eğitim sayfaları ve `app/pages/3_Job_Monitor.py` | Recovery alanları, validation ve job kontrolleri |
| `app/portable_notebook.py`, `src/tracking/external_bundle.py`, `app/external_run_import.py` | Drive recovery yolu, resume notebook akışı, provenance aktarımı |
| `app/experiment_removal.py`, `scripts/reset_experiments.py` | Yerel recovery dosyaları ve bağımlılık koruması; bulut silme yok |
| Yeni `src/evaluation/experiment_comparison.py` | Kaynakları normalize etme, uyum kuralları, tablo ve rapor üretimi |
| Yeni `app/comparison_view.py` | Karşılaştırma seçimleri ve indirme kontrolleri |
| `app/pages/4_Results_Explorer.py` | Yeni sekmeyi mevcut sayfaya bağlama |
| `README.md` | Resume sınırları, Drive davranışı, rapor üretimi kullanımı |

Tek uygulamalı factory/policy hiyerarşisi kurulmaz. Ortak dosya işlemleri bir kez yazılır; mevcut trainer'ları bu iş için tek büyük trainer'a dönüştürmek yoktur.

## 8. Uygulama sırası ve test kapıları

Her görevde önce belirtilen başarısız test, ardından minimum uygulama ve aynı testlerin tekrar çalıştırılması yapılır. Test verileri geçici dizinler ve küçük CPU modelleridir; gerçek dataset eğitimi başlatılmaz.

### A1 — Protokol metadata'sı ve recovery dosya formatı

- [ ] `tests/test_training_recovery.py` ve `tests/test_run_metadata.py` ekle.
- [ ] `write_generation(root: Path, payload: dict) -> Path`, `load_generation(path: Path) -> dict`, `mirror_generation(source: Path, destination: Path) -> Path` arayüzlerini recovery modülünde kur.
- [ ] `protocol_fingerprint(metadata: dict) -> str` ve `validate_resume(saved: dict, requested: dict) -> None` ekle; validation farkları alan isimleriyle ValueError üretir.
- [ ] Round-trip state, yanlış hash, yarım generation, kopyalama hatası, önceki generation'ın korunması ve yönetilen root dışı silmenin reddini test et.
- [ ] Aynı split'in Windows/Colab kökleri değişince canonical hash'inin eşit; üyelik/etiket/class order değişince farklı olduğunu test et.

Örnek başlangıç testi (yeni arayüz için):

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from src.training.recovery import write_generation, load_generation

def test_generation_round_trip():
    payload = {"schema_version": 1, "attempt_id": "a1", "epoch": 2,
               "phase": "training", "history": [{"epoch": 2, "val_loss": 0.4}]}
    with TemporaryDirectory() as directory:
        path = write_generation(Path(directory), payload)
        assert load_generation(path)["epoch"] == 2
```

### A2 — Üç trainer'da gerçek epoch-boundary resume

- [ ] `tests/test_resume_training.py` ekle; küçük dropout modeli, deterministik sentetik veri ve augmentation RNG içeren testler kur.
- [ ] Standard fit, single-KD ve multi-KD için restore sırasını uygula: modeller/strategy, optimizer/scheduler/scaler, ilerleme/history, RNG; loader iterator oluşturulmadan önce restore tamamlanır.
- [ ] İki epoch kesintisiz koşuyu bir epoch + save/reload + bir epoch ile karşılaştır: state tensor'ları, LR, best epoch ve patience eşit olmalı. CPU kontrollü ortamda tam/uygun toleranslı eşitlik aranır.
- [ ] Feature adapter state/optimizer parametre gruplarını, relation/logit akışlarını ve teacher gradientlerinin kapalı kalmasını test et.
- [ ] Persistent worker seçiminin kurtarma modunda kapandığını ve num_workers 0/2 için örnek sırası/seed davranışını test et.
- [ ] Hedef epoch ve patience bitmiş snapshot yalnızca finalizasyona girmeli; best epoch geçmiş attempt'te kaldığında test yine doğru best ağırlıklarını kullanmalı.

### A3 — CLI, job, MLflow ve kaldırma entegrasyonu

- [ ] `tests/test_recovery_jobs.py` ekle; mevcut command builder/removal testlerini genişlet.
- [ ] Her giriş scriptine recovery config ve immutable provenance bağla; eski config'leri optional alanlarla çalışır tut.
- [ ] Yeni attempt/parent metadata'sı, tek aktif resume kontrolü ve idempotent summary yazımını uygula.
- [ ] Stop-request'i trainer'larda kontrol et; zorla durdurmayı ayrı tut. Mock süreç/MLflow ile FINISHED/KILLED/FAILED davranışlarını doğrula.
- [ ] Teacher/split/config uyuşmazlığında subprocess başlatılmadığını test et.
- [ ] Aktif resume kaynağını silme engeli, child snapshot bağımsızlığı ve yalnızca yerel recovery temizliği için regresyon testlerini çalıştır.

### A4 — Streamlit ve portable notebook kurtarma

- [ ] `tests/test_portable_notebook.py` ve launcher AppTest testlerini genişlet.
- [ ] Yeni recovery alanları ve Monitor kontrollerini ekle; checkpoint yoksa veya eski formatlıysa nedenini göstererek resume butonunu kapat.
- [ ] Drive bağlantısı ve yazılabilir hedef kontrolü eğitimden önce; resume seçili notebook'ta kaynak commit/config hazırlığı state yüklemesinden önce gerçekleşmeli.
- [ ] Final bundle paketlemesini mevcut tamamlanan sonuç formatıyla koru; provenance alanlarını bundle/import'a geçir.
- [ ] Colab mount başarısı/başarısızlığı, local yol, Kaggle `/kaggle/working`, eksik teacher bağımlılığı ve tüm üretilen hücrelerin derlenmesini test et.
- [ ] Kaggle runtime kaybına karşı `/kaggle/working` tek başına garanti değildir; kullanıcıya sürüm çıktısı kalıcılığı ile epoch yedeği ayrımını göster.

### B1 — Karşılaştırma çekirdeği

- [ ] `tests/test_experiment_comparison.py` ekle.
- [ ] `load_comparison_runs(root: Path) -> list[dict]`, `build_comparison(runs: list[dict], selected_ids: list[str], baseline_id: str | None) -> dict` ve `export_comparison(report: dict, format: str) -> bytes` arayüzlerini kur.
- [ ] Beş summary kaynağı ve artifactlerden ortak kayıtları üret. Report sözleşmesi `rows`, `class_rows`, `warnings`, `provenance` alanlarından oluşur.
- [ ] Dataset aynı/split farklı, class order farklı, eksik metadata, stage alanı farklı, dry-run, batch-limit, farklı teacher checkpoint ve silinmiş run senaryolarını doğrula.
- [ ] Delta testi: baseline F1 0.80 ve KD F1 0.83 için sonuç +3.0 pp olmalı; +3.75% olarak etiketlenmemeli.
- [ ] Otomatik baseline seçiminde test F1 yüksek fakat validation F1 düşük aday seçilmemeli; farklı student/seed adayları havuza girmemeli.
- [ ] Ensemble satırı öğretmen skor ortalamasıyla doldurulmamalı; eksik metrik `N/A` olmalı. Resume attempt'leri bağımsız deney diye sayılmamalı.

### B2 — Results Explorer ve makale export'u

- [ ] `tests/test_comparison_view.py` ile AppTest ve export fixture testleri ekle.
- [ ] Yeni sekmede dataset/run/job seçimlerini, protokol uyarılarını, baseline adayını, genel ve sınıf tablolarını göster.
- [ ] Markdown/CSV/LaTeX/JSON'u aynı report nesnesinden üret; UI ve indirilen dosya değerlerinin eşitliğini test et.
- [ ] LaTeX özel karakterleri (`_`, `%`, `&`, `#`) ve uzun teacher adlarını test et. Eksik veriler ve negatif delta doğru gösterilmeli.
- [ ] Karşılaştırma/indirme sırasında eğitim, model forward, dataset download veya MLflow write çağrısı yapılmadığını test et.
- [ ] Kaynak hash'leri, rapor seçim politikası ve yeniden üretilebilirlik açıklamasını manifestte doğrula.

### Son doğrulama

- [ ] `conda activate masterthesis` ardından `python -m pytest tests/test_training_recovery.py tests/test_run_metadata.py tests/test_resume_training.py tests/test_recovery_jobs.py tests/test_experiment_comparison.py tests/test_comparison_view.py -q` çalıştır.
- [ ] Mevcut notebook, single/multi KD, teacher/baseline, job monitor, import/export, kaldırma ve reset regresyon testlerini çalıştır.
- [ ] Küçük CPU fixture ile bir kesinti → doğrulanmış yedek → yeni attempt → tek final sonuç → karşılaştırma export zincirini uçtan uca doğrula.
- [ ] Eğitim sonucu/kullanıcı dosyalarını değiştirmeden test izolasyonunu kontrol et. Gerçek Drive/Kaggle uçtan uca doğrulama yapılmadıysa bunu teslimde açıkça belirt.
- [ ] README'yi güncelle; mevcut kullanıcı değişikliklerini koru. Uygulama sonunda Graphify/codebase-memory güncellemelerini kod değişikliklerinden ayrı raporla. Commit/merge/push yalnızca ayrıca istendiğinde yapılır.

## 9. Kabul kriterleri

1. Ani kesintide son doğrulanmış generation yüklenir; yarım yedek seçilmez.
2. Son tamamlanan epoch'tan devam edilir; optimizer/scheduler/patience sıfırlanmaz.
3. Eski best-only checkpoint yanlışlıkla tam kurtarma dosyası kabul edilmez.
4. Colab'da final paketlemeden önce epoch yedekleri oluşabilir; son yedek bilgisi kullanıcıya görünür.
5. Resume kaynakları ve gerekli teacher dosyaları doğrulanmadan eğitim kuyruğa alınmaz.
6. Tamamlanan aynı attempt için birden fazla sonuç satırı yazılmaz; eski failed/stopped geçmişi korunur.
7. Uyuşmayan split/protokol ve eksik kanıtlar raporda saklanmaz veya tahminle tamamlanmaz.
8. Student baseline, gerçek teacher run'ları, single-KD, multi-KD ve kayıtlı ensemble kaynağı izlenebilir tabloda karşılaştırılır.
9. UI, Markdown, CSV ve LaTeX aynı sayıları/sınıf sırasını verir; delta yüzde puan cinsindedir.
10. Bu geliştirmeler yeni uzun eğitim veya otomatik veri silme başlatmadan test edilebilir.
