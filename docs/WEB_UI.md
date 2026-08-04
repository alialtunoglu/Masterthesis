# Streamlit Web UI

Bu projeye yerel bir Streamlit dashboard eklendi. Amaç, baseline deney komutlarını terminalden sürekli yazmak yerine arayüzden hazırlamak, başlatmak ve sonuç dosyalarını hızlıca incelemektir.

Bu arayüz yalnızca local kullanım içindir. Public network'e açılmamalıdır.

## Streamlit Dashboard

```powershell
streamlit run app/streamlit_app.py
```

Streamlit varsayılan olarak şu portu kullanır:

```text
http://127.0.0.1:8501
```

## MLflow UI

MLflow UI ayrı bir terminalde başlatılır:

```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

Tarayıcı adresi:

```text
http://127.0.0.1:5000
```

Yeni MLflow sürümlerinde SQLite backend kullanılması önerilir:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

## Sayfalar

- `Dashboard`: CUDA durumu, veri seti özeti, split JSON özetleri ve son baseline sonuçlarını gösterir.
- `Baseline Experiments`: baseline eğitim komutlarını hazırlar, dry-run veya gerçek eğitim job'ı başlatabilir.
- `Job Monitor`: seçilen job için status, PID, süre, log satır sayısı, son görülen epoch/metrik bilgileri ve canlı yenilenen log görüntüsü sağlar.
- `Results Explorer`: baseline ve CNN teacher sonuçlarını filtreler, artifact görsellerini gösterir ve tamamlanmış tek bir deneyi güvenli şekilde kaldırabilir.
- `MLflow Helper`: MLflow UI komutlarını ve loglanan içerikleri özetler.

## Baseline Deney Başlatma

`Baseline Experiments` sayfasında dataset, model, config ve eğitim parametreleri seçilir. Komut önizlemesi kontrol edildikten sonra:

- `Dry Run Başlat`: eğitim yapmadan pipeline kontrolü başlatır.
- `Eğitimi Başlat`: gerçek baseline eğitim sürecini subprocess olarak başlatır.

Gerçek eğitim GPU kullanabilir ve uzun sürebilir.

Dashboard varsayılan olarak training komutlarına şu MLflow tracking URI değerini ekler:

```text
sqlite:///mlflow.db
```

## Dosya Konumları

- Job metadata: `runs/jobs/`
- Job logları: `runs/logs/`
- Baseline sonuçları: `results/baseline/`
- Baseline run artifactleri: `results/baseline/runs/{dataset_name}/{model_name}/{run_name}/`
- Checkpoint dosyaları: `checkpoints/`
- MLflow kayıtları: `mlruns/`

`runs/jobs/`, `runs/logs/`, `mlruns/` ve checkpoint dosyaları Git'e eklenmemelidir.

## Tek Bir Deneyi Kaldırma

`Results Explorer > Run Artifacts` sekmesinde baseline veya CNN teacher sonuç kaynağı seçilir. Kaldırma işlemi yalnızca sonuç satırında hem `run_name` hem de `mlflow_run_id` bulunduğunda etkinleşir.

İki mod vardır:

- `Arşivle (önerilen)`: run klasörü, checkpoint ve kesin olarak eşleşen job/log dosyaları `_archive/experiment_removals/` altına taşınır.
- `Yerel dosyaları kalıcı sil`: aynı yerel dosyalar geri alınamayacak şekilde silinir.

Her iki mod da run'ı MLflow'da soft-delete eder ve aynı run'ı içeren sonuç CSV satırlarını kaldırır. Böylece kayıt normal MLflow ve Streamlit görünümlerinde görünmez. MLflow garbage collection otomatik çalıştırılmaz.

Yanlış deneyi kaldırmayı önlemek için kullanıcıdan `KALDIR {run_name}` metnini birebir yazması ve ayrıca onay kutusunu işaretlemesi istenir. Eşleşen job `queued` veya `running` durumundaysa işlem başlamaz. Dosya veya MLflow işlemlerinden biri commit öncesinde başarısız olursa CSV ve taşınmış dosyalar geri alınır; daha önce soft-delete edilen MLflow run'ı restore edilir.

## Job Queue

Streamlit dashboard job'ları doğrudan sınırsız paralel başlatmaz. `app/dashboard_settings.json` dosyasındaki değer kadar job aynı anda çalışır:

```json
{
  "max_parallel_jobs": 1
}
```

Varsayılan değer `1` olduğu için yeni eğitimler önce `queued` durumuna alınır. Çalışan job bittiğinde veya durdurulduğunda sıradaki job otomatik olarak `running` durumuna geçer.

Queue ilerlemesi Streamlit refresh'ine bağlı değildir. Bunun için ayrı worker kullanılır:

```powershell
python app/queue_worker.py
```

Worker ayrıca Job Monitor sayfasındaki `Queue Worker Başlat` butonuyla da başlatılabilir. Streamlit kapatılsa bile worker ayrı process olarak arka planda devam eder. Worker durumu `runs/queue_worker.json`, heartbeat bilgisi `runs/queue_worker_heartbeat.json`, worker logları ise `runs/logs/queue_worker.log` altında tutulur.

Aynı anda iki worker çalıştırılmaz. Worker zaten çalışıyorsa terminalden veya Streamlit üzerinden ikinci worker başlatma isteği reddedilir.

Queued durumundaki yanlış bir job, Job Monitor sayfasında `Queue iptalini onayla` kutusu işaretlenip `Queued Job'ı İptal Et` butonuyla iptal edilebilir. Bu işlem çalışan process'e dokunmaz; yalnızca henüz başlamamış job metadata'sını `cancelled` olarak günceller.

## Job Monitor

`Job Monitor` sayfası, Streamlit üzerinden başlatılan deneylerin loglarını dosyadan okuyarak gösterir. `Auto-refresh` açıkken sayfa birkaç saniyede bir yenilenir ve seçilen job için son log satırları güncellenir.

Training script log dosyasına epoch başına şu formatta tek satırlık özet yazar:

```text
EPOCH_SUMMARY epoch=1/30 train_loss=... val_loss=... val_accuracy=... val_macro_f1=...
```

Job Monitor bu satırı okuyarak epoch progress bar ve metrik kartlarını günceller. Tqdm progress barları dosyaya yönlendirilmiş loglarda kapatılır; böylece log dosyası tekrar eden `train` satırlarıyla dolmaz.

Çalışan bir job gerektiğinde `Durdurmayı onayla` kutusu işaretlenip `Seçili Job'ı Durdur` butonuyla durdurulabilir. Bu işlem yalnızca seçili job metadata'sındaki PID için uygulanır.

Logların fiziksel konumu:

```text
runs/logs/{job_id}.log
```

Terminalden aynı logu canlı izlemek için:

```powershell
Get-Content runs/logs/{job_id}.log -Wait
```

## CPU Mini Deney ve CUDA Full Deney

CPU mini deneyler genellikle batch limitleriyle smoke test amacıyla çalıştırılır. CUDA full deneylerde batch limiti kullanılmadan hedef epoch sayısı tamamlanır. Run isimleri ve sonuç dosyaları dataset, model, seed, cihaz ve çalışma modu bilgileriyle ayrışacak şekilde üretilir.
