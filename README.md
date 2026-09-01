# MasterThesis Plant Disease KD

Bu proje, bitki yaprak hastalığı teşhisinde derin öğrenme modelleri için bilgi damıtma yöntemlerinin karşılaştırmalı analizini yürütmek için hazırlanmıştır.

Kısa proje konusu:

```text
Bitki Yaprak Hastalığı Teşhisinde Derin Öğrenme Modelleri için
Bilgi Damıtma Yöntemlerinin Karşılaştırmalı Analizi
```

## Mevcut Aşamalar

Altyapı:

- Veri seti analizi ve deterministic split üretimi hazır.
- PyTorch Dataset/DataLoader katmanı hazır.
- MLflow destekli student baseline pipeline hazır.
- Streamlit local dashboard hazır (9 sayfa) ve queue worker hazır.
- CNN teacher training altyapısı hazır.
- ViT teacher training altyapısı hazır (config, Streamlit sayfası ve testler).
- Plant Pathology 2021 için logit, feature ve relation tabanlı single-teacher KD altyapısı hazır.
- Multi-teacher KD altyapısı hazır.
- Colab/Kaggle için portable notebook üretimi ve harici run import altyapısı hazır.

Üretilmiş sonuçlar:

- Student baseline deneyleri tamamlandı ve raporlandı (3 veri seti x 4 model).
- CNN teacher deneyleri tamamlandı (AppleLeaf9 ve PlantVillage 10 model, PlantPathology2021 8 model).
- Single-teacher KD: yalnızca PlantPathology2021 `logit_based` koşuldu.
- Multi-teacher KD: yalnızca PlantPathology2021 `logit_based` koşuldu.
- ViT teacher, feature/relation KD ve quantization deneyleri henüz koşulmadı.

## Knowledge Distillation

Streamlit'teki `Knowledge Distillation` sayfası MobileNetV3-Small student ile
ResNet50, DenseNet201 veya RegNet-Y-8GF teacher arasında tek koşuluk deneyleri
kuyruğa ekler. Hazır şablonlar
`configs/knowledge_distillation/plantpathology2021/` altındadır; özel JSON da
yüklenebilir. Kuyruğa gönderilen çözülmüş config değişmez bir snapshot olarak
`runs/configs/{job_id}.json` altında saklanır.

Eğitim hattı üç stratejiyi destekler:

- Logit-based: hard-label cross entropy ve temperature-scaled KL.
- Feature-based: kayıtlı ara katmanlar, otomatik 1×1 adapter ve feature loss.
- Relation-based: batch içi distance/angle ilişkileri.

Teacher checkpoint, dataset sınıf sırası, output boyutu ve feature katmanları
başlatma öncesinde doğrulanır. Sonuçlar `results/knowledge_distillation/`, en iyi
student checkpointleri `checkpoints/knowledge_distillation/` altında tutulur.
Results Explorer teacher, aynı seed'deki student baseline ve KD student sonucunu
birlikte gösterir.

## Multi-Teacher Knowledge Distillation

`Multi-Teacher Knowledge Distillation` sayfası Plant Pathology 2021 üzerinde
MobileNetV3-Small student için ResNet50, DenseNet201 ve RegNet-Y-8GF
teacher'larından iki veya üçünü birlikte kullanır. Logit, feature ve relation KD;
eşit, validation macro-F1 tabanlı veya manuel teacher ağırlıklandırmasıyla
çalıştırılabilir.

On iki hazır şablon
`configs/multi_teacher_knowledge_distillation/plantpathology2021/` altındadır.
Sonuçlar `results/multi_teacher_knowledge_distillation/`, checkpointler
`checkpoints/multi_teacher_knowledge_distillation/` altında saklanır. Dry run,
bir train batch'inde loss ve backward ile bir validation batch'ini gerçekten
çalıştırır.

## Ortam

Önerilen ortam adı:

```powershell
mamba activate masterthesis
```

CUDA kontrolü:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Bu projede doğrulanan CUDA kurulumu:

```text
torch: 2.6.0+cu124
GPU: NVIDIA GeForce RTX 4070
```

## Bağımlılıklar

Paketler `requirements.txt` ve `pyproject.toml` içinde tutulur.

Gerekirse:

```powershell
pip install -r requirements.txt
```

PyTorch CUDA kurulumu ayrıca uygun CUDA build ile yapılmalıdır.

## Veri Setleri

Ham veri setleri şu konumdadır:

```text
datasets/
  AppleLeaf9/raw/
  PlantVillage/raw/color/
  PlantPathology2021/train.csv
  PlantPathology2021/train_images/
```

Önemli:

- Ham veri setleri taşınmaz, silinmez, yeniden adlandırılmaz.
- Fiziksel `train/val/test` klasörleri oluşturulmaz.
- Splitler JSON dosyalarından okunur.

Split dosyaları:

```text
splits/
```

## Veri Seti Analizi

```powershell
python src/datasets/analyze_datasets.py
```

Çıktılar:

```text
results/dataset_analysis/
```

## Split Oluşturma

```powershell
python src/datasets/create_splits.py --dataset all
```

Var olan splitlerin üzerine yazmak için:

```powershell
python src/datasets/create_splits.py --dataset all --overwrite
```

## MLflow UI

Yeni MLflow sürümleri için SQLite backend kullan:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Tarayıcı:

```text
http://127.0.0.1:5000
```

## Streamlit Dashboard

```powershell
streamlit run app/streamlit_app.py
```

Tarayıcı:

```text
http://127.0.0.1:8501
```

Dashboard sayfaları:

- Dashboard
- Baseline Experiments
- Job Monitor
- Results Explorer
- CNN Teacher Experiments
- Vision Transformer Teacher Experiments
- MLflow Helper
- Knowledge Distillation
- Multi-Teacher Knowledge Distillation

## Queue Worker

Deneylerin Streamlit refresh edilmeden sırayla çalışması için queue worker kullanılır.

Streamlit içinden:

```text
Job Monitor -> Queue Worker Başlat
```

Terminalden:

```powershell
python app/queue_worker.py
```

Varsayılan paralel eğitim sayısı:

```json
{
  "max_parallel_jobs": 1
}
```

Ayar dosyası:

```text
app/dashboard_settings.json
```

Worker durumu:

```text
runs/queue_worker.json
runs/queue_worker_heartbeat.json
runs/logs/queue_worker.log
```

## Student Baseline Eğitimi

Configler:

```text
configs/baseline/
```

Örnek dry-run:

```powershell
python src/training/train_baseline.py --config configs/baseline/mobilenetv3_small_appleleaf9.json --dry-run
```

Örnek gerçek eğitim:

```powershell
python src/training/train_baseline.py --config configs/baseline/mobilenetv3_small_appleleaf9.json
```

Streamlit üzerinden:

```text
Baseline Experiments -> config seç -> Eğitimi Başlat
```

Sonuçlar:

```text
results/baseline/baseline_results.csv
results/baseline/runs/{dataset}/{model}/{run_name}/
checkpoints/{dataset}/{model}/
```

Student baseline raporu:

```text
docs/STUDENT_BASELINE_REPORT.md
```

## CNN Teacher Eğitimi

Configler:

```text
configs/teachers/cnn/{dataset}/{model}.json
```

Örnek dry-run:

```powershell
python src/training/train_teacher.py --config configs/teachers/cnn/appleleaf9/resnet50.json --dry-run
```

Örnek gerçek eğitim:

```powershell
python src/training/train_teacher.py --config configs/teachers/cnn/appleleaf9/resnet50.json
```

Streamlit üzerinden:

```text
CNN Teacher Experiments -> config seç -> Teacher Eğitimi Başlat
```

Sonuçlar:

```text
results/teachers/cnn/teacher_results.csv
results/teachers/cnn/runs/{dataset}/{model}/{run_name}/
checkpoints/teachers/cnn/{dataset}/{model}/
```

MLflow experiment:

```text
MasterThesis-CNN-Teachers
```

## ViT Teacher Eğitimi

ViT teacher'lar aynı `train_teacher.py` girişini kullanır; config yolu aileyi belirler.

Configler:

```text
configs/teachers/vision_transformers/{dataset}/{model}.json
```

Desteklenen modeller:

```text
vit_b_16, swin_v2_t, maxvit_t, dinov2_vitb14
```

Örnek dry-run:

```powershell
python src/training/train_teacher.py --config configs/teachers/vision_transformers/appleleaf9/vit_b_16.json --dry-run
```

Streamlit üzerinden:

```text
Vision Transformer Teacher Experiments -> config seç -> Teacher Eğitimi Başlat
```

Sonuçlar:

```text
results/teachers/vision_transformers/teacher_results.csv
results/teachers/vision_transformers/runs/{dataset}/{model}/{run_name}/
checkpoints/teachers/vision_transformers/{dataset}/{model}/
```

MLflow experiment:

```text
MasterThesis-Vision-Transformer-Teachers
```

## Portable Notebook (Colab / Kaggle)

Baseline, CNN teacher ve ViT teacher sayfaları seçili komuttan çalıştırılabilir bir
`.ipynb` üretir. Üretilen notebook repoyu klonlar, çalıştırıldığı commit'e checkout
eder, veri setini hazırlar, eğitimi canlı log akışıyla koşturur ve sonucu
`scripts/package_external_run.py` ile bundle olarak paketler.

Export yalnızca şu koşullarda etkinleşir:

- `origin` public bir `https://github.com/...` adresi olmalı.
- Çalışma ağacı temiz olmalı.
- Yerel HEAD, origin üzerindeki aktif branch'e push edilmiş olmalı.
- Komut `--dry-run` içermemeli.

Sonuç bundle'ı kalıcı bir konuma yazılır; çalışma zamanı kapanınca kaybolmaz:

| Ortam | Bundle konumu |
|---|---|
| Google Colab | `/content/drive/MyDrive/MasterThesis/bundles/` |
| Kaggle | `/kaggle/working/` |
| Yerel | Çalışma dizini |

Colab'da Drive bağlama onayı notebook'un başındaki `Result destination` hücresinde
istenir. Onayı defteri başlatırken bir kez verirsiniz; eğitim bittiğinde bilgisayar
başında olmanız gerekmez. Drive bağlanamazsa hücre çökmez, `/content` altına düşer ve
bundle'ın kalıcı olmadığını uyarır.

Kaggle'da interaktif oturum boşta kalıp düşerse `/kaggle/working` de kaybolabilir.
Ekran başında beklememek için defteri `Save Version` ile çalıştırın: notebook headless
koşar, tarayıcıyı kapatabilirsiniz ve `/kaggle/working` oturum sonunda kalıcı output
olarak saklanır.

Veri seti hazırlığı ortama göre değişir:

- AppleLeaf9 ve PlantVillage: pinned commit ile GitHub checkout.
- PlantPathology2021: önce mevcut veri, sonra Kaggle mounted input, sonra Kaggle CLI
  indirmesi denenir. Colab'da kimlik yoksa `kaggle.json` upload widget'ıyla istenir.

Kimlik bilgileri notebook'a hiçbir zaman yazılmaz.

Dışarıda koşan run'ı geri almak için:

```text
Results Explorer -> external run import
```

## Pretrained Ağırlık Notu

TorchVision modelleri `pretrained=true` ise ağırlık indirmek isteyebilir.

Eğer Windows/network izni şu hatayı verirse:

```text
WinError 10013
```

çözüm seçenekleri:

- Python için firewall/Defender izinlerini düzenle.
- Ağırlıkları elle indirip Torch cache klasörüne koy.
- Sadece altyapı dry-run için geçici olarak `--no-pretrained` kullan.

Gerçek karşılaştırmalı eğitimlerde mümkünse `pretrained=true` kullanılmalıdır.

## Sonuç Analizi

Student baseline analiz raporu üretmek için:

```powershell
python scripts/analyze_student_baselines.py
```

Çıktılar:

```text
results/baseline/student_baseline_clean_results.csv
results/baseline/student_baseline_overall_comparison.csv
results/baseline/student_selection_recommendation.csv
docs/STUDENT_BASELINE_REPORT.md
```

## Deney Çıktılarını Temizleme

Güvenli arşivli reset:

```powershell
python scripts/reset_experiments.py --archive --yes --keep-dataset-analysis
```

Bu komut deney çıktılarını şuraya taşır:

```text
_archive/experiment_reset_YYYYMMDD_HHMMSS/
```

Korunanlar:

```text
datasets/
splits/
src/
configs/
docs/
.agent/
results/dataset_analysis/
```

Temizlenen/arşivlenenler:

```text
mlruns/
mlflow.db
mlruns.db
checkpoints/
runs/jobs/
runs/logs/
results/baseline/
results/teachers/
results/knowledge_distillation/
results/multi_teacher_knowledge_distillation/
results/quantization/
```

Eski isimlendirmeden kalan `results/vit_teachers/`, `results/kd_single/`,
`results/kd_multi/` ve `results/experiments/` de temizlenir, ancak yeniden
oluşturulmaz. Reset sonrası yalnızca eğitim hatlarının gerçekten yazdığı dizinler
boş olarak geri kurulur.

## Önerilen Çalıştırma Sırası

1. Ortamı aktive et.
2. MLflow UI başlat.
3. Streamlit dashboard başlat.
4. Job Monitor’dan queue worker başlat.
5. İlgili deney sayfasından config seç.
6. Önce dry-run yap.
7. Gerçek eğitimi queue’ya ekle.
8. Job Monitor’dan log ve durum takip et.
9. Results Explorer’dan sonuçları incele.

## Daha Fazla Dokümantasyon

- `docs/DATASETS.md`
- `docs/ENVIRONMENT.md`
- `docs/MLFLOW.md`
- `docs/WEB_UI.md`
- `docs/STUDENT_BASELINE_REPORT.md`
- `docs/CNN_TEACHERS.md`
- `docs/EXPERIMENT_PLAN.md`
- `docs/EXPERIMENT_RESULTS_ANALYSIS.md`
- `docs/KD_RESULTS_ANALYSIS.md`
