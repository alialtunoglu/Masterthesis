# Experiment Plan

Bu doküman temiz deney başlangıcından sonra izlenecek sırayı tanımlar. Ham veri setleri fiziksel olarak bölünmez; tüm deneyler mevcut split JSON dosyaları üzerinden yürütülür.

## Aşama 1: Altyapı Doğrulama

- CUDA ve PyTorch kurulumu doğrulanır.
- Split JSON dosyaları ve DataLoader katmanı dry-run ile test edilir.
- MLflow UI, CSV/JSON çıktıları, checkpoint yazımı ve Streamlit dashboard kontrol edilir.

## Aşama 2: AppleLeaf9 Student Baseline

- Aynı split dosyasıyla baseline student modeller eğitilir.
- Öncelikli modeller: MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0, ResNet18.
- Her run için MLflow parametreleri, metrikleri, artifactleri, CSV sonuçları ve checkpoint kaydedilir.

Tamamlanma kriterleri:

- 3 veri seti x 4 student model için 12 baseline sonucu üretilmiş olmalıdır.
- `results/baseline/student_baseline_clean_results.csv` her dataset-model çifti için seçilen en iyi run'ı içermelidir.
- `results/baseline/student_baseline_overall_comparison.csv` ve `results/baseline/student_selection_recommendation.csv` oluşturulmuş olmalıdır.
- Teacher aşamasından önce recommended KD student netleşmiş olmalıdır.

## Aşama 3: PlantVillage Student Baseline

- AppleLeaf9 üzerinde doğrulanan pipeline PlantVillage için çalıştırılır.
- Aynı model ailesi ve karşılaştırılabilir hiperparametreler kullanılır.
- Sınıf sayısı daha yüksek olduğu için confusion matrix okunabilirliği ayrıca kontrol edilir.

## Aşama 4: CNN Teacher Deneyleri

- Baseline sonuçlarına göre güçlü CNN teacher adayları seçilir.
- Teacher modeller validation macro F1 ve test metrikleriyle değerlendirilir.
- Checkpointler ileride KD deneylerinde kullanılacak şekilde saklanır.
- Student baseline aşaması tamamlandıktan sonra CNN teacher training altyapısına geçilir.
- Teacher modeller KD için öğretmen adayı olarak eğitilir.

## Aşama 5: ViT Teacher Deneyleri

- Vision Transformer tabanlı teacher adayları eğitilir.
- CNN teacher sonuçlarıyla aynı split ve metrik seti üzerinden karşılaştırılır.
- Parametre sayısı ve model boyutu raporlanır.

## Aşama 6: Teacher/Student Seçimi

- Teacher ve student adayları accuracy, macro F1, weighted F1, model boyutu ve pratik çalıştırma maliyetiyle karşılaştırılır.
- KD aşamalarında kullanılacak teacher checkpointleri ve student mimarileri sabitlenir.

## Aşama 7: Single-Teacher KD

- Seçilen her teacher ile student model için single-teacher knowledge distillation deneyleri yapılır.
- Temperature, alpha ve loss bileşenleri MLflow ve CSV çıktılarında açıkça loglanır.
- PlantVillage'da ceiling effect beklendiği için KD etkisi özellikle AppleLeaf9 ve PlantPathology2021 üzerinde daha dikkatli yorumlanır.

## Aşama 8: Multi-Teacher KD

- Birden fazla teacher modelden gelen bilgi birleştirilir.
- Multi-teacher stratejisi, teacher ağırlıkları ve distillation ayarları deney çıktılarında saklanır.

## Aşama 9: Adaptive Temperature, Self-Distillation, Quantization

- Adaptive temperature KD varyantları denenir.
- Self-distillation deneyleri student model ailesi üzerinde değerlendirilir.
- Quantization sonrası accuracy/F1 kaybı ve model boyutu kazancı raporlanır.

## Takip Edilecek Çıktılar

Her aşamada üç kanal birlikte korunur:

- MLflow: arayüzden experiment, run, params, metrics ve artifacts takibi.
- CSV/JSON: tez tabloları, tekrar üretilebilirlik ve toplu analiz için `results/` altında makine-okunabilir çıktılar.
- Run artifactleri: baseline aşamasında `results/baseline/runs/{dataset_name}/{model_name}/{run_name}/` altında config, history, per-class metrics, confusion matrix ve learning curve dosyaları.
- Checkpoint: en iyi model ağırlıkları `checkpoints/` altında, Git dışında tutulacak şekilde saklanır.

Deneyler bittikten sonra ilgili history CSV, per-class metrics, confusion matrix ve learning curve dosyaları MLflow artifact olarak da loglanmalıdır.
