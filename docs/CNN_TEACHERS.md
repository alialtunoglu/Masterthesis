# CNN Teacher Training

Bu aşamanın amacı, student baseline sonuçları tamamlandıktan sonra KD deneylerinde öğretmen adayı olarak kullanılacak CNN teacher modellerinin eğitim altyapısını kurmaktır.

Bu aşamada ViT teacher veya KD altyapısı yer almaz.

## CNN Teacher Modelleri

- `resnet50`
- `resnet101`
- `densenet121`
- `densenet201`
- `vgg19_bn`
- `efficientnet_b3`
- `efficientnet_b4`
- `convnext_tiny`
- `convnext_base`
- `regnet_y_8gf`

## Config Konumu

Config dosyaları şu yapıda tutulur:

```text
configs/teachers/cnn/{dataset_name}/{model_name}.json
```

Toplam config sayısı:

```text
3 dataset x 10 CNN teacher = 30 config
```

## Training Script

Ana script:

```powershell
python src/training/train_teacher.py --config configs/teachers/cnn/appleleaf9/resnet50.json
```

Dry-run örneği:

```powershell
python src/training/train_teacher.py --config configs/teachers/cnn/appleleaf9/resnet50.json --dry-run
```

Gerçek eğitim örneği:

```powershell
python src/training/train_teacher.py --config configs/teachers/cnn/appleleaf9/resnet50.json
```

## MLflow

Experiment adı:

```text
MasterThesis-CNN-Teachers
```

MLflow UI için önerilen komut:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

## Sonuç Dosyaları

Özet CSV:

```text
results/teachers/cnn/teacher_results.csv
```

Run artifact klasörü:

```text
results/teachers/cnn/runs/{dataset_name}/{model_name}/{run_name}/
```

Her run klasöründe şunlar tutulur:

- `config.json`
- `history.csv`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `learning_curves.png`

Checkpoint klasörü:

```text
checkpoints/teachers/cnn/{dataset_name}/{model_name}/
```

## Streamlit Dashboard

Streamlit içinde `CNN Teacher Experiments` sayfasından:

- dataset seçilebilir,
- CNN teacher model seçilebilir,
- config dosyası seçilebilir,
- dry-run kuyruğa eklenebilir,
- gerçek teacher eğitimi kuyruğa eklenebilir,
- job durumu `Job Monitor` sayfasından takip edilebilir.

## Sonuçların Yorumlanması

Teacher modeller KD aşamasında öğretmen adayı olarak kullanılacaktır. Seçimde yalnızca accuracy değil, özellikle macro F1, model kapasitesi, checkpoint kararlılığı ve dataset bazlı davranış dikkate alınmalıdır.

PlantVillage sonuçlarında ceiling effect beklenebileceği için teacher farkları AppleLeaf9 ve PlantPathology2021 üzerinde daha belirgin yorumlanmalıdır.

## Sonraki Aşama

CNN teacher altyapısı ve ilk teacher deneyleri tamamlandıktan sonra sıradaki aşama ViT teacher training altyapısıdır.
