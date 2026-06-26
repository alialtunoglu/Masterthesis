# Student Baseline Report

## Amaç
Student baseline aşamasının amacı, KD öncesinde her veri seti için bağımsız öğrenci model performansını ölçmek ve KD ile neyin iyileştirileceğini netleştirmektir.

## Veri Setleri
appleleaf9, plantvillage, plantpathology2021

## Student Modeller
mobilenet_v3_small, mobilenet_v3_large, efficientnet_b0, resnet18

## Eğitim Ayarları
Tüm seçilen run'larda seed=42, ImageNet pretrained başlangıç, AdamW optimizer, image_size=224 ve aynı split JSON mantığı kullanılmıştır.

## Okunan Dosyalar
- `results/baseline/baseline_results.csv`
- `results/baseline/runs/{dataset}/{model}/{run_name}/`

## Veri Seti Bazlı Sonuçlar

### appleleaf9
| model_name | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | best_val_accuracy | best_val_macro_f1 | params | model_size_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mobilenet_v3_large | 0.9699 | 0.9596 | 0.9594 | 0.9592 | 0.9699 | 0.9748 | 0.9677 | 4213561 | 16.1669 |
| efficientnet_b0 | 0.9644 | 0.9450 | 0.9527 | 0.9485 | 0.9645 | 0.9748 | 0.9659 | 4019077 | 15.4922 |
| resnet18 | 0.9612 | 0.9428 | 0.9431 | 0.9414 | 0.9615 | 0.9647 | 0.9511 | 11181129 | 42.6894 |
| mobilenet_v3_small | 0.9539 | 0.9378 | 0.9431 | 0.9401 | 0.9539 | 0.9574 | 0.9485 | 1527081 | 5.8718 |

### plantvillage
| model_name | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | best_val_accuracy | best_val_macro_f1 | params | model_size_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet18 | 0.9948 | 0.9943 | 0.9913 | 0.9926 | 0.9948 | 0.9951 | 0.9931 | 11196006 | 42.7461 |
| mobilenet_v3_large | 0.9946 | 0.9928 | 0.9921 | 0.9924 | 0.9946 | 0.9939 | 0.9914 | 4250710 | 16.3086 |
| mobilenet_v3_small | 0.9942 | 0.9929 | 0.9919 | 0.9923 | 0.9942 | 0.9945 | 0.9914 | 1556806 | 5.9852 |
| efficientnet_b0 | 0.9953 | 0.9949 | 0.9871 | 0.9903 | 0.9953 | 0.9962 | 0.9939 | 4056226 | 15.6339 |

### plantpathology2021
| model_name | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | best_val_accuracy | best_val_macro_f1 | params | model_size_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| efficientnet_b0 | 0.9595 | 0.9588 | 0.9444 | 0.9504 | 0.9587 | 0.9494 | 0.9297 | 4015234 | 15.4776 |
| resnet18 | 0.9437 | 0.9383 | 0.9290 | 0.9322 | 0.9426 | 0.9293 | 0.9086 | 11179590 | 42.6835 |
| mobilenet_v3_large | 0.9448 | 0.9387 | 0.9262 | 0.9317 | 0.9440 | 0.9347 | 0.9179 | 4209718 | 16.1522 |
| mobilenet_v3_small | 0.9402 | 0.9341 | 0.9195 | 0.9256 | 0.9390 | 0.9344 | 0.9202 | 1524006 | 5.8601 |

## Genel Ortalama Performans
| model_name | num_datasets | avg_test_accuracy | avg_test_macro_f1 | avg_model_size_mb | avg_params | performance_size_score |
| --- | --- | --- | --- | --- | --- | --- |
| efficientnet_b0 | 3 | 0.9731 | 0.9631 | 15.5346 | 4030179.0000 | 0.0620 |
| mobilenet_v3_large | 3 | 0.9698 | 0.9611 | 16.2092 | 4224663.0000 | 0.0593 |
| resnet18 | 3 | 0.9666 | 0.9554 | 42.7064 | 11185575.0000 | 0.0224 |
| mobilenet_v3_small | 3 | 0.9628 | 0.9527 | 5.9057 | 1535964.3333 | 0.1613 |

## Model Boyutu ve Parametre Karşılaştırması
| model_name | avg_model_size_mb | avg_params | avg_test_macro_f1 |
| --- | --- | --- | --- |
| mobilenet_v3_small | 5.9057 | 1535964.3333 | 0.9527 |
| efficientnet_b0 | 15.5346 | 4030179.0000 | 0.9631 |
| mobilenet_v3_large | 16.2092 | 4224663.0000 | 0.9611 |
| resnet18 | 42.7064 | 11185575.0000 | 0.9554 |

## Veri Seti Bazında En İyi Student
| dataset_name | model_name | test_macro_f1 | test_accuracy | model_size_mb |
| --- | --- | --- | --- | --- |
| plantvillage | resnet18 | 0.9926 | 0.9948 | 42.7461 |
| appleleaf9 | mobilenet_v3_large | 0.9592 | 0.9699 | 16.1669 |
| plantpathology2021 | efficientnet_b0 | 0.9504 | 0.9595 | 15.4776 |

## Student Seçim Önerileri
| recommendation_type | selected_model | reason | avg_test_accuracy | avg_test_macro_f1 | avg_model_size_mb | avg_params |
| --- | --- | --- | --- | --- | --- | --- |
| Best performance student | efficientnet_b0 | Highest average test macro F1 across datasets. | 0.9731 | 0.9631 | 15.5346 | 4030179.0000 |
| Best accuracy student | efficientnet_b0 | Highest average test accuracy across datasets. | 0.9731 | 0.9631 | 15.5346 | 4030179.0000 |
| Best lightweight student | mobilenet_v3_small | Smallest model within 0.02 average macro F1 of the best model. | 0.9628 | 0.9527 | 5.9057 | 1535964.3333 |
| Recommended KD student | mobilenet_v3_small | MobileNetV3-Small is the lightest student and stays within 0.02 avg macro F1 of the best model, making KD gains easier to interpret. | 0.9628 | 0.9527 | 5.9057 | 1535964.3333 |
| Strong compact baseline | efficientnet_b0 | Compact but stronger reference student for secondary KD comparisons. | 0.9731 | 0.9631 | 15.5346 | 4030179.0000 |
| Best performance-size balance | mobilenet_v3_small | Highest avg_test_macro_f1 / avg_model_size_mb score. | 0.9628 | 0.9527 | 5.9057 | 1535964.3333 |

## PlantVillage Ceiling Effect Yorumu
PlantVillage sonuçları tüm modellerde çok yüksek olduğu için bu veri setinde ceiling effect beklenir. KD etkisi burada sınırlı görünebilir; AppleLeaf9 ve PlantPathology2021 sonuçları KD kazanımlarını yorumlamak için daha ayırt edici olacaktır.

## KD Seçim Kriterleri
KD aşamasında sadece accuracy değil macro F1, model boyutu, parametre sayısı ve ileride ölçülecek inference maliyeti birlikte değerlendirilecektir. Macro F1 özellikle sınıf dengesizliği ve az temsil edilen hastalık sınıfları için daha açıklayıcıdır.

## Artifact Kontrolü
Eksik artifact sayısı: 0

## Sonraki Aşama
Student baseline aşaması kapatıldıktan sonra CNN teacher training altyapısına geçilmelidir.
