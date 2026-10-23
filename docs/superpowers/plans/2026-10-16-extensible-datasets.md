# Extensible Dataset and Experiment Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline önerilir) or superpowers:subagent-driven-development after user approval. Steps use checkbox syntax. This document is a plan, not execution authorization.

**Goal:** Yeni single-label klasör verilerini kod değişikliği olmadan kaydetmek ve bütün baseline/KD akışlarında doğrulanmış modellerle kullanmak.
**Architecture:** Mevcut dataset/model katmanları genişletilir; küçük dosya tabanlı dataset registry, immutable content manifest/split ve ortak runtime/teacher çözümleyicileri kullanılır. Streamlit, CLI, notebook ve import aynı kimlik ve validation kurallarını tüketir. Eski kayıtlar/artefact yolları korunur.
**Tech Stack:** Mevcut Python/conda masterthesis, pathlib/json/hashlib/Pillow, PyTorch/torchvision/timm, Streamlit, MLflow, unittest/AppTest. Yeni bağımlılık yok.
**Spec:** `docs/superpowers/specs/2026-10-16-extensible-datasets-design.md`.

## Global Constraints

- Önce test RED, minimal uygulama, hedef test GREEN; her task sonunda regresyon.
- Sadece geçici fixture ve `pretrained=False` testleri; gerçek eğitim, indirme, data migration veya dosya silme ayrı kullanıcı onayı ister.
- Dirty checkout korunur. Bu plan commit/merge/push/worktree yetkisi vermez.
- Eski `DATASETS`, model setleri, stage kodları, split sıraları ve PlantPathology etiketi işleme korunur.
- Yeni veri: single-label, >=2 sınıf; imagefolder veya hazır train/val/test. Yeni CSV/multi-label görevleri yok.
- Yeni dataset/config referansı `{dataset_id, revision, split_id}`; strict içerik kimliği ve v1 provenance ayrıdır.
- Mevcut seed42,70/15/15 ve224 varsayılanları kalır; split/training seed birbirinden bağımsızdır.
- Katman/shape kabiliyeti doğrulanmadan özellik aktarımı açılmaz; CNN legacy objective sessizce değiştirilmez.
- Her task aşağıdaki somut testleri ekler; doğrulama komutları masterthesis ortamında çalıştırılır.

## Review Focus

1. Dosya/etiket değişimi, aynı adla farklı veri ve queue sırasında varsayılan değişimi (Task1/2/4).
2. Root dışı symlink/junction, case çakışması, bozuk/duplicate görüntü ve yetersiz sınıf (Task1/2).
3. Legacy hash'in içerik hash'i sanılması veya yanlış teacher/baseline eşleşmesi (Task3/5/9).
4. Logit'in feature registry'ye gereksiz bağımlılığı; CNN/ViT layout ve adapter gradientleri (Task3/6).
5. Uzak veri/teacher dosyasının yerelde sanılması, import traversal ve run silerken dataset kaybı (Task8/9).

## Ortak test komutu

PowerShell: `& C:/Users/alial/miniforge3/envs/masterthesis/python.exe -m unittest discover -s tests`.
Task testlerini `-p test_<name>.py` ile daralt. GREEN, exit0 ve unittest OK demektir.
Test dosyaları fixture kökünü dependency injection ile almalı; gerçek datasets/runs/mlflow.db'ye yazmamalı.

## Task1 — Dataset tanımı, keşif ve doğrulama

**Files:** Extend `src/datasets/dataset_registry.py`, `imagefolder_utils.py`; create `src/datasets/dataset_registration.py`; test `tests/test_dataset_registration.py`.
**Consumes:** Mevcut `DatasetConfig`, sınıf/görüntü toplama yardımcıları.
**Produces:** `DatasetRevisionRef(dataset_id:str,revision:str)` ve tam `DatasetRef(dataset_id:str,revision:str,split_id:str)` tipleri; `DatasetDefinition` (descriptor alanları ve geçerli default referanslarından hesaplanan `ready:bool`); `DatasetScan` (classes,samples,issues,content hash). `discover_datasets(root:Path)->list[Path]`, `scan_dataset(root:Path,candidate:Path,layout:str)->DatasetScan`, `register_dataset_revision(root:Path,definition:DatasetDefinition,scan:DatasetScan)->DatasetRevisionRef`, `list_datasets(root:Path)->list[DatasetDefinition]`, `resolve_dataset(root:Path,ref:DatasetRef)->DatasetConfig`. İçerik kaydı tek başına ready değildir. `DatasetConfig` yeni kimlik ve manifest path taşır; legacy alanlar optional/default ile korunur.

- [ ] Test: Potato yeni adla keşfedilir; class order deterministic; metadata/hidden klasör sınıf olmaz; nested class görüntüsü doğru etiketlenir.
- [ ] Test: aday revision içindeki duplicate bytes/cross-label duplicate, broken image, empty class, case collision, root escape ve slug collision kayıt oluşturmaz. Önceki revision'daki görüntülerin yeniden kullanımı kabul edilir. Yeniden aynı kayıt idempotent; yarım tarama commit olamaz; split'i olmayan revision eğitim-ready değildir.
- [ ] Task testini çalıştır, eksik davranış nedeniyle RED gözle.
- [ ] Spec'teki descriptor/manifest sınırlarında implement et; sınıf isimlerini koru, byte hash'leri üret; dosyaları taşımadan atomic metadata kaydı yap. Aynı dataset_id yazımlarını mevcut result-store lock yaklaşımıyla serileştir; eşzamanlı iki farklı root kaydı kimliği overwrite edemez. Commit öncesi scan'in hâlâ güncel olduğunu kontrol et.
- [ ] Test: sadece display/root location değişimi içerik revision'ını değiştirmez; görüntü byte veya label değişimi yeni revision ister. Eşzamanlı register, immutable manifest ve eski default'u bozmaz. Task GREEN.

## Task2 — Split ve runtime veri çözümleme

**Files:** Extend `src/datasets/create_splits.py`, `torch_datasets.py`, `dataloaders.py`; create `src/datasets/dataset_runtime.py`; test `tests/test_dataset_runtime.py`.
**Consumes:** Task1 DatasetRef/manifest.
**Produces:** `create_registered_split(root:Path,dataset_id:str,revision:str,ratios:dict,seed:int)->DatasetRef`; `resolve_dataset_runtime(root:Path,ref:DatasetRef|None,legacy_name:str|None)->dict` returns dataset_ref, split_file, class_names, class_mapping_hash, provenance_level. `verify_dataset_content(root:Path,ref:DatasetRef)->None`.

- [ ] Test: üç sınıflı fixture'da aynı seed aynı assignment; oranlar1 ve sınıf/split toplamları doğru. Legacy largest-remainder sonuçları birebir korunur.
- [ ] Test: hazır train/val/test assignment değişmez; örtüşen hash veya eksik sınıf reddedilir. Yetersiz örnek ve geçersiz oranlar anlaşılır hatadır.
- [ ] Testleri RED çalıştır. Oranları mevcut stratified fonksiyonuna optional parametre olarak geçir; global default çağrıları bozma.
- [ ] Manifest'i mevcut SplitJsonImageDataset formatına bağla; runtime kökü yeniden eşlenebilir olsun. Split yazımı başarılı olunca descriptor'ın default revision/split çiftini tek atomic güncellemeyle yayımla. Hata halinde önceki default korunur veya yeni kayıt not-ready kalır. Tam content verification preflight'ı açıkça çağrılır; bütünlüğü yalnızca mtime'a dayandırma.
- [ ] Test: root değişimi aynı identity; içerik mutasyonu reddedilir; split seed ile experiment seed ayrıdır. Task GREEN.

## Task3 — Model kabiliyetleri ve ortak deney config çözümü

**Files:** Extend `src/models/model_catalog.py`, `model_factory.py`, `src/training/kd_config.py`, `multi_kd_config.py`, `src/tracking/experiment_taxonomy.py`; create `src/training/experiment_config.py`; test `tests/test_dynamic_experiment_config.py`.
**Consumes:** DatasetRuntime; mevcut model sets/factory, training config dataclass'ları.
**Produces:** `get_model_spec(name:str)->ModelSpec` (roles,family,input constraints,feature spec); eski public sets korunur. `resolve_experiment_config(root:Path,payload:dict)->dict` legacy/modern config'i normalize edip pin'lenmiş dataset_ref ile runtime payload döndürür. FeatureSpec path/layout/pooling bilgisi Task6'da tamamlanır; ilan edilmeden doğrulanır.

- [ ] Test: yeni dataset tüm baseline rollerinde config olarak kabul edilir; bilinmeyen model/role, task_type, input size ve sınıf sayısı hataları reddedilir. Eski tüm9KD/12multi presetleri yüklenir.
- [ ] Test: logit config `features.12` gibi student'a özel katmanı zorunlu tutmaz; multi config listesi registry'deki uygun farklı modellerle2/3/>3 teacher destekler, tekrarları reddeder.
- [ ] RED çalıştır. Dataset/student/teacher sabit validation'larını registry tabanlı validation'a çevir. Modern layer default'ları modelden çözülür; eski explicit değerler aynen korunur.
- [ ] Test: descriptor default'u değişse bile normalize config'teki revision/split aynı kalır. Eski imports/stage/paths API regresyonlarıyla GREEN.

## Task4 — Baseline/teacher CLI, snapshot ve provenance

**Files:** Extend `src/training/train_baseline.py`, `train_teacher.py`, `recovery.py`; `src/tracking/run_metadata.py`; `app/experiment_runner.py`; test `tests/test_dynamic_baselines.py`, existing recovery/run_metadata tests.
**Consumes:** Task2 runtime, Task3 normalized config.
**Produces:** CLI/config'te optional dataset_ref alanları; job/config/result/checkpoint/run_metadata içinde dataset_id,revision,split_id,class_mapping_hash,identity_version,provenance_level. `dataset_fingerprint` v1 semantics korunur; yeni content hash ayrı alandır.

- [ ] Test: Potato Student CNN/ViT ve Teacher CNN/ViT command/config resolve eder; classifier sınıf sayısını metadata'dan alır. Kısa fixture/spy ile gerçek veri/MLflow'a dokunma.
- [ ] Test: enqueue→worker arasında dataset default değişimi snapshot'ı değiştirmez; içerik değişimi eğitimden önce reddedilir. RED gözle.
- [ ] CLI dataset seçimlerini registry'den çöz; queue snapshot'ını tüm referanslar çözülünce yaz, worker da content preflight yapar. Legacy CLI/config protokolünü değiştirme.
- [ ] Identity'yi MLflow/config/checkpoint/result/recovery'ye ekle. Test: farklı revision/split resume reddedilir; eski resume testi geçer. GREEN.

## Task5 — Ortak teacher adayları ve genelleştirilmiş logit KD

**Files:** Create `src/training/teacher_catalog.py`; extend `train_kd.py`, `train_multi_kd.py`, ilgili configler; test `tests/test_teacher_catalog.py`, extend `test_knowledge_distillation.py`, `test_multi_teacher_kd.py`.
**Consumes:** Runtime dataset ref ve model spec; her iki teacher summary kaynağı.
**Produces:** `find_compatible_teacher_runs(root:Path,dataset_ref:DatasetRef,student_model:str,kd_type:str)->list[dict]`, `validate_teacher_selection(root:Path,config:dict)->None`. Legacy çağrılar named compatibility yolundan geçer, yeni strict revision'a otomatik atanmaz.

- [ ] Test: aynı sınıf sayısı ama farklı class order/content/split teacher'ı dışlar; dry-run/kısmi/failed/eksik checkpoint ve hash mismatch dışlanır; uygun CNN+ViT bulunur. Legacy kanıt seviyesi ayrı kalır.
- [ ] Test: bir teacher ile single,2 ve4 farklı teacher ile multi logit CPU mini epoch; student CNN/ViT; teacher gradients kapalı, student tek forward, sabit3 üst sınırı yok. RED çalıştır.
- [ ] Aday servisini UI'dan bağımsız kur. Logit koşusunda feature-layer çözümü/capture/probe yapma; validation UI ve CLI'da ortak olsun.
- [ ] Checkpoint/dry-run/final sonuç formatına kimlik ekle, mevcut loss formüllerini değiştirme. Teacher eksik/uyumsuzken MLflow başarısızlık durumları ve artifact davranışıyla GREEN.

## Task6 — Tüm kayıtlı modellerde feature/relation kabiliyetleri

**Files:** Extend `src/training/kd_features.py`, `kd_strategies.py`, `multi_kd_strategies.py`, `kd_trainer.py`, `multi_kd_trainer.py`, iki KD CLI; model catalog specs. Test `tests/test_kd_model_capabilities.py` and existing KD tests.
**Consumes:** Task3 ModelSpec; Task5 validated selections.
**Produces:** `extract_feature_vector(tensor,feature_spec)->Tensor[B,D]`; `create_feature_adapter(student_spec,teacher_spec,shapes)->nn.Module`. Legacy Conv1x1 ve modern Linear adapter backend seçimi config/checkpoint'te açıkça saklanır.

- [ ] Test: her kayıtlı modelden `pretrained=False` ile doğrulanmış feature layout/output shape. Modelleri sırayla oluştur/serbest bırak; büyük tüm çiftler kartesyen test edilmez.
- [ ] Test: NCHW↔NCHW eski loss/adapter aynı; CNN↔ViT ve ViT↔ViT pooled-vector Linear; tokens/NHWC yanlış eksende pool edilmez. Relation vectors batch ilişkilerini korur. RED çalıştır.
- [ ] Factory classifier sınırları veya doğrulanmış feature node'larıyla registry'yi doldur. Mimariye özel CLS/mean/layout kararını model testinde sabitle; shape tahminine dayanan otomatik eksen seçimi yapma.
- [ ] Mini single/multi feature ve relation train+val batch: student/adapter gradientleri var, teacher gradient yok; her teacher ayrı adapter; inference checkpoint adapter'ı student modeline katmaz. Yeni representation alanları MLflow/protocol/history'de ayrılır. GREEN.

## Task7 — Veri Setleri sayfası ve dinamik launcher'lar

**Files:** Create `app/pages/12_Datasets.py`, `app/dataset_view.py`; extend `streamlit_app.py`, `standard_launcher.py`, baseline sayfaları2/5/6/11, KD sayfaları8/9, `experiment_ui.py`; create `src/training/experiment_presets.py`, `configs/presets/{student_baseline,teacher_baseline,knowledge_distillation,multi_teacher_knowledge_distillation}/` JSONları. Tests `test_dataset_page.py`, `test_dynamic_launcher_pages.py`.
**Consumes:** Task1 registration, Task2 runtime, Task3 config/model capabilities, Task5 teacher candidates.
**Produces:** `list_presets(stage:str,model_name:str|None,kd_type:str|None)->list[dict]`, `compose_preset(preset:dict,dataset_ref:DatasetRef,selections:dict)->dict` consumed by existing builders. Genel presetler dataset/checkpoint içermez; eski specific config yükleme korunur.

- [ ] AppTest RED: Potato kaydı restart olmadan dört baseline dalı ve iki KD sayfasında görünür; model seçimi configten önce; count/class names doğru; validation başarısızken kayıt ve eğitim pasif.
- [ ] AppTest: yeni dataset'te teacher yoksa sebep+teacher-baseline bağlantısı; bir teacher'da single açık/multi pasif; yeterli uygun teacher'da doğru seçenekler; teacher sayısı dinamik.
- [ ] Native Streamlit ile beşinci `Veri Setleri` navigasyonu ve aşamalı kayıt kur; ağır scan yalnızca butonla ve ilerlemeyle, her widget rerun'ında değil. Belirsiz layout kullanıcıya seçtirilir.
- [ ] Bütün launcher'larda sabit dataset/split/config köklerini kaldır, registry/services kullan. Student/teacher rollerini ve onay/queue/notebook/recovery kontrollerini koru; dynamic loss ve validation panelini güncelle.
- [ ] AppTest snapshot'ta dataset_ref, teacher kimliği ve preset/UI önceliğini doğrula; dependent job filters kayıtlı işlerden beslenmeye devam eder. GREEN.

## Task8 — Özel veri ve KD için taşınabilir notebook/bundle

**Files:** Extend `app/portable_notebook.py`, `src/tracking/external_bundle.py`, `app/external_run_import.py`, `src/training/train_kd.py`, `train_multi_kd.py` (baseline/teacher bundle callers dahil); tests `test_portable_notebook.py`, `test_external_run_transfer.py`, new `test_dynamic_dataset_transfer.py`.
**Consumes:** Immutable dataset manifest/split, resolved config, teacher SHA kimlikleri.
**Produces:** version2 bundle manifest; notebook builder optional `dataset_package:dict`, `teacher_inputs:list[dict]`, `runtime_config:dict` accepts tüm5stage; v1 signature callers çalışır. v2 import'ta class/content/split identities güvenli manifest üzerinden doğrulanır.

- [ ] RED: custom dataset notebook'u sabit indirme URL'si üretmez, Drive/Kaggle mount yolu ister; farklı root'ta aynı manifest/split korunur; hatalı veri/checkpoint training çağrısına ulaşmaz.
- [ ] Notebook'a JSON runtime config/manifest/split ekle; uzak mount yollarını input hücresinde eşle. Başlangıçta gereken v2 kod kabiliyetini kontrol et, eski remote commit'te anlaşılır hata üret; dirty-worktree indirme yasağı ekleme. Teacher checkpoint'leri ayrı kullanıcı mount input'udur, repoya veya notebook'a ağırlık gömme. Yerel MLflow run ID'yi uzak tracking'de arama; kaynak kimliği olarak sakla.
- [ ] v2'de KD stage/path/summary/multi teacher kimliklerini destekle; mevcut zip limits/hash/path kontrollerini zayıflatma. Test: traversal, symlink, id collision, teacher substitution, class order ve duplicate bundle/idempotency.
- [ ] v1 legacy import regresyonu; v2 raw veri yerelde yokken result manifest görünür ama dataset training-ready/teacher-eligible sayılmaz. Yeni registration gerekir. Unpickle yok; özgün kaynak kimliği korunur.
- [ ] Colab/Kaggle hücrelerini fake mount/filesystem/subprocess ile test et; ağ/GPU/credential olmadan canlı stdout, Drive recovery ve final bundle davranışı GREEN.

## Task9 — Sonuç, karşılaştırma, recovery ve kaldırma sınırları

**Files:** Extend `src/evaluation/experiment_comparison.py`, `app/comparison_view.py`, `app/pages/4_Results_Explorer.py`, `app/job_filters.py`, `experiment_removal.py`, `scripts/reset_experiments.py`, recovery metadata tüketicileri; related comparison/removal/reset/filter tests.
**Consumes:** v2 identity+representation fields; stage/schema backward compatibility.
**Produces:** UI/report grupları dataset_id+revision+split+class mapping+eval protocol ile ayrılır; baseline kıyaslamasında student ve training seed de eşleşir.

- [ ] RED: aynı display_name farklı içerik/split yanlış eşleşmez; v1 legacy'ye yüksek güven etiketi verilmez; external-only metadata eğitim açmaz.
- [ ] Result kaynak/filtreleri yeni kayıt ve sonuçlardan doldur, dataset revision/split kısa etiketi ve provenance göster. Raporlara dataset identity ve representation ekle, mevcut veri setlerini birleştirip ortalama alma.
- [ ] Test: interrupted/completed yeni dataset/KD koşusu güvenli arşivlenirken raw images, registry, shared splits ve dataset descriptors değişmez. Aktif/resumed job korumaları korunur; reset de registry'yi korur. GREEN.

## Task10 — Tam kabul ve teslim

**Files:** Add `tests/test_extensible_platform_acceptance.py`; add `docs/DATASET_TEMPLATE.md`; update README/WEB_UI ve etkilenen schema/config açıklamaları. Uygulanan yüzey brief'i güncellenir; native design sistemi değiştirilmez.

- [ ] RED: geçici Potato fixture'ında keşif→kayıt→split→preset→queue snapshot→kısa CLI train+val→summary→transfer/import→compare→archive zinciri. Dört baseline dalı ve üçKD türü single/multi kapsanır; büyük model eğitimleri değil oyuncak ağlar kullanılır, gerçek model testleri Task6'dadır.
- [ ] Eski üç dataset/split/class mapping fixture'ları ve mevcut tüm testleri çalıştır. Başlangıçta mevcut167 test vardı; sayı sabit assertion yapılmaz, eklenen testlerle tüm suite exit0 gerekir.
- [ ] masterthesis ortamında fresh full unittest ve git diff --check; masaüstü/mobil gerçek arayüzde Potato geçici test köküyle review. Gerçek araştırma verisini kayıt/yeniden bölme yok; training/stop/delete UI aksiyonları tetiklenmez.
- [ ] Bağımsız final code review ve Impeccable native-surface review; material bulgular RED→GREEN ile düzeltilir. Dataset şablonu, known limits, hash tarama maliyeti, legacy provenance ve remote-input gereksinimi belgelenir.
- [ ] Ancak uygulama bittiğinde mevcut dışlama kurallarıyla Graphify güncelle; raw datasets/results/checkpoints/logs dahil edilmez. MCP/Graphify erişimi yoksa dürüstçe raporla, veri veya ortamı keyfi değiştirme.
- [ ] Kullanıcıya değişiklik özeti, test kanıtı, migration yapılmadığı ve kalan sınırları bildir; commit/merge/push ayrıca kararlaştırılır.

## Uygulama sırası ve durma kapıları

Task1→2→3→4→5→6→7→8→9→10. Bunlar tek platform genişletmesinin aşamalarıdır;
Task4 sonunda yeni veri baseline'ı, Task5 sonunda dinamik logitKD, Task6 sonunda
geniş feature/relation ve Task8 sonunda uzak aktarım ayrı kabul kapılarıdır.
Bir kabiliyet tamamlanmadan UI'da varmış gibi açılmaz. Eski protokole etkisi olan
bir test başarısızsa sonraki aşamaya geçilmez.

## Plan hazırlama sırasında yapılan tasarım önerileri

- Hazır train/val/test şablonunu da kabul etmek; mevcut test bölümünü yeniden bölmemek.
- Beşinci ana menü `Veri Setleri`; kayıt için yeni DB/servis kurmamak.
- Modern ViT/cross-family feature aktarımında pooled-vector Linear; eski CNN hedefini korumak.
- Özel veri için mount tabanlı notebook girişi; upload sunucusu, internet kaynak tahmini ve otomatik veri Git push'u yok.
- Yeni CSV/multi-label ve model eklentisi bu planda yok. Bunlar kullanıcı tarafından ayrıca seçilecek genişletmelerdir.

Onaydan sonra önerilen uygulama biçimi: aynı oturumda benim taskları sırayla
uygulamam ve sonunda bağımsız inceleme. Her taskta ayrı uygulayıcı/reviewer
çalıştırmak bu birbirine bağlı kapsam için gereksiz tekrar yaratır.

## Plan inceleme kaydı

Bağımsız builder check iki önemli belirsizlik buldu: registration'ın split'ten
önce tam DatasetRef döndürmesi ve duplicate kapsamı. Birincisi revision-only
kayıt → split → atomic ready-publication olarak netleştirildi. Duplicate kontrolü
yalnızca aday revision'ın içindedir; önceki revision veya başka dataset'te aynı
görüntünün bulunması yasak değildir. Bunlar kullanıcı kapsamını değiştirmeyen
uygulama kararlarıdır. Kaynak/test yolları ve task arası tipler yeniden kontrol
edildi; bu plan hazırlanırken test veya eğitim çalıştırılmadı.
