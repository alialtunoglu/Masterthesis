# Graph Report - MasterThesis  (2026-08-04)

## Corpus Check
- 172 files · ~870,220 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 991 nodes · 1976 edges · 74 communities (50 shown, 24 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 274 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5199f199`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- experiment_runner.py
- 4_Results_Explorer.py
- ConvFeatureAdapter
- main
- fit
- analyze_student_baselines.py
- create_splits.py
- reset_experiments.py
- get_model_summary_dict
- dataloaders.py
- 3_Job_Monitor.py
- model_factory.py
- RunTrackingGateway
- load_kd_config
- main
- main
- 4_Results_Explorer.py
- train_teacher.py
- ensure_dir
- create_model
- experiment_naming.py
- Deney Sonuçları Analizi ve Bildiri Potansiyeli
- external_run_import.py
- 6_MLflow_Helper.py
- streamlit_app.py
- Knowledge Distillation Sonuçları: Karşılaştırmalı Analiz
- 4_Results_Explorer.py
- get_project_root
- test_vision_transformer_teachers.py
- kd_features.py
- portable_notebook.py
- main
- get_queue_worker_status
- kd_trainer.py
- main
- multi_kd_aggregation.py
- optimization.py
- 5_CNN_Teacher_Experiments.py
- __init__.py
- __init__.py
- __init__.py
- __init__.py
- __init__.py
- __init__.py
- masterthesis-plant-disease-kd
- seed.py
- StreamlitVitPageTests
- Student Baseline Report
- MasterThesis Plant Disease KD
- Experiment Plan
- CNN Teacher Training
- Streamlit Web UI
- Dataset Preparation Notes
- MLflow Tracking
- Graphify Corpus Exclusions Design
- Global Constraints
- Environment Setup
- 00-project-principles.md
- 10-clean-code.md
- 20-clean-architecture.md
- 30-ml-experiment-rules.md
- 40-git-rules.md
- experiment-tracking-skill.md
- git-commit-skill.md
- python-project-skill.md
- pytorch-ml-skill.md
- 01-feature-development.md
- 02-dataset-pipeline.md
- 03-training-experiment.md
- 04-result-logging.md
- __init__.py

## God Nodes (most connected - your core abstractions)
1. `main()` - 35 edges
2. `main()` - 35 edges
3. `ExperimentRemovalService` - 31 edges
4. `main()` - 30 edges
5. `main()` - 29 edges
6. `ExperimentRemovalServiceTests` - 26 edges
7. `get_project_root()` - 23 edges
8. `load_kd_config()` - 21 edges
9. `ensure_dir()` - 21 edges
10. `load_multi_kd_config()` - 20 edges

## Surprising Connections (you probably didn't know these)
- `ExternalRunTransferTests` --uses--> `RemovalMode`  [INFERRED]
  tests/test_external_run_transfer.py → app/experiment_removal.py
- `ExternalRunTransferTests` --uses--> `ExperimentRemovalService`  [INFERRED]
  tests/test_external_run_transfer.py → app/experiment_removal.py
- `model_profile()` --calls--> `create_model()`  [INFERRED]
  app/pages/7_Knowledge_Distillation.py → src/models/model_factory.py
- `matching_configs()` --calls--> `load_multi_kd_config()`  [INFERRED]
  app/pages/8_Multi_Teacher_Knowledge_Distillation.py → src/training/multi_kd_config.py
- `model_profile()` --calls--> `create_model()`  [INFERRED]
  app/pages/8_Multi_Teacher_Knowledge_Distillation.py → src/models/model_factory.py

## Import Cycles
- None detected.

## Communities (74 total, 24 thin omitted)

### Community 0 - "experiment_runner.py"
Cohesion: 0.09
Nodes (38): build_baseline_command(), build_multi_kd_command(), cancel_queued_job(), enqueue_job(), _ensure_run_dirs(), get_job_runtime_seconds(), _job_path(), list_jobs() (+30 more)

### Community 1 - "4_Results_Explorer.py"
Cohesion: 0.22
Nodes (13): TrainingConfig, load_multi_kd_config(), _mapping(), merge_multi_kd_config(), MultiExperimentConfig, MultiKDConfig, Any, Path (+5 more)

### Community 2 - "ConvFeatureAdapter"
Cohesion: 0.17
Nodes (16): ConvFeatureAdapter, Align student channels and spatial dimensions to a teacher feature map., MultiDistillationConfig, _angles(), create_multi_teacher_strategy(), _distances(), MultiFeatureStrategy, MultiLogitStrategy (+8 more)

### Community 3 - "main"
Cohesion: 0.23
Nodes (21): add_runtime_fields(), append_summary_row(), build_artifact_paths(), build_final_config(), build_run_name_extra(), build_summary_row(), cli_overrides(), load_config_file() (+13 more)

### Community 4 - "fit"
Cohesion: 0.20
Nodes (17): Optimizer, _batch_limit_reached(), evaluate(), fit(), device, MetricLogger, Module, Path (+9 more)

### Community 5 - "analyze_student_baselines.py"
Cohesion: 0.22
Nodes (22): add_artifact_paths(), artifact_check(), build_overall(), build_recommendations(), _format_markdown_value(), load_results(), main(), parse_args() (+14 more)

### Community 6 - "create_splits.py"
Cohesion: 0.09
Nodes (36): Counter, analyze_imagefolder(), analyze_plantpathology2021(), main(), print_summary(), Analyze raw datasets without modifying them., build_split_payload(), create_split() (+28 more)

### Community 7 - "reset_experiments.py"
Cohesion: 0.19
Nodes (19): NamedTuple, archive_path(), confirm_or_exit(), delete_paths(), find_project_root(), main(), move_to_archive(), parse_args() (+11 more)

### Community 8 - "get_model_summary_dict"
Cohesion: 0.22
Nodes (14): model_profile(), model_profile(), count_parameters(), count_trainable_parameters(), estimate_model_size_mb(), get_model_compute_dict(), get_model_summary_dict(), Module (+6 more)

### Community 9 - "dataloaders.py"
Cohesion: 0.15
Nodes (12): Dataset, build_eval_transform(), build_train_transform(), create_dataloaders(), Path, DataLoader creation from split JSON files., Create train, validation, and test DataLoaders from a split file., Any (+4 more)

### Community 10 - "3_Job_Monitor.py"
Cohesion: 0.27
Nodes (8): choose_monitor_job_id(), find_job_by_id(), is_kd_job(), is_multi_kd_job(), job_option_label(), job_type_label(), latest_running_job(), Live-ish job monitor for Streamlit-launched experiments.

### Community 11 - "model_factory.py"
Cohesion: 0.06
Nodes (38): _CsvMutation, ExperimentActiveError, ExperimentIdentity, ExperimentRemovalError, ExperimentRemovalService, _file_contains(), find_jobs_for_run(), InterruptedJobRemovalReport (+30 more)

### Community 12 - "RunTrackingGateway"
Cohesion: 0.20
Nodes (14): compute_classification_metrics(), Classification metrics for multi-class experiments., Compute aggregate multi-class classification metrics., _epoch_summary(), evaluate_teacher_ensemble(), fit_multi_teacher_kd(), device, MetricLogger (+6 more)

### Community 13 - "load_kd_config"
Cohesion: 0.05
Nodes (45): build_kd_command(), Build a Windows-safe KD command from a resolved config snapshot., config_label(), config_paths(), load_uploaded(), loss_preview(), matching_config_paths(), Any (+37 more)

### Community 14 - "main"
Cohesion: 0.15
Nodes (20): end_run(), _import_mlflow(), log_artifact(), log_artifacts(), log_metrics(), log_params(), log_text(), Any (+12 more)

### Community 15 - "main"
Cohesion: 0.24
Nodes (17): append_summary(), build_paths(), build_run_name(), flatten_params(), load_teacher_checkpoint(), main(), model_latency_ms(), parse_args() (+9 more)

### Community 16 - "4_Results_Explorer.py"
Cohesion: 0.11
Nodes (21): _config_label(), _filter_configs(), _load_config_defaults(), _load_config_index(), Any, Baseline experiment launcher page., Vision Transformer teacher experiment launcher., load_baseline_results() (+13 more)

### Community 17 - "train_teacher.py"
Cohesion: 0.26
Nodes (19): add_runtime_fields(), append_summary_row(), build_artifact_paths(), build_run_name(), build_summary_row(), load_config_file(), main(), normalize_config_keys() (+11 more)

### Community 18 - "ensure_dir"
Cohesion: 0.18
Nodes (17): compute_per_class_metrics(), DataFrame, Return per-class precision, recall, F1, and support as a DataFrame., Path, Evaluation report artifacts for baseline classification runs., Save confusion matrix as a CSV table and a heatmap PNG., Save loss and validation metric curves for a training run., save_confusion_matrix_artifacts() (+9 more)

### Community 19 - "create_model"
Cohesion: 0.14
Nodes (16): checkpoint_model_compatible(), expected_classes(), feature_shape_profile(), matching_configs(), DataFrame, Path, Series, Multi-teacher knowledge-distillation experiment launcher. (+8 more)

### Community 20 - "experiment_naming.py"
Cohesion: 0.33
Nodes (6): build_experiment_name(), build_run_name(), _normalize(), Consistent experiment and run naming helpers., Build a stable MLflow run name., Build a project-level MLflow experiment name.

### Community 21 - "Deney Sonuçları Analizi ve Bildiri Potansiyeli"
Cohesion: 0.05
Nodes (37): 10. Sonuçların güvenilirliği ve sınırlamalar, 11. Bu sonuçlardan bildiri olur mu?, 12. Gönderim öncesi gerekli deneyler, 13. Önerilen bildiri çerçevesi, 14. Nihai değerlendirme, 1. Yönetici özeti, 2. İncelenen deney envanteri, 3.1 AppleLeaf9 (+29 more)

### Community 22 - "external_run_import.py"
Cohesion: 0.14
Nodes (24): _default_register_run(), ExternalRunImportError, import_external_bundle(), Path, Validate and import external experiment bundles without loading model code., _read_manifest(), _safe_relative(), scan_import_inbox() (+16 more)

### Community 28 - "Knowledge Distillation Sonuçları: Karşılaştırmalı Analiz"
Cohesion: 0.06
Nodes (30): 10. Bilimsel yorum ve sınırlamalar, 11. Önerilen takip deneyleri, 12. Nihai değerlendirme, 1. Kısa cevap: Student iyileşti mi?, 2. Baseline student ve teacher sonuçları, 3. Single-teacher KD sonuçları, 4. Single-teacher sınıf bazlı analiz, 5. Multi-teacher KD sonuçları (+22 more)

### Community 29 - "4_Results_Explorer.py"
Cohesion: 0.23
Nodes (19): _comparison_row(), _filter_multiselect(), _job_belongs_to_source(), _json_list(), _linked_jobs(), DataFrame, Path, Series (+11 more)

### Community 30 - "get_project_root"
Cohesion: 0.15
Nodes (15): Project overview page for the local Streamlit dashboard., display_path_status(), file_exists(), format_metric(), get_project_root(), Any, DataFrame, Path (+7 more)

### Community 31 - "test_vision_transformer_teachers.py"
Cohesion: 0.17
Nodes (10): build_teacher_command(), Build a Windows-safe teacher training command., make_command(), get_model_type(), Return the project model type for a supported model., build_final_config(), cli_overrides(), parse_args() (+2 more)

### Community 32 - "kd_features.py"
Cohesion: 0.18
Nodes (10): default_feature_layer(), FeatureCapture, FeatureShapes, infer_feature_shapes(), device, Module, Tensor, Feature extraction boundaries used by feature and relational KD strategies. (+2 more)

### Community 33 - "portable_notebook.py"
Cohesion: 0.26
Nodes (11): build_portable_notebook(), _cell(), _dataset_setup(), _git(), Path, Generate portable Colab/Kaggle notebooks from the selected local command., Render a guarded Streamlit notebook download control., render_notebook_download() (+3 more)

### Community 34 - "main"
Cohesion: 0.25
Nodes (13): append_summary(), build_paths(), build_run_name(), flatten_params(), main(), parse_args(), Any, device (+5 more)

### Community 35 - "get_queue_worker_status"
Cohesion: 0.20
Nodes (12): get_log_file_info(), get_queue_worker_status(), Path, Read the last n lines of a log file., Resolve a job log path relative to the project root., Return compact file metadata for a job log., Return queue worker state and heartbeat metadata., Stop only the queue worker process, not training jobs. (+4 more)

### Community 36 - "kd_trainer.py"
Cohesion: 0.32
Nodes (11): _epoch_summary(), evaluate_student(), fit_kd(), freeze_teacher(), device, MetricLogger, Module, Path (+3 more)

### Community 37 - "main"
Cohesion: 0.25
Nodes (10): is_process_running(), Return whether a process id is currently alive., claim_worker_slot(), main(), mark_state(), parse_args(), Namespace, Background queue worker for Streamlit-launched experiment jobs. (+2 more)

### Community 38 - "multi_kd_aggregation.py"
Cohesion: 0.29
Nodes (6): _non_negative(), _positive_finite(), Teacher-weight resolution policies for multi-teacher distillation., resolve_teacher_weights(), TeacherConfig, AggregationTests

### Community 39 - "optimization.py"
Cohesion: 0.32
Nodes (7): Parameter, create_optimizer(), create_scheduler(), is_improvement(), monitor_mode(), Optimizer, scheduler, and monitored-metric factories shared by KD training., step_scheduler()

### Community 40 - "5_CNN_Teacher_Experiments.py"
Cohesion: 0.40
Nodes (4): config_label(), load_config_index(), Any, CNN teacher experiment launcher page.

### Community 48 - "seed.py"
Cohesion: 0.50
Nodes (3): Reproducibility helpers., Set common random seeds used in experiments and data preparation., set_seed()

### Community 76 - "Student Baseline Report"
Cohesion: 0.11
Nodes (18): Amaç, appleleaf9, Artifact Kontrolü, Eğitim Ayarları, Genel Ortalama Performans, KD Seçim Kriterleri, Model Boyutu ve Parametre Karşılaştırması, Okunan Dosyalar (+10 more)

### Community 77 - "MasterThesis Plant Disease KD"
Cohesion: 0.10
Nodes (19): Bağımlılıklar, CNN Teacher Eğitimi, Daha Fazla Dokümantasyon, Deney Çıktılarını Temizleme, Knowledge Distillation, MasterThesis Plant Disease KD, Mevcut Aşamalar, MLflow UI (+11 more)

### Community 78 - "Experiment Plan"
Cohesion: 0.17
Nodes (11): Aşama 1: Altyapı Doğrulama, Aşama 2: AppleLeaf9 Student Baseline, Aşama 3: PlantVillage Student Baseline, Aşama 4: CNN Teacher Deneyleri, Aşama 5: ViT Teacher Deneyleri, Aşama 6: Teacher/Student Seçimi, Aşama 7: Single-Teacher KD, Aşama 8: Multi-Teacher KD (+3 more)

### Community 79 - "CNN Teacher Training"
Cohesion: 0.20
Nodes (9): CNN Teacher Modelleri, CNN Teacher Training, Config Konumu, MLflow, Sonraki Aşama, Sonuç Dosyaları, Sonuçların Yorumlanması, Streamlit Dashboard (+1 more)

### Community 80 - "Streamlit Web UI"
Cohesion: 0.18
Nodes (10): Baseline Deney Başlatma, CPU Mini Deney ve CUDA Full Deney, Dosya Konumları, Job Monitor, Job Queue, MLflow UI, Sayfalar, Streamlit Dashboard (+2 more)

### Community 81 - "Dataset Preparation Notes"
Cohesion: 0.25
Nodes (7): Analyze Datasets, Create Splits, Dataset Preparation Notes, Existing Dataset Structure, PlantPathology2021 Single-Label Policy, Used Folders, Why Raw Datasets Are Not Physically Split

### Community 82 - "MLflow Tracking"
Cohesion: 0.25
Nodes (7): Baseline Real Run, Baseline Smoke Runs, CUDA AppleLeaf9 Full Baseline, Logged MLflow Content, MLflow Tracking, `results/baseline/` Files, Start the MLflow UI

### Community 83 - "Graphify Corpus Exclusions Design"
Cohesion: 0.25
Nodes (7): Data Flow, Design, Failure Handling, Goal, Graphify Corpus Exclusions Design, Scope, Verification

### Community 85 - "Global Constraints"
Cohesion: 0.33
Nodes (5): Global Constraints, Graphify Corpus Exclusions Implementation Plan, Task 1: Add and verify persistent corpus exclusions, Task 2: Rebuild the filtered graph safely, Task 3: Verify excluded paths and final artifacts

### Community 86 - "Environment Setup"
Cohesion: 0.40
Nodes (4): CPU-Only Fallback, Environment Setup, If CPU PyTorch Was Installed Accidentally, Recommended CUDA Environment on Windows

## Knowledge Gaps
- **148 isolated node(s):** `masterthesis-plant-disease-kd`, `Project Principles`, `Clean Code Rules`, `Clean Architecture Rules`, `Machine Learning Experiment Rules` (+143 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **24 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ExperimentRemovalService` connect `model_factory.py` to `4_Results_Explorer.py`, `external_run_import.py`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Why does `main()` connect `main` to `kd_features.py`, `4_Results_Explorer.py`, `ConvFeatureAdapter`, `kd_trainer.py`, `optimization.py`, `get_model_summary_dict`, `dataloaders.py`, `RunTrackingGateway`, `main`, `main`, `seed.py`, `ensure_dir`, `create_model`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `create_model()` connect `create_model` to `main`, `main`, `get_model_summary_dict`, `load_kd_config`, `main`, `train_teacher.py`, `test_vision_transformer_teachers.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `main()` (e.g. with `create_dataloaders()` and `create_model()`) actually correct?**
  _`main()` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `main()` (e.g. with `create_dataloaders()` and `create_model()`) actually correct?**
  _`main()` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `ExperimentRemovalService` (e.g. with `_render_removal_controls()` and `ExperimentRemovalServiceTests`) actually correct?**
  _`ExperimentRemovalService` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `main()` (e.g. with `create_dataloaders()` and `create_model()`) actually correct?**
  _`main()` has 15 INFERRED edges - model-reasoned connections that need verification._