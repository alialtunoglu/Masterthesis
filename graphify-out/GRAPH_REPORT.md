# Graph Report - MasterThesis  (2026-09-01)

## Corpus Check
- 174 files · ~871,842 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1013 nodes · 2036 edges · 72 communities (47 shown, 25 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 235 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8a911a0f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- experiment_runner.py
- ExperimentRemovalServiceTests
- load_multi_kd_config
- train_baseline.py
- load_kd_config
- analyze_student_baselines.py
- create_splits.py
- reset_experiments.py
- 7_Knowledge_Distillation.py
- dataloaders.py
- 3_Job_Monitor.py
- 4_Results_Explorer.py
- 2_Baseline_Experiments.py
- kd_strategies.py
- mlflow_tracker.py
- main
- get_project_root
- train_teacher.py
- ensure_dir
- Portable Notebook Dataset Bootstrap Design
- experiment_naming.py
- Deney Sonuçları Analizi ve Bildiri Potansiyeli
- external_run_import.py
- 6_MLflow_Helper.py
- streamlit_app.py
- Knowledge Distillation Sonuçları: Karşılaştırmalı Analiz
- KDStrategyTests
- format_metric
- 9_Vision_Transformer_Teacher_Experiments.py
- kd_features.py
- portable_notebook.py
- main
- read_log_tail
- kd_trainer.py
- list_jobs
- Global Constraints
- optimization.py
- __init__.py
- __init__.py
- __init__.py
- __init__.py
- __init__.py
- __init__.py
- masterthesis-plant-disease-kd
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
5. `get_project_root()` - 29 edges
6. `main()` - 29 edges
7. `ExperimentRemovalServiceTests` - 26 edges
8. `load_kd_config()` - 21 edges
9. `ensure_dir()` - 21 edges
10. `load_multi_kd_config()` - 20 edges

## Surprising Connections (you probably didn't know these)
- `loss_preview()` --uses--> `KDConfig`  [INFERRED]
  app/pages/7_Knowledge_Distillation.py → src/training/kd_config.py
- `ExperimentRemovalServiceTests` --uses--> `RemovalMode`  [INFERRED]
  tests/test_experiment_removal.py → app/experiment_removal.py
- `ExternalRunTransferTests` --uses--> `RemovalMode`  [INFERRED]
  tests/test_external_run_transfer.py → app/experiment_removal.py
- `ExperimentRemovalServiceTests` --uses--> `ExperimentRemovalError`  [INFERRED]
  tests/test_experiment_removal.py → app/experiment_removal.py
- `ExperimentRemovalServiceTests` --uses--> `ExperimentActiveError`  [INFERRED]
  tests/test_experiment_removal.py → app/experiment_removal.py

## Import Cycles
- None detected.

## Communities (72 total, 25 thin omitted)

### Community 0 - "experiment_runner.py"
Cohesion: 0.11
Nodes (34): build_kd_command(), build_multi_kd_command(), cancel_queued_job(), enqueue_job(), _ensure_run_dirs(), get_job_runtime_seconds(), get_queue_worker_status(), _job_path() (+26 more)

### Community 1 - "ExperimentRemovalServiceTests"
Cohesion: 0.13
Nodes (3): ExperimentRemovalServiceTests, FakeRunTrackingGateway, Path

### Community 2 - "load_multi_kd_config"
Cohesion: 0.06
Nodes (44): TrainingConfig, _non_negative(), _positive_finite(), Teacher-weight resolution policies for multi-teacher distillation., resolve_teacher_weights(), load_multi_kd_config(), _mapping(), merge_multi_kd_config() (+36 more)

### Community 3 - "train_baseline.py"
Cohesion: 0.23
Nodes (21): add_runtime_fields(), append_summary_row(), build_artifact_paths(), build_final_config(), build_run_name_extra(), build_summary_row(), cli_overrides(), load_config_file() (+13 more)

### Community 4 - "load_kd_config"
Cohesion: 0.19
Nodes (14): ExperimentConfig, KDConfig, load_kd_config(), _mapping(), merge_kd_config(), Any, Path, Typed configuration and validation for knowledge-distillation experiments. (+6 more)

### Community 5 - "analyze_student_baselines.py"
Cohesion: 0.22
Nodes (22): add_artifact_paths(), artifact_check(), build_overall(), build_recommendations(), _format_markdown_value(), load_results(), main(), parse_args() (+14 more)

### Community 6 - "create_splits.py"
Cohesion: 0.09
Nodes (36): Counter, analyze_imagefolder(), analyze_plantpathology2021(), main(), print_summary(), Analyze raw datasets without modifying them., build_split_payload(), create_split() (+28 more)

### Community 7 - "reset_experiments.py"
Cohesion: 0.19
Nodes (19): NamedTuple, archive_path(), confirm_or_exit(), delete_paths(), find_project_root(), main(), move_to_archive(), parse_args() (+11 more)

### Community 8 - "7_Knowledge_Distillation.py"
Cohesion: 0.07
Nodes (41): checkpoint_classes(), config_label(), config_paths(), expected_classes(), load_uploaded(), loss_preview(), matching_config_paths(), model_profile() (+33 more)

### Community 9 - "dataloaders.py"
Cohesion: 0.15
Nodes (12): Dataset, build_eval_transform(), build_train_transform(), create_dataloaders(), Path, DataLoader creation from split JSON files., Create train, validation, and test DataLoaders from a split file., Any (+4 more)

### Community 10 - "3_Job_Monitor.py"
Cohesion: 0.27
Nodes (8): choose_monitor_job_id(), find_job_by_id(), is_kd_job(), is_multi_kd_job(), job_option_label(), job_type_label(), latest_running_job(), Live-ish job monitor for Streamlit-launched experiments.

### Community 11 - "4_Results_Explorer.py"
Cohesion: 0.06
Nodes (56): _CsvMutation, ExperimentActiveError, ExperimentIdentity, ExperimentRemovalError, ExperimentRemovalService, _file_contains(), find_jobs_for_run(), InterruptedJobRemovalReport (+48 more)

### Community 12 - "2_Baseline_Experiments.py"
Cohesion: 0.24
Nodes (8): build_baseline_command(), Build a Windows-safe baseline training command., _config_label(), _filter_configs(), _load_config_defaults(), _load_config_index(), Any, Baseline experiment launcher page.

### Community 13 - "kd_strategies.py"
Cohesion: 0.17
Nodes (14): DistillationConfig, _angles(), create_distillation_strategy(), DistillationLoss, DistillationStrategy, _feature_vectors(), FeatureDistillationStrategy, LogitDistillationStrategy (+6 more)

### Community 14 - "mlflow_tracker.py"
Cohesion: 0.15
Nodes (20): end_run(), _import_mlflow(), log_artifact(), log_artifacts(), log_metrics(), log_params(), log_text(), Any (+12 more)

### Community 15 - "main"
Cohesion: 0.24
Nodes (17): append_summary(), build_paths(), build_run_name(), flatten_params(), load_teacher_checkpoint(), main(), model_latency_ms(), parse_args() (+9 more)

### Community 16 - "get_project_root"
Cohesion: 0.10
Nodes (30): Project overview page for the local Streamlit dashboard., config_label(), load_config_index(), Any, CNN teacher experiment launcher page., load_baseline_results(), load_csv_if_exists(), load_dataset_summary() (+22 more)

### Community 17 - "train_teacher.py"
Cohesion: 0.07
Nodes (52): Optimizer, compute_classification_metrics(), Classification metrics for multi-class experiments., Compute aggregate multi-class classification metrics., create_model(), get_model_type(), Module, Factory for supported student and teacher models. (+44 more)

### Community 18 - "ensure_dir"
Cohesion: 0.18
Nodes (17): compute_per_class_metrics(), DataFrame, Return per-class precision, recall, F1, and support as a DataFrame., Path, Evaluation report artifacts for baseline classification runs., Save confusion matrix as a CSV table and a heatmap PNG., Save loss and validation metric curves for a training run., save_confusion_matrix_artifacts() (+9 more)

### Community 19 - "Portable Notebook Dataset Bootstrap Design"
Cohesion: 0.22
Nodes (8): Code structure, Error handling, Goal, Plant Pathology bootstrap flow, Portable Notebook Dataset Bootstrap Design, Scope, Success criteria, Tests

### Community 20 - "experiment_naming.py"
Cohesion: 0.33
Nodes (6): build_experiment_name(), build_run_name(), _normalize(), Consistent experiment and run naming helpers., Build a stable MLflow run name., Build a project-level MLflow experiment name.

### Community 21 - "Deney Sonuçları Analizi ve Bildiri Potansiyeli"
Cohesion: 0.05
Nodes (37): 10. Sonuçların güvenilirliği ve sınırlamalar, 11. Bu sonuçlardan bildiri olur mu?, 12. Gönderim öncesi gerekli deneyler, 13. Önerilen bildiri çerçevesi, 14. Nihai değerlendirme, 1. Yönetici özeti, 2. İncelenen deney envanteri, 3.1 AppleLeaf9 (+29 more)

### Community 22 - "external_run_import.py"
Cohesion: 0.14
Nodes (24): _default_register_run(), ExternalRunImportError, import_external_bundle(), Path, RuntimeError, Validate and import external experiment bundles without loading model code., _read_manifest(), _safe_relative() (+16 more)

### Community 28 - "Knowledge Distillation Sonuçları: Karşılaştırmalı Analiz"
Cohesion: 0.06
Nodes (30): 10. Bilimsel yorum ve sınırlamalar, 11. Önerilen takip deneyleri, 12. Nihai değerlendirme, 1. Kısa cevap: Student iyileşti mi?, 2. Baseline student ve teacher sonuçları, 3. Single-teacher KD sonuçları, 4. Single-teacher sınıf bazlı analiz, 5. Multi-teacher KD sonuçları (+22 more)

### Community 30 - "format_metric"
Cohesion: 0.67
Nodes (3): format_metric(), Any, Format a metric value for display.

### Community 31 - "9_Vision_Transformer_Teacher_Experiments.py"
Cohesion: 0.33
Nodes (4): build_teacher_command(), Build a Windows-safe teacher training command., make_command(), Vision Transformer teacher experiment launcher.

### Community 32 - "kd_features.py"
Cohesion: 0.13
Nodes (12): ConvFeatureAdapter, default_feature_layer(), FeatureCapture, FeatureShapes, infer_feature_shapes(), device, Module, Tensor (+4 more)

### Community 33 - "portable_notebook.py"
Cohesion: 0.21
Nodes (12): build_portable_notebook(), _cell(), _dataset_setup(), _git(), Path, Generate portable Colab/Kaggle notebooks from the selected local command., Render a guarded Streamlit notebook download control., render_notebook_download() (+4 more)

### Community 34 - "main"
Cohesion: 0.18
Nodes (16): append_summary(), build_paths(), build_run_name(), flatten_params(), main(), parse_args(), Any, device (+8 more)

### Community 35 - "read_log_tail"
Cohesion: 0.33
Nodes (7): get_log_file_info(), Path, Read the last n lines of a log file., Resolve a job log path relative to the project root., Return compact file metadata for a job log., read_log_tail(), resolve_log_path()

### Community 36 - "kd_trainer.py"
Cohesion: 0.32
Nodes (11): _epoch_summary(), evaluate_student(), fit_kd(), freeze_teacher(), device, MetricLogger, Module, Path (+3 more)

### Community 37 - "list_jobs"
Cohesion: 0.14
Nodes (20): is_process_running(), list_jobs(), load_dashboard_settings(), log_has_error(), promote_queued_jobs(), Update a job status if its process is no longer running., List known jobs from metadata files., Start queued jobs while respecting the max_parallel_jobs setting. (+12 more)

### Community 38 - "Global Constraints"
Cohesion: 0.50
Nodes (3): Global Constraints, Portable Notebook Dataset Bootstrap Implementation Plan, Task 1: Generate an environment-aware Plant Pathology setup cell

### Community 39 - "optimization.py"
Cohesion: 0.23
Nodes (8): Parameter, create_optimizer(), create_scheduler(), is_improvement(), monitor_mode(), Optimizer, scheduler, and monitored-metric factories shared by KD training., step_scheduler(), KDFeatureAndOptimizationTests

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
- **156 isolated node(s):** `masterthesis-plant-disease-kd`, `Project Principles`, `Clean Code Rules`, `Clean Architecture Rules`, `Machine Learning Experiment Rules` (+151 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ExperimentRemovalService` connect `4_Results_Explorer.py` to `ExperimentRemovalServiceTests`, `external_run_import.py`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `get_project_root()` connect `get_project_root` to `experiment_runner.py`, `read_log_tail`, `list_jobs`, `4_Results_Explorer.py`, `2_Baseline_Experiments.py`, `9_Vision_Transformer_Teacher_Experiments.py`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Why does `ExperimentRemovalServiceTests` connect `ExperimentRemovalServiceTests` to `4_Results_Explorer.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `main()` (e.g. with `create_dataloaders()` and `create_model()`) actually correct?**
  _`main()` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `main()` (e.g. with `create_dataloaders()` and `create_model()`) actually correct?**
  _`main()` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ExperimentRemovalService` (e.g. with `_render_removal_controls()` and `ExperimentRemovalServiceTests`) actually correct?**
  _`ExperimentRemovalService` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `main()` (e.g. with `create_dataloaders()` and `create_model()`) actually correct?**
  _`main()` has 15 INFERRED edges - model-reasoned connections that need verification._