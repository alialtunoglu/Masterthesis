# Workflow: Result Logging

Every experiment should produce structured outputs.

Required fields for classification results:

- dataset_name
- model_name
- model_type
- num_classes
- seed
- split_file
- image_size
- batch_size
- epochs
- optimizer
- learning_rate
- scheduler
- best_val_accuracy
- best_val_macro_f1
- test_accuracy
- test_macro_precision
- test_macro_recall
- test_macro_f1
- test_weighted_f1
- params
- model_size_mb
- inference_time_ms
- checkpoint_path

For KD experiments, additionally log:

- kd_type
- teacher_model
- teacher_checkpoint
- student_model
- temperature
- alpha
- feature_loss_weight
- multi_teacher_strategy
- adaptive_temperature_enabled
