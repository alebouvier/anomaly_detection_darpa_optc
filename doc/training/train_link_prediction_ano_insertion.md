# training/train_link_prediction_ano_insertion.py

## Purpose
This file is the anomaly-aware variant of the standard temporal link-prediction training script. It trains the same family of dynamic graph models while explicitly separating normal and anomalous edges: normal edges are treated as positive training links, while anomalous edges are injected into the negative-side training set and evaluated with an anomaly-insertion-aware validation routine.

## Responsibilities
- load train/validation data that has already been split into normal and anomaly subsets
- initialize temporal samplers with optional pattern masking
- build the dynamic backbone and link predictor for the selected model
- train on batches containing normal positive edges and anomalous negative edges
- store temporal embeddings for both regular and negative samples during training
- evaluate the model using the anomaly-aware validation function
- save logs, loss curves, and the best model checkpoint for the anomaly-insertion pipeline

## Inputs
- runtime arguments such as:
  - dataset_name, start_train, start_val, start_test, end_test
  - model_name, optimizer, learning_rate, batch_size, num_epochs, num_runs
  - negative_sample_strategy, sample_neighbor_strategy, time_scaling_factor, pattern_masking
  - inductive, calibration, anomaly_injection, device, temperature
- processed temporal data from utils.DataLoader.get_link_prediction_data with normal and anomaly splits
- optional anomaly pattern metadata carried by pattern_ids and labels

## Outputs
- experiment directory under experiments/{dataset_name}/{model_name.lower()}/{save_model_name}/
- saved loss plots and batch-level loss CSV files
- checkpointed best model returned by EarlyStopping
- temporal embedding dumps saved in BASE/temporal_embeddings_data/{dataset_name}/{model_name.lower()}/train/
  - temporal_embeddings.pt
  - negative_temporal_embeddings.pt
- training and validation logs with per-epoch metrics

## Key Components
### Functions / Classes
- main(args):
  - Purpose: executes the full anomaly-aware temporal link-training pipeline.
  - Inputs: high-level configuration and dataset splits.
  - Outputs: trained model, logs, and saved artifacts.

- get_link_prediction_data:
  - Purpose: returns the normal/anomaly train and validation sets required by the anomaly-insertion workflow.
  - Inputs: dataset configuration and anomaly settings.
  - Outputs: a richer data structure than the standard training script, including normal and anomaly subsets.

- get_neighbor_sampler / NegativeEdgeSampler:
  - Purpose: produce temporal neighborhoods and negative samples while respecting the anomaly-aware settings.
  - Inputs: temporal graph data, masking options, and negative-sampling strategy.
  - Outputs: samplers used for training and validation.

- evaluate_model_link_prediction_ano_insertion:
  - Purpose: evaluate normal and anomalous validation edges separately.
  - Inputs: the current model and anomaly-aware validation loaders.
  - Outputs: validation losses and metrics computed with a specialized anomaly-aware evaluation pipeline.

### Important Workflow
1. The script loads data where some edges are marked as normal and some as anomalous, then builds separate loaders for each subset.
2. It configures training and validation neighbor samplers, as well as negative edge samplers for evaluation.
3. For each epoch, it processes normal positive edges and anomalous edges together, using anomalous edges as part of the negative-side training signal.
4. It records temporal embeddings and evaluates the model on anomaly-aware validation data before selecting the best checkpoint.

## Dependencies
- torch, torch.nn
- numpy, pandas
- matplotlib
- scipy.ndimage.median_filter
- tqdm
- models.TGAT, models.GraphMixer, models.DyGFormer, models.MemoryModel
- utils.utils, utils.DataLoader, utils.EarlyStopping
- evaluation.evaluate_models_utils
- utils.metrics

## Notes
- This script is intentionally specialized for anomaly-injection experiments: the negative training examples are not only random or historical negatives, but often include anomalous edges that capture the abnormal behavior being modeled.
- It also records temporal embedding dumps for source/destination nodes and negative samples, which is useful for later analysis of how anomalies modify the learned representations.
- Pattern IDs can be passed into the temporal embedding computations, so the model may incorporate anomaly pattern information when the corresponding model supports it.
- Memory-based models are handled with the same snapshot/reload mechanism as in the standard training file, ensuring that validation and new-node evaluation operate on the correct memory state.
