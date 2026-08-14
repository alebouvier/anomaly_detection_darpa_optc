# training/train_link_prediction_diffusion.py

## Purpose
This script trains a temporal link-prediction model with an additional conditional diffusion module that generates synthetic negative embeddings during training. The goal is to improve the model's ability to distinguish true edges from hard negatives by injecting diffusion-generated negatives into the learning objective.

## Responsibilities
- load the processed temporal graph data for training, validation, and optional inductive evaluation
- initialize temporal neighbor samplers and negative edge samplers for the selected sampling strategy
- build the chosen dynamic backbone model (TGAT, GraphMixer, DyGFormer, or a memory-based model) and a MergeLayer link predictor
- instantiate a conditional diffusion model (`Diffusion_Cond`) and optimize it jointly with the main model
- for each training batch, compute positive and negative edge probabilities and add diffusion-generated negative embeddings to the loss
- evaluate validation performance after each epoch, trigger early stopping, and keep the best checkpoint
- save training logs, loss curves, and per-batch loss summaries for each run

## Inputs
The script expects runtime arguments such as:
- dataset_name, start_train, start_val, start_test, end_test
- model_name, optimizer, learning_rate, weight_decay, batch_size, num_epochs
- negative_sample_strategy, sample_neighbor_strategy, time_scaling_factor, num_neighbors, time_gap
- inductive, calibration, anomaly_injection, device, temperature, patience
- num_runs, seed configuration

It also relies on:
- processed temporal data produced by `utils.DataLoader.get_link_prediction_data`
- a temporal neighborhood sampler from `utils.utils.get_neighbor_sampler`
- a negative sampler from `utils.utils.NegativeEdgeSampler`
- a diffusion generator defined in `utils.diffusion.Diffusion_Cond`

## Outputs
For each run, the script writes artifacts under:
- experiments/{dataset_name}/{model_name.lower()}/{save_model_name}/

Typical outputs include:
- training logs under `logs/`
- loss curves and summaries under `loss/`
- saved model checkpoints under `saved_models/`
- a final best model restored via `EarlyStopping`

## Key Components
### main(args)
This is the main entry point. It:
1. loads the temporal dataset and prepares validation/test splits,
2. builds the dynamic backbone and link predictor,
3. initializes the diffusion model and optimizer,
4. loops over runs and epochs,
5. trains on batches, generates diffusion negatives, and updates the model,
6. evaluates validation metrics and saves the best checkpoint.

### Diffusion model
The script creates a conditional diffusion model with:
- `in_feat_dim = 33`
- `out_feat_dim = 33`
- `timesteps = 50`

It initializes:
- `diffusion = Diffusion_Cond(in_feat_dim, out_feat_dim, timesteps, y).to(args.device)`
- `d_optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)`

This is not a standalone diffusion model pipeline; it is used as an auxiliary generator of negative temporal embeddings inside the learning loop.

### Dynamic graph backbone
The model is selected from the same family as the regular training script:
- TGAT
- GraphMixer
- DyGFormer
- MemoryModel for JODIE / DyRep / TGN variants

The selected backbone is wrapped in a `MergeLayer` to produce edge probabilities.

### Negative sampling and temporal neighborhoods
The script uses:
- `get_neighbor_sampler(...)` for historical neighborhood extraction
- `NegativeEdgeSampler(...)` for positive/negative edge generation

It includes special handling for `2hop_neighbor` sampling and for model-specific historical neighbor logic.

## Important Workflow
1. Load temporal data and initialize train/validation samplers.
2. Create the diffusion model and the main dynamic graph model.
3. For each training batch:
   - sample negative destination nodes,
   - extract historical neighbors,
   - compute source and destination temporal embeddings for positive and negative edges,
   - compute diffusion-conditioned negative embeddings from the source/neighbor context,
   - combine positive probabilities, sampled negative probabilities, and diffusion negative probabilities,
   - compute binary cross-entropy on the concatenated predictions and labels.
4. Update the main model and diffusion optimizer.
5. Evaluate the current model on validation data.
6. Save the best checkpoint with `EarlyStopping`.
7. Plot and save the training loss curves at the end of each run.

## Key Training Detail
A distinctive part of this script is the generation of diffusion negatives:
- it first trains the conditional diffusion model on neighbor embeddings,
- then it samples synthetic destination embeddings,
- then it computes a `negative_diffusion_probabilities` term,
- and finally concatenates:
  - positive probabilities,
  - sampled negative probabilities,
  - diffusion-generated negative probabilities

This makes the loss more robust to hard negatives created from the learned temporal structure.

## Dependencies
- torch, torch.nn
- numpy, pandas
- matplotlib
- scipy.ndimage.median_filter
- tqdm
- models.TGAT, models.GraphMixer, models.DyGFormer, models.MemoryModel, models.modules.MergeLayer
- utils.utils, utils.DataLoader, utils.EarlyStopping, utils.diffusion.Diffusion_Cond
- evaluation.evaluate_models_utils
- utils.metrics

## Notes
- This script is not a pure anomaly-injection workflow; it is a diffusion-enhanced temporal link prediction trainer.
- The diffusion component is used as an internal negative generator rather than as a standalone generative model.
- The per-epoch validation logic follows the standard training pipeline, including optional inductive evaluation for new nodes.
- The script also preserves the memory-bank behavior used by JODIE/DyRep/TGN models, reloading the correct memory state when moving between training and validation phases.
- Loss curves are smoothed with a median filter before plotting, and batch-loss data is exported to CSV for later inspection.
