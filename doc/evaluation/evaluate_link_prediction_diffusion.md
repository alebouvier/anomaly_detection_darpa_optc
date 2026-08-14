# evaluation/evaluate_link_prediction_diffusion.py

## Purpose
Evaluate the link prediction model on the validation and test data using diffusion model to create negatives

## Responsibilities
- evaluate the link prediciton model on the validation set.
- evaluate the link prediciton model on the validation set.
- evaluate the link prediciton model on the inductive setting for validation and test if inductive is active
- evaluate the link prediciton model on the calibration set if calibration is active
- store the results in the data folder

## Inputs
- load all necessary data from dataloader

## Outputs
- For each evaluation it stores:
  - predicted_links : predicted scores for actual links only
  - actual_links : actuals links with their labels (0=normal, 1=anomaly)
  - non_exist_links : predicted scores for negative sampling links


## Dependencies
- NegativeEdgeSampler in utils.utils
- evaluate_model_link_prediction_ano_insertion in evaluation.evaluate_models_utils
- get_link_prediction_data in utils.DataLoader
- EarlyStopping in utils.EarlyStopping

## Notes

