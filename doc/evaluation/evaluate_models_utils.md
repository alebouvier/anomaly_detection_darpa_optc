# evaluation/evaluate_models_utils.py

## Purpose
It contains different functions to evaluate link prediction models  

## Responsibilities
- evaluate a link_prediction model on validation or test set 
- store the scores

## Inputs
-  Describe required inputs, parameters, or configuration values.
-  Note any expected file formats, data structures, or environment assumptions.

## Outputs
-  Describe produced results, artifacts, or return values.
-  Mention any files written, metrics computed, or models trained.

## Key Components
### Functions / Classes
- evaluate_model_link_prediction: evaluate the link prediction model in the classic way
- evaluate_model_link_prediction_ano_insertion: evaluate the link prediction model with the use of incorporated anomalies as negatives.
- evaluate_model_node_classification: evaluate node classification model (not used)
- evaluate_edge_bank_link_prediction: evaluate the edge bank link prediction model (not used)


## Dependencies
- utils.utils
- utils.metrics
- utils.DataLoader

## Notes
- When using anomalies as negatives, we usually use evaluate_model_link_prediction_ano_insertion for validation data and evaluate_model_link_prediction for test data as we don't incorporate anomalies in the test.
