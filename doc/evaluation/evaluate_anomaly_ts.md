# evaluation/evaluate_anomaly_ts.py

## Purpose
This script compute the metrics and plots for the link prediction task and the anomaly detection task.

## Responsibilities
- [ ] compute link_prediction metrics from the validation results
- [ ] compute anomaly detection metrics from the test results at the edge and graph level
- [ ] store the most badly predicted data (false positive, false negative) by the model in a file for the validation and test set.
- [ ] compute miscoverage rate if calibration is selected.

## Inputs
- [ ] load the validation, test and calibration results if calibration is selected.

## Outputs
- link prediction results at the edge level
  - report file with metrics
  - ROC curve
  - Precision recall curve
  - calibration curve
  - confusion matrix
  - distribution histogram of scores for positive and negative examples
- anomaly detection results 
  - report file with metrics at the edge level
  - ROC curve at the edge level
  - Precision recall curve at the edge level
  - calibration curve at the edge level
  - confusion matrix at the graph level
  - distribution histogram of scores for positive and negative examples at the edge level.


## Dependencies
- utils.utils
- evaluation.calibration

## Notes
- anomaly detection at the graph level means that we consider graph windows of 15 minutes and associate for each graph a score computed as the mean of the 1% highest anomaly scores.
- The adaptive miscoverage level plot was never used in practice.