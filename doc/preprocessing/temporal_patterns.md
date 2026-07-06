# preprocessing/temporal_patterns.py

## Purpose
Extract temporal pattern from anomalies present in the test and insert them in train data as negative samples.


## Responsibilities
- [ ] extract anomaly patterns
- [ ] normalize all command lines to make them more generic
- [ ] modify csv files to incorporate patterns


## Inputs
- [ ] clients: a list of clients/machines
- [ ] start_val: date of validation start
- [ ] start_test: date of test start

## Outputs
- Outputs are stored in {BASE}/processed_data/optc_{client}/
- anomalies.csv : file conataining anomaly pattern
- ml_optc_{client}.csv : file modified with anomaly patterns
- edge_features.csv : file modified with anomaly patterns and normalized command lines.

## Key Components

### Important Workflow
- executed after preprocessing_data.py

## Dependencies
- utility package
- preprocessing.extract_all_anomaly : implement one way of extracting and incorporating tempora patterns.

## Notes
- [ ] Many different implementation of temporal patterns have been tested. Some function in the script may be unused or obsolete.
