# preprocessing/temporal_patterns.py

## Purpose
This module studies temporal structures in anomalous interactions and builds pattern-based augmentations used to stress the training set. It focuses on the idea that real anomalies often repeat a recognizable temporal sequence, and those sequences can be inserted into earlier data to create harder learning conditions.

## Responsibilities
- select anomaly edges from the test split
- identify sequences of two-hop temporal paths in anomalous interactions
- normalize command-line strings so equivalent behaviors are grouped together
- summarize repeated anomaly patterns in a DataFrame
- prepare the data for injecting those patterns into train/validation data

## Inputs
- client-specific processed edge lists and feature tables
- validation start and test start timestamps
- anomaly edges already identified in the test period

## Outputs
- unique anomaly path summaries saved in processed_data/optc_{client}/unique_anomaly_paths.csv
- intermediate pattern tables used to decide how to inject abnormal temporal structures into training data

## Key Components
### Functions / Classes
- get_unique_stats_features:
  - Purpose: loads the relevant processed files and computes anomaly path statistics for a client.

- _command_line_key:
  - Purpose: provides a normalized key used to merge similar command-line patterns despite minor noise.

- find_non_present_edges / find_similar_edges / find_similar_paths:
  - Purpose: compare sets of edges or paths across splits to detect repeated or missing patterns.

- build_edge_signature_df:
  - Purpose: constructs a signature DataFrame describing edges by source type, target type, action, and command line.

- find_anomaly_paths_in_train:
  - Purpose: checks whether anomaly patterns are already present in the train split.

- normalize_cmdline:
  - Purpose: same canonicalization logic used in extract_all_anomaly.py, with the same goal of making patterns comparable.

- find_temporal_paths_length_2:
  - Purpose: extracts two-hop temporal paths from the anomaly edges.

- create_unique_paths_df:
  - Purpose: condenses multiple observed temporal paths into unique patterns with statistics like count and average time gap.

- add_anomaly_paths_in_train_val:
  - Purpose: injects discovered temporal anomaly patterns back into the training data.

### Important Workflow
1. Anomalies are isolated from the test split.
2. Their temporal two-hop paths are extracted and aggregated.
3. Similar patterns are matched and summarized.
4. The repeated patterns can be reintroduced into earlier data to create anomaly-aware training samples.

## Dependencies
- pandas
- numpy
- tqdm
- preprocessing.extract_all_anomaly
- utils.utils

## Notes
- This file is one of the more experimental parts of the pipeline and contains several functions that were used for different pattern-mining strategies.
- Some helpers are historical or unused, but the core workflow is to identify anomaly signatures that can be injected into training time windows.
