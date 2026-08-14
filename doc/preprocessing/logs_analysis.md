# preprocessing/logs_analysis.py

## Purpose
This module performs a first-pass analysis of raw OpTC log files. It summarizes object types, action frequencies, process relationships, and anomaly distributions.

## Responsibilities
- load log files from one or several clients
- count how many times each object type and action type appears
- track process-to-process adjacency and process roles
- detect object-type conflicts and process IDs appearing in different roles
- report anomaly statistics and client-wise repartition of malicious events

## Inputs
- raw log files or folders from the OpTC dataset
- a list of clients
- a dataset module exposing the appropriate log loading function
- the malicious.json label data used for anomaly counting

## Outputs
- console summaries of raw log statistics
- counts of logs by object type and action type
- process path statistics of length 2 and 3 relationships
- a breakdown of anomaly occurrences by client/day

## Key Components
### Functions / Classes
- get_log_stats:
  - Purpose: aggregate statistics from a raw log iterator.
  - Inputs: iterable of log dictionaries.
  - Outputs: object summaries, action counters, process statistics, and type-conflict information.

- extract_data / iter_logs:
  - Purpose: read gzip or plain-text JSON logs efficiently and iterate over records.

- main:
  - Purpose: run the analysis for the requested clients and the malicious label dataset.

- repartition_anomaly_by_client:
  - Purpose: summarize which client and date contain each malicious log.

- count_process_paths:
  - Purpose: estimate process path counts at lengths 2 and 3.

- print_stats:
  - Purpose: format the final analysis report in a readable way.

### Important Workflow
1. Logs are loaded from the client data folder.
2. Object and action counts are accumulated over the full corpus.
3. Process adjacency and cross-role relationships are tracked.
4. The program prints diagnostic summaries useful for preprocessing validation.

## Dependencies
- networkx
- pandas
- pandasql
- numpy
- tqdm
- utils.utils

## Notes
- This script is diagnostic and exploratory rather than a final data-conversion step.
- It helps validate whether the raw event stream is consistent enough to build the graph data model used later by the preprocessing pipeline.
