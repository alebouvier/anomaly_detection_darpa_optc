# preprocessing/preprocessing_data.py

## Purpose
This is the main data-preparation script of the project. It converts raw OpTC JSON logs into the processed CSV tables that the graph model and feature-embedding pipeline expects: edge lists, edge features, node features, and a list of path/command text samples.

## Responsibilities
- read raw log files for one or several clients
- filter out irrelevant events and keep only supported object types
- build node identifiers for process/file/shell entities
- create edge records linking source and destination nodes with timestamps and anomaly labels
- export edge and node metadata into CSV tables
- collect path and command-line strings for later Word2Vec/BERT training

## Inputs
- a raw logs directory
- a dataset identifier such as optc
- a list of clients to process
- malicious labels from the label_data/malicious.json file

## Outputs
- processed_data/optc_{client}/ml_optc_{client}.csv
  - u: source node id
  - i: destination node id
  - ts: timestamp
  - label: 1 for anomaly, 0 otherwise
  - pattern_id: pattern identifier, usually 0 for base data
  - idx: unique edge index
- processed_data/optc_{client}/edge_features.csv
  - action_type and command_line associated with each edge
- processed_data/optc_{client}/node_features.csv
  - object_type and path associated with each node
- processed_data/optc_{client}/path_cmd_list.pkl
  - raw path and command strings collected for embedding training

## Key Components
### Functions / Classes
- LogStats:
  - Purpose: lightweight dataclass holding the generated edge list, edge features, node features, and path/command text list.

- safe_get / parse_timestamp:
  - Purpose: robustly read JSON values and convert timestamps to floating-point Unix time.

- read_log_file / iter_logs:
  - Purpose: stream raw JSON log files row by row.

- collect_log_stats:
  - Purpose: aggregate all edge and node data from a log iterator.

- extract_ids:
  - Purpose: recover malicious log IDs by client from the malicious label file.

- build_dataframes / save_preprocessing_data:
  - Purpose: convert the aggregated stats into DataFrames and write the CSV exports.

- process_client / main:
  - Purpose: orchestrate processing for each client and dataset.

### Important Workflow
1. Raw logs are read and filtered by object type.
2. Nodes are assigned stable indices and path metadata is collected.
3. Each log event becomes an edge (actor to object) with timestamp and anomaly flag.
4. The graph tables are exported to disk and later used by feature extraction and model training.

## Dependencies
- pandas
- numpy
- tqdm
- scipy.sparse
- utils.utils

## Notes
- This script is the first step in the data pipeline and is typically run before any feature generation or temporal pattern injection.
- Only events whose object type belongs to PROCESS, FILE, or SHELL are kept in the final tables.
- The edge list is sorted by timestamp before being written, and the feature rows are aligned with the sorted edges to keep indices consistent.
