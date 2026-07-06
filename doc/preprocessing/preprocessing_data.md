# preprocessing/preprocessing_data.py

## Purpose
Preprocess raw log data into csv files containing edges list and non-encoded features.
It can handle multiple clients

## Responsibilities
- [ ] Read raw data
- [ ] extract pertinent information
- [ ] Create csv files used as entries for DyGLib libraries

## Inputs
- [ ] logs: path to log data folder
- [ ] dataset: "optc"
- [ ] clients: list of clients, e.g., ["051", "201"]

## Outputs
- outputs are stored in {BASE}/processed_data/optc_{client}/
- [ ] ml_optc_{client}.csv
  - "u": source node id, 
  - "i": destination node id, 
  - "ts": timestamp, 
  - "label": 1 for anomaly, 0 otherwise,
  - "pattern_id": id of the temporal pattern the edge belongs to, 0 for every edge, 
  - "idx": unique index of the edge (start with 1)
- [ ] edge_feature.csv
  - the row n corresponds to the features of the edge of idx=n
  - the first row (row 0) is empty
  - "action_type": type of the event
  - "command_line": command line associated to the event.
- [ ] node_feature.csv
  - the row n corresponds to the features of the node of id=n
  - the first row (row 0) is empty
  - "object_type": type of the node
  - "path": path associated to the node.

## Workflow
- first script to be executed


## Dependencies
- utility package

## Notes
- [ ] Only events involving processes, files and shells are kept in the csv files. This can be modified with the constant variable OBJECT_TYPE.
