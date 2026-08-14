# features/extract_features.py

## Purpose
This module prepares final node and edge feature matrices from processed CSV files. It combines one-hot type information with semantic embeddings from either Word2Vec or BERT and writes the result as memory-mapped NumPy arrays for downstream graph learning.

## Responsibilities
- read processed CSV rows for node or edge features
- build a text representation from the raw path/command data
- encode that text with the selected model
- concatenate the type one-hot vector with the semantic embedding
- save the generated embeddings as .npy arrays

## Inputs
- a dataset name and client identifier used to locate processed data folders
- a configuration object loaded from the dataset config
- a Word2Vec model or BERT tokenizer + model
- edge_features.csv or node_features.csv files containing type and text information

## Outputs
- ml_{dataset}_{client}.npy for edge features
- ml_{dataset}_{client}_node.npy for node features
- in-memory feature matrices for the current client and split

## Key Components
### Functions / Classes
- read_line_csv_file:
  - Purpose: stream CSV rows while skipping empty lines and the header.
  - Inputs: path to a CSV or gzipped CSV file.
  - Outputs: iterator of [type, text] rows.

- count_csv_rows:
  - Purpose: count the number of valid rows in a feature CSV before allocating matrix storage.
  - Inputs: CSV path.
  - Outputs: number of rows.

- one_hot_encoding:
  - Purpose: encode the object or action type as a one-hot vector.
  - Inputs: type label and boolean is_edge.
  - Outputs: list of binary values.

- build_text_sequence:
  - Purpose: normalize a raw line into a comparable token sequence using the Word2Vec preprocessing pipeline.
  - Inputs: raw line and config.
  - Outputs: joined token string.

- w2v_encoding / bert_encoding:
  - Purpose: convert a text field to a semantic embedding using the selected encoder.
  - Inputs: model or tokenizer/model and the raw path/command value.
  - Outputs: fixed-size embedding list.

- create_features:
  - Purpose: build the full feature matrix for a node or edge CSV file.
  - Inputs: iterator over rows, model, config, type flag, output path, and model type.
  - Outputs: NumPy memmap array containing concatenated type + semantic features.

- process_client:
  - Purpose: create edge and node matrices for one client.
  - Inputs: dataset name, client, config, and model.
  - Outputs: writes both .npy files to the processed_data folder.

- main:
  - Purpose: orchestrates all clients for a dataset using either W2V or BERT embedding models.

### Important Workflow
1. The input CSV is read row-by-row.
2. Each row contributes a type one-hot representation and a semantic embedding of the textual part.
3. The concatenated vector is written into a preallocated NumPy memmap.
4. The resulting arrays are saved for use in graph models.

## Dependencies
- numpy
- pandas
- tqdm
- features.train_bert128
- features.w2v
- utils.utils
- transformers (for BERT mode)

## Notes
- This file is the final feature-construction layer before model training.
- It supports both Word2Vec and BERT, but the BERT path uses the fine-tuned model from the variable bert_dir or the default Bert_ft directory.
- The output arrays are intentionally made memory-friendly through numpy memmap.
