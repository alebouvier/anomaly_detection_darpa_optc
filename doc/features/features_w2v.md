# features/features_w2v.py

## Purpose
This module builds node features using Word2Vec embeddings instead of BERT. the process metadata is embedded with a trained Word2Vec model rather than a transformer.

## Responsibilities
- embed image paths and command lines using the Word2Vec model
- combine semantic node-level features
- save feature bundles for downstream model training

## Inputs
- a process graph object
- a path to the trained Word2Vec model
- metadata flag
- dataset configuration and selected node definitions
- a list of clients and graph files

## Outputs
- features_e: per-node feature entries containing the embedded process metadata and structural summaries
- saved .pkl graph feature files for each client

## Key Components
### Functions / Classes
- Encoding_builder:
  - Purpose: computes structural and semantic feature encodings for process graphs.
  - Inputs: graph, model, metadata, selected nodes, config.
  - Outputs: graph and node features.

- encoding_process:
  - Purpose: for each process node, computes embeddings for image path, parent image path, and command line with Word2Vec.
  - Inputs: graph and nodes.
  - Outputs: per-node feature vectors.

- get_degree, lifetimenodes, node_time_for_node_type, edge_type_for_node_type:
  - Purpose: compute the same structural descriptors as in the BERT version, but using Word2Vec semantics rather than transformer embeddings.

- main:
  - Purpose: iterates over all graph files, generates features, and writes feature metadata files.

### Important Workflow
1. Structural statistics are computed from the graph topology and process lifetimes.
2. Process metadata is turned into Word2Vec embeddings using the trained model.
3. These semantic embeddings are combined with structural values for each node.
4. The aggregated feature bundles are serialized to disk.

## Dependencies
- networkx
- numpy
- tqdm
- others.w2v
- utils.utils

## Notes
- This is the baseline non-transformer version of the feature pipeline.
- It is conceptually similar to the BERT-based builder, but the semantic channel relies on the Word2Vec embeddings trained in features/w2v.py.
- The per-node embedding size is determined by the model configuration, especially LEN_ENCODE_PATH.
- The structural graph level features arer not used in the next steps of the pipeline.
