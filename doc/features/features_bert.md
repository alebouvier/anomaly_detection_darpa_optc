# features/features_bert.py

## Purpose
This module builds node-level structural features for process graphs, enriched with BERT-based semantic descriptors for process metadata. It is one of the main feature-generation pipelines used before temporal graph learning.

## Responsibilities
- select the process nodes involved in the graph

- encode process attributes such as image path, parent image path, command line, and SID using BERT embeddings
- aggregate node-level features into two dictionaries per-node features

## Inputs
- a graph object built from processed OpTC data
- a BERT tokenizer and model
- a metadata flag controlling whether semantic process embeddings are added
- a dataset configuration object specifying node types, action types, and encoding lengths
- a selected node type used to identify process nodes in the graph

## Outputs
- features_e: a dictionary mapping node IDs to their per-node feature vectors
- optionally saves .pkl files for each client and split using the main() routine

## Key Components
### Functions / Classes
- Encoding_builder:
  - Purpose: central object that computes all graph and node encodings.
  - Inputs: graph, tokenizer/model, metadata, selected nodes, config.
  - Outputs: feature vector and per-node feature map.

- build_encoding:
  - Purpose: orchestrates the encoding workflow for a graph.
  - Inputs: none beyond object state.
  - Outputs: global graph features and node features.

- encoding_process:
  - Purpose: generate semantic embeddings for each process node using the BERT model on process metadata.
  - Inputs: graph and node list.
  - Outputs: process-level features per node.

- get_degree:
  - Purpose: computes degree, in-degree, and out-degree statistics for graph nodes.
  - Inputs: graph and list of nodes.
  - Outputs: global and per-node degree features.

- lifetimenodes:
  - Purpose: computes lifetime statistics over the selected nodes.

- node_time_for_node_type:
  - Purpose: measures temporal activity patterns aggregated by node type.

- edge_type_for_node_type:
  - Purpose: counts outgoing/incoming edge types and neighboring node types to generate structural features.

### Important Workflow
1. The graph is inspected and selected process nodes are extracted from the graph edges.
2. Structural statistics are computed: degree, edge types, node type frequencies, lifetime, and temporal activity.
3. If metadata is enabled, each process is encoded with BERT-based text embeddings from its image path and related metadata.
4. The result is combined into graph-level features and per-node feature arrays.

## Dependencies
- networkx
- numpy
- tqdm
- features.bert
- peft
- transformers
- utils.utils

## Notes
- This file is a graph-feature builder for the BERT-enhanced representation pipeline.
- It mixes structural features and semantic metadata, which makes it a central component for anomaly detection on process graphs.
- The global features are mostly structural summaries, while per-node features are richer and include semantic information when metadata=True.
