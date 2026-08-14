# preprocessing/two_hop_non_neighbors.py

## Purpose
This module computes a lightweight graph feature describing, for each edge, the set of nodes reachable in two hops while excluding the direct neighbors of the source node. It is meant to enrich temporal graph data with neighborhood context before model training or graph-based analysis.

## Responsibilities
- stream a chronological edge list from a CSV file
- maintain neighbor information up to the current timestamp
- compute candidate two-hop non-neighbors for each edge
- write the result as a new CSV file for downstream use

## Inputs
- a processed edge CSV file such as ml_optc_{client}.csv
- an optional output path; otherwise the file is saved next to the input CSV

## Outputs
- a CSV file named two_hop_non_neighbors.csv containing, for each original edge, the list of two-hop non-neighbors

## Key Components
### Functions / Classes
- compute_2hop_non_neighbors:
  - Purpose: compute the two-hop non-neighbor list in a streaming fashion without fully materializing the whole graph in memory.
  - Inputs: CSV edge path and optional output path.
  - Outputs: output CSV path.

- main:
  - Purpose: process each client in the requested dataset and generate its two-hop summary file.

### Important Workflow
1. The edge list is read in time order.
2. Neighbor sets are incrementally maintained for the graph built so far.
3. For each edge, a limited sample of direct neighbors is used to explore the second hop.
4. The direct neighbors and the sampled second-hop nodes are filtered out, leaving the two-hop non-neighbors.
5. The final list is written to disk.

## Dependencies
- pandas
- numpy
- tqdm
- utils.utils

## Notes
- This is a graph-context enrichment utility rather than a full feature extraction pipeline.
- It uses a sampled direct-neighbor strategy to keep memory and runtime manageable.
- The result is stored in a compact string format separated by semicolons, which is easy to consume in additional analysis steps.
