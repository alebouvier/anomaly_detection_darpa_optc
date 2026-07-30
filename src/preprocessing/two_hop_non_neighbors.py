
import csv
import datetime as dt
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm



from utils.utils import BASE, create_folder, save_pkl

OBJECT_TYPES = {"PROCESS", "FILE", "SHELL"}
EDGE_COLUMNS = ["u", "i", "ts", "label", "idx"]
EDGE_FEATURE_COLUMNS = ["action_type", "command_line"]
NODE_COLUMNS = ["object_type", "path"]

def compute_2hop_non_neighbors(csv_path: str, output_csv_path: Optional[str] = None) -> str:
    """
    For each edge (u, v, t), find all nodes reachable in exactly 2 hops
    from u in the graph of edges with timestamp < t, excluding direct neighbors of u.

    This implementation streams the CSV file and writes results row-by-row to avoid
    building the full output in RAM.
    """
    if output_csv_path is None:
        output_csv_path = str(Path(csv_path).with_name("two_hop_non_neighbors.csv"))

    create_folder(Path(output_csv_path).parent)
    neighbors: dict[int, set[int]] = defaultdict(set)

    prev_ts = None
    pending_edges = []  # edges to add after processing current timestamp group

    def add_edges_to_graph(edges):
        """Add the current batch of edges to the temporary graph view."""
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)

    with open(csv_path, newline="", encoding="utf-8") as csv_file, open(output_csv_path, "w", newline="", encoding="utf-8") as out_file:
        reader = csv.DictReader(csv_file)
        writer = csv.DictWriter(out_file, fieldnames=["u", "i", "ts", "two_hop_non_neighbors"])
        writer.writeheader()

        for row in tqdm(reader, desc="Computing 2-hop non-neighbors", unit="edge"):
            try:
                u = int(row["u"])
                v = int(row["i"])
                t = float(row["ts"])
            except (KeyError, ValueError):
                continue

            if prev_ts is not None and t != prev_ts:
                add_edges_to_graph(pending_edges)
                pending_edges = []

            direct_neighbors_u = neighbors[u]
            sampled_neighbors = set()
            if direct_neighbors_u:
                sample_size = min(100, len(direct_neighbors_u))
                sampled_neighbors = set(random.sample(list(direct_neighbors_u), sample_size))

            two_hop = set()
            for w in sampled_neighbors:
                two_hop |= neighbors[w]
                if len(two_hop) > 20:
                    break

            two_hop.discard(u)
            two_hop -= sampled_neighbors

            writer.writerow({
                "u": u,
                "i": v,
                "ts": t,
                "two_hop_non_neighbors": ";".join(str(node) for node in sorted(two_hop)),
            })

            pending_edges.append((u, v))
            prev_ts = t

    add_edges_to_graph(pending_edges)
    return output_csv_path

def main( dataset, clients) -> None:
    """Perform the work of main."""
    for client in clients:
        print(f"Processing client {client} for dataset {dataset}")
        neighbor_csv_path = Path(BASE) / "processed_data" / f"{dataset}_{client}" / f"ml_{dataset}_{client}.csv"
        output_path = Path(BASE) / "processed_data" / f"{dataset}_{client}" / "two_hop_non_neighbors.csv"
        output_path = compute_2hop_non_neighbors(str(neighbor_csv_path), str(output_path))
        print(f"Saved 2-hop non-neighbors for client {client} at {output_path}")



