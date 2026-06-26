import argparse
import csv
import gzip
import importlib
import json
import datetime as dt
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from scipy.sparse import lil_matrix, csr_matrix
from itertools import groupby


from utils.utils import BASE, create_folder, save_pkl

OBJECT_TYPES = {"PROCESS", "FILE", "SHELL"}
EDGE_COLUMNS = ["u", "i", "ts", "label", "pattern_id", "idx"]
EDGE_FEATURE_COLUMNS = ["action_type", "command_line"]
NODE_COLUMNS = ["object_type", "path"]


@dataclass
class LogStats:
    edge_list: List[List]
    edge_features: List[List]
    node_features: List[List]
    path_cmd_list: List[str]


def safe_get(data: Optional[dict], key: str) -> Optional[str]:
    return data.get(key) if isinstance(data, dict) else None


def parse_timestamp(timestamp: Optional[str]) -> float:
    if timestamp is None:
        raise ValueError("Missing timestamp in log record")

    fmt = "%Y-%m-%dT%H:%M:%S.%f%z" if "." in timestamp else "%Y-%m-%dT%H:%M:%S%z"
    return dt.datetime.strptime(timestamp, fmt).timestamp()


def read_log_file(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as log_file:
        for line in tqdm(log_file, desc=f"Reading {path.name}", unit="lines"):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_logs(file_paths: Iterable[str]) -> Iterator[dict]:
    for file_path in file_paths:
        yield from read_log_file(Path(file_path))


def collect_log_stats(log_iter: Iterator[dict], anomalies) -> LogStats:
    objects = {}
    edge_list: List[List] = []
    edge_features: List[List] = [[None, None]]
    node_features: List[List] = [[None, None]]
    path_cmd_list: List[str] = []
    node_idx = 1
    edge_idx = 1
    pattern_id = 0

    for log in log_iter:
        object_type = safe_get(log, "object")
        if object_type not in OBJECT_TYPES:
            continue

        actor_id = safe_get(log, "actorID")
        object_id = safe_get(log, "objectID")
        timestamp_str = safe_get(log, "timestamp")
        properties = safe_get(log, "properties") or {}
        action_type = safe_get(log, "action")
        image_path = safe_get(properties, "image_path")
        file_path = safe_get(properties, "file_path")
        command_line = safe_get(properties, "command_line")
        log_id = safe_get(log, "id")

        if actor_id is None or object_id is None or timestamp_str is None:
            continue


        for candidate in (image_path, file_path, command_line):
            if candidate is not None:
                path_cmd_list.append(candidate)

        actor_key = ("PROCESS", actor_id)
        if actor_key not in objects:
            objects[actor_key] = node_idx
            node_features.append(["PROCESS", image_path])
            node_idx += 1
        else:
            actor_index = objects[actor_key]
            if node_features[actor_index][1] is None and image_path is not None:
                node_features[actor_index][1] = image_path

        object_key = (object_type, object_id)
        if object_key not in objects:
            objects[object_key] = node_idx
            node_features.append([object_type, file_path])
            node_idx += 1
        else:
            object_index = objects[object_key]
            if node_features[object_index][1] is None and file_path is not None:
                node_features[object_index][1] = file_path
        
        if log_id in anomalies:
            label = 1
        else:
            label = 0
        

        edge_list.append([objects[actor_key], objects[object_key], parse_timestamp(timestamp_str), label, pattern_id, edge_idx])
        edge_features.append([action_type, command_line])
        edge_idx += 1

    # Sort edges by timestamp while keeping the corresponding edge feature row aligned.
    sorted_edges = sorted(edge_list, key=lambda edge: edge[2])
    sorted_edge_features: List[List] = [[None, None]]
    for new_idx, edge in enumerate(sorted_edges, start=1):
        old_idx = edge[-1]
        edge[-1] = new_idx
        sorted_edge_features.append(edge_features[old_idx])

    return LogStats(
        edge_list=sorted_edges,
        edge_features=sorted_edge_features,
        node_features=node_features,
        path_cmd_list=path_cmd_list,
    )

def extract_ids(log_iter: Iterator[dict]):
    anomaly_dict: dict = {}

    for log in log_iter:
        client = safe_get(log, "hostname").lower()
        log_id = safe_get(log, "id")

        if client not in anomaly_dict:
            anomaly_dict[client] = []
        
        anomaly_dict[client].append(log_id)
    
    return anomaly_dict




def build_dataframes(stats: LogStats) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_edge_list = pd.DataFrame(stats.edge_list, columns=EDGE_COLUMNS)
    df_edge_features = pd.DataFrame(stats.edge_features, columns=EDGE_FEATURE_COLUMNS)
    df_node_features = pd.DataFrame(stats.node_features, columns=NODE_COLUMNS)
    return df_edge_list, df_edge_features, df_node_features


def save_preprocessing_data(output_dir: Path, stats: LogStats, dataset, client) -> None:
    create_folder(output_dir)
    df_edge_list, df_edge_features, df_node_features = build_dataframes(stats)

    df_edge_list.to_csv(output_dir / f"ml_{dataset}_{client}.csv", index=False)
    df_edge_features.to_csv(output_dir / "edge_features.csv", index=False)
    df_node_features.to_csv(output_dir / "node_features.csv", index=False)
    save_pkl(stats.path_cmd_list, str(output_dir / "path_cmd_list.pkl"))


def process_client(logs_dir: str, dataset: str, client: str, anomalies) -> None:
    dataset_module = importlib.import_module(f"data.{dataset}_utils")
    log_files = dataset_module.logs_from_folder(logs_dir, client)
    stats = collect_log_stats(iter_logs(log_files), anomalies)
    output_dir = Path(BASE) / "processed_data" / f"{dataset}_{client}"
    save_preprocessing_data(output_dir, stats, dataset, client)


def main(logs, dataset, clients) -> None:
    anomaly_path = [Path(BASE) / "label_data" / "malicious.json"]
    anomalies = extract_ids(iter_logs(anomaly_path))
    for client in clients:
        hostname = f"sysclient0{client}.systemia.com"
        print(f"Processing client {client} for dataset {dataset}")
        process_client(logs, dataset, client, anomalies[hostname])



