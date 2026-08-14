import gzip
import json
import importlib
import re
import traceback
import datetime as dt

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.utils import period, round_duration, save_pkl, open_config, BASE

def get_unique_stats_features(client, start_val, start_test):
    """Build the anomaly-path summary features for a client from the processed logs."""
    val_time = dt.datetime.strptime(start_val, "%Y-%m-%dT%H:%M").timestamp()
    test_time = dt.datetime.strptime(start_test, "%Y-%m-%dT%H:%M").timestamp()

    # open node_features.csv
    node_feature_file_path = f"{BASE}/processed_data/optc_{client}/node_features.csv"
    node_features = pd.read_csv(node_feature_file_path, header=0)
    edge_feature_file_path = f"{BASE}/processed_data/optc_{client}/edge_features.csv"
    edge_features = pd.read_csv(edge_feature_file_path, header=0)
    edge_list_file_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_file_path, header=0)

    node_features["idx"] = node_features.index
    edge_features["idx"] = edge_features.index

    node_features["path_clean"] = node_features["path"].apply(normalize_cmdline)


    # unique_edges = create_unique_edges_df(edge_list, node_features, edge_features)
    # unique_edges.to_csv(f"{BASE}/processed_data/unique_edges.csv")

    train_edges = edge_list[edge_list["ts"] < val_time]
    val_edges = edge_list[(edge_list["ts"] >= val_time) & (edge_list["ts"] < test_time)]
    test_edges = edge_list[edge_list["ts"] >= test_time]
    anomaly_edges = test_edges[test_edges["label"] == 1]

    edge_features["command_line_clean"] = edge_features["command_line"].apply(normalize_cmdline)

    # unique_train_edges = create_unique_edges_df(train_edges, node_features, edge_features)
    # unique_val_edges = create_unique_edges_df( val_edges, node_features, edge_features)
    # unique_test_edges = create_unique_edges_df(test_edges, node_features, edge_features)
    # unique_anomaly_edges = create_unique_edges_df(anomaly_edges, node_features, edge_features)

    # val_similar_edges = find_similar_edges(unique_train_edges, unique_val_edges)
    # test_similar_edges = find_similar_edges(unique_train_edges, unique_test_edges)
    # anomaly_similar_edges = find_similar_edges(unique_train_edges, unique_anomaly_edges)

    anomaly_paths = find_temporal_paths_length_2(anomaly_edges)
    unique_anomaly_paths = create_unique_paths_df(anomaly_paths, node_features, edge_features, edge_list)
    unique_anomaly_paths.to_csv(f"{BASE}/processed_data/optc_{client}/unique_anomaly_paths.csv")

    # anomaly_paths_in_train = find_anomaly_paths_in_train(
    #     anomaly_paths, train_edges, node_features, edge_features
    # )
    # anomaly_paths_in_train.to_csv(f"{BASE}/processed_data/optc_{client}/anomaly_paths_in_train.csv", index=False)

    # print("-- ANOMALY PATHS --")
    # print(f"num unique anomaly paths: {unique_anomaly_paths.shape[0]}")
    # print(f"num unique anomaly paths present in train: {anomaly_paths_in_train.shape[0]}")

    # anomaly_edges_not_in_train = find_non_present_edges(unique_train_edges, unique_anomaly_edges)
    # anomaly_edges_not_in_train.to_csv(f"{BASE}/processed_data/optc_{client}/anomaly_edges_not_in_train.csv")

    # print("-- TRAIN --")
    # print(f"num total edges: {train_edges.shape[0]}")
    # print(f"num distinct edges: {unique_train_edges.shape[0]}")
    
    # print("-- VAL --")
    # print(f"num total edges: {val_edges.shape[0]}")
    # print(f"num distinct edges: {unique_val_edges.shape[0]}")
    # print(f"num edges present in train: {val_similar_edges['n_edges'].sum()}" )
    # print(f"num distinct edges present in train: {val_similar_edges.shape[0]}" )

    # print("-- TEST --")
    # print(f"num total edges: {test_edges.shape[0]}")
    # print(f"num distinct edges: {unique_test_edges.shape[0]}")
    # print(f"num edges present in train: {test_similar_edges['n_edges'].sum()}" )
    # print(f"num distinct edges present in train: {test_similar_edges.shape[0]}" )

    # print("-- ANOMALY --")
    # print(f"num total edges: {anomaly_edges.shape[0]}")
    # print(f"num distinct edges: {unique_anomaly_edges.shape[0]}")
    # print(f"num edges present in train: {anomaly_similar_edges['n_edges'].sum()}" )
    # print(f"num distinct edges present in train: {anomaly_similar_edges.shape[0]}" )
    return unique_anomaly_paths


def _command_line_key(series):
    """Normalize a command-line value into a stable key for grouping."""
    return series.fillna("__COMMAND_LINE_NULL__")


def find_non_present_edges(split_edges_1, split_edges_2):
    """Return the edges seen in one split but absent from the other."""
    split_edges_1 = split_edges_1.copy()
    split_edges_2 = split_edges_2.copy()
    split_edges_1["_command_line_key"] = _command_line_key(split_edges_1["command_line"])
    split_edges_2["_command_line_key"] = _command_line_key(split_edges_2["command_line"])

    merged = split_edges_2.merge(
        split_edges_1[["object_type_src", "object_type_dst", "action_type", "_command_line_key"]],
        on=["object_type_src", "object_type_dst", "action_type", "_command_line_key"],
        how="left",
        indicator=True,
    )

    return merged.loc[
        merged["_merge"] == "left_only",
        ["object_type_src", "object_type_dst", "action_type", "command_line", "n_edges"],
    ].copy()


def create_unique_edges_df(split_edges, node_features, edge_features):
    """Aggregate repeated edges into a unique table with counts and anomaly totals."""
    source_nodes = node_features[["idx", "object_type"]].rename(
        columns={"idx": "node_idx", "object_type": "object_type_src"}
    )
    target_nodes = node_features[["idx", "object_type"]].rename(
        columns={"idx": "node_idx", "object_type": "object_type_dst"}
    )

    df = split_edges.merge(source_nodes, left_on="u", right_on="node_idx", how="left").drop(columns=["node_idx"])
    df = df.merge(target_nodes, left_on="i", right_on="node_idx", how="left").drop(columns=["node_idx"])
    df = df.merge(
        edge_features[["idx", "action_type", "command_line_clean"]],
        on="idx",
        how="left",
    )

    return (
        df.groupby(
            ["object_type_src", "object_type_dst", "action_type", "command_line_clean"],
            dropna=False,
            as_index=False,
        )
        .agg(n_edges=("idx", "size"), n_anomalies=("label", "sum"))
        .rename(columns={"command_line_clean": "command_line"})
    )


def find_similar_edges(split_edges_1, split_edges_2):
    """Return the edges that appear with the same signature in both splits."""
    split_edges_1 = split_edges_1.copy()
    split_edges_2 = split_edges_2.copy()
    split_edges_1["_command_line_key"] = _command_line_key(split_edges_1["command_line"])
    split_edges_2["_command_line_key"] = _command_line_key(split_edges_2["command_line"])

    merged = split_edges_1.merge(
        split_edges_2[["object_type_src", "object_type_dst", "action_type", "command_line", "n_edges", "_command_line_key"]],
        on=["object_type_src", "object_type_dst", "action_type", "_command_line_key"],
        how="inner",
    )

    return merged[["object_type_src", "object_type_dst", "action_type", "command_line", "n_edges"]].copy()


def find_similar_paths(split_paths_1, split_paths_2):
    """Find two-hop paths with the same structure and commands in both splits."""
    split_paths_1 = split_paths_1.copy()
    split_paths_2 = split_paths_2.copy()

    split_paths_1["_command_line_1_key"] = _command_line_key(split_paths_1["command_line_1"])
    split_paths_1["_command_line_2_key"] = _command_line_key(split_paths_1["command_line_2"])
    split_paths_2["_command_line_1_key"] = _command_line_key(split_paths_2["command_line_1"])
    split_paths_2["_command_line_2_key"] = _command_line_key(split_paths_2["command_line_2"])

    merged = split_paths_2.merge(
        split_paths_1[
            [
                "object_type_start",
                "object_type_middle",
                "object_type_end",
                "action_type_1",
                "action_type_2",
                "_command_line_1_key",
                "_command_line_2_key",
            ]
        ],
        on=[
            "object_type_start",
            "object_type_middle",
            "object_type_end",
            "action_type_1",
            "action_type_2",
            "_command_line_1_key",
            "_command_line_2_key",
        ],
        how="inner",
    )

    return merged[
        [
            "object_type_start",
            "object_type_middle",
            "object_type_end",
            "action_type_1",
            "command_line_1",
            "action_type_2",
            "command_line_2",
            "n_edges",
        ]
    ].copy()


def build_edge_signature_df(edges, node_features, edge_features):
    """Add object-type and command-line signatures to the edge table."""
    source_nodes = node_features[["idx", "object_type"]].rename(
        columns={"idx": "u", "object_type": "object_type_src"}
    )
    target_nodes = node_features[["idx", "object_type"]].rename(
        columns={"idx": "i", "object_type": "object_type_dst"}
    )

    df = edges.merge(source_nodes, on="u", how="left")
    df = df.merge(target_nodes, on="i", how="left")
    df = df.merge(
        edge_features[["idx", "action_type", "command_line_clean"]],
        on="idx",
        how="left",
    )
    df["_command_line_key"] = _command_line_key(df["command_line_clean"])
    return df


def find_anomaly_paths_in_train(anomaly_paths, train_edges, node_features, edge_features):
    """Find anomaly paths that already appear in the training data."""
    train_sig = build_edge_signature_df(train_edges, node_features, edge_features)

    anomaly_sig = anomaly_paths.merge(
        node_features[["idx", "object_type"]].rename(
            columns={"idx": "start", "object_type": "object_type_start"}
        ),
        on="start",
        how="left",
    )
    anomaly_sig = anomaly_sig.merge(
        node_features[["idx", "object_type"]].rename(
            columns={"idx": "middle", "object_type": "object_type_middle"}
        ),
        on="middle",
        how="left",
    )
    anomaly_sig = anomaly_sig.merge(
        node_features[["idx", "object_type"]].rename(
            columns={"idx": "end", "object_type": "object_type_end"}
        ),
        on="end",
        how="left",
    )
    anomaly_sig = anomaly_sig.merge(
        edge_features[["idx", "action_type", "command_line_clean"]].rename(
            columns={
                "idx": "idx_1",
                "action_type": "action_type_1",
                "command_line_clean": "command_line_1",
            }
        ),
        on="idx_1",
        how="left",
    )
    anomaly_sig = anomaly_sig.merge(
        edge_features[["idx", "action_type", "command_line_clean"]].rename(
            columns={
                "idx": "idx_2",
                "action_type": "action_type_2",
                "command_line_clean": "command_line_2",
            }
        ),
        on="idx_2",
        how="left",
    )

    anomaly_sig["_command_line_1_key"] = _command_line_key(anomaly_sig["command_line_1"])
    anomaly_sig["_command_line_2_key"] = _command_line_key(anomaly_sig["command_line_2"])

    matched_first = anomaly_sig.merge(
        train_sig,
        left_on=[
            "middle",
            "object_type_start",
            "object_type_middle",
            "action_type_1",
            "_command_line_1_key",
        ],
        right_on=[
            "i",
            "object_type_src",
            "object_type_dst",
            "action_type",
            "_command_line_key",
        ],
        how="inner",
    )

    matched_both = matched_first.merge(
        train_sig,
        left_on=[
            "middle",
            "object_type_middle",
            "object_type_end",
            "action_type_2",
            "_command_line_2_key",
        ],
        right_on=[
            "u",
            "object_type_src",
            "object_type_dst",
            "action_type",
            "_command_line_key",
        ],
        how="inner",
        suffixes=("", "_train2"),
    )

    return matched_both[
        [
            "object_type_start",
            "object_type_middle",
            "object_type_end",
            "action_type_1",
            "command_line_1",
            "action_type_2",
            "command_line_2",
        ]
    ].drop_duplicates().copy()


def normalize_cmdline(cmd):
    """Normalize a command-line string so similar commands can be compared reliably."""
    if pd.isna(cmd):
        return None

    s = cmd.lower()
    s = re.sub(r'\\\\\?\\', '', s)                                          # UNC prefix
    s = re.sub(r'(c:\\users\\)[^\\]+', r'\1<USER>', s)                      # usernames
    s = re.sub(r'(\\temp\\)[a-z0-9]+\.tmp', r'\1<TMPDIR>', s)               # temp dirs
    s = re.sub(r'rust_mozprofile\.\S+', 'rust_mozprofile.<TMPPROFILE>', s)  # FF profiles
    s = re.sub(r'[\w\s\-_.]+\.(pdf|doc|docx|xls|xlsx|txt)', r'<DOC>.\1', s)# documents
    s = re.sub(r'--channel="[^"]+"', '--channel=<CHANNEL>', s)              # Chrome channel
    s = re.sub(r'(windowtitle eq )[^"*]+(\*?")', r'\1<WINTITLE>\2', s)      # taskkill titles
    s = re.sub(r'(/d\s+)"?[^"/\s][^"]*"?', r'\1<REGVAL>', s)               # registry values
    s = re.sub(r'global\\[a-z_]+\d+_?', r'global\\<PIPE>', s)              # named pipes  ← NEW
    s = re.sub(r'\{[0-9a-f<>\-_A-Z]+\}', '<GUID>', s)                      # GUIDs        ← NEW
    s = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', s)                  # IPs          ← NEW
    s = re.sub(r'0x[0-9a-f]+', '<HEX>', s)                                 # hex literals
    s = re.sub(r'\b[0-9a-f]{4,}\b', '<HEXID>', s)                          # long hex IDs
    s = re.sub(r'\b\d{3,}\b', '<NUM>', s)                                   # standalone numbers
    s = re.sub(r'(?<=[a-z_])\d+', '<NUM>', s)                              # embedded numbers ← NEW
    s = re.sub(r'  +', ' ', s).strip()

    return s

def find_temporal_paths_length_2(edge_list):
    """Extract all length-2 temporal paths whose edges occur in chronological order."""
    start_node_list = []
    middle_node_list = []
    end_node_list = []
    edge_idx_1_list = []
    edge_idx_2_list = []
    for i, edge in edge_list.iterrows():
        src_node = edge["u"]
        dst_node = edge["i"]
        timestamp = edge["ts"]
        following_edges = edge_list[(edge_list["u"] == dst_node) & (edge_list["ts"] > timestamp)]
        for j, following_edge in following_edges.iterrows():
            start_node_list.append(src_node)
            middle_node_list.append(dst_node)
            end_node_list.append(following_edge["i"])
            edge_idx_1_list.append(edge["idx"])
            edge_idx_2_list.append(following_edge["idx"])
    return pd.DataFrame({"start": start_node_list, 
                         "middle": middle_node_list, 
                         "end": end_node_list, 
                         "idx_1": edge_idx_1_list, 
                         "idx_2": edge_idx_2_list})


def create_unique_paths_df(split_path, node_features, edge_features, edge_list):
    """Aggregate repeated temporal paths into a single summary table."""
    path_df = split_path.merge(
        node_features[["idx", "object_type"]].rename(
            columns={"idx": "start", "object_type": "object_type_start"}
        ),
        on="start",
        how="left",
    )
    path_df = path_df.merge(
        node_features[["idx", "object_type"]].rename(
            columns={"idx": "middle", "object_type": "object_type_middle"}
        ),
        on="middle",
        how="left",
    )
    path_df = path_df.merge(
        node_features[["idx", "object_type"]].rename(
            columns={"idx": "end", "object_type": "object_type_end"}
        ),
        on="end",
        how="left",
    )
    path_df = path_df.merge(
        edge_features[["idx", "action_type", "command_line_clean"]].rename(
            columns={
                "idx": "idx_1",
                "action_type": "action_type_1",
                "command_line_clean": "command_line_1",
            }
        ),
        on="idx_1",
        how="left",
    )
    path_df = path_df.merge(
        edge_features[["idx", "action_type", "command_line_clean"]].rename(
            columns={
                "idx": "idx_2",
                "action_type": "action_type_2",
                "command_line_clean": "command_line_2",
            }
        ),
        on="idx_2",
        how="left",
    )
    path_df = path_df.merge(
        edge_list[["idx", "ts"]].rename(columns={"idx": "idx_1", "ts": "ts_1"}),
        on="idx_1",
        how="left",
    )
    path_df = path_df.merge(
        edge_list[["idx", "ts"]].rename(columns={"idx": "idx_2", "ts": "ts_2"}),
        on="idx_2",
        how="left",
    )
    path_df["delta_t"] = path_df["ts_2"] - path_df["ts_1"]

    if "label" in edge_features.columns:
        label_1 = edge_features[["idx", "label"]].rename(
            columns={"idx": "idx_1", "label": "label_1"}
        )
        label_2 = edge_features[["idx", "label"]].rename(
            columns={"idx": "idx_2", "label": "label_2"}
        )
        path_df = path_df.merge(label_1, on="idx_1", how="left")
        path_df = path_df.merge(label_2, on="idx_2", how="left")
        path_df["n_anomalies"] = path_df[["label_1", "label_2"]].sum(axis=1, skipna=True)
    else:
        path_df["n_anomalies"] = 0

    return path_df.groupby(
        [
            "object_type_start",
            "object_type_middle",
            "object_type_end",
            "action_type_1",
            "command_line_1",
            "action_type_2",
            "command_line_2",
        ],
        dropna=False,
        as_index=False,
    ).agg(
        n_edges=("start", "size"),
        n_anomalies=("n_anomalies", "sum"),
        delta_t=("delta_t", "mean"),
    )








def add_anomaly_paths_in_train_val(unique_anomaly_paths, client, start_val, start_test):
    """Inject synthetic anomaly paths into the training/validation set and return the updated tables."""
    val_time = dt.datetime.strptime(start_val, "%Y-%m-%dT%H:%M").timestamp()
    test_time = dt.datetime.strptime(start_test, "%Y-%m-%dT%H:%M").timestamp()

    # open node_features.csv
    node_feature_file_path = f"{BASE}/processed_data/optc_{client}/node_features.csv"
    node_features = pd.read_csv(node_feature_file_path, header=0)
    edge_feature_file_path = f"{BASE}/processed_data/optc_{client}/edge_features.csv"
    edge_features = pd.read_csv(edge_feature_file_path, header=0)
    edge_list_file_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_file_path, header=0)

    node_features["idx"] = node_features.index
    edge_features["idx"] = edge_features.index


    train_val_edges = edge_list[edge_list["ts"] < test_time].copy()

    if unique_anomaly_paths.empty or train_val_edges.empty:
        return edge_list, edge_features

    required_columns = [
        "object_type_start",
        "object_type_middle",
        "object_type_end",
        "action_type_1",
        "command_line_1",
        "action_type_2",
        "command_line_2",
    ]
    missing = [col for col in required_columns if col not in unique_anomaly_paths.columns]
    if missing:
        raise ValueError(f"unique_anomaly_paths is missing required columns: {missing}")

    nodes_by_type = node_features.groupby("object_type")["idx"].apply(list).to_dict()
    weights = unique_anomaly_paths.get("n_edges", pd.Series(1.0, index=unique_anomaly_paths.index)).astype(float)
    weights = weights.fillna(1.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=unique_anomaly_paths.index)
    weights = weights / weights.sum()

    new_edge_list = []
    new_edge_features = []
    edge_list_columns = edge_list.columns.tolist()
    feature_columns = edge_features.columns.tolist()

    unique_anomaly_paths = unique_anomaly_paths.reset_index(drop=True)
    num_paths = len(unique_anomaly_paths)
    path_choices = np.random.choice(num_paths, size=int(len(train_val_edges)/10), p=weights.values)
    train_val_rows = list(train_val_edges.itertuples(index=False, name=None))[::10]

    path_values = unique_anomaly_paths.loc[path_choices, [
        "object_type_start",
        "object_type_middle",
        "object_type_end",
        "action_type_1",
        "command_line_1",
        "action_type_2",
        "command_line_2",
        "delta_t",
    ]].to_dict(orient="records")

    node_pools = {k: np.asarray(v, dtype=int) for k, v in nodes_by_type.items()}

    for train_val_edge, path in tqdm(zip(train_val_rows, path_values)):
        start_nodes = node_pools.get(path["object_type_start"])
        middle_nodes = node_pools.get(path["object_type_middle"])
        end_nodes = node_pools.get(path["object_type_end"])
        if start_nodes is None or middle_nodes is None or end_nodes is None:
            continue

        start_node = np.random.choice(start_nodes)
        middle_node = np.random.choice(middle_nodes)
        if len(middle_nodes) > 1 and middle_node == start_node:
            candidates = middle_nodes[middle_nodes != start_node]
            if len(candidates):
                middle_node = np.random.choice(candidates)

        end_node = np.random.choice(end_nodes)
        if len(end_nodes) > 1 and end_node == middle_node:
            candidates = end_nodes[end_nodes != middle_node]
            if len(candidates):
                end_node = np.random.choice(candidates)

        ts1 = train_val_edge[2] + 0.001
        ts2 = ts1 + max(path.get("delta_t", 0.001), 0.001)

        edge1 = {col: getattr(train_val_edge, col) if hasattr(train_val_edge, col) else np.nan for col in edge_list_columns}
        edge1.update({"u": start_node, "i": middle_node, "ts": ts1, "label": 1})
        edge2 = {col: getattr(train_val_edge, col) if hasattr(train_val_edge, col) else np.nan for col in edge_list_columns}
        edge2.update({"u": middle_node, "i": end_node, "ts": ts2, "label": 1})

        if ts1 < test_time:
            new_edge_list.append(edge1)

        if ts2 < test_time:
            new_edge_list.append(edge2)

        feature1 = {col: np.nan for col in feature_columns}
        feature1.update(
            {
                "action_type": path["action_type_1"],
                "command_line": path["command_line_1"],
            }
        )
        feature2 = {col: np.nan for col in feature_columns}
        feature2.update(
            {
                "action_type": path["action_type_2"],
                "command_line": path["command_line_2"],
            }
        )

        new_edge_features.append(feature1)
        new_edge_features.append(feature2)

    if not new_edge_list:
        return edge_list, edge_features

    edge_list = pd.concat([edge_list, pd.DataFrame(new_edge_list)], ignore_index=True, sort=False)
    edge_features = pd.concat([edge_features, pd.DataFrame(new_edge_features)], ignore_index=True, sort=False)

    edge_list["_original_row"] = np.arange(len(edge_list)) + 1
    edge_list = edge_list.sort_values(by="ts", kind="mergesort").reset_index(drop=True)

    edge_features = edge_features.loc[edge_list["_original_row"]].reset_index(drop=True)
    edge_list = edge_list.drop(columns=["_original_row"])

    edge_list["idx"] = edge_list.index + 1

    edge_features = edge_features.drop(columns=["idx"])

    edge_features = pd.concat([pd.DataFrame([{"action_type": np.nan, "command_line": np.nan}]), edge_features], ignore_index=True, sort=False)

    return edge_list, edge_features









def main(clients, start_val, start_test):
    """Generate the temporal-pattern anomaly features for the selected clients."""
    for client in clients:
        unique_anomaly_paths = get_unique_stats_features(client, start_val, start_test)
        # unique_anomaly_paths = pd.read_csv(f"{BASE}/processed_data/optc_{client}/unique_anomaly_paths.csv")
        edge_list, edge_features = add_anomaly_paths_in_train_val(unique_anomaly_paths, client, start_val, start_test)
        edge_list.to_csv(f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv", index=False)
        edge_features.to_csv(f"{BASE}/processed_data/optc_{client}/edge_features.csv", index=False)

