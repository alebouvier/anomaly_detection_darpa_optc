from collections import defaultdict
import gzip
import json
import importlib
import re
import traceback
import datetime as dt

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.extract_all_anomaly import add_anomaly_whole_in_train_val, extract_whole_anomaly, add_anomaly_nodes_in_train_val

try:
    from utils.utils import period, round_duration, save_pkl, open_config, BASE
except ModuleNotFoundError:  # pragma: no cover - fallback for repo-root execution
    from src.utils.utils import period, round_duration, save_pkl, open_config, BASE

def get_unique_stats_features(client, start_val, start_test):
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



    # unique_edges = create_unique_edges_df(edge_list, node_features, edge_features)
    # unique_edges.to_csv(f"{BASE}/processed_data/unique_edges.csv")

    train_edges = edge_list[edge_list["ts"] < val_time]
    val_edges = edge_list[(edge_list["ts"] >= val_time) & (edge_list["ts"] < test_time)]
    test_edges = edge_list[edge_list["ts"] >= test_time]
    anomaly_edges = test_edges[test_edges["label"] == 1]

    edge_features["command_line"] = edge_features["command_line"].apply(normalize_cmdline)

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
    return series.fillna("__COMMAND_LINE_NULL__")


def find_non_present_edges(split_edges_1, split_edges_2):
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
    source_nodes = node_features[["idx", "object_type"]].rename(
        columns={"idx": "node_idx", "object_type": "object_type_src"}
    )
    target_nodes = node_features[["idx", "object_type"]].rename(
        columns={"idx": "node_idx", "object_type": "object_type_dst"}
    )

    df = split_edges.merge(source_nodes, left_on="u", right_on="node_idx", how="left").drop(columns=["node_idx"])
    df = df.merge(target_nodes, left_on="i", right_on="node_idx", how="left").drop(columns=["node_idx"])
    df = df.merge(
        edge_features[["idx", "action_type", "command_line"]],
        on="idx",
        how="left",
    )

    return (
        df.groupby(
            ["object_type_src", "object_type_dst", "action_type", "command_line"],
            dropna=False,
            as_index=False,
        )
        .agg(n_edges=("idx", "size"), n_anomalies=("label", "sum"))
    )


def find_similar_edges(split_edges_1, split_edges_2):
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
    source_nodes = node_features[["idx", "object_type"]].rename(
        columns={"idx": "u", "object_type": "object_type_src"}
    )
    target_nodes = node_features[["idx", "object_type"]].rename(
        columns={"idx": "i", "object_type": "object_type_dst"}
    )

    df = edges.merge(source_nodes, on="u", how="left")
    df = df.merge(target_nodes, on="i", how="left")
    df = df.merge(
        edge_features[["idx", "action_type", "command_line"]],
        on="idx",
        how="left",
    )
    df["_command_line_key"] = _command_line_key(df["command_line"])
    return df


def find_anomaly_paths_in_train(anomaly_paths, train_edges, node_features, edge_features):
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
        edge_features[["idx", "action_type", "command_line"]].rename(
            columns={
                "idx": "idx_1",
                "action_type": "action_type_1",
                "command_line": "command_line_1",
            }
        ),
        on="idx_1",
        how="left",
    )
    anomaly_sig = anomaly_sig.merge(
        edge_features[["idx", "action_type", "command_line"]].rename(
            columns={
                "idx": "idx_2",
                "action_type": "action_type_2",
                "command_line": "command_line_2",
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
    if pd.isna(cmd):
        return None

    # case-insensitive
    s = cmd.lower() 

    # strips the "\\?\" extended-length path prefix some processes add
    s = re.sub(r'\\\\\?\\', '', s) 

    # replaces the Windows username in user profile paths
    s = re.sub(r'(c:\\users\\)[^\\]+', r'\1<USER>', s)
    
    # replace randomly-named temp folders
    s = re.sub(r'(\\temp\\)[a-z0-9]+\.tmp', r'\1<TMPDIR>', s)

    # replace Firefox's randomly-generated profile suffix
    s = re.sub(r'rust_mozprofile\.\S+', 'rust_mozprofile.<TMPPROFILE>', s)

    # replaces document filenames, keeping only the extension
    s = re.sub(r'[\w\s\-_.]+\.(pdf|doc|docx|xls|xlsx|txt)', r'<DOC>.\1', s)

    # replace Chrome's channel ID 
    s = re.sub(r'--channel="[^"]+"', '--channel=<CHANNEL>', s)

    # replaces the specific window title in taskkill filters
    s = re.sub(r'(windowtitle eq )[^"*]+(\*?")', r'\1<WINTITLE>\2', s)

    # replaces the data value passed to "reg add /d" 
    s = re.sub(r'(/d\s+)"?[^"/\s][^"]*"?', r'\1<REGVAL>', s)

    # replaces named pipe/kernel object names that end in a varying numeric ID
    s = re.sub(r'global\\[a-z_]+\d+_?', r'global\\<PIPE>', s)

    # replaces GUIDs 
    s = re.sub(r'\{[0-9a-f<>\-_A-Z]+\}', '<GUID>', s)

    # replaces IP addresses 
    s = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', s)

    # replaces hexadecimal literals
    s = re.sub(r'0x[0-9a-f]+', '<HEX>', s)

    # replaces long hex-looking standalone IDs (4+ chars)
    s = re.sub(r'\b[0-9a-f]{4,}\b', '<HEXID>', s)

    # replaces standalone numbers of 3+ digits
    s = re.sub(r'\b\d{3,}\b', '<NUM>', s)

    # replaces numbers glued onto a word
    s = re.sub(r'(?<=[a-z_])\d+', '<NUM>', s)

    # collapses repeated spaces left behind by the substitutions above
    s = re.sub(r'  +', ' ', s).strip()

    return s

def find_temporal_paths_length_2(edge_list):
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
        edge_features[["idx", "action_type", "command_line"]].rename(
            columns={
                "idx": "idx_1",
                "action_type": "action_type_1",
                "command_line": "command_line_1",
            }
        ),
        on="idx_1",
        how="left",
    )
    path_df = path_df.merge(
        edge_features[["idx", "action_type", "command_line"]].rename(
            columns={
                "idx": "idx_2",
                "action_type": "action_type_2",
                "command_line": "command_line_2",
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

    edge_features["command_line"] = edge_features["command_line"].apply(normalize_cmdline)

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
    ratio = 10
    path_choices = np.random.choice(num_paths, size=int(len(train_val_edges)/ratio), p=weights.values)
    train_val_rows = list(train_val_edges.itertuples(index=False, name=None))[::ratio]

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

    for id, (train_val_edge, path) in tqdm(enumerate(zip(train_val_rows, path_values))):
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
        edge1.update({"u": start_node, "i": middle_node, "ts": ts1, "label": 1, "pattern_id": id+1})
        edge2 = {col: getattr(train_val_edge, col) if hasattr(train_val_edge, col) else np.nan for col in edge_list_columns}
        edge2.update({"u": middle_node, "i": end_node, "ts": ts2, "label": 1, "pattern_id": id+1})

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





def add_anomaly_frames_in_train_val(anomaly_frames, client, start_val, start_test):
    """Inject anomaly-frame edges into the train/validation split.

    The function samples a portion of the historical edges, then inserts the
    edges contained in each anomaly frame using the frame's node and temporal
    structure. It also appends matching edge-feature rows so the resulting
    dataframes stay aligned.
    """
    _ = dt.datetime.strptime(start_val, "%Y-%m-%dT%H:%M").timestamp()
    test_time = dt.datetime.strptime(start_test, "%Y-%m-%dT%H:%M").timestamp()

    node_feature_file_path = f"{BASE}/processed_data/optc_{client}/node_features.csv"
    node_features = pd.read_csv(node_feature_file_path, header=0)
    edge_feature_file_path = f"{BASE}/processed_data/optc_{client}/edge_features.csv"
    edge_features = pd.read_csv(edge_feature_file_path, header=0)
    edge_list_file_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_file_path, header=0)

    node_features["idx"] = node_features.index
    edge_features["idx"] = edge_features.index

    if "command_line" in edge_features.columns:
        edge_features["command_line"] = edge_features["command_line"].apply(normalize_cmdline)

    train_val_edges = edge_list[edge_list["ts"] < test_time].copy()

    if anomaly_frames is None or anomaly_frames.empty or train_val_edges.empty:
        return edge_list, edge_features

    anomaly_ratio = 1
    num_max_anomalies = int(len(train_val_edges) * anomaly_ratio)
    if num_max_anomalies <= 0:
        return edge_list, edge_features

    new_edge_list = []
    new_edge_features = []
    edge_list_columns = edge_list.columns.tolist()
    feature_columns = edge_features.columns.tolist()

    nodes_by_type = node_features.groupby("object_type")["idx"].apply(list).to_dict()

    id_pattern = 0

    while len(new_edge_list) < num_max_anomalies:
        # sample one window id
        window_ids = anomaly_frames["window_id"].unique()
        sampled_window_id = np.random.choice(window_ids, size=1)[0]
        sampled_frames = anomaly_frames[anomaly_frames["window_id"] == sampled_window_id]
        id_pattern += 1

        # sample one edge from the train/validation edges to use as a base for the new edges
        base_edge = train_val_edges.sample(n=1).iloc[0]

        # substract min timestamp from all frames to keep the relative temporal structure
        sampled_frames["ts"] = sampled_frames["ts"] - sampled_frames["ts"].min() + base_edge["ts"]

        node_attribution = {}

        for frame_idx, frame in sampled_frames.iterrows():
            if "u" not in frame.index or "i" not in frame.index:
                continue

            frame_start_node = int(frame["u"])
            frame_end_node = int(frame["i"])
            frame_ts = float(frame.get("ts", base_edge["ts"]))

            start_nodes_candidates = nodes_by_type.get(node_features.loc[frame_start_node, "object_type"], [])
            end_nodes_candidates = nodes_by_type.get(node_features.loc[frame_end_node, "object_type"], [])

            if frame_start_node in node_attribution:
                start_node = node_attribution[frame_start_node]
            else:
                start_node = np.random.choice(start_nodes_candidates)
                node_attribution[frame_start_node] = start_node
            
            if frame_end_node in node_attribution:
                end_node = node_attribution[frame_end_node]
            else:
                end_node = np.random.choice(end_nodes_candidates)
                node_attribution[frame_end_node] = end_node

            new_edge = {col: getattr(base_edge, col) if hasattr(base_edge, col) else np.nan for col in edge_list_columns}
            new_edge.update(
                {
                    "u": start_node,
                    "i": end_node,
                    "ts": frame_ts,
                    "label": 1,
                    "pattern_id": id_pattern,
                }
            )
            new_edge_list.append(new_edge)

            feature_row = {col: np.nan for col in feature_columns}
            feature_row.update(
                {
                    "action_type": frame.get("action_type", np.nan),
                    "command_line": frame.get("command_line", np.nan),
                }
            )
            new_edge_features.append(feature_row)

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
    edge_features = pd.concat(
        [pd.DataFrame([{"action_type": np.nan, "command_line": np.nan}]), edge_features],
        ignore_index=True,
        sort=False,
    )

    return edge_list, edge_features

class DSU:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank.get(ra, 0) < self.rank.get(rb, 0): ra, rb = rb, ra
        self.parent[rb] = ra
        self.rank[ra] = self.rank.get(ra, 0) + 1



def get_anomaly_connected_components(client):
    edge_list_file_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_file_path, header=0)
    anomaly_edges = edge_list[edge_list["label"] == 1]

    dsu = DSU()
    for id , edge  in anomaly_edges.iterrows():
        src = edge["u"]
        dst = edge["i"]
        dsu.union(src, dst)
    components = defaultdict(list)
    
    for node in dsu.parent:
        components[dsu.find(node)].append(node)
    for comp in components.values():
        print(len(comp))


def extract_anomalies_by_sliding_windows(edge_list, window_minutes=15, step_minutes=15, time_col="ts", label_col="label"):
    """Return anomalous edges annotated with their overlapping time windows.

    The input dataframe is sorted by time and split into overlapping windows of
    ``window_minutes`` duration, sliding forward by ``step_minutes``.
    Only rows where ``label_col`` equals ``1`` are returned, each with the window
    boundaries they belong to.
    """
    if edge_list is None:
        raise ValueError("edge_list cannot be None")

    if time_col not in edge_list.columns:
        raise KeyError(f"Missing time column: {time_col}")
    if label_col not in edge_list.columns:
        raise KeyError(f"Missing label column: {label_col}")

    edge_list = edge_list[[time_col, label_col] + [col for col in edge_list.columns if col not in {time_col, label_col}]].copy()
    edge_list = edge_list.sort_values(by=time_col, kind="mergesort").reset_index(drop=True)

    edge_list[time_col] = pd.to_numeric(edge_list[time_col], errors="coerce")
    valid_edges = edge_list.dropna(subset=[time_col]).copy()

    if valid_edges.empty:
        return pd.DataFrame(columns=[*edge_list.columns, "window_id", "window_start", "window_end"])

    window_seconds = int(window_minutes) * 60
    step_seconds = int(step_minutes) * 60
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_minutes and step_minutes must be positive")

    start_ts = int(np.floor(valid_edges[time_col].min() / step_seconds) * step_seconds)
    end_ts = int(np.ceil(valid_edges[time_col].max() / step_seconds) * step_seconds)

    anomaly_frames = []
    current_ts = start_ts
    window_id = 0
    while current_ts < end_ts:
        window_end = current_ts + window_seconds
        window_edges = valid_edges[(valid_edges[time_col] >= current_ts) & (valid_edges[time_col] < window_end)]
        if not window_edges.empty:
            anomalies = window_edges[window_edges[label_col] == 1]
            if not anomalies.empty:
                anomalies = anomalies.copy()
                anomalies["window_id"] = window_id
                anomalies["window_start"] = current_ts
                anomalies["window_end"] = window_end
                anomaly_frames.append(anomalies)

        current_ts += step_seconds
        window_id += 1

    if not anomaly_frames:
        return pd.DataFrame(columns=[*edge_list.columns, "window_id", "window_start", "window_end"])

    return pd.concat(anomaly_frames, ignore_index=True)


def extract_anomalies_from_test_data(client, start_test, window_minutes=15, step_minutes=15):
    """Load the processed test edges and extract anomalous rows by sliding windows."""
    edge_list_file_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_file_path, header=0)

    test_time = dt.datetime.strptime(start_test, "%Y-%m-%dT%H:%M").timestamp()
    test_edges = edge_list[edge_list["ts"] >= test_time].copy()

    return extract_anomalies_by_sliding_windows(
        test_edges,
        window_minutes=window_minutes,
        step_minutes=step_minutes,
    )


def temporal_metapath_mining(client, start_val, start_test):
    val_time = dt.datetime.strptime(start_val, "%Y-%m-%dT%H:%M").timestamp()
    test_time = dt.datetime.strptime(start_test, "%Y-%m-%dT%H:%M").timestamp()

    node_feature_file_path = f"{BASE}/processed_data/optc_{client}/node_features.csv"
    node_features = pd.read_csv(node_feature_file_path, header=0)
    edge_feature_file_path = f"{BASE}/processed_data/optc_{client}/edge_features.csv"
    edge_features = pd.read_csv(edge_feature_file_path, header=0)
    edge_list_file_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_file_path, header=0)

    if edge_list.empty:
        return pd.DataFrame(columns=["path_length", "start_node", "end_node", "start_ts", "end_ts", "type_seq"])

    anomaly_edges = edge_list[ (edge_list["ts"] > test_time) & (edge_list["label"] == 1)]

    anomaly_edges = anomaly_edges.copy()
    edge_features = edge_features.copy()
    node_features = node_features.copy()

    if "idx" not in anomaly_edges.columns:
        anomaly_edges["idx"] = np.arange(len(anomaly_edges))
    if "idx" not in edge_features.columns:
        edge_features["idx"] = np.arange(len(edge_features))
    if "idx" not in node_features.columns:
        node_features["idx"] = np.arange(len(node_features))

    source_nodes = node_features[["idx", "object_type"]].rename(columns={"idx": "src", "object_type": "src_type"})
    target_nodes = node_features[["idx", "object_type"]].rename(columns={"idx": "dst", "object_type": "dst_type"})

    edges = (
        anomaly_edges.rename(columns={"u": "src", "i": "dst", "idx": "edge_idx"})
        .merge(source_nodes, on="src", how="left")
        .merge(target_nodes, on="dst", how="left")
        .merge(
            edge_features[["idx", "action_type", "command_line"]].rename(columns={"idx": "edge_idx", "action_type": "edge_type", "command_line": "cmd_line"}),
            on="edge_idx",
            how="left",
        )[["src", "dst", "ts", "src_type", "dst_type", "edge_type", "cmd_line"]]
        .copy()
    )

    frontier = pd.DataFrame(
        {
            "start_node": edges["src"],
            "end_node": edges["dst"],
            "start_ts": edges["ts"],
            "end_ts": edges["ts"],
            "type_seq": [
                ((src_type, edge_type, cmd_line, dst_type),)
                for src_type, edge_type, cmd_line, dst_type in zip(edges["src_type"], edges["edge_type"], edges["cmd_line"], edges["dst_type"])
            ],
        }
    )

    min_support = 20
    max_hops = 5
    max_span = 5 * 3600

    results = []
    current_frontier = frontier
    for hop in range(1, max_hops + 1):
        if current_frontier.empty:
            break

        current_frontier = current_frontier.copy()
        current_frontier["path_length"] = hop
        results.append(
            current_frontier[["path_length", "start_node", "end_node", "start_ts", "end_ts", "type_seq"]].copy()
        )

        if hop == max_hops:
            break
        
        # prune low-support prefixes BEFORE extending (anti-monotonic pruning)
        support_counts = (
            current_frontier.groupby("type_seq", dropna=False)
            .agg(support=("start_node", lambda s: s.nunique()))
            .reset_index()
        )
        valid_prefixes = support_counts.loc[support_counts["support"] >= min_support, "type_seq"]
        if valid_prefixes.empty:
            break

        current_frontier = current_frontier.loc[current_frontier["type_seq"].isin(valid_prefixes)].copy()
        if current_frontier.empty:
            break

        extended_rows = []
        for _, path in tqdm(current_frontier.iterrows(), total=current_frontier.shape[0]):
            candidates = edges.loc[
                (edges["src"] == path["end_node"])
                & (edges["ts"] >= path["end_ts"])
                & ((edges["ts"] - path["start_ts"]) <= max_span)
            ]
            for _, candidate in candidates.iterrows():
                extended_rows.append(
                    {
                        "start_node": path["start_node"],
                        "end_node": candidate["dst"],
                        "start_ts": path["start_ts"],
                        "end_ts": candidate["ts"],
                        "type_seq": path["type_seq"] + ((candidate["src_type"], candidate["edge_type"], candidate["cmd_line"], candidate["dst_type"]),),
                    }
                )

        current_frontier = pd.DataFrame(
            extended_rows,
            columns=["start_node", "end_node", "start_ts", "end_ts", "type_seq"],
        )

    if not results:
        return pd.DataFrame(columns=["path_length", "start_node", "end_node", "start_ts", "end_ts", "type_seq"])

    return pd.concat(results, ignore_index=True)

def main(clients, start_val, start_test):
    for client in clients:
        # unique_anomaly_paths = get_unique_stats_features(client, start_val, start_test)
        # unique_anomaly_paths = pd.read_csv(f"{BASE}/processed_data/optc_{client}/unique_anomaly_paths.csv")
        # edge_list, edge_features = add_anomaly_paths_in_train_val(unique_anomaly_paths, client, start_val, start_test)
        # edge_list.to_csv(f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv", index=False)
        # edge_features.to_csv(f"{BASE}/processed_data/optc_{client}/edge_features.csv", index=False)
        # get_anomaly_connected_components(client)
        # results = temporal_metapath_mining(client, start_val, start_test)
        # results.to_csv(f"{BASE}/processed_data/optc_{client}/metapath.csv", index=False)
        anomalies = extract_whole_anomaly(client, start_test)
        anomalies.to_csv(f"{BASE}/processed_data/optc_{client}/anomalies.csv", index=False)
        edge_list, edge_features = add_anomaly_nodes_in_train_val(anomalies, client, start_val, start_test)
        edge_list.to_csv(f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv", index=False)
        edge_features.to_csv(f"{BASE}/processed_data/optc_{client}/edge_features.csv", index=False)
