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

try:
    from utils.utils import period, round_duration, save_pkl, open_config, BASE
except ModuleNotFoundError:  # pragma: no cover - fallback for repo-root execution
    from src.utils.utils import period, round_duration, save_pkl, open_config, BASE


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



def extract_whole_anomaly(client, start_test):
    """Load the processed test edges and extract anomalous rows without sliding windows."""
    edge_list_file_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_file_path, header=0)

    test_time = dt.datetime.strptime(start_test, "%Y-%m-%dT%H:%M").timestamp()
    test_edges = edge_list[edge_list["ts"] >= test_time].copy()

    anomalies = test_edges[test_edges["label"] == 1].copy()

    return anomalies


def add_anomaly_whole_in_train_val(anomalies, client, start_val, start_test):
    """Inject anomalous edges into the train/validation split without sliding windows.

    The function samples a portion of the historical edges, then inserts the
    anomalous edges using their original node and temporal structure. It also
    appends matching edge-feature rows so the resulting dataframes stay aligned.
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

    if anomalies is None or anomalies.empty or train_val_edges.empty:
        return edge_list, edge_features
    
    anomaly_ratio = 1
    num_max_anomalies = int(len(train_val_edges) * anomaly_ratio)
    if num_max_anomalies <= 0:
        return edge_list, edge_features

    new_edge_list = []
    new_edge_features = []
    edge_list_columns = edge_list.columns.tolist()
    feature_columns = edge_features.columns.tolist()

    # group nodes by their object type and by 15 minutes sliding windows
    window_minutes = 15
    nodes_by_type_and_ts = group_node_by_type_and_ts(node_features, edge_list, time_col="ts", window_minutes=window_minutes)



    id_pattern = 0

    anomalies["ts"] = anomalies["ts"] - anomalies["ts"].min()

    while len(new_edge_list) < num_max_anomalies:
        if id_pattern % 100 == 0:
            print(f"Injected {len(new_edge_list)} anomalies out of {num_max_anomalies} (pattern_id={id_pattern})")
        id_pattern += 1

        # sample one edge from the train/validation edges to use as a base for the new edges
        base_edge = train_val_edges.sample(n=1).iloc[0]

        # substract min timestamp from all frames to keep the relative temporal structure

        node_attribution = {}

        for idx, anomaly in anomalies.iterrows():
            if "u" not in anomaly.index or "i" not in anomaly.index:
                continue

            ano_start_node = int(anomaly["u"])
            ano_end_node = int(anomaly["i"])
            ano_ts = float(anomaly.get("ts", base_edge["ts"]))

            if ano_ts + base_edge["ts"] >= test_time:
                continue

            start_nodes_candidates = []
            end_nodes_candidates = []
            candidate_window_ts = int(np.floor(ano_ts + base_edge["ts"]) / (window_minutes * 60)) * (window_minutes * 60)
            while not start_nodes_candidates or not end_nodes_candidates:
                start_nodes_candidates = nodes_by_type_and_ts.get(
                    ("src", node_features.loc[ano_start_node, "object_type"], candidate_window_ts),
                    []
                )
                end_nodes_candidates = nodes_by_type_and_ts.get(
                    ("dst", node_features.loc[ano_end_node, "object_type"], candidate_window_ts),
                    []
                )
                candidate_window_ts += window_minutes * 60
                

  

            if ano_start_node in node_attribution:
                start_node = node_attribution[ano_start_node]
            else:
                start_node = np.random.choice(start_nodes_candidates)
                node_attribution[ano_start_node] = start_node

            if ano_end_node in node_attribution:
                end_node = node_attribution[ano_end_node]
            else:
                end_node = np.random.choice(end_nodes_candidates)
                node_attribution[ano_end_node] = end_node

            new_edge = {col: getattr(base_edge, col) if hasattr(base_edge, col) else np.nan for col in edge_list_columns}
            new_edge.update(
                {
                    "u": start_node,
                    "i": end_node,
                    "ts": ano_ts + base_edge["ts"],
                    "label": 1,
                    "pattern_id": id_pattern,
                }
            )
            new_edge_list.append(new_edge)

            feature_row = {col: np.nan for col in feature_columns}
            feature_row.update(
                {
                    "action_type": anomaly.get("action_type", np.nan),
                    "command_line": anomaly.get("command_line", np.nan),
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



def add_anomaly_nodes_in_train_val(anomalies, client, start_val, start_test):
    """Inject anomalous edges into the train/validation split without sliding windows.

    The function samples a portion of the historical edges, then inserts the
    anomalous edges using their original node and temporal structure. It also
    appends matching edge-feature rows so the resulting dataframes stay aligned.
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

    if anomalies is None or anomalies.empty or train_val_edges.empty:
        return edge_list, edge_features
    
    anomaly_ratio = 1
    num_max_anomalies = int(len(train_val_edges) * anomaly_ratio)
    if num_max_anomalies <= 0:
        return edge_list, edge_features

    new_edge_list = []
    new_edge_features = []
    edge_list_columns = edge_list.columns.tolist()
    feature_columns = edge_features.columns.tolist()


    id_pattern = 0

    # substract min timestamp from all frames to keep the relative temporal structure
    anomalies["ts"] = anomalies["ts"] - anomalies["ts"].min()

    while len(new_edge_list) < num_max_anomalies:
        if id_pattern % 100 == 0:
            print(f"Injected {len(new_edge_list)} anomalies out of {num_max_anomalies} (pattern_id={id_pattern})")
        id_pattern += 1

        # sample one edge from the train/validation edges to use as a base timestamp for the new edges
        base_ts = train_val_edges.sample(n=1).iloc[0]["ts"]


        for idx, anomaly in anomalies.iterrows():
            if "u" not in anomaly.index or "i" not in anomaly.index:
                continue

            ano_start_node = int(anomaly["u"])
            ano_end_node = int(anomaly["i"])
            ano_ts = float(anomaly.get("ts", base_ts))

            if ano_ts + base_ts >= test_time:
                continue
            

            new_edge = {}
            new_edge.update(
                {
                    "u": ano_start_node,
                    "i": ano_end_node,
                    "ts": ano_ts + base_ts,
                    "label": 1,
                    "pattern_id": id_pattern,
                }
            )
            new_edge_list.append(new_edge)

            feature_row = {col: np.nan for col in feature_columns}
            feature_row.update(
                {
                    "action_type": anomaly.get("action_type", np.nan),
                    "command_line": anomaly.get("command_line", np.nan),
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
    
def group_node_by_type_and_ts(node_features, edge_list, time_col="ts", window_minutes=15):
    """Group nodes by src or dst, their object type and by 15 minutes sliding windows."""
    if node_features is None or node_features.empty:
        return {}

    if edge_list is None or edge_list.empty:
        return {}

    if time_col not in edge_list.columns:
        raise KeyError(f"Missing time column: {time_col}")

    window_seconds = int(window_minutes) * 60
    if window_seconds <= 0:
        raise ValueError("window_minutes must be positive")

    start_ts = int(np.floor(edge_list[time_col].min() / window_seconds) * window_seconds)
    end_ts = int(np.ceil(edge_list[time_col].max() / window_seconds) * window_seconds)

    nodes_by_type_and_ts = defaultdict(list)

    for ts in range(start_ts, end_ts, window_seconds):
        window_edges = edge_list[(edge_list[time_col] >= ts) & (edge_list[time_col] < ts + window_seconds)]
        if not window_edges.empty:
            src_nodes = window_edges["u"].unique()
            dst_nodes = window_edges["i"].unique()

            for node in src_nodes:
                if node in node_features.index:
                    obj_type = node_features.loc[node, "object_type"]
                    nodes_by_type_and_ts[("src", obj_type, ts)].append(node)

            for node in dst_nodes:
                if node in node_features.index:
                    obj_type = node_features.loc[node, "object_type"]
                    nodes_by_type_and_ts[("dst", obj_type, ts)].append(node)

    return nodes_by_type_and_ts


def deconnect_test_anomaly(client, start_val, start_test):

    val_time = dt.datetime.strptime(start_val, "%Y-%m-%dT%H:%M").timestamp()
    test_time = dt.datetime.strptime(start_test, "%Y-%m-%dT%H:%M").timestamp()

    node_feature_file_path = f"{BASE}/processed_data/optc_{client}/node_features.csv"
    node_features = pd.read_csv(node_feature_file_path, header=0)
    edge_feature_file_path = f"{BASE}/processed_data/optc_{client}/edge_features.csv"
    edge_features = pd.read_csv(edge_feature_file_path, header=0)
    edge_list_file_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_file_path, header=0)

    anomaly_list = edge_list[(edge_list["ts"] >= test_time) & (edge_list["label"] == 1)]
    normal_train_list = edge_list[(edge_list["ts"] < val_time) & (edge_list["label"] == 0)]


    anomalous_nodes = np.unique(np.concat([anomaly_list["u"].to_numpy(), anomaly_list["i"].to_numpy()]))
    normal_train_nodes = np.unique(np.concat([normal_train_list["u"].to_numpy(), normal_train_list['i'].to_numpy()]))

    ano_nodes_not_in_train = np.setdiff1d(anomalous_nodes, normal_train_nodes)

    edges_to_remove = []
    for idx, edge in edge_list[(edge_list["ts"] >= test_time)].iterrows():
        src_node = edge["u"]
        dst_node = edge["i"]

        if (src_node in ano_nodes_not_in_train and dst_node not in ano_nodes_not_in_train) or (src_node not in ano_nodes_not_in_train and dst_node in ano_nodes_not_in_train):
            edges_to_remove.append(idx)
    
    print(f"{len(edges_to_remove)} edges removed")
    
    edge_list.drop(edges_to_remove, axis=0)

    return edge_list

def stats_test_anomaly(client, start_val, start_test):

    val_time = dt.datetime.strptime(start_val, "%Y-%m-%dT%H:%M").timestamp()
    test_time = dt.datetime.strptime(start_test, "%Y-%m-%dT%H:%M").timestamp()

    node_feature_file_path = f"{BASE}/processed_data/optc_{client}/node_features.csv"
    node_features = pd.read_csv(node_feature_file_path, header=0)
    edge_feature_file_path = f"{BASE}/processed_data/optc_{client}/edge_features.csv"
    edge_features = pd.read_csv(edge_feature_file_path, header=0)
    edge_list_file_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_file_path, header=0)

    anomaly_list = edge_list[(edge_list["ts"] >= test_time) & (edge_list["label"] == 1)]
    normal_test_list = edge_list[(edge_list["ts"] >= test_time) & (edge_list["label"] == 0)]
    normal_train_list = edge_list[(edge_list["ts"] < val_time) & (edge_list["label"] == 0)]


    anomalous_nodes = np.unique(np.concat([anomaly_list["u"].to_numpy(), anomaly_list["i"].to_numpy()]))
    normal_train_nodes = np.unique(np.concat([normal_train_list["u"].to_numpy(), normal_train_list['i'].to_numpy()]))
    normal_test_nodes = np.unique(np.concat([normal_test_list["u"].to_numpy(), normal_test_list['i'].to_numpy()]))

    ano_nodes_not_in_train = np.setdiff1d(anomalous_nodes, normal_train_nodes)
    ano_nodes_not_in_test =  np.setdiff1d(anomalous_nodes, normal_test_nodes)
    test_nodes_not_in_train = np.setdiff1d(normal_test_nodes, normal_train_nodes)
    
    ano_nodes_in_train_not_in_test = np.setdiff1d(ano_nodes_not_in_test, ano_nodes_not_in_train)



    print(f"n normal train edges: {len(normal_train_list)}")
    print(f"n normal test edges: {len(normal_test_list)}")
    print(f"n anomalous edges: {len(anomaly_list)}")
    print(f"n normal train nodes: {len(normal_train_nodes)}")
    print(f"n normal test nodes: {len(normal_test_nodes)}")
    print(f"n anomalous nodes: {len(anomalous_nodes)}")
    print(f"n anomalous nodes not in normal train nodes: {len(ano_nodes_not_in_train)}")
    print(f"n anomalous not in normal test nodes: {len(ano_nodes_not_in_test)}")
    print(f"n normal test nodes not in train nodes: {len(test_nodes_not_in_train)}")
    print(f"n anomalous nodes in train not in test: {len(ano_nodes_in_train_not_in_test)}")

    return edge_list