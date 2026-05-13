import datetime as dt

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from utils.utils import create_folder, BASE, open_config


def preprocess(dataset_name: str):
    """
    read the original data file and return the DataFrame that has columns ['u', 'i', 'ts', 'label', 'idx']
    :param dataset_name: str, dataset name
    :return:
    """
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(dataset_name) as f:
        # skip the first line
        next(f)
        previous_time = -1
        for idx, line in enumerate(f):
            e = line.strip().split(",")
            # user_id
            u = int(e[0])
            # item_id
            i = int(e[1])

            # timestamp
            ts = float(e[2])
            # check whether time in ascending order
            assert ts >= previous_time
            previous_time = ts
            # state_label
            label = float(e[3])

            # edge features
            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            # edge index
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame(
        {"u": u_list, "i": i_list, "ts": ts_list, "label": label_list, "idx": idx_list}
    ), np.array(feat_l)


def reindex(df: pd.DataFrame, bipartite: bool = True):
    """
    reindex the ids of nodes and edges
    :param df: DataFrame
    :param bipartite: boolean, whether the graph is bipartite or not
    :return:
    """
    new_df = df.copy()
    if bipartite:
        # check the ids of users and items
        assert df.u.max() - df.u.min() + 1 == len(df.u.unique())
        assert df.i.max() - df.i.min() + 1 == len(df.i.unique())
        assert df.u.min() == df.i.min() == 0

        # if bipartite, discriminate the source and target node by unique ids (target node id is counted based on source node id)
        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i

    # make the id start from 1
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    return new_df


def sanity_check(
    dataset_name: str, val_start, test_start
):
    """
    preprocess the data
    :param dataset_name: str, dataset name
    :param bipartite: boolean, whether the graph is bipartite or not
    :param node_feat_dim: int, dimension of node features
    :return:
    """

    PATH = f"{BASE}/DG_data/{dataset_name}/{dataset_name}.csv"

    df, edge_feats = preprocess(PATH)

    val_time = dt.datetime.strptime(val_start, "%Y-%m-%dT%H:%M").timestamp()
    test_time = dt.datetime.strptime(test_start, "%Y-%m-%dT%H:%M").timestamp()

    train_data = df[df["ts"] < val_time]
    test_data = df[df["ts"] > test_time]

    anomaly_data = test_data[test_data["label"] == 1]


    # find unique nodes in test not present in train
    unique_node_test = pd.unique(test_data[['u', 'i']].values.ravel('K'))
    unique_node_train = pd.unique(train_data[['u', 'i']].values.ravel('K'))
    test_node_not_in_train = np.setdiff1d(unique_node_test, unique_node_train, assume_unique=True)

    unique_edge_anomaly = anomaly_data[["u", "i"]].drop_duplicates()
    unique_edge_train = train_data[["u", "i"]].drop_duplicates()
    anomaly_edge_not_in_train = (
        unique_edge_anomaly.merge(unique_edge_train, how="left", indicator=True)
            .query('_merge == "left_only"')
            .drop(columns="_merge")
    )   


    print("number of edges ", edge_feats.shape[0] - 1)
    print("number of edge features ", edge_feats.shape[1])
    print("nb unique node in train ", len(unique_node_train))
    print("nb unique node in test ", len(unique_node_test))
    print("nb unique node in test not present in train ", len(test_node_not_in_train))
    print("nb unique edge in train ", len(unique_edge_train))
    print("nb unique anomalous edge in test", len(unique_edge_anomaly))
    print("nb unique anomalous edge in test not present in train ", len(anomaly_edge_not_in_train))


def compute_split_stats(dataset_name, val_start, test_start):
    """
    df columns:
        src, dest, timestamp, label, idx

    start_val:
        timestamp where validation starts

    start_test:
        timestamp where test starts
    """

    PATH = f"{BASE}/DG_data/{dataset_name}/{dataset_name}.csv"

    df, edge_feats = preprocess(PATH)

    val_time = dt.datetime.strptime(val_start, "%Y-%m-%dT%H:%M").timestamp()
    test_time = dt.datetime.strptime(test_start, "%Y-%m-%dT%H:%M").timestamp()

    # -------------------------
    # Splits
    # -------------------------
    train_df = df[df["ts"] < val_time]

    val_df = df[
        (df["ts"] >= val_time)
        & (df["ts"] < test_time)
    ]

    test_df = df[df["ts"] >= test_time]

    anomaly_df = test_df[test_df["label"] == 1]

    # -------------------------
    # Helper functions
    # -------------------------
    def get_nodes(split_df):
        return set(split_df["u"]).union(set(split_df["i"]))

    def get_edges(split_df):
        return set(zip(split_df["u"], split_df["i"]))

    def stats(name, split_df):
        nodes = get_nodes(split_df)
        edges = get_edges(split_df)

        return {
            "split": name,
            "num_edges": len(split_df),
            "num_unique_nodes": len(nodes),
            "num_unique_edges": len(edges),
        }

    # -------------------------
    # Compute base stats
    # -------------------------
    train_stats = stats("train", train_df)
    val_stats = stats("val", val_df)
    test_stats = stats("test", test_df)
    anomaly_stats = stats("anomaly", anomaly_df)

    # -------------------------
    # Train reference sets
    # -------------------------
    train_nodes = get_nodes(train_df)
    train_edges = get_edges(train_df)

    # -------------------------
    # Val novelty
    # -------------------------
    val_nodes = get_nodes(val_df)
    val_edges = get_edges(val_df)

    new_val_nodes = val_nodes - train_nodes
    new_val_edges = val_edges - train_edges

    # -------------------------
    # Test novelty
    # -------------------------
    test_nodes = get_nodes(test_df)
    test_edges = get_edges(test_df)

    new_test_nodes = test_nodes - train_nodes
    new_test_edges = test_edges - train_edges

    # -------------------------
    # anomaly novelty
    # -------------------------
    anomaly_nodes = get_nodes(anomaly_df)
    anomaly_edges = get_edges(anomaly_df)

    new_anomaly_nodes = anomaly_nodes - train_nodes
    new_anomaly_edges = anomaly_edges - train_edges

    # -------------------------
    # Print summary
    # -------------------------
    print("\n=== SPLIT STATS ===")

    for s in [train_stats, val_stats, test_stats, anomaly_stats]:
        print(f"\n{s['split'].upper()}")
        print(f"Edges:          {s['num_edges']}")
        print(f"Unique nodes:   {s['num_unique_nodes']}")
        print(f"Unique edges:   {s['num_unique_edges']}")

    print("\n=== NOVELTY VS TRAIN ===")

    print("\nVAL")
    print(f"New nodes:      {len(new_val_nodes)}")
    print(f"New edges:      {len(new_val_edges)}")

    print("\nTEST")
    print(f"New nodes:      {len(new_test_nodes)}")
    print(f"New edges:      {len(new_test_edges)}")

    print("\nANOMALY")
    print(f"New nodes:      {len(new_anomaly_nodes)}")
    print(f"New edges:      {len(new_anomaly_edges)}")

    # Optional return
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "stats": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
            "new_val_nodes": len(new_val_nodes),
            "new_val_edges": len(new_val_edges),
            "new_test_nodes": len(new_test_nodes),
            "new_test_edges": len(new_test_edges),
        }
    }


def main(dataset_name, val_start, test_start):

    compute_split_stats(dataset_name, val_start, test_start)
