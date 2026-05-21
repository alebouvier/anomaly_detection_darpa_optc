import numpy as np


def get_nodes_from_data(d):
    return set(d.src_node_ids) | set(d.dst_node_ids)

def get_edges_from_data(d):
    return set(zip(d.src_node_ids, d.dst_node_ids))

def get_stats_by_node(d, nodes):
    # for each node compute
    # - number of edges where it is source
    # - number of edges where it is destination
    # - number of edges total
    # - number of distinct timestamp
    # - timestamp delta between the first and last timestamp
    # return a dictionary
    src_node_ids = np.asarray(d.src_node_ids)
    dst_node_ids = np.asarray(d.dst_node_ids)
    timestamps = np.asarray(d.node_interact_times)

    stats_by_node = {
        node: {
            "num_src_edges": 0,
            "num_dst_edges": 0,
            "timestamps": [],
        }
        for node in nodes
    }

    for src, dst, ts in zip(src_node_ids, dst_node_ids, timestamps):
        if src in stats_by_node:
            stats_by_node[src]["num_src_edges"] += 1
            stats_by_node[src]["timestamps"].append(ts)
        if dst in stats_by_node:
            stats_by_node[dst]["num_dst_edges"] += 1
            stats_by_node[dst]["timestamps"].append(ts)

    for node, node_stats in stats_by_node.items():
        num_src = node_stats["num_src_edges"]
        num_dst = node_stats["num_dst_edges"]
        timestamp_list = np.asarray(node_stats["timestamps"])

        if timestamp_list.size > 0:
            num_distinct_timestamps = int(np.unique(timestamp_list).shape[0])
            timestamp_delta = float(timestamp_list.max() - timestamp_list.min())
        else:
            num_distinct_timestamps = 0
            timestamp_delta = 0.0

        stats_by_node[node] = {
            "num_src_edges": num_src,
            "num_dst_edges": num_dst,
            "num_edges": num_src + num_dst,
            "num_distinct_timestamps": num_distinct_timestamps,
            "timestamp_delta": timestamp_delta,
        }

    return stats_by_node
      

def stats(name, nodes, edges, num_edges):
    return {
        "split": name,
        "num_edges": num_edges,
        "num_unique_nodes": len(nodes),
        "num_unique_edges": len(edges),
    }


def _print_distribution(name, values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        print(f"    {name}: empty")
        return

    percentiles = np.percentile(values, [0, 25, 50, 75, 100])
    print(f"    {name}")
    print(f"      min:    {percentiles[0]:.4f}")
    print(f"      25%%:    {percentiles[1]:.4f}")
    print(f"      median: {percentiles[2]:.4f}")
    print(f"      75%%:    {percentiles[3]:.4f}")
    print(f"      max:    {percentiles[4]:.4f}")
    print(f"      mean:   {values.mean():.4f}")


def compute_sanity_check(train_data, val_data, test_data):

    train_nodes = get_nodes_from_data(train_data)
    val_nodes = get_nodes_from_data(val_data)
    test_nodes = get_nodes_from_data(test_data)

    train_edges = get_edges_from_data(train_data)
    val_edges = get_edges_from_data(val_data)
    test_edges = get_edges_from_data(test_data)

    train_stats_by_node = get_stats_by_node(train_data, train_nodes)
    val_stats_by_node = get_stats_by_node(val_data, val_nodes)
    test_stats_by_node = get_stats_by_node(test_data, test_nodes)


    # anomaly data from test_data
    anomaly_mask = test_data.labels == 1
    anomaly_src = test_data.src_node_ids[anomaly_mask]
    anomaly_dst = test_data.dst_node_ids[anomaly_mask]
    anomaly_nodes = set(anomaly_src) | set(anomaly_dst)
    anomaly_edges = set(zip(anomaly_src, anomaly_dst))


    train_stats = stats("train", train_nodes, train_edges, train_data.num_interactions)
    val_stats = stats("val", val_nodes, val_edges, val_data.num_interactions)
    test_stats = stats("test", test_nodes, test_edges, test_data.num_interactions)
    anomaly_stats = stats("anomaly", anomaly_nodes, anomaly_edges, int(np.sum(anomaly_mask)))

    new_val_nodes = val_nodes - train_nodes
    new_val_edges = val_edges - train_edges

    new_test_nodes = test_nodes - train_nodes
    new_test_edges = test_edges - train_edges

    new_anomaly_nodes = anomaly_nodes - train_nodes
    new_anomaly_edges = anomaly_edges - train_edges

    # print summary
    print("\n=== SPLIT STATS ===")
    for s in [train_stats, val_stats, test_stats, anomaly_stats]:
        print(f"\n{s['split'].upper()}")
        print(f"Edges:          {s['num_edges']}")
        print(f"Unique nodes:   {s['num_unique_nodes']}")
        print(f"Unique edges:   {s['num_unique_edges']}")

    for split_name, split_stats in [
        ("TRAIN", train_stats_by_node),
        ("VAL", val_stats_by_node),
        ("TEST", test_stats_by_node),
    ]:
        print(f"\n=== NODE METRIC DISTRIBUTIONS ({split_name}) ===")
        metric_values = {
            "num_src_edges": [m["num_src_edges"] for m in split_stats.values()],
            "num_dst_edges": [m["num_dst_edges"] for m in split_stats.values()],
            "num_edges": [m["num_edges"] for m in split_stats.values()],
            "num_distinct_timestamps": [m["num_distinct_timestamps"] for m in split_stats.values()],
            "timestamp_delta": [m["timestamp_delta"] for m in split_stats.values()],
        }
        for metric_name, values in metric_values.items():
            _print_distribution(metric_name, values)

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


