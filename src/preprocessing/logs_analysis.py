import gzip
import json
import importlib
import re
import traceback
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import pandasql as ps
from tqdm import tqdm
import datetime as dt


from utils.utils import period, round_duration, save_pkl, open_config, BASE


def get_log_stats(log_iter):
    # for each entity (type, id): count logs and verify consistency
    objects = {}
    objects_actions = defaultdict(Counter)
    logs_by_object_type = Counter()
    object_type_by_id = {}
    object_type_conflicts = {}

    actor_ids = set()
    process_object_ids = set()
    actor_in_diff_log = set()
    process_object_in_diff_log = set()
    process_adj = defaultdict(set)

    for log in log_iter:
        actor_id = log["actorID"]
        object_id = log["objectID"]
        object_type = log["object"]
        action_type = log["action"]

        # Count process logs for the actor
        actor_key = ("PROCESS", actor_id)
        if actor_key not in objects:
            objects[actor_key] = {"object_type": "PROCESS", "num_logs": 1}
        else:
            objects[actor_key]["num_logs"] += 1

        # Count object logs for the destination object
        object_key = (object_type, object_id)
        if object_key not in objects:
            objects[object_key] = {"object_type": object_type, "num_logs": 1}
        else:
            objects[object_key]["num_logs"] += 1

        # Track actor and PROCESS object IDs
        actor_ids.add(actor_id)
        if object_type == "PROCESS":
            process_object_ids.add(object_id)
            if actor_id != object_id:
                process_adj[actor_id].add(object_id)

        if actor_id != object_id:
            actor_in_diff_log.add(actor_id)
            if object_type == "PROCESS":
                process_object_in_diff_log.add(object_id)

        # Track object type consistency for each object ID
        if object_id in object_type_by_id:
            if object_type_by_id[object_id] != object_type:
                object_type_conflicts.setdefault(object_id, set()).update(
                    {object_type_by_id[object_id], object_type}
                )
        else:
            object_type_by_id[object_id] = object_type

        objects_actions[object_type][action_type] += 1
        logs_by_object_type[object_type] += 1

    process_ids_cross = {
        idx
        for idx in actor_ids & process_object_ids
        if idx in actor_in_diff_log or idx in process_object_in_diff_log
    }

    return (
        objects,
        objects_actions,
        logs_by_object_type,
        object_type_conflicts,
        process_ids_cross,
        process_adj,
    )


def extract_data(file):
    print("File: ", file)
    open_fn = gzip.open if file.endswith(".gz") else open
    with open_fn(file, "rt") as f:
        for line in tqdm(f):
            yield json.loads(line)


def iter_logs(files):
    for file in files:
        yield from extract_data(file)


def main(clients, logs, dataset, start_val, start_test):
    print("Start task graph")
    utils_dataset = importlib.import_module(f"data.{dataset}_utils")
    for c in clients:
        print("Client : ", c, logs)
        data = utils_dataset.logs_from_folder(logs, c)
        print(data)
        objects, objects_actions, logs_by_object_type, object_type_conflicts, process_ids_cross, process_adj = get_log_stats(
            iter_logs(data)
        )

        print_stats(
            objects,
            objects_actions,
            logs_by_object_type,
            object_type_conflicts,
            process_ids_cross,
            process_adj,
        )

    anomaly_data = [f"{BASE}/label_data/malicious.json"]

    objects, objects_actions, logs_by_object_type, object_type_conflicts, process_ids_cross, process_adj = get_log_stats(
        iter_logs(anomaly_data)
    )

    print("ANOMALY_RESULTS")
    print_stats(
        objects,
        objects_actions,
        logs_by_object_type,
        object_type_conflicts,
        process_ids_cross,
        process_adj,
    )

    client_anomalies = repartition_anomaly_by_client(anomaly_data)
    for client, anomalies in client_anomalies.items():
        print(f"\n=== Anomalies for client {client} ===")
        day_counts = Counter(day for day, _ in anomalies)
        for day, count in day_counts.items():
            print(f"{day}: {count} anomalies")


    return


def repartition_anomaly_by_client(anomaly_data):
    # find number of anomalies by client and by day
    client_anomalies = defaultdict(list)
    for log in iter_logs(anomaly_data):
        client = log["hostname"].lower()
        timestamp_str = log["timestamp"]
        day = timestamp_str[:10]
        client_anomalies[client].append((day, log))
        if log["object"] == "SHELL":
            print(f"Anomaly log with SHELL object: {log}")
    return client_anomalies


def count_process_paths(process_adj):
    process_in = defaultdict(set)
    for src, targets in process_adj.items():
        for dst in targets:
            process_in[dst].add(src)

    length_2 = 0
    for mid in set(process_adj) | set(process_in):
        length_2 += len(process_in[mid]) * len(process_adj.get(mid, []))

    length_3 = 0
    for mid1, mids in process_adj.items():
        for mid2 in mids:
            length_3 += len(process_in[mid1]) * len(process_adj.get(mid2, []))

    return length_2, length_3


def print_stats(
    objects,
    objects_actions,
    logs_by_object_type,
    object_type_conflicts,
    process_ids_cross,
    process_adj,
):
    totals_by_type = Counter()
    for (otype, _), stats in objects.items():
        totals_by_type[otype] += stats["num_logs"]

    cross_count = len(process_ids_cross)
    print(f"\nPROCESS object IDs present as actor_id in one log and object_id in another: {cross_count}")
    if cross_count > 0:
        sample_ids = sorted(process_ids_cross)[:20]
        print(f"Sample PROCESS ids: {sample_ids}")

    length_2, length_3 = count_process_paths(process_adj)
    print(f"PROCESS path count length 2: {length_2}")
    print(f"PROCESS path count length 3: {length_3}")

    for object_type in objects_actions:
        logs_count = np.array(
            [
                obj["num_logs"]
                for obj in objects.values()
                if obj["object_type"] == object_type
            ]
        )
        percentiles = np.percentile(logs_count, [0, 25, 50, 75, 100])
        total_logs = np.sum(logs_count)
        total_actions = sum(objects_actions[object_type].values())

        print(f"\n=== type: {object_type} ===\n")
        print(f"total logs: {total_logs}\n")
        print(f"total actions: {total_actions}\n")
        if object_type != "PROCESS" and total_logs != total_actions:
            print(
                f"WARNING: total logs ({total_logs}) != total actions ({total_actions}) for {object_type}"
            )
        if object_type == "PROCESS":
            print("Note: PROCESS counts are based on actors, not destination object logs.")
        print(f"number of objects: {len(logs_count)}")
        print(f"min num of log by object: {percentiles[0]}")
        print(f"25% num of log by object: {percentiles[1]}")
        print(f"median num of log by object: {percentiles[2]}")
        print(f"75% num of log by object: {percentiles[3]}")
        print(f"max num of log by object: {percentiles[4]}")
        print(f"mean num of log by object: {np.mean(logs_count)}\n")
        for action_type in objects_actions[object_type]:
            print(f"{action_type}: {objects_actions[object_type][action_type]}")

    if object_type_conflicts:
        print(f"\n {len(object_type_conflicts)} Object ID type conflicts detected:")




