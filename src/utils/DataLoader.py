from torch.utils.data import Dataset, DataLoader
from utils.sanity_check import compute_sanity_check
import numpy as np
import random
import pandas as pd
import datetime as dt
import os

BASE = os.getenv("DATA_BASE", "./data")

class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )
    return data_loader


class Data:
    def __init__(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray,
        labels: np.ndarray,
        pattern_ids: np.ndarray,
    ):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.pattern_ids = pattern_ids
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)





def get_link_prediction_data(
            dataset_name: str,
            train_start: float, 
            val_start: float, 
            test_start: float, 
            test_end: float,
            inductive: bool = False, 
            calibration: bool = False, 
            anomaly_injection: bool = False,
            sanity_check: bool = False):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param train_start: float, start time of training set
    :param val_start: float, start time of validation set
    :param test_start: float, start time of test set
    :param test_end: float, end time of test set
    :param inductive: boolean, whether to prepare data for inductive setting 
    :param calibration: boolean, whether to prepare calibration data for conformal prediction
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv(
        f"{BASE}/processed_data/{dataset_name}/ml_{dataset_name}.csv"
    )
    edge_raw_features = np.load(
        f"{BASE}/processed_data/{dataset_name}/ml_{dataset_name}.npy"
    )
    node_raw_features = np.load(
        f"{BASE}/processed_data/{dataset_name}/ml_{dataset_name}_node.npy"
    )

    NODE_FEAT_DIM = EDGE_FEAT_DIM = edge_raw_features.shape[1]
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], (
        f"Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!"
    )
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], (
        f"Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!"
    )
    # padding the features of edges and nodes to the same dimension (790 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros(
            (node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1])
        )
        node_raw_features = np.concatenate(
            [node_raw_features, node_zero_padding], axis=1
        )
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros(
            (edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1])
        )
        edge_raw_features = np.concatenate(
            [edge_raw_features, edge_zero_padding], axis=1
        )

    assert (
        NODE_FEAT_DIM == node_raw_features.shape[1]
        and EDGE_FEAT_DIM == edge_raw_features.shape[1]
    ), "Unaligned feature dimensions after feature padding!"


    # get the timestamp of validate and test set
    train_time = dt.datetime.strptime(train_start, "%Y-%m-%dT%H:%M").timestamp()
    val_time = dt.datetime.strptime(val_start, "%Y-%m-%dT%H:%M").timestamp()
    test_time = dt.datetime.strptime(test_start, "%Y-%m-%dT%H:%M").timestamp()
    end_test_time = dt.datetime.strptime(test_end, "%Y-%m-%dT%H:%M").timestamp()

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values
    pattern_ids = graph_df.pattern_id.values

    full_data = Data(
        src_node_ids=src_node_ids,
        dst_node_ids=dst_node_ids,
        node_interact_times=node_interact_times,
        edge_ids=edge_ids,
        labels=labels,
        pattern_ids=pattern_ids,
    )

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time (> val_time, < end_test_time) 
    test_node_set = set(src_node_ids[(node_interact_times > val_time) & (node_interact_times < end_test_time)]).union(
        set(dst_node_ids[(node_interact_times > val_time) & (node_interact_times < end_test_time)])
    )


    if inductive:
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
        new_test_node_set = set(
            random.sample(sorted(test_node_set), int(0.1 * num_total_unique_node_ids))
        ) 

        # mask for each source and destination to denote whether they are new test nodes
        new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
        new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

        # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
        observed_edges_mask = np.logical_and(
            ~new_test_source_mask, ~new_test_destination_mask
        )
    # for train  and calibration data, we keep edges happening before the validation time and after train_time which do not involve any new node, used for inductiveness
        train_cal_mask = np.logical_and(np.logical_and(node_interact_times <= val_time, node_interact_times >= train_time), observed_edges_mask)
    else:
    # for train and calibration data, we keep edges happening before the validation time and after train_time, used for transductive setting
        train_cal_mask = np.logical_and(node_interact_times <= val_time, node_interact_times >= train_time)

    # randomly sample 10% of the train_cal_mask as calibration data, and the rest 90% as training data
    cal_ids = random.sample(
        list(np.where(train_cal_mask)[0]), int(0.1 * np.sum(train_cal_mask))
    )

    cal_mask = np.zeros_like(train_cal_mask, dtype=bool)
    cal_mask[cal_ids] = True

    if calibration:
        # if we want to prepare calibration data for conformal prediction, then we use the sampled 10% edges as calibration data and the rest 90% edges as training data        
        train_mask = np.logical_and(train_cal_mask, ~cal_mask)
    else:
        # if we do not want to prepare calibration data, then we use all the edges before validation time as training data
        train_mask = train_cal_mask

    if anomaly_injection:
        train_nodes = set(src_node_ids[train_mask]) | set(dst_node_ids[train_mask])

        anomaly_mask = labels == 1
        anomalous_src = src_node_ids[anomaly_mask]
        anomalous_dst = dst_node_ids[anomaly_mask]

        # Split by leakage level
        both_seen = []
        one_seen = []
        none_seen = []

        for s, d in zip(anomalous_src, anomalous_dst):
            s_in = s in train_nodes
            d_in = d in train_nodes
            if s_in and d_in:
                both_seen.append((s, d))
            elif s_in or d_in:
                one_seen.append((s, d))
            else:
                none_seen.append((s, d))

        print(f"Both nodes in train : {len(both_seen)}  → zero leakage")
        print(f"One node in train   : {len(one_seen)}   → small leakage")
        print(f"No node in train    : {len(none_seen)}  → full leakage")

        anomalous_nodes = set(anomalous_src) | set(anomalous_dst)



        for i, node in enumerate(anomalous_nodes):
            if node not in train_nodes:
                if node in src_node_ids:
                    edge_index = np.where(src_node_ids == node)[0][0]
                else:
                    edge_index = np.where(dst_node_ids == node)[0][0]
                train_mask[edge_index] = True
        

    train_data = Data(
        src_node_ids=src_node_ids[train_mask],
        dst_node_ids=dst_node_ids[train_mask],
        node_interact_times=node_interact_times[train_mask],
        edge_ids=edge_ids[train_mask],
        labels=labels[train_mask],
        pattern_ids=pattern_ids[train_mask],
    )
    train_data_normal = Data(
        src_node_ids=src_node_ids[np.logical_and(train_mask, labels == 0)],
        dst_node_ids=dst_node_ids[np.logical_and(train_mask, labels == 0)],
        node_interact_times=node_interact_times[np.logical_and(train_mask, labels == 0)],
        edge_ids=edge_ids[np.logical_and(train_mask, labels == 0)],
        labels=labels[np.logical_and(train_mask, labels == 0)],
        pattern_ids=pattern_ids[np.logical_and(train_mask, labels == 0)],
    )
    train_data_anomaly = Data(
        src_node_ids=src_node_ids[np.logical_and(train_mask, labels == 1)],
        dst_node_ids=dst_node_ids[np.logical_and(train_mask, labels == 1)],
        node_interact_times=node_interact_times[np.logical_and(train_mask, labels == 1)],
        edge_ids=edge_ids[np.logical_and(train_mask, labels == 1)],
        labels=labels[np.logical_and(train_mask, labels == 1)],
        pattern_ids=pattern_ids[np.logical_and(train_mask, labels == 1)],
    )

    if calibration:
        cal_data = Data(
            src_node_ids=src_node_ids[cal_mask],
            dst_node_ids=dst_node_ids[cal_mask],
            node_interact_times=node_interact_times[cal_mask],
            edge_ids=edge_ids[cal_mask],
            labels=labels[cal_mask],
            pattern_ids=pattern_ids[cal_mask],
        )
    else:
        cal_data = None

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    if inductive:
        assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(
        node_interact_times <= test_time, node_interact_times > val_time
    )
    test_mask = np.logical_and(
        node_interact_times <= end_test_time, node_interact_times > test_time
    )
    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array(
        [
            (src_node_id in new_node_set or dst_node_id in new_node_set)
            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)
        ]
    )
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(
        src_node_ids=src_node_ids[val_mask],
        dst_node_ids=dst_node_ids[val_mask],
        node_interact_times=node_interact_times[val_mask],
        edge_ids=edge_ids[val_mask],
        labels=labels[val_mask],
        pattern_ids=pattern_ids[val_mask],
    )
    val_data_normal = Data(
        src_node_ids=src_node_ids[np.logical_and(val_mask, labels == 0)],
        dst_node_ids=dst_node_ids[np.logical_and(val_mask, labels == 0)],
        node_interact_times=node_interact_times[np.logical_and(val_mask, labels == 0)],
        edge_ids=edge_ids[np.logical_and(val_mask, labels == 0)],
        labels=labels[np.logical_and(val_mask, labels == 0)],
        pattern_ids=pattern_ids[np.logical_and(val_mask, labels == 0)],
    )
    val_data_anomaly = Data(
        src_node_ids=src_node_ids[np.logical_and(val_mask, labels == 1)],
        dst_node_ids=dst_node_ids[np.logical_and(val_mask, labels == 1)],
        node_interact_times=node_interact_times[np.logical_and(val_mask, labels == 1)],
        edge_ids=edge_ids[np.logical_and(val_mask, labels == 1)],
        labels=labels[np.logical_and(val_mask, labels == 1)],
        pattern_ids=pattern_ids[np.logical_and(val_mask, labels == 1)],
    )

    test_data = Data(
        src_node_ids=src_node_ids[test_mask],
        dst_node_ids=dst_node_ids[test_mask],
        node_interact_times=node_interact_times[test_mask],
        edge_ids=edge_ids[test_mask],
        labels=labels[test_mask],
        pattern_ids=pattern_ids[test_mask],
    )

    if inductive:
        # validation and test with edges that at least has one new node (not in training set)
        new_node_val_data = Data(
            src_node_ids=src_node_ids[new_node_val_mask],
            dst_node_ids=dst_node_ids[new_node_val_mask],
            node_interact_times=node_interact_times[new_node_val_mask],
            edge_ids=edge_ids[new_node_val_mask],
            labels=labels[new_node_val_mask],
            pattern_ids=pattern_ids[new_node_val_mask],
        )

        new_node_test_data = Data(
            src_node_ids=src_node_ids[new_node_test_mask],
            dst_node_ids=dst_node_ids[new_node_test_mask],
            node_interact_times=node_interact_times[new_node_test_mask],
            edge_ids=edge_ids[new_node_test_mask],
            labels=labels[new_node_test_mask],
            pattern_ids=pattern_ids[new_node_test_mask],
        )
    else:
        new_node_val_data = None
        new_node_test_data = None

    print(
        "The dataset has {} interactions, involving {} different nodes".format(
            full_data.num_interactions, full_data.num_unique_nodes
        )
    )
    print(
        "The training dataset has {} interactions, involving {} different nodes".format(
            train_data.num_interactions, train_data.num_unique_nodes
        )
    )
    print(
        "The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.num_interactions, val_data.num_unique_nodes
        )
    )
    print(
        "The test dataset has {} interactions, involving {} different nodes".format(
            test_data.num_interactions, test_data.num_unique_nodes
        )
    )
    if inductive:
        print(
            "The new node validation dataset has {} interactions, involving {} different nodes".format(
                new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes
            )
        )
        print(
            "The new node test dataset has {} interactions, involving {} different nodes".format(
                new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes
            )
        )
        print(
            "{} nodes were used for the inductive testing, i.e. are never seen during training".format(
                len(new_test_node_set)
            )
        )
    
    if sanity_check:
        compute_sanity_check(train_data, val_data, test_data)

    return (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        train_data_normal,
        train_data_anomaly,
        val_data,
        val_data_normal,
        val_data_anomaly,
        test_data,
        new_node_val_data,
        new_node_test_data,
        cal_data,
    )


def get_node_classification_data(
    dataset_name: str, val_ratio: float, test_ratio: float
):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv(
        "data/processed_data/{}/ml_{}.csv".format(dataset_name, dataset_name)
    )
    edge_raw_features = np.load(
        "data/processed_data/{}/ml_{}.npy".format(dataset_name, dataset_name)
    )
    node_raw_features = np.load(
        "data/processed_data/{}/ml_{}_node.npy".format(dataset_name, dataset_name)
    )

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 2000
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], (
        f"Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!"
    )
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], (
        f"Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!"
    )
    # padding the features of edges and nodes to the same dimension (278 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros(
            (node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1])
        )
        node_raw_features = np.concatenate(
            [node_raw_features, node_zero_padding], axis=1
        )
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros(
            (edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1])
        )
        edge_raw_features = np.concatenate(
            [edge_raw_features, edge_zero_padding], axis=1
        )

    assert (
        NODE_FEAT_DIM == node_raw_features.shape[1]
        and EDGE_FEAT_DIM == edge_raw_features.shape[1]
    ), "Unaligned feature dimensions after feature padding!"

    # get the timestamp of validate and test set
    val_time, test_time = list(
        np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)])
    )

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(
        node_interact_times <= test_time, node_interact_times > val_time
    )
    test_mask = node_interact_times > test_time

    full_data = Data(
        src_node_ids=src_node_ids,
        dst_node_ids=dst_node_ids,
        node_interact_times=node_interact_times,
        edge_ids=edge_ids,
        labels=labels,
    )
    train_data = Data(
        src_node_ids=src_node_ids[train_mask],
        dst_node_ids=dst_node_ids[train_mask],
        node_interact_times=node_interact_times[train_mask],
        edge_ids=edge_ids[train_mask],
        labels=labels[train_mask],
    )
    val_data = Data(
        src_node_ids=src_node_ids[val_mask],
        dst_node_ids=dst_node_ids[val_mask],
        node_interact_times=node_interact_times[val_mask],
        edge_ids=edge_ids[val_mask],
        labels=labels[val_mask],
    )
    test_data = Data(
        src_node_ids=src_node_ids[test_mask],
        dst_node_ids=dst_node_ids[test_mask],
        node_interact_times=node_interact_times[test_mask],
        edge_ids=edge_ids[test_mask],
        labels=labels[test_mask],
    )

    return (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
    )
