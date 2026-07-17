import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import umap
import umap.plot
import warnings
warnings.filterwarnings('ignore')

from utils.utils import BASE


def load_temporal_embeddings_dataframe(temporal_embeddings_path):
    temporal_embeddings_batch = torch.load(temporal_embeddings_path, map_location="cpu")

    if isinstance(temporal_embeddings_batch, dict):
        batch_records = [temporal_embeddings_batch]
    else:
        batch_records = temporal_embeddings_batch

    batch_dataframes = []
    for batch in batch_records:
        if "src_temporal_embeddings" in batch and "dst_temporal_embeddings" in batch:
            src_embeddings = np.asarray(batch["src_temporal_embeddings"])
            dst_embeddings = np.asarray(batch["dst_temporal_embeddings"])
            src_ids = np.asarray(batch["src_node_ids"])
            dst_ids = np.asarray(batch["dst_node_ids"])
            edge_ids = np.asarray(batch["edge_ids"])
        elif "neg_src_temporal_embeddings" in batch and "neg_dst_temporal_embeddings" in batch:
            src_embeddings = np.asarray(batch["neg_src_temporal_embeddings"])
            dst_embeddings = np.asarray(batch["neg_dst_temporal_embeddings"])
            src_ids = np.asarray(batch["neg_src_node_ids"])
            dst_ids = np.asarray(batch["neg_dst_node_ids"])
            edge_ids = np.asarray(batch["neg_edge_ids"])
        else:
            raise ValueError(         
                "Unsupported temporal embedding batch format. Expected either positive or negative embedding keys."
            )

        if src_embeddings.ndim == 1:
            src_embeddings = src_embeddings[np.newaxis, :]
        if dst_embeddings.ndim == 1:
            dst_embeddings = dst_embeddings[np.newaxis, :]

        batch_df = pd.DataFrame(
            {
                "edge_ids": edge_ids.reshape(-1),
                "src_node_ids": src_ids.reshape(-1),
                "dst_node_ids": dst_ids.reshape(-1),
                "node_interact_times": np.asarray(batch["node_interact_times"]).reshape(-1),
                "src_temporal_embeddings": [row for row in src_embeddings],
                "dst_temporal_embeddings": [row for row in dst_embeddings],
            }
        )
        batch_dataframes.append(batch_df)

    return pd.concat(batch_dataframes, ignore_index=True)


def test():
    # Load the Olivetti faces dataset
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data
    y = faces.target


    print(f"Dataset shape: {X.shape}")
    print(f"Number of individuals: {len(np.unique(y))}")
    print(f"Number of images per individual: {X.shape[0] // len(np.unique(y))}")


    # Display sample faces
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(64, 64), cmap='gray')
        ax.set_title(f'Person {y[i]}')
        ax.axis('off')
    plt.suptitle('Sample Faces from the Dataset')
    plt.tight_layout()
    plt.show()

    # Create UMAP instance with default parameters
    reducer = umap.UMAP(random_state=42)

    # Fit and transform the data
    embedding = reducer.fit_transform(X)

    # Create a custom colormap for 40 distinct classes
    colors = cm.get_cmap('hsv', 40)  

    # Create visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y,
                        cmap=colors, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Person ID', ticks=np.arange(0, 40, 5))
    plt.title('UMAP Projection of Olivetti Faces (Default Parameters)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Create subplots for different parameter settings
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    # Different parameter configurations
    param_configs = [
    {'n_neighbors': 5, 'min_dist': 0.1},
    {'n_neighbors': 15, 'min_dist': 0.1},
    {'n_neighbors': 50, 'min_dist': 0.1},
    {'n_neighbors': 15, 'min_dist': 0.0},
    {'n_neighbors': 15, 'min_dist': 0.5},
    {'n_neighbors': 15, 'min_dist': 0.99}
    ]

    # Apply UMAP with different parameters
    for idx, params in enumerate(param_configs):
        reducer = umap.UMAP(random_state=42, **params)
        embedding = reducer.fit_transform(X)
    
    ax = axes[idx]
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y,
                        cmap='tab20', s=30, alpha=0.8)
    ax.set_title(f"n_neighbors={params['n_neighbors']}, "
                    f"min_dist={params['min_dist']}")
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualisation_node_features(client):
    node_features_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}_node.npy"
    node_features = np.load(node_features_path)

    print(f"data shape: {node_features.shape}")

    nb_nodes = 10000

    chosen_ids = np.random.choice(list(range(node_features.shape[0])), size=nb_nodes)

    X = node_features[chosen_ids,:]

    types = X[:,0] + 2 * X[:,1]
    print(types[:10])

    # Create UMAP instance with default parameters
    reducer = umap.UMAP(random_state=42, n_neighbors=15)

    # Fit and transform the data
    embedding = reducer.fit_transform(X)
    process_embedding = embedding[X[:,0] == 1,:]
    file_embedding = embedding[X[:,1] == 1,:]


    fig, ax = plt.subplots(figsize=(12, 10))

    # 1. 
    ax.scatter(
        process_embedding[:, 0], process_embedding[:, 1],
        s=8, c="green", alpha=0.5,
        edgecolors="none", zorder=1, label="Process destination node"
    )

    # 2. 
    ax.scatter(
        file_embedding[:, 0], file_embedding[:, 1],
        s=8, c="orange",
        edgecolors="none",
        zorder=1, label="File destination node"
    )

    ax.legend()
    ax.set_title(f'UMAP node_features')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"experiments/optc_{client}/node_features_map.png")

    print("map saved")

def visualisation_edge_features(client):
    edge_features_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.npy"
    edge_features = np.load(edge_features_path)
    edge_list_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_path, header=0)
    node_features_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}_node.npy"
    node_features = np.load(node_features_path)

    print(f"data shape: {edge_features.shape}")

    window_size = 3600
    stride = 12 * 3600
    start = int(edge_list["ts"].min())
    end = int(edge_list["ts"].max())
    
    for window_start in range(start, end, stride):
        window_end = window_start + window_size

        id_start = np.searchsorted(edge_list["ts"], window_start)
        id_end = np.searchsorted(edge_list['ts'], window_end)

        if id_start == id_end:
            continue

        dst_node_id = edge_list["i"][id_start:id_end]
        edge_id = edge_list["idx"][id_start:id_end]

        X = edge_features[edge_id, :]

        types = node_features[dst_node_id, 0] + 2 * node_features[dst_node_id, 1]        

        # Create UMAP instance with default parameters
        reducer = umap.UMAP(random_state=42, n_neighbors=15)

        # Fit and transform the data
        embedding = reducer.fit_transform(X)

        process_embedding = embedding[node_features[dst_node_id, 0] == 1,:]
        file_embedding = embedding[node_features[dst_node_id, 1] == 1,:]


        fig, ax = plt.subplots(figsize=(12, 10))

        # 1. 
        ax.scatter(
            process_embedding[:, 0], process_embedding[:, 1],
            s=8, c="green", alpha=0.5,
            edgecolors="none", zorder=1, label="Process destination node"
        )

        # 2. 
        ax.scatter(
            file_embedding[:, 0], file_embedding[:, 1],
            s=8, c="orange",
            edgecolors="none",
            zorder=1, label="File destination node"
        )

        ax.legend()
        ax.set_title(f'UMAP edge_features (ts: {window_start}, nb_edge: {id_end - id_start})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"experiments/optc_{client}/edge_features_map_{int(window_start)}_{int(window_end)}.png")

        print("map saved")

def visualisation_edge_features_anomaly_compare(client):
    edge_features_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.npy"
    edge_features = np.load(edge_features_path)
    edge_list_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_path, header=0)
    node_features_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}_node.npy"
    node_features = np.load(node_features_path)

    anomaly_list = edge_list[edge_list["label"] == 1]

    print(f"data shape: {edge_features.shape}")

    window_size = 3600
    stride = 1 * 3600
    start = int(anomaly_list["ts"].min())
    end = int(anomaly_list["ts"].max())
    
    for window_start in range(start, end, stride):
        window_end = window_start + window_size

        id_start = np.searchsorted(edge_list["ts"], window_start)
        id_end = np.searchsorted(edge_list['ts'], window_end)

        if id_start == id_end:
            continue

        dst_node_id = edge_list["i"][id_start:id_end]
        edge_id = edge_list["idx"][id_start:id_end]

        X = edge_features[edge_id, :]

        anomaly_label = edge_list["label"][id_start:id_end]     

        # Create UMAP instance with default parameters
        reducer = umap.UMAP(random_state=42, n_neighbors=15)

        # Fit and transform the data
        embedding = reducer.fit_transform(X)

        normal_embedding = embedding[np.logical_not(anomaly_label),:]
        ano_embedding = embedding[anomaly_label,:]

        fig, ax = plt.subplots(figsize=(12, 10))

        # 1. Normal points: small, transparent, drawn first (low zorder)
        ax.scatter(
            normal_embedding[:, 0], normal_embedding[:, 1],
            s=8, c="steelblue", alpha=0.25,
            edgecolors="none", zorder=1, label="Normal"
        )

        # 2. Anomalies: larger, distinct marker, opaque, drawn last (high zorder)
        ax.scatter(
            ano_embedding[:, 0], ano_embedding[:, 1],
            s=120, c="crimson", marker="X",
            edgecolors="black", linewidths=1.2,
            zorder=3, label="Anomaly"
        )

        # optional: halo ring behind anomalies for extra pop in dense areas
        ax.scatter(
            ano_embedding[:, 0], ano_embedding[:, 1],
            s=300, facecolors="none", edgecolors="crimson",
            linewidths=1.5, alpha=0.6, zorder=2
        )

        ax.legend()
        ax.set_title(f'UMAP edge_features (ts: {window_start}, nb_edge: {id_end - id_start}, nb_anomalies: {anomaly_label.sum()})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"experiments/optc_{client}/edge_features_anomaly_compare_map_{int(window_start)}_{int(window_end)}.png")

        print("map saved")
    

def visualisation_temporal_embeddings(client, model_name):
    edge_features_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.npy"
    edge_features = np.load(edge_features_path)
    edge_list_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_path, header=0)
    node_features_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}_node.npy"
    node_features = np.load(node_features_path)
    temporal_embeddings_path = f"{BASE}/temporal_embeddings_data/optc_{client}/{model_name.lower()}/train/temporal_embeddings.pt"
    temporal_embeddings = load_temporal_embeddings_dataframe(temporal_embeddings_path)

    print(f"data shape: {temporal_embeddings.shape}")

    window_size = 3600
    stride = 12 * 3600
    start = int(temporal_embeddings["node_interact_times"].min())
    end = int(temporal_embeddings["node_interact_times"].max())
    
    for window_start in range(start, end, stride):
        window_end = window_start + window_size

        id_start = np.searchsorted(temporal_embeddings["node_interact_times"], window_start)
        id_end = np.searchsorted(temporal_embeddings['node_interact_times'], window_end)

        if id_start == id_end:
            continue

        dst_node_id = temporal_embeddings["dst_node_ids"][id_start:id_end]

        src_embeddings = np.vstack(temporal_embeddings["src_temporal_embeddings"].iloc[id_start:id_end].tolist())
        dst_embeddings = np.vstack(temporal_embeddings["dst_temporal_embeddings"].iloc[id_start:id_end].tolist())
        X = np.concatenate([src_embeddings, dst_embeddings], axis=1)

        # Create UMAP instance with default parameters
        reducer = umap.UMAP(random_state=42, n_neighbors=15)

        # Fit and transform the data
        embedding = reducer.fit_transform(X)

        process_embedding = embedding[node_features[dst_node_id, 0] == 1,:]
        file_embedding = embedding[node_features[dst_node_id, 1] == 1,:]


        fig, ax = plt.subplots(figsize=(12, 10))

        # 1. 
        ax.scatter(
            process_embedding[:, 0], process_embedding[:, 1],
            s=8, c="green", alpha=0.5,
            edgecolors="none", zorder=1, label="Process destination node"
        )

        # 2. 
        ax.scatter(
            file_embedding[:, 0], file_embedding[:, 1],
            s=8, c="orange",
            edgecolors="none",
            zorder=1, label="File destination node"
        )

        ax.legend()
        ax.set_title(f'UMAP temporal_embedding (ts: {window_start}, nb_edge: {id_end - id_start})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"experiments/optc_{client}/{model_name.lower()}/temporal_embedding_map_{int(window_start)}_{int(window_end)}.png")

        print("map saved")


def visualisation_temporal_embeddings_negative_compare(client, model_name, mode):
    edge_features_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.npy"
    edge_features = np.load(edge_features_path)
    edge_list_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}.csv"
    edge_list = pd.read_csv(edge_list_path, header=0)
    node_features_path = f"{BASE}/processed_data/optc_{client}/ml_optc_{client}_node.npy"
    node_features = np.load(node_features_path)
    temporal_positive_embeddings_path = f"{BASE}/temporal_embeddings_data/optc_{client}/{model_name.lower()}/{mode}/temporal_embeddings.pt"
    temporal_positive_embeddings = load_temporal_embeddings_dataframe(temporal_positive_embeddings_path)
    temporal_negative_embeddings_path = f"{BASE}/temporal_embeddings_data/optc_{client}/{model_name.lower()}/{mode}/negative_temporal_embeddings.pt"
    temporal_negative_embeddings = load_temporal_embeddings_dataframe(temporal_negative_embeddings_path)

    temporal_positive_embeddings["positive"] = np.ones((temporal_positive_embeddings.shape[0],), dtype=np.int32)
    temporal_negative_embeddings["positive"] = np.zeros((temporal_negative_embeddings.shape[0],), dtype=np.int32)

    temporal_positive_embeddings["anomaly"] = temporal_positive_embeddings["edge_ids"].isin(edge_list[edge_list["label"] == 1]["idx"])
    temporal_negative_embeddings["anomaly"] = np.zeros((temporal_negative_embeddings.shape[0],), dtype=np.int32)

    temporal_embeddings = pd.concat([temporal_positive_embeddings, temporal_negative_embeddings])
    temporal_embeddings.sort_values(by=["node_interact_times"], inplace=True, ignore_index=True)


    print(f"data shape: {temporal_embeddings.shape}")

    window_size = 3600

    if mode == "test":
        stride = 1 * 3600
        start = int(temporal_embeddings[temporal_embeddings["anomaly"] == 1]["node_interact_times"].min())
        end = int(temporal_embeddings[temporal_embeddings["anomaly"] == 1]["node_interact_times"].max())
    else:
        stride = 12 * 3600
        start = int(temporal_embeddings["node_interact_times"].min())
        end = int(temporal_embeddings["node_interact_times"].max())
    
    for window_start in range(start, end, stride):
        window_end = window_start + window_size

        id_start = np.searchsorted(temporal_embeddings["node_interact_times"], window_start)
        id_end = np.searchsorted(temporal_embeddings['node_interact_times'], window_end)

        if id_start == id_end:
            continue

        src_embeddings = np.vstack(temporal_embeddings["src_temporal_embeddings"].iloc[id_start:id_end].tolist())
        dst_embeddings = np.vstack(temporal_embeddings["dst_temporal_embeddings"].iloc[id_start:id_end].tolist())
        X = np.concatenate([src_embeddings, dst_embeddings], axis=1)

        # Create UMAP instance with default parameters
        reducer = umap.UMAP(random_state=42, n_neighbors=15)

        # Fit and transform the data
        embedding = reducer.fit_transform(X)

        positive_embedding = embedding[temporal_embeddings['positive'][id_start:id_end] == 1,:]
        negative_embedding = embedding[temporal_embeddings['positive'][id_start:id_end] == 0,:]
        anomaly_embedding = embedding[temporal_embeddings["anomaly"][id_start:id_end] == 1,:]


        fig, ax = plt.subplots(figsize=(12, 10))

        # 1. 
        ax.scatter(
            positive_embedding[:, 0], positive_embedding[:, 1],
            s=8, c="steelblue", alpha=0.5,
            edgecolors="none", zorder=1, label="positive edge"
        )

        # 2. 
        ax.scatter(
            negative_embedding[:, 0], negative_embedding[:, 1],
            s=8, c="pink",
            edgecolors="none",
            zorder=1, label="negative edge"
        )

        ax.scatter(
            anomaly_embedding[:, 0], anomaly_embedding[:, 1],
            s=64, c="crimson", marker="X",
            edgecolors="black", linewidths=1.2,
            zorder=3, label="anomalous edge"
        )

        ax.legend()
        ax.set_title(f'UMAP temporal_embedding (ts: {window_start}, nb_edge: {id_end - id_start})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"experiments/optc_{client}/{model_name.lower()}/temporal_embedding_negative_compare_{mode}_map_{int(window_start)}_{int(window_end)}.png")

        print("map saved")

def main(clients, model_name="GraphMixer"):
    for client in clients:
        visualisation_node_features(client)
        visualisation_edge_features(client)
        visualisation_edge_features_anomaly_compare(client)
        visualisation_temporal_embeddings(client, model_name=model_name)
        visualisation_temporal_embeddings_negative_compare(client, model_name=model_name, mode="train")
        visualisation_temporal_embeddings_negative_compare(client, model_name=model_name, mode="val")
        visualisation_temporal_embeddings_negative_compare(client, model_name=model_name, mode="test")