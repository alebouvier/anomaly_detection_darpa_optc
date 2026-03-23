import argparse
import os
import time
import random
import torch
import numpy as np
import psutil

from utils.utils import analyze_graphs, BASE
from utils.load_configs import get_link_prediction_args

from preprocessing.graphs import main as building_graphs
from features.w2v import main as training_w2v
from features.features import main as extract_features
from preprocessing.optc_data_preprocessor import main as graph_to_csv_preprocessor
from preprocessing.preprocess_data import main as ml_data_preprocessor
from training.train_link_prediction import main as training_link_prediction
from evaluation.evaluate_link_prediction import main as validate_link_prediction
from evaluation.evaluate_anomaly_ts import main as testing_anomaly_detection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build graphs from json files.")

    parser.add_argument(
        "-c",
        "--clients",
        dest="clients",
        action="store",
        nargs="+",
        default=["051"],
        help="List of clients, e.g., -c 051 201 501",
    )
    parser.add_argument(
        "-d",
        "--duration-graph",
        dest="duration",
        action="store",
        default=15,
        type=int,
        help="Graph duration in minutes",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        action="store",
        default="complete",
        help="complete, graph, feature, training, prediction",
    )
    parser.add_argument(
        "--start_val",
        dest="start_val",
        action="store",
        default="2019-09-22T12:00",
        type=str,
        help="start of the validation",
    )
    parser.add_argument(
        "--start_test",
        dest="start_test",
        action="store",
        default="2019-09-23T00:00",
        type=str,
        help="start of the test",
    )

    parser.add_argument(
        '--with_features',
        action='store_true',
        help='whether or not include features '
    )
    
    args = parser.parse_args()

    try:
        buser = BASE
        print(buser)
        dataset = "optc"
        print("Dataset: ", dataset)
        logs = buser + "log_data"
        duration = args.duration
        print("Selected duration: ", duration)
        clients = args.clients
        nb = len(clients)
        base = buser
        print("Selected clients: ", clients)
        start_val = args.start_val
        start_test = args.start_test
        
        graphs = base + "graph_data/graphs.pkl"
        cmds = base + "feature_data/cmds.pkl"
        paths = base + "feature_data/paths.pkl"
        model_w2v_path = base + "feature_data/w2v_model_path.pt"
        features_e = base + "feature_data/features_e.pkl"
        features_g = base + "feature_data/features_g.pkl"
        label_path = base + "label_data/malicious.json"



        task = args.task
        if task == "parse":
            parse_wget()
        if task == "graph":  # construction of all graphs
            building_graphs(base, clients, duration, logs, dataset)

        elif task == "w2v":  # extraction of features
            training_w2v(base, clients, cmds, data=dataset, is_path=False)
            training_w2v(base, clients, paths, data=dataset)

        elif task == "feature":  # extraction of features
            extract_features(base, clients, graphs, model_w2v_path, dataset)

        elif task == "preprocessing":
            for client in clients:
                dataset_name = f"optc_{client}"
                input_path = base + f"graph_data/{dataset_name}/"
                output_path = base + f"DG_data/{dataset_name}/{dataset_name}.csv"
                graph_to_csv_preprocessor(client, input_path, output_path, label_path, start_val, start_test, args.with_features)
                ml_data_preprocessor(dataset_name, bipartite=False, node_feat_dim=25)

        elif task == "train_link_prediction":
            train_link_prediction_args = get_link_prediction_args(is_evaluation=False)
            for client in clients:
                train_link_prediction_args.dataset_name = f"optc_{client}"
                training_link_prediction(train_link_prediction_args)
            
        elif task == "validate_link_prediction":
            validation_link_prediction_args = get_link_prediction_args(is_evaluation=True)
            for client in clients:
                validation_link_prediction_args.dataset_name = f"optc_{client}"
                validate_link_prediction(validation_link_prediction_args)

        elif task == "test_anomaly_detection":
            test_anomaly_detection_args = get_link_prediction_args(is_evaluation=True)
            for client in clients:
                test_anomaly_detection_args.dataset_name = f"optc_{client}"
                testing_anomaly_detection(test_anomaly_detection_args)

        elif task == "ML":
            ML_args = get_link_prediction_args(is_evaluation=False)
            for client in clients:
                ML_args.dataset_name = f"optc_{client}"
                training_link_prediction(ML_args)
                validate_link_prediction(ML_args)
                testing_anomaly_detection(ML_args)

        elif task == "all_preprocessing":
            building_graphs(base, clients, duration, logs, dataset)
            training_w2v(base, clients, cmds, data=dataset, is_path=False)
            training_w2v(base, clients, paths, data=dataset)
            extract_features(base, clients, graphs, model_w2v_path, dataset)
            for client in clients:
                dataset_name = f"optc_{client}"
                input_path = base + f"graph_data/{dataset_name}/"
                output_path = base + f"DG_data/{dataset_name}/{dataset_name}.csv"
                graph_to_csv_preprocessor(client, input_path, output_path, label_path, start_val, start_test, args.with_features)
                ml_data_preprocessor(dataset_name, bipartite=False, node_feat_dim=25)
        
        elif task == "complete":
            building_graphs(base, clients, duration, logs, dataset)
            training_w2v(base, clients, cmds, data=dataset, is_path=False)
            training_w2v(base, clients, paths, data=dataset)
            extract_features(base, clients, graphs, model_w2v_path, dataset)
            ML_args = get_link_prediction_args(is_evaluation=False)
            for client in clients:
                dataset_name = f"optc_{client}"
                input_path = base + f"graph_data/{dataset_name}/"
                output_path = base + f"DG_data/{dataset_name}/{dataset_name}.csv"
                graph_to_csv_preprocessor(client, input_path, output_path, label_path, start_val, start_test, args.with_features)
                ml_data_preprocessor(dataset_name, bipartite=False, node_feat_dim=25)
                ML_args.dataset_name = f"optc_{client}"
                training_link_prediction(ML_args)
                validate_link_prediction(ML_args)
                testing_anomaly_detection(ML_args)



    except Exception as e:
        print(f"Error {e}")
        exit(1)
