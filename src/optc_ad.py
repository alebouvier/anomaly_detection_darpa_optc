import argparse

from utils.utils import BASE, open_config
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
        "--with_features", action="store_true", help="whether or not include features "
    )

    args = parser.parse_known_args()[0]

    try:
        dataset = "optc"
        print("Dataset: ", dataset)
        logs = f"{BASE}/log_data"
        duration = args.duration
        print("Selected duration: ", duration)
        clients = args.clients
        nb = len(clients)
        print("Selected clients: ", clients)
        start_val = args.start_val
        start_test = args.start_test

        graphs = f"{BASE}/graph_data/graphs.pkl"
        cmds = f"{BASE}/feature_data/cmds.pkl"
        paths = f"{BASE}/feature_data/paths.pkl"
        model_w2v_path = f"{BASE}/feature_data/w2v_model_path.pt"
        features_e = f"{BASE}/feature_data/features_e.pkl"
        features_g = f"{BASE}/feature_data/features_g.pkl"
        label_path = f"{BASE}/label_data/malicious.json"

        cfg_dataset = open_config(dataset)

        task = args.task
        if task == "graph":  # construction of all graphs
            building_graphs(clients, duration, logs, dataset)

        elif task == "w2v":  # extraction of features
            training_w2v(clients, cmds, data=dataset, is_path=False)
            training_w2v(clients, paths, data=dataset)

        elif task == "feature":  # extraction of features
            extract_features(clients, graphs, model_w2v_path, dataset)

        elif task == "preprocessing":
            for client in clients:
                dataset_name = f"optc_{client}"
                input_path = f"{BASE}/graph_data/{dataset_name}/"
                output_path = f"{BASE}/DG_data/{dataset_name}/{dataset_name}.csv"
                graph_to_csv_preprocessor(
                    client,
                    input_path,
                    output_path,
                    label_path,
                    start_val,
                    start_test,
                    args.with_features,
                    cfg_dataset["MODEL"]["LEN_ENCODE_PATH"],
                )
                ml_data_preprocessor(dataset_name, bipartite=False, node_feat_dim=cfg_dataset["MODEL"]["LEN_ENCODE_PATH"])

        elif task == "preprocessing_2":
            for client in clients:
                dataset_name = f"optc_{client}"
                ml_data_preprocessor(dataset_name, bipartite=False, node_feat_dim=cfg_dataset["MODEL"]["LEN_ENCODE_PATH"])

        elif task == "train_link_prediction":
            train_link_prediction_args = get_link_prediction_args(is_evaluation=False)
            for client in clients:
                train_link_prediction_args.dataset_name = f"optc_{client}"
                training_link_prediction(train_link_prediction_args)

        elif task == "validate_link_prediction":
            validation_link_prediction_args = get_link_prediction_args(
                is_evaluation=True
            )
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
            building_graphs( clients, duration, logs, dataset)
            training_w2v(clients, cmds, data=dataset, is_path=False)
            training_w2v(clients, paths, data=dataset)
            extract_features(clients, graphs, model_w2v_path, dataset)
            for client in clients:
                dataset_name = f"optc_{client}"
                input_path = f"{BASE}/graph_data/{dataset_name}/"
                output_path = f"{BASE}/DG_data/{dataset_name}/{dataset_name}.csv"
                graph_to_csv_preprocessor(
                    client,
                    input_path,
                    output_path,
                    label_path,
                    start_val,
                    start_test,
                    args.with_features,
                    cfg_dataset["MODEL"]["LEN_ENCODE_PATH"],
                )
                ml_data_preprocessor(dataset_name, bipartite=False, node_feat_dim=cfg_dataset["MODEL"]["LEN_ENCODE_PATH"])

        elif task == "complete":
            building_graphs( clients, duration, logs, dataset)
            training_w2v(clients, cmds, data=dataset, is_path=False)
            training_w2v(clients, paths, data=dataset)
            extract_features(clients, graphs, model_w2v_path, dataset)
            ML_args = get_link_prediction_args(is_evaluation=False)
            for client in clients:
                dataset_name = f"optc_{client}"
                input_path = f"{BASE}/graph_data/{dataset_name}/"
                output_path = f"{BASE}/DG_data/{dataset_name}/{dataset_name}.csv"
                graph_to_csv_preprocessor(
                    client,
                    input_path,
                    output_path,
                    label_path,
                    start_val,
                    start_test,
                    args.with_features,
                    cfg_dataset["MODEL"]["LEN_ENCODE_PATH"],
                )
                ml_data_preprocessor(dataset_name, bipartite=False, node_feat_dim=cfg_dataset["MODEL"]["LEN_ENCODE_PATH"])
                ML_args.dataset_name = f"optc_{client}"
                training_link_prediction(ML_args)
                validate_link_prediction(ML_args)
                testing_anomaly_detection(ML_args)

    except Exception as e:
        print(f"Error {e}")
        exit(1)
