import argparse

from utils.utils import BASE, open_config
from utils.load_configs import get_link_prediction_args

from preprocessing.logs_analysis import main as analyse_log
from preprocessing.preprocessing_data import main as test_preproc
from preprocessing.temporal_patterns import main as temporal_patterns
from features.w2v import main as training_w2v
from features.extract_features import main as extract_features
from features.train_bert128 import main as train_bert128
from training.train_link_prediction import main as training_link_prediction
from training.train_link_prediction_ano_insertion import main as training_link_prediction_check
from training.train_link_prediction_diffusion import main as training_link_prediction_diffusion
from evaluation.evaluate_link_prediction import main as validate_link_prediction
from evaluation.evaluate_link_prediction_ano_insertion import main as validate_link_prediction_check
from evaluation.evaluate_anomaly_ts import main as testing_anomaly_detection
from preprocessing.two_hop_non_neighbors import main as compute_2hop_non_neighbors


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

    parser.add_argument(
        "--aggregation", action="store_true", help="whether or not aggregate timestamps"
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
        if task == "log_analysis":
            analyse_log(clients, logs, dataset, start_val, start_test)

        elif task == "temporal_pattern":
            temporal_patterns(clients, start_val, start_test)

        elif task == "preprocessing":
            test_preproc(logs, dataset, clients)
        
        elif task == "two_hop_non_neighbors":
            compute_2hop_non_neighbors(dataset, clients)
        
        elif task == "train_w2v":
            training_w2v(clients, data=dataset)
        
        elif task == "train_bert128":
            train_bert128(clients, data=dataset)
            
        elif task == "feature_w2v":  
            extract_features(dataset, clients, model_type="w2v")

        elif task == "feature_bert":  
            extract_features(dataset, clients, model_type="bert")


        elif task == "train_link_prediction":
            train_link_prediction_args = get_link_prediction_args(is_evaluation=False)
            for client in clients:
                train_link_prediction_args.dataset_name = f"optc_{client}"
                training_link_prediction(train_link_prediction_args)
        
        elif task == "train_link_prediction_ano_insertion":
            train_link_prediction_args = get_link_prediction_args(is_evaluation=False)
            for client in clients:
                train_link_prediction_args.dataset_name = f"optc_{client}"
                training_link_prediction_check(train_link_prediction_args)
        
        elif task == "train_link_prediction_diffusion":
            train_link_prediction_args = get_link_prediction_args(is_evaluation=False)
            for client in clients:
                train_link_prediction_args.dataset_name = f"optc_{client}"
                training_link_prediction_diffusion(train_link_prediction_args)


        elif task == "validate_link_prediction":
            validation_link_prediction_args = get_link_prediction_args(
                is_evaluation=True
            )
            for client in clients:
                validation_link_prediction_args.dataset_name = f"optc_{client}"
                validate_link_prediction(validation_link_prediction_args)

        elif task == "validate_link_prediction_ano_insertion":
            validation_link_prediction_args = get_link_prediction_args(
                is_evaluation=True
            )
            for client in clients:
                validation_link_prediction_args.dataset_name = f"optc_{client}"
                validate_link_prediction_check(validation_link_prediction_args)

        elif task == "test_anomaly_detection":
            test_anomaly_detection_args = get_link_prediction_args(is_evaluation=True)
            for client in clients:
                test_anomaly_detection_args.dataset_name = f"optc_{client}"
                testing_anomaly_detection(test_anomaly_detection_args)




    except Exception as e:
        print(f"Error {e}")
        exit(1)
