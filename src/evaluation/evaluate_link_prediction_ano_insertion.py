import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch.nn as nn
import pandas as pd

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import (
    set_random_seed,
    convert_to_gpu,
    get_parameter_sizes,
    create_folder,
    save_pkl,
)
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler, BASE
from evaluation.evaluate_models_utils import (
    evaluate_model_link_prediction,
    evaluate_model_link_prediction_ano_insertion,
    evaluate_edge_bank_link_prediction,
)
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping


def main(args):

    warnings.filterwarnings("ignore")

    # get data for training, validation and testing
    (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
        cal_data,
    ) = get_link_prediction_data(
        dataset_name=args.dataset_name,
        train_start=args.start_train,
        val_start=args.start_val,
        test_start=args.start_test,
        test_end=args.end_test,
        inductive=args.inductive,
        calibration=args.calibration,
    )


    if args.negative_sample_strategy == "2hop_neighbor":
        # read csv
        neighbors_2hop = pd.read_csv(f"{BASE}/processed_data/{args.dataset_name}/two_hop_non_neighbors.csv")
    else:
        neighbors_2hop = None

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
    )

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, interact_times=full_data.node_interact_times, negative_sample_strategy=args.negative_sample_strategy, neighbors_2hop=neighbors_2hop, seed=0
    )
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, interact_times=full_data.node_interact_times, negative_sample_strategy=args.negative_sample_strategy, neighbors_2hop=neighbors_2hop, seed=2
    )
    if args.inductive:
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=new_node_val_data.src_node_ids,
            dst_node_ids=new_node_val_data.dst_node_ids,
            negative_sample_strategy=args.negative_sample_strategy,
            neighbors_2hop=neighbors_2hop,
            seed=1,
        )
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=new_node_test_data.src_node_ids,
            dst_node_ids=new_node_test_data.dst_node_ids,
            negative_sample_strategy=args.negative_sample_strategy,
            neighbors_2hop=neighbors_2hop,
            seed=3,
        )
    if args.calibration:
        cal_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=cal_data.src_node_ids, dst_node_ids=cal_data.dst_node_ids, negative_sample_strategy=args.negative_sample_strategy, neighbors_2hop=neighbors_2hop, seed=1
        )

    # get data loaders
    val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.inductive:
        new_node_val_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(new_node_val_data.src_node_ids))),
            batch_size=args.batch_size,
            shuffle=False,
        )

        new_node_test_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(new_node_test_data.src_node_ids))),
            batch_size=args.batch_size,
            shuffle=False,
        )
    if args.calibration:
        cal_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(cal_data.src_node_ids))),
            batch_size=args.batch_size,
            shuffle=False,
        )

    # we separately evaluate EdgeBank, since EdgeBank does not contain any trainable parameters and has a different evaluation pipeline
    if args.model_name == "EdgeBank":
        evaluate_edge_bank_link_prediction(
            args=args,
            train_data=train_data,
            val_data=val_data,
            test_idx_data_loader=test_idx_data_loader,
            test_neg_edge_sampler=test_neg_edge_sampler,
            test_data=test_data,
        )

    else:
        (
            val_metric_all_runs,
            new_node_val_metric_all_runs,
            test_metric_all_runs,
            new_node_test_metric_all_runs,
        ) = [], [], [], []

        for run in range(args.num_runs):
            set_random_seed(seed=run)

            args.seed = run
            args.load_model_name = f"{args.model_name}_seed{args.seed}"
            args.save_result_name = f"{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}"
            args.experiment_folder = (
                f"experiments/{args.dataset_name}/{args.model_name.lower()}"
            )
            create_folder(args.experiment_folder)

            # set up logger
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            create_folder(f"{args.experiment_folder}/{args.save_result_name}/logs")
            # create file handler that logs debug and higher level messages
            fh = logging.FileHandler(
                f"{args.experiment_folder}/{args.save_result_name}/logs/{str(time.time())}.log"
            )
            fh.setLevel(logging.DEBUG)
            # create console handler with a higher log level
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            # create formatter and add it to the handlers
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            # add the handlers to logger
            logger.addHandler(fh)
            logger.addHandler(ch)

            run_start_time = time.time()
            logger.info(f"********** Run {run + 1} starts. **********")

            logger.info(f"configuration is {args}")

            # create model
            if args.model_name == "TGAT":
                dynamic_backbone = TGAT(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=full_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    device=args.device,
                )
            elif args.model_name == "GraphMixer":
                dynamic_backbone = GraphMixer(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=full_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    num_tokens=args.num_neighbors,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    device=args.device,
                )
            elif args.model_name == "DyGFormer":
                dynamic_backbone = DyGFormer(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=full_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    channel_embedding_dim=args.channel_embedding_dim,
                    patch_size=args.patch_size,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    max_input_sequence_length=args.max_input_sequence_length,
                    device=args.device,
                )
            elif args.model_name in ["JODIE", "DyRep", "TGN"]:
                # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
                (
                    src_node_mean_time_shift,
                    src_node_std_time_shift,
                    dst_node_mean_time_shift_dst,
                    dst_node_std_time_shift,
                ) = compute_src_dst_node_time_shifts(
                    train_data.src_node_ids,
                    train_data.dst_node_ids,
                    train_data.node_interact_times,
                )
                dynamic_backbone = MemoryModel(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=full_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    model_name=args.model_name,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    src_node_mean_time_shift=src_node_mean_time_shift,
                    src_node_std_time_shift=src_node_std_time_shift,
                    dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                    dst_node_std_time_shift=dst_node_std_time_shift,
                    device=args.device,
                )
            else:
                raise ValueError(f"Wrong value for model_name {args.model_name}!")
            link_predictor = MergeLayer(
                input_dim1=node_raw_features.shape[1],
                input_dim2=node_raw_features.shape[1],
                hidden_dim=node_raw_features.shape[1],
                output_dim=1,
            )
            model = nn.Sequential(dynamic_backbone, link_predictor)
            logger.info(f"model -> {model}")
            logger.info(
                f"model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, "
                f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB."
            )

            # load the saved model
            load_model_folder = (
                f"{args.experiment_folder}/{args.load_model_name}/saved_models/"
            )
            early_stopping = EarlyStopping(
                patience=0,
                save_model_folder=load_model_folder,
                save_model_name=args.load_model_name,
                logger=logger,
                model_name=args.model_name,
            )
            early_stopping.load_checkpoint(model, map_location="cpu")

            model = convert_to_gpu(model, device=args.device)

            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                for node_id, node_raw_messages in model[
                    0
                ].memory_bank.node_raw_messages.items():
                    new_node_raw_messages = []
                    for node_raw_message in node_raw_messages:
                        new_node_raw_messages.append(
                            (node_raw_message[0].to(args.device), node_raw_message[1])
                        )
                    model[0].memory_bank.node_raw_messages[node_id] = (
                        new_node_raw_messages
                    )

            loss_func = nn.BCELoss()

            # evaluate the best model
            logger.info(f"get final performance on dataset {args.dataset_name}...")

            # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
            if args.model_name not in ["JODIE", "DyRep", "TGN"]:
                (
                    val_losses,
                    val_metrics,
                    val_predicted_links,
                    val_actual_links,
                    non_exist_links,
                ) = evaluate_model_link_prediction_ano_insertion(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=val_idx_data_loader,
                    evaluate_neg_edge_sampler=val_neg_edge_sampler,
                    evaluate_data=val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    full_return=True,
                    temp=args.temperature,
                )
                val_score_folder = (
                f"{BASE}/val_result_data/{args.dataset_name}/{args.model_name}"
                )
                create_folder(val_score_folder)
                save_pkl(
                    val_predicted_links, f"{val_score_folder}/val_predicted_links.pkl"
                )
                save_pkl(val_actual_links, f"{val_score_folder}/val_actual_links.pkl")
                save_pkl(non_exist_links, f"{val_score_folder}/non_exist_links.pkl")

                if args.inductive:
                    new_node_val_losses, new_node_val_metrics = (
                        evaluate_model_link_prediction_ano_insertion(
                            model_name=args.model_name,
                            model=model,
                            neighbor_sampler=full_neighbor_sampler,
                            evaluate_idx_data_loader=new_node_val_idx_data_loader,
                            evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                            evaluate_data=new_node_val_data,
                            loss_func=loss_func,
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap,
                            temp=args.temperature,
                        )
                    )

            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            if args.calibration:
                (
                    cal_losses,
                    cal_metrics,
                    cal_predicted_links,
                    cal_actual_links,
                    non_exist_links,
                ) = evaluate_model_link_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=cal_idx_data_loader,
                    evaluate_neg_edge_sampler=cal_neg_edge_sampler,
                    evaluate_data=cal_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    full_return=True,
                    temp=args.temperature,
                )
                cal_score_folder = (
                    f"{BASE}/cal_result_data/{args.dataset_name}/{args.model_name}"
                )
                create_folder(cal_score_folder)
                save_pkl(
                    cal_predicted_links, f"{cal_score_folder}/cal_predicted_links.pkl"
                )
                save_pkl(cal_actual_links, f"{cal_score_folder}/cal_actual_links.pkl")
                save_pkl(non_exist_links, f"{cal_score_folder}/non_exist_links.pkl")

            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                # reload validation memory bank for new testing nodes
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            (
                test_losses,
                test_metrics,
                test_predicted_links,
                test_actual_links,
                non_exist_links,
            ) = evaluate_model_link_prediction(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=test_idx_data_loader,
                evaluate_neg_edge_sampler=test_neg_edge_sampler,
                evaluate_data=test_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
                full_return=True,
                temp=args.temperature,
            )
            test_score_folder = (
                f"{BASE}/test_result_data/{args.dataset_name}/{args.model_name}"
            )
            create_folder(test_score_folder)
            save_pkl(
                test_predicted_links, f"{test_score_folder}/test_predicted_links.pkl"
            )
            save_pkl(test_actual_links, f"{test_score_folder}/test_actual_links.pkl")
            save_pkl(non_exist_links, f"{test_score_folder}/non_exist_links.pkl")

            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                # reload validation memory bank for new testing nodes
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            if args.inductive:
                new_node_test_losses, new_node_test_metrics = (
                    evaluate_model_link_prediction(
                        model_name=args.model_name,
                        model=model,
                        neighbor_sampler=full_neighbor_sampler,
                        evaluate_idx_data_loader=new_node_test_idx_data_loader,
                        evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                        evaluate_data=new_node_test_data,
                        loss_func=loss_func,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                        temp=args.temperature,
                    )
                )
            # store the evaluation metrics at the current run
            (
                val_metric_dict,
                new_node_val_metric_dict,
                test_metric_dict,
                new_node_test_metric_dict,
            ) = {}, {}, {}, {}

            if args.model_name not in ["JODIE", "DyRep", "TGN"]:
                logger.info(f"val loss: {np.mean(val_losses):.4f}")
                for metric_name in val_metrics[0].keys():
                    average_val_metric = np.mean(
                        [val_metric[metric_name] for val_metric in val_metrics]
                    )
                    logger.info(f"val {metric_name}, {average_val_metric:.4f}")
                    val_metric_dict[metric_name] = average_val_metric

                if args.inductive:
                    logger.info(f"new node val loss: {np.mean(new_node_val_losses):.4f}")
                    for metric_name in new_node_val_metrics[0].keys():
                        average_new_node_val_metric = np.mean(
                            [
                                new_node_val_metric[metric_name]
                                for new_node_val_metric in new_node_val_metrics
                            ]
                        )
                        logger.info(
                            f"new node val {metric_name}, {average_new_node_val_metric:.4f}"
                        )
                        new_node_val_metric_dict[metric_name] = average_new_node_val_metric

            logger.info(f"test loss: {np.mean(test_losses):.4f}")
            for metric_name in test_metrics[0].keys():
                average_test_metric = np.mean(
                    [test_metric[metric_name] for test_metric in test_metrics]
                )
                logger.info(f"test {metric_name}, {average_test_metric:.4f}")
                test_metric_dict[metric_name] = average_test_metric

            if args.inductive:
                logger.info(f"new node test loss: {np.mean(new_node_test_losses):.4f}")
                for metric_name in new_node_test_metrics[0].keys():
                    average_new_node_test_metric = np.mean(
                        [
                            new_node_test_metric[metric_name]
                            for new_node_test_metric in new_node_test_metrics
                        ]
                    )
                    logger.info(
                        f"new node test {metric_name}, {average_new_node_test_metric:.4f}"
                    )
                    new_node_test_metric_dict[metric_name] = average_new_node_test_metric

            single_run_time = time.time() - run_start_time
            logger.info(f"Run {run + 1} cost {single_run_time:.2f} seconds.")

            if args.model_name not in ["JODIE", "DyRep", "TGN"]:
                val_metric_all_runs.append(val_metric_dict)
                new_node_val_metric_all_runs.append(new_node_val_metric_dict)

            test_metric_all_runs.append(test_metric_dict)
            new_node_test_metric_all_runs.append(new_node_test_metric_dict)

            # avoid the overlap of logs
            if run < args.num_runs - 1:
                logger.removeHandler(fh)
                logger.removeHandler(ch)

            # save model result
            if args.model_name not in ["JODIE", "DyRep", "TGN"]:
                result_json = {
                    "val metrics": {
                        metric_name: f"{val_metric_dict[metric_name]:.4f}"
                        for metric_name in val_metric_dict
                    },
                    "new node val metrics": {
                        metric_name: f"{new_node_val_metric_dict[metric_name]:.4f}"
                        for metric_name in new_node_val_metric_dict
                    },
                }
                result_json = json.dumps(result_json, indent=4)

            result_json = {
                "test metrics": {
                    metric_name: f"{test_metric_dict[metric_name]:.4f}"
                    for metric_name in test_metric_dict
                },
                "new node test metrics": {
                    metric_name: f"{new_node_test_metric_dict[metric_name]:.4f}"
                    for metric_name in new_node_test_metric_dict
                },
            }
            result_json = json.dumps(result_json, indent=4)

            save_result_folder = f"{args.experiment_folder}/saved_results"
            create_folder(save_result_folder)
            save_result_path = os.path.join(
                save_result_folder, f"{args.save_result_name}.json"
            )
            with open(save_result_path, "w") as file:
                file.write(result_json)
            logger.info(f"save negative sampling results at {save_result_path}")

        # store the average metrics at the log of the last run
        logger.info(f"metrics over {args.num_runs} runs:")

        if args.model_name not in ["JODIE", "DyRep", "TGN"]:
            for metric_name in val_metric_all_runs[0].keys():
                logger.info(
                    f"validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}"
                )
                logger.info(
                    f"average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} "
                    f"± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}"
                )

            if args.inductive:
                for metric_name in new_node_val_metric_all_runs[0].keys():
                    logger.info(
                        f"new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}"
                    )
                    logger.info(
                        f"average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} "
                        f"± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}"
                    )

        if args.evaluate_test:
            for metric_name in test_metric_all_runs[0].keys():
                logger.info(
                    f"test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}"
                )
                logger.info(
                    f"average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} "
                    f"± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}"
                )

            if args.inductive:
                for metric_name in new_node_test_metric_all_runs[0].keys():
                    logger.info(
                        f"new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}"
                    )
                    logger.info(
                        f"average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} "
                        f"± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}"
                    )

    sys.exit()
