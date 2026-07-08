import logging
import time
import sys

# import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import median_filter

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import (
    set_random_seed,
    convert_to_gpu,
    get_parameter_sizes,
    create_optimizer,
    create_folder,
)
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler, BASE
from evaluation.evaluate_models_utils import evaluate_model_link_prediction, evaluate_model_link_prediction_ano_insertion
from utils.metrics import get_link_prediction_metrics
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
        anomaly_injection=args.anomaly_injection,
        sanity_check=True
    )
    

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        pattern_masking=args.pattern_masking,
        seed=0,
    )

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        pattern_masking=args.pattern_masking,
        seed=1,
    )

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need a seed for random sampling, but non-random training samplers should be seeded
    train_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids,
        dst_node_ids=train_data.dst_node_ids,
        interact_times=train_data.node_interact_times,
        negative_sample_strategy=args.negative_sample_strategy,
        seed=0 if args.negative_sample_strategy != "random" else None,
    )
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, interact_times=full_data.node_interact_times, negative_sample_strategy=args.negative_sample_strategy, seed=1
    )
    if args.inductive:
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=new_node_val_data.src_node_ids,
            dst_node_ids=new_node_val_data.dst_node_ids,
            negative_sample_strategy=args.negative_sample_strategy,
            seed=1,
        )
    # test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    # new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)


    # get data loaders
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.edge_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    if args.inductive:
        new_node_val_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(new_node_val_data.src_node_ids))),
            batch_size=args.batch_size,
            shuffle=False,
        )

    # test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)


    # val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):
        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f"{args.model_name}_seed{args.seed}"
        args.experiment_folder = (
            f"experiments/{args.dataset_name}/{args.model_name.lower()}/{args.save_model_name}"
        )
        create_folder(args.experiment_folder)

        batch_historical_ratios = []
        global_historical_edges = 0
        global_random_edges = 0

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        create_folder(f"{args.experiment_folder}/logs/")
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"{args.experiment_folder}/logs/{str(time.time())}.log"
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
        # Variables for tracking loss
        train_loss_history = []
        val_loss_history = []
        train_loss_per_batch = []
        val_loss_per_batch = []

        loss_save_folder = f"{args.experiment_folder}/loss"
        create_folder(loss_save_folder)

        # create model
        if args.model_name == "TGAT":
            dynamic_backbone = TGAT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
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
                neighbor_sampler=train_neighbor_sampler,
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
                neighbor_sampler=train_neighbor_sampler,
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
                neighbor_sampler=train_neighbor_sampler,
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

        optimizer = create_optimizer(
            model=model,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"{args.experiment_folder}/saved_models"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        create_folder(save_model_folder)

        early_stopping = EarlyStopping(
            patience=args.patience,
            save_model_folder=save_model_folder,
            save_model_name=args.save_model_name,
            logger=logger,
            model_name=args.model_name,
        )

        loss_func = nn.BCELoss()

        for epoch in range(args.num_epochs):
            model.train()
            if args.model_name in [
                "DyRep",
                "TGAT",
                "TGN",
                "CAWN",
                "TCL",
                "GraphMixer",
                "DyGFormer",
            ]:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(
                train_idx_data_loader, ncols=120, mininterval=2
            )
            nb_normal_edges_sampled = 0
            nb_anomalous_edges_sampled = 0
            nb_random_edges_sampled = 0
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                # keep train_data_indices for which the label is zero
                normal_train_data_indices = train_data_indices[train_data.labels[train_data_indices] == 0]
                ano_train_data_indices = train_data_indices[train_data.labels[train_data_indices] == 1]

                if len(normal_train_data_indices) == 0 or len(ano_train_data_indices) == 0:
                    continue

                (
                    batch_src_node_ids,
                    batch_dst_node_ids,
                    batch_node_interact_times,
                    batch_edge_ids,
                    batch_labels,
                    batch_pattern_ids,
                ) = (
                    train_data.src_node_ids[normal_train_data_indices],
                    train_data.dst_node_ids[normal_train_data_indices],
                    train_data.node_interact_times[normal_train_data_indices],
                    train_data.edge_ids[normal_train_data_indices],
                    train_data.labels[normal_train_data_indices],
                    train_data.pattern_ids[normal_train_data_indices],
                )

                (
                    batch_ano_src_node_ids,
                    batch_ano_dst_node_ids,
                    batch_ano_node_interact_times,
                    batch_ano_edge_ids,
                    batch_ano_labels,
                    batch_ano_pattern_ids,
                ) = (
                    train_data.src_node_ids[ano_train_data_indices],
                    train_data.dst_node_ids[ano_train_data_indices],
                    train_data.node_interact_times[ano_train_data_indices],
                    train_data.edge_ids[ano_train_data_indices],
                    train_data.labels[ano_train_data_indices],
                    train_data.pattern_ids[ano_train_data_indices],
                )

                if len(batch_src_node_ids) < len(batch_ano_src_node_ids):
                    # repeat the normal edges to match the number of anomalous edges
                    repeat_factor = len(batch_ano_src_node_ids) // len(batch_src_node_ids) + 1
                    batch_src_node_ids = np.tile(batch_src_node_ids, repeat_factor)
                    batch_dst_node_ids = np.tile(batch_dst_node_ids, repeat_factor)
                    batch_node_interact_times = np.tile(batch_node_interact_times, repeat_factor)
                    batch_edge_ids = np.tile(batch_edge_ids, repeat_factor)
                    batch_labels = np.tile(batch_labels, repeat_factor)
                    batch_pattern_ids = np.tile(batch_pattern_ids, repeat_factor)
                elif len(batch_src_node_ids) > len(batch_ano_src_node_ids):
                    # repeat the anomalous edges to match the number of normal edges
                    repeat_factor = len(batch_src_node_ids) // len(batch_ano_src_node_ids) + 1
                    batch_ano_src_node_ids = np.tile(batch_ano_src_node_ids, repeat_factor)
                    batch_ano_dst_node_ids = np.tile(batch_ano_dst_node_ids, repeat_factor)
                    batch_ano_node_interact_times = np.tile(batch_ano_node_interact_times, repeat_factor)
                    batch_ano_edge_ids = np.tile(batch_ano_edge_ids, repeat_factor)
                    batch_ano_labels = np.tile(batch_ano_labels, repeat_factor)
                    batch_ano_pattern_ids = np.tile(batch_ano_pattern_ids, repeat_factor)

                neg_edge_sample_size = max(0, len(batch_src_node_ids) - len(batch_ano_src_node_ids))

                # if neg_edge_sample_size > 0:
                #     _, batch_neg_dst_node_ids, num_preferred_sample_edges, num_random_sample_edges = (
                #         train_neg_edge_sampler.sample(
                #             size=neg_edge_sample_size,
                #             batch_src_node_ids=batch_src_node_ids,
                #             batch_dst_node_ids=batch_dst_node_ids,
                #             current_batch_start_time=float(np.min(batch_node_interact_times)),
                #             current_batch_end_time=float(np.max(batch_node_interact_times)),
                #             return_sampling_counts=True,
                #         )
                #     )
                #     batch_neg_src_node_ids = batch_src_node_ids[:neg_edge_sample_size]
                #     batch_neg_labels = np.zeros(neg_edge_sample_size)
                #     batch_neg_pattern_ids = np.zeros(neg_edge_sample_size)
                #     batch_neg_node_interact_times = batch_node_interact_times[:neg_edge_sample_size]
                # else:
                batch_neg_src_node_ids = np.array([], dtype=np.int64)
                batch_neg_dst_node_ids = np.array([], dtype=np.int64)
                batch_neg_labels = np.array([], dtype=np.int64)
                batch_neg_pattern_ids = np.array([], dtype=np.int64)
                batch_neg_node_interact_times = np.array([], dtype=np.float32)
                num_preferred_sample_edges = 0
                num_random_sample_edges = 0

                nb_normal_edges_sampled += len(batch_src_node_ids)
                nb_anomalous_edges_sampled += len(batch_ano_src_node_ids)
                nb_random_edges_sampled += num_random_sample_edges

                # extend the negative samples with anomalous edges (the node_ids list are ndarrays)
                if len(batch_ano_src_node_ids) > 0:
                    batch_neg_src_node_ids = np.concatenate([batch_neg_src_node_ids, batch_ano_src_node_ids])
                    batch_neg_dst_node_ids = np.concatenate([batch_neg_dst_node_ids, batch_ano_dst_node_ids])
                    batch_neg_node_interact_times = np.concatenate([batch_neg_node_interact_times, batch_ano_node_interact_times])
                    batch_neg_labels = np.concatenate([batch_neg_labels, batch_ano_labels])
                    batch_neg_pattern_ids = np.concatenate([batch_neg_pattern_ids, batch_ano_pattern_ids])



                if args.negative_sample_strategy == "historical":
                    global_historical_edges += num_preferred_sample_edges
                    batch_historical_ratios.append(
                        num_preferred_sample_edges / len(batch_src_node_ids)
                    )
                else:
                    batch_historical_ratios.append(0.0)
                global_random_edges += num_random_sample_edges

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ["TGAT", "CAWN", "TCL"]:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        node_pattern_ids=batch_pattern_ids,
                        num_neighbors=args.num_neighbors,
                    )

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = (
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_neg_node_interact_times,
                            node_pattern_ids=batch_neg_pattern_ids,
                            num_neighbors=args.num_neighbors,
                        )
                    )
                elif args.model_name in ["GraphMixer"]:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        node_pattern_ids=batch_pattern_ids,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                    )

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = (
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_neg_node_interact_times,
                            node_pattern_ids=batch_neg_pattern_ids,
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap,
                        )
                    )
                elif args.model_name in ["DyGFormer"]:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    if len(batch_src_node_ids) > 0:
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[
                            0
                        ].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            node_pattern_ids=batch_pattern_ids,
                        )
                    else:
                        batch_src_node_embeddings = torch.empty(
                            (0, node_raw_features.shape[1]),
                            device=node_raw_features.device,
                        )
                        batch_dst_node_embeddings = torch.empty(
                            (0, node_raw_features.shape[1]),
                            device=node_raw_features.device,
                        )

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    if len(batch_neg_src_node_ids) > 0:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = (
                            model[0].compute_src_dst_node_temporal_embeddings(
                                src_node_ids=batch_neg_src_node_ids,
                                dst_node_ids=batch_neg_dst_node_ids,
                                node_interact_times=batch_neg_node_interact_times,
                                node_pattern_ids=batch_neg_pattern_ids,
                            )
                        )
                    else:
                        batch_neg_src_node_embeddings = torch.empty(
                            (0, batch_src_node_embeddings.shape[1]),
                            device=batch_src_node_embeddings.device,
                        )
                        batch_neg_dst_node_embeddings = torch.empty(
                            (0, batch_dst_node_embeddings.shape[1]),
                            device=batch_dst_node_embeddings.device,
                        )
                elif args.model_name in ["JODIE", "DyRep", "TGN"]:
                    # note that negative nodes do not change the memories while the positive nodes change the memories,
                    # we need to first compute the embeddings of negative nodes for memory-based models
                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = (
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_neg_node_interact_times,
                            edge_ids=None,
                            edges_are_positive=False,
                            num_neighbors=args.num_neighbors,
                        )
                    )

                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=batch_edge_ids,
                        edges_are_positive=True,
                        num_neighbors=args.num_neighbors,
                    )
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                # get positive and negative probabilities, shape (batch_size, )
                positive_probabilities = (
                    model[1](
                        input_1=batch_src_node_embeddings,
                        input_2=batch_dst_node_embeddings,
                    )
                    .squeeze(dim=-1)
                    .sigmoid()
                )
                negative_probabilities = (
                    model[1](
                        input_1=batch_neg_src_node_embeddings,
                        input_2=batch_neg_dst_node_embeddings,
                    )
                    .squeeze(dim=-1)
                    .sigmoid()
                )

                predicts = torch.cat(
                    [positive_probabilities, negative_probabilities], dim=0
                )
                labels = torch.cat(
                    [
                        torch.ones_like(positive_probabilities),
                        torch.zeros_like(negative_probabilities),
                    ],
                    dim=0,
                )

                loss = loss_func(input=predicts, target=labels)

                train_losses.append(loss.item())

                train_metrics.append(
                    get_link_prediction_metrics(
                        predicts=predicts, labels=labels, threshold=0.5
                    )
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 1000 == 0:
                    train_idx_data_loader_tqdm.set_description(
                        f"evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}"
                    )
                train_loss_per_batch.append(loss.item())

                if args.model_name in ["JODIE", "DyRep", "TGN"]:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()

            logger.info(f"nb_normal_edges_sampled: {nb_normal_edges_sampled}, nb_anomalous_edges_sampled: {nb_anomalous_edges_sampled}, nb_random_edges_sampled: {nb_random_edges_sampled}")  

            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                # backup memory bank after training so it can be used for new validation nodes
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            epoch_train_loss = np.mean(train_losses)
            train_loss_history.append(epoch_train_loss)

            val_losses, val_metrics, val_batch_losses = evaluate_model_link_prediction_ano_insertion(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_neg_edge_sampler=val_neg_edge_sampler,
                evaluate_data=val_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
                temp=args.temperature,
                return_batch_losses=True,
            )

            val_loss_per_batch.extend(val_batch_losses)
            epoch_val_loss = np.mean(val_losses)
            val_loss_history.append(epoch_val_loss)

            total_sampled_edges = global_historical_edges + global_random_edges
            average_batch_historical_proportion = np.mean(batch_historical_ratios) if len(batch_historical_ratios) > 0 else 0.0

            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                # reload training memory bank for new validation nodes
                model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)

            if args.inductive:
                new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(
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

            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                # reload validation memory bank for testing nodes or saving models
                # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            logger.info(
                f"Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]['lr']}, train loss: {np.mean(train_losses):.4f}"
            )
            for metric_name in train_metrics[0].keys():
                logger.info(
                    f"train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}"
                )
            logger.info(f"validate loss: {np.mean(val_losses):.4f}")
            for metric_name in val_metrics[0].keys():
                logger.info(
                    f"validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}"
                )
            if args.inductive:
                logger.info(f"new node validate loss: {np.mean(new_node_val_losses):.4f}")
                for metric_name in new_node_val_metrics[0].keys():
                    logger.info(
                        f"new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}"
                    )

            # perform testing once after test_interval_epochs
            # if (epoch + 1) % args.test_interval_epochs == 0:
            #     test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
            #                                                                model=model,
            #                                                                neighbor_sampler=full_neighbor_sampler,
            #                                                                evaluate_idx_data_loader=test_idx_data_loader,
            #                                                                evaluate_neg_edge_sampler=test_neg_edge_sampler,
            #                                                                evaluate_data=test_data,
            #                                                                loss_func=loss_func,
            #                                                                num_neighbors=args.num_neighbors,
            #                                                                time_gap=args.time_gap, temp=args.temperature)

            #     new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
            #                                                                                  model=model,
            #                                                                                  neighbor_sampler=full_neighbor_sampler,
            #                                                                                  evaluate_idx_data_loader=new_node_test_idx_data_loader,
            #                                                                                  evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
            #                                                                                  evaluate_data=new_node_test_data,
            #                                                                                  loss_func=loss_func,
            #                                                                                  num_neighbors=args.num_neighbors,
            #                                                                                  time_gap=args.time_gap, temp=args.temperature)

            #     logger.info(f'test loss: {np.mean(test_losses):.4f}')
            #     for metric_name in test_metrics[0].keys():
            #         logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
            #     logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
            #     for metric_name in new_node_test_metrics[0].keys():
            #         logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append(
                    (
                        metric_name,
                        np.mean(
                            [val_metric[metric_name] for val_metric in val_metrics]
                        ),
                        True,
                    )
                )
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        kernel_size = min(1000, len(train_loss_per_batch))
        if kernel_size >= 3:
            train_loss_smoothed = median_filter(train_loss_per_batch, size=kernel_size)
        else:
            train_loss_smoothed = train_loss_per_batch

        # Dataframe for loss per batch
        df_train_loss_per_batch = pd.DataFrame(
            {
                "batch": range(1, len(train_loss_per_batch) + 1),
                "raw_loss": train_loss_per_batch,
                "smoothed_loss": train_loss_smoothed,
            }
        )

        # Save csv
        df_train_loss_per_batch.to_csv(
            f"{loss_save_folder}/train_loss_per_batch_run_{run}.csv", index=False
        )

        plt.figure(figsize=(15, 10))

        # Subplot 1: training loss per batch
        plt.subplot(3, 1, 1)
        batch_numbers_train = range(1, len(train_loss_per_batch) + 1)
        plt.plot(
            batch_numbers_train,
            train_loss_per_batch,
            "b-",
            alpha=0.3,
            linewidth=0.8,
            label="Train Loss (raw)",
        )
        plt.plot(
            batch_numbers_train,
            train_loss_smoothed,
            "b-",
            linewidth=2,
            label="Train Loss (smoothed)",
        )
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        plt.title(
            f"Training Loss per Batch - {args.model_name} on {args.dataset_name} (Run {run + 1})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{loss_save_folder}/loss_curves_single_epoch_run_{run}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(
            batch_numbers_train,
            train_loss_smoothed,
            "b-",
            linewidth=2,
            label="Train Loss (smoothed)",
        )
        plt.fill_between(batch_numbers_train, train_loss_smoothed, alpha=0.3)
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        plt.title(
            f"Training Loss Evolution - {args.model_name} on {args.dataset_name} (Run {run + 1})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add stats
        train_final_loss = (
            train_loss_smoothed[-1] if len(train_loss_smoothed) > 0 else 0
        )
        train_min_loss = min(train_loss_smoothed) if len(train_loss_smoothed) > 0 else 0
        train_max_loss = max(train_loss_smoothed) if len(train_loss_smoothed) > 0 else 0

        plt.text(
            0.02,
            0.98,
            f"Final Loss: {train_final_loss:.4f}\nMin Loss: {train_min_loss:.4f}\nMax Loss: {train_max_loss:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(
            f"{loss_save_folder}/loss_evolution_with_stats_run_{run}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(f"Single epoch loss data and plots saved in {loss_save_folder}")
        logger.info(
            f"Training batches: {len(train_loss_per_batch)}, Validation batches: {len(val_loss_per_batch)}"
        )
        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f"get final performance on dataset {args.dataset_name}...")

        # # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        # if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        #     val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
        #                                                              model=model,
        #                                                              neighbor_sampler=full_neighbor_sampler,
        #                                                              evaluate_idx_data_loader=val_idx_data_loader,
        #                                                              evaluate_neg_edge_sampler=val_neg_edge_sampler,
        #                                                              evaluate_data=val_data,
        #                                                              loss_func=loss_func,
        #                                                              num_neighbors=args.num_neighbors,
        #                                                              time_gap=args.time_gap, temp=args.temperature)

        #     new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
        #                                                                                model=model,
        #                                                                                neighbor_sampler=full_neighbor_sampler,
        #                                                                                evaluate_idx_data_loader=new_node_val_idx_data_loader,
        #                                                                                evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
        #                                                                                evaluate_data=new_node_val_data,
        #                                                                                loss_func=loss_func,
        #                                                                                num_neighbors=args.num_neighbors,
        #                                                                                time_gap=args.time_gap, temp=args.temperature)

        # test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
        #                                                            model=model,
        #                                                            neighbor_sampler=full_neighbor_sampler,
        #                                                            evaluate_idx_data_loader=test_idx_data_loader,
        #                                                            evaluate_neg_edge_sampler=test_neg_edge_sampler,
        #                                                            evaluate_data=test_data,
        #                                                            loss_func=loss_func,
        #                                                            num_neighbors=args.num_neighbors,
        #                                                            time_gap=args.time_gap, temp=args.temperature)

        # new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
        #                                                                              model=model,
        #                                                                              neighbor_sampler=full_neighbor_sampler,
        #                                                                              evaluate_idx_data_loader=new_node_test_idx_data_loader,
        #                                                                              evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
        #                                                                              evaluate_data=new_node_test_data,
        #                                                                              loss_func=loss_func,
        #                                                                              num_neighbors=args.num_neighbors,
        #                                                                              time_gap=args.time_gap, temp=args.temperature)
        # # store the evaluation metrics at the current run
        # val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        # if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        #     logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        #     for metric_name in val_metrics[0].keys():
        #         average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
        #         logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
        #         val_metric_dict[metric_name] = average_val_metric

        #     logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
        #     for metric_name in new_node_val_metrics[0].keys():
        #         average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
        #         logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
        #         new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        # logger.info(f'test loss: {np.mean(test_losses):.4f}')
        # for metric_name in test_metrics[0].keys():
        #     average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
        #     logger.info(f'test {metric_name}, {average_test_metric:.4f}')
        #     test_metric_dict[metric_name] = average_test_metric

        # logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        # for metric_name in new_node_test_metrics[0].keys():
        #     average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
        #     logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
        #     new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f"Run {run + 1} cost {single_run_time:.2f} seconds.")

        # if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        #     val_metric_all_runs.append(val_metric_dict)
        #     new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        # test_metric_all_runs.append(test_metric_dict)
        # new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # # save model result
        # if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        #     result_json = {
        #         "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
        #         "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
        #         "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
        #         "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        #     }
        # else:
        #     result_json = {
        #         "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
        #         "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        #     }
        # result_json = json.dumps(result_json, indent=4)

        # save_result_folder = f"{args.experiment_folder}/saved_results"
        # create_folder(save_result_folder)
        # save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        # with open(save_result_path, 'w') as file:
        #     file.write(result_json)

    # # store the average metrics at the log of the last run
    # logger.info(f'metrics over {args.num_runs} runs:')

    # if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
    #     for metric_name in val_metric_all_runs[0].keys():
    #         logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
    #         logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
    #                     f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    #     for metric_name in new_node_val_metric_all_runs[0].keys():
    #         logger.info(f'new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}')
    #         logger.info(f'average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} '
    #                     f'± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}')

    # for metric_name in test_metric_all_runs[0].keys():
    #     logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
    #     logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
    #                 f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    # for metric_name in new_node_test_metric_all_runs[0].keys():
    #     logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
    #     logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
    #                 f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
