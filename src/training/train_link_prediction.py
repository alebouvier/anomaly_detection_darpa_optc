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
    load_pickle_file,
    set_random_seed,
    convert_to_gpu,
    get_parameter_sizes,
    create_optimizer,
    create_folder,
)
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler, BASE
from evaluation.evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping


def main(args):

    """Perform the work of main."""
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

    if args.negative_sample_strategy == "2hop_neighbor":
        # read csv
        neighbors_2hop = pd.read_csv(f"{BASE}/processed_data/{args.dataset_name}/two_hop_non_neighbors.csv")
    else:
        neighbors_2hop = None
    

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
    )

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
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
        neighbors_2hop=neighbors_2hop,
        seed=0 if args.negative_sample_strategy != "random" else None,
    )
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, interact_times=full_data.node_interact_times, negative_sample_strategy=args.negative_sample_strategy, neighbors_2hop=neighbors_2hop, seed=1
    )
    if args.inductive:
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=new_node_val_data.src_node_ids,
            dst_node_ids=new_node_val_data.dst_node_ids,
            negative_sample_strategy=args.negative_sample_strategy,
            neighbors_2hop=neighbors_2hop,
            seed=1,
        )
    # test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    # new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)


    # get data loaders
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
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

        batch_preferred_ratios = []
        global_preferred_edges = 0
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
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                (
                    batch_src_node_ids,
                    batch_dst_node_ids,
                    batch_node_interact_times,
                    batch_edge_ids,
                ) = (
                    train_data.src_node_ids[train_data_indices],
                    train_data.dst_node_ids[train_data_indices],
                    train_data.node_interact_times[train_data_indices],
                    train_data.edge_ids[train_data_indices],
                )

                
                _, batch_neg_dst_node_ids, num_preferred_sample_edges, num_random_sample_edges = (
                    train_neg_edge_sampler.sample(
                        size=len(batch_src_node_ids),
                        batch_src_node_ids=batch_src_node_ids,
                        batch_dst_node_ids=batch_dst_node_ids,
                        current_batch_start_time=float(np.min(batch_node_interact_times)),
                        current_batch_end_time=float(np.max(batch_node_interact_times)),
                        return_sampling_counts=True,
                    )
                )
                batch_neg_src_node_ids = batch_src_node_ids

                if args.negative_sample_strategy != "random":
                    global_preferred_edges += num_preferred_sample_edges
                    batch_preferred_ratios.append(
                        num_preferred_sample_edges / len(batch_src_node_ids)
                    )
                else:
                    batch_preferred_ratios.append(0.0)
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
                        num_neighbors=args.num_neighbors,
                    )

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = (
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
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
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                    )

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = (
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap,
                        )
                    )
                elif args.model_name in ["DyGFormer"]:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                    )

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = (
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                        )
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
                            node_interact_times=batch_node_interact_times,
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

            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                # backup memory bank after training so it can be used for new validation nodes
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            epoch_train_loss = np.mean(train_losses)
            train_loss_history.append(epoch_train_loss)

            val_losses, val_metrics, val_batch_losses = evaluate_model_link_prediction(
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

            total_sampled_edges = global_preferred_edges + global_random_edges
            average_batch_preferred_proportion = np.mean(batch_preferred_ratios) if len(batch_preferred_ratios) > 0 else 0.0
            logger.info(
                f"Training negative sampling proportions: global preferred edges = {global_preferred_edges}, "
                f"global random edges = {global_random_edges}, "
                f"global preferred proportion = {(global_preferred_edges / total_sampled_edges if total_sampled_edges > 0 else 0.0):.4f}, "
                f"average batch preferred proportion = {average_batch_preferred_proportion:.4f}"
            )

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

        single_run_time = time.time() - run_start_time
        logger.info(f"Run {run + 1} cost {single_run_time:.2f} seconds.")


        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

    sys.exit()
