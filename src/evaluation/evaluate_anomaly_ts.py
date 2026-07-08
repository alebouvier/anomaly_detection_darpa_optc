import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
)

from sklearn.calibration import calibration_curve

from collections import defaultdict

from utils.utils import create_folder, load_pickle_file, BASE

from evaluation.calibration import ConformalForecastingEvaluator



def print_results(dataset_name, model_name, metrics, conf_evaluator, mode, level="edge"):
    """Write a text report and save all figures"""
    output_folder = f"experiments/{dataset_name}/{model_name.lower()}/{mode}"
    create_folder(output_folder)
 
    # ------------------------------------------------------------------
    # 1. Text report
    # ------------------------------------------------------------------
    report_path = os.path.join(output_folder, f"report_{level}.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"  Evaluation Report for {mode}\n")
        f.write(f"  Dataset : {dataset_name}\n")
        f.write(f"  Model   : {model_name}\n")
        f.write(f"  Level   : {level}\n")
        f.write("=" * 60 + "\n\n")

        if conf_evaluator is not None:
            f.write("--- Miscoverage ---\n")
            f.write(f"  Method : {conf_evaluator.method}\n")
            f.write(f"  Miscoverage Rate : {metrics['miscoverage']:.4f}\n")
            if conf_evaluator.method == "adaptive":
                f.write(f" Initial Miscoverage Level: {metrics['miscoverage_level'][-1]:.4f}\n")
            else:
                f.write(f" Miscoverage Level: {metrics['miscoverage_level']:.4f}\n")
            f.write("\n")

        if conf_evaluator is not None and mode == "anomaly_detection":
            f.write(f"--- Classification Metrics (threshold = {conf_evaluator.get_threshold():.4f}) ---\n")
        elif mode == "anomaly_detection":
            f.write(f"--- Classification Metrics (threshold = {metrics['best_threshold']:.4f}) ---\n")
        else:
            f.write(f"--- Classification Metrics (threshold = 0.5) ---\n")
        if "auc" in metrics:
            f.write(f"  AUC       : {metrics['auc']:.4f}\n")
        if "ap" in metrics:
            f.write(f"  AP        : {metrics['ap']:.4f}\n")
        if "accuracy" in metrics:
            f.write(f"  Accuracy  : {metrics['accuracy']:.4f}\n")
        if "precision" in metrics:
            f.write(f"  Precision : {metrics['precision']:.4f}\n")
        if "recall" in metrics:
            f.write(f"  Recall    : {metrics['recall']:.4f}\n")
        if "f1" in metrics:
            f.write(f"  F1-Score  : {metrics['f1']:.4f}\n\n")

        if "confusion_matrix" in metrics:
            f.write("--- Confusion Matrix ---\n")
            cm = metrics['confusion_matrix']
            f.write(f"  TN={cm[0,0]}  FP={cm[0,1]}\n")
            f.write(f"  FN={cm[1,0]}  TP={cm[1,1]}\n\n")
 
    print(f"Report saved to {report_path}")
 
    # ------------------------------------------------------------------
    # 2. ROC curve
    # ------------------------------------------------------------------
    if "list_fpr"  in metrics and "list_tpr" in metrics and "auc" in metrics:
        fig, ax = plt.subplots()
        ax.plot(metrics['list_fpr'], metrics['list_tpr'],
                label=f"AUC = {metrics['auc']:.4f}")
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve — {model_name} on {dataset_name}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f"roc.png"))
        plt.close(fig)
 
    # ------------------------------------------------------------------
    # 3. Precision-Recall curve
    # ------------------------------------------------------------------
    if "list_recall" in metrics and "list_precision" in metrics and "ap" in metrics:
        fig, ax = plt.subplots()
        ax.plot(metrics['list_recall'], metrics['list_precision'],
                label=f"AP = {metrics['ap']:.4f}")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve — {model_name} on {dataset_name}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f"pr_curve.png"))
        plt.close(fig)
 
    # ------------------------------------------------------------------
    # 4. Calibration curve
    # ------------------------------------------------------------------
    if "list_prob_pred" in metrics and "list_prob_true" in metrics:
        fig, ax = plt.subplots()
        ax.plot(metrics['list_prob_pred'], metrics['list_prob_true'],
                marker='o', label='Model')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Curve — {model_name} on {dataset_name} ')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f"calibration.png"))
        plt.close(fig)
 
    # ------------------------------------------------------------------
    # 5. Confusion matrix heatmap
    # ------------------------------------------------------------------
    if "confusion_matrix" in metrics:
        fig, ax = plt.subplots()
        cm = metrics['confusion_matrix']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred Neg', 'Pred Pos'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['True Neg', 'True Pos'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        ax.set_title(f'Confusion Matrix — {model_name} on {dataset_name}')
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f"confusion_matrix.png"))
        plt.close(fig)

    # ------------------------------------------------------------------
    # 6. 2 figures side by side: histogram of scores in the calibration bins for positive samples and negative samples (abscisse [0,1])
    # ------------------------------------------------------------------
    if "calibration_bin_counts_pos" in metrics and "calibration_bin_counts_neg" in metrics and "link_prediction" in mode:
        fig, ax = plt.subplots()
        bin_counts_pos = metrics['calibration_bin_counts_pos']
        bin_counts_neg = metrics['calibration_bin_counts_neg']
        bins = np.linspace(0, 1, len(bin_counts_pos) + 1)
        # make transparency for better visualization
        ax.bar(bins[:-1], bin_counts_pos, width=0.02, align='edge', label='Positive Samples', alpha=0.6)
        ax.bar(bins[:-1], bin_counts_neg, width=0.02, align='edge', label='Negative Samples', alpha=0.6)
        ax.set_xlabel('Predicted Probability Bin')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Histogram of Scores in Calibration Bins — {model_name} on {dataset_name}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f"calibration_bin_histogram.png"))
        plt.close(fig)
    
    if "calibration_bin_counts_pos" in metrics and "calibration_bin_counts_neg" in metrics and mode == "anomaly_detection":
        fig, ax = plt.subplots(figsize=(10, 5))
        n_bins = len(metrics['calibration_bin_counts_neg'])
        x = np.arange(n_bins) / n_bins
        width = 0.9 / n_bins

        ax.bar(
            x,
            metrics['calibration_bin_counts_neg'],
            width=width,
            align='center',
            alpha=0.45,
            color='tab:blue',
            label='Normal samples'
        )
        ax.set_xlabel('Score bin')
        ax.set_ylabel('Count (normal samples)', color='tab:blue')
        ax.set_title(f'Calibration bin counts — {model_name} on {dataset_name}')
        ax.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax.twinx()
        ax2.bar(
            x,
            metrics['calibration_bin_counts_pos'],
            width=width,
            align='center',
            alpha=0.45,
            color='orange',
            label='Anomalous samples'
        )
        ax2.set_ylabel('Count (anomalous samples)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.8)
        ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), framealpha=0.8)

        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f"calibration_bin_counts.png"))
        plt.close(fig)

    # ------------------------------------------------------------------
    # 7. adaptive miscoverage level plot (if applicable)
    # ------------------------------------------------------------------
    if conf_evaluator is not None and conf_evaluator.method == "adaptive": 
        fig, ax = plt.subplots()
        ax.plot(metrics['miscoverage_level'], label='Miscoverage Level')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Miscoverage Level')
        ax.set_title(f'Adaptive Miscoverage Level — {model_name} on {dataset_name}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f"adaptive_miscoverage_level.png"))
        plt.close(fig)

def link_prediciton_metrics(
    predicted_links,
    actual_links,
    non_exist_links,
    conf_evaluator,
    names,
    mode="link_prediction",
    level="edge",
):
    dataset_name, model_name = names

    # y_score are scores predicited by the model
    # y_true is 1 for real links and 0 for non-existing links 
    y_score = []
    y_true = []
    for src, dst, score, ts in predicted_links:
        y_score.append(score)
        y_true.append(1)
    for src, dst, score, ts in non_exist_links:
        y_score.append(score)
        y_true.append(0)        


    metrics = {}
    # check if scores are outside [0, 1]
    if np.any(np.array(y_score) < 0) or np.any(np.array(y_score) > 1):
        # min max squashing to [0, 1]
        y_score = (y_score - np.min(y_score)) / (np.max(y_score) - np.min(y_score))

    y_score = np.array(y_score, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)

    # compute miscoverage
    if conf_evaluator is not None:
        miscoverage, _, miscoverage_level = conf_evaluator.evaluate(predicted_links, actual_links, non_exist_links)
        metrics["miscoverage"] = miscoverage
        metrics["miscoverage_level"] = miscoverage_level

    # compute AUC and AP
    metrics["auc"] = roc_auc_score(y_true, y_score)
    metrics["ap"] = average_precision_score(y_true, y_score)

    # ROC and Precision-recall curve
    metrics["list_fpr"], metrics["list_tpr"], metrics["list_roc_threshold"] = roc_curve(y_true, y_score)  
    metrics["list_precision"], metrics["list_recall"], metrics["list_pr_threshold"] = precision_recall_curve(y_true, y_score)

    # calibration curve
    metrics["list_prob_true"], metrics["list_prob_pred"] = calibration_curve(y_true, y_score, n_bins=50)

    # count number of scores in each bin of the calibration curve for positive samples and negative samples
    bin_counts_pos = np.histogram(y_score[y_true == 1], bins=50, range=(0, 1))[0]
    bin_counts_neg = np.histogram(y_score[y_true == 0], bins=50, range=(0, 1))[0]
    metrics["calibration_bin_counts_pos"] = bin_counts_pos
    metrics["calibration_bin_counts_neg"] = bin_counts_neg


    # compute predicted class for threshold = 0.5
    threshold = 0.5
    y_pred = (y_score > threshold).astype(np.int32)

    # compute accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred)

    # compute confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)


    print_results(dataset_name, model_name, metrics, conf_evaluator, mode=mode, level=level)

def anomaly_detection_edge_level_metrics(predicted_links, actual_links, non_exist_links, conf_evaluator, names):
    dataset_name, model_name = names
    score_map = {(src, dst, ts): 1 - score for src, dst, score, ts in predicted_links}

    y_score = []
    y_true = []

    for src, dst, label, ts in actual_links:
        if (src, dst, ts) in score_map:
            y_score.append(score_map[(src, dst, ts)])
            y_true.append(label)
    
    y_score = np.array(y_score, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)

    metrics = {}

    # compute AUC and AP
    metrics["auc"] = roc_auc_score(y_true, y_score)
    metrics["ap"] = average_precision_score(y_true, y_score)

    # ROC and Precision-recall curve
    metrics["list_fpr"], metrics["list_tpr"], metrics["list_roc_threshold"] = roc_curve(y_true, y_score)  
    metrics["list_precision"], metrics["list_recall"], metrics["list_pr_threshold"] = precision_recall_curve(y_true, y_score)

    # find threshold that maximize f1-score
    tp = metrics["list_tpr"] * np.sum(y_true)
    fp = metrics["list_fpr"] * np.sum(1 - y_true)
    fn = (1 - metrics["list_tpr"]) * np.sum(y_true)
    f1_scores = 2 * tp / (2 * tp + fp + fn)
    best_th_idx = np.argmax(f1_scores)
    best_th = metrics["list_roc_threshold"][best_th_idx]
    metrics["best_threshold"] = best_th


    # calibration curve
    metrics["list_prob_true"], metrics["list_prob_pred"] = calibration_curve(y_true, y_score, n_bins=50)

    # count number of scores in each bin of the calibration curve for positive samples and negative samples
    bin_counts_pos = np.histogram(y_score[y_true == 1], bins=50, range=(0, 1))[0]
    bin_counts_neg = np.histogram(y_score[y_true == 0], bins=50, range=(0, 1))[0]
    metrics["calibration_bin_counts_pos"] = bin_counts_pos
    metrics["calibration_bin_counts_neg"] = bin_counts_neg
    
    if conf_evaluator is not None:
        miscoverage, miscoverage_mask, miscoverage_level = conf_evaluator.evaluate(predicted_links, actual_links, non_exist_links)
        metrics["miscoverage"] = miscoverage
        metrics["miscoverage_level"] = miscoverage_level

        y_pred = miscoverage_mask.astype(np.int32)
    else:
        # compute predicted class for threshold with best f1-score
        y_pred = y_score > best_th

    # compute f1_score
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred)

    # compute confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    print_results(dataset_name, model_name, metrics, conf_evaluator, mode="anomaly_detection")

def anomaly_detection_graph_level_metrics(predicted_links, actual_links, non_exist_links, conf_evaluator, names):
    dataset_name, model_name = names
    

    # floor timestamp to 15minutes level
    timestamp_to_label = {}
    for src, dst, label, ts in actual_links:
        ts = int(ts) // (15 * 60) * (15 * 60)
        timestamp_to_label[ts] = max(timestamp_to_label.get(ts, 0), label)
    
    
    # group and keep the 1% lowest scores by timestamp
    timestamp_to_scores = {}
    for src, dst, score, ts in predicted_links:
        ts = int(ts) // (15 * 60) * (15 * 60)
        if ts not in timestamp_to_scores:
            timestamp_to_scores[ts] = []
        timestamp_to_scores[ts].append(score)
    
    for ts in timestamp_to_scores:
        timestamp_to_scores[ts] = sorted(timestamp_to_scores[ts])[:max(1, int(len(timestamp_to_scores[ts]) * 0.01))]

    # compute average score by timestamp
    timestamp_to_avg_score = {ts: np.mean(scores) for ts, scores in timestamp_to_scores.items()}

    # compute AUC and AP
    y_true = np.array(list(timestamp_to_label.values()), dtype=np.int32)
    y_score_avg = np.array([1 - timestamp_to_avg_score.get(ts, 1.0) for ts in timestamp_to_label.keys()], dtype=np.float32)


    metrics = {}

    metrics["num_timestamps"] = len(timestamp_to_label)

    # compute anomaly proportion in the test set
    anomaly_proportion = np.mean(y_true)
    metrics["anomaly_proportion"] = anomaly_proportion

    metrics["auc"] = roc_auc_score(y_true, y_score_avg)
    metrics["ap"] = average_precision_score(y_true, y_score_avg)


    # find best threshold for average score and compute classification metrics at the graph level 
    precision, recall, thresholds = precision_recall_curve(y_true, y_score_avg)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    metrics["best_threshold"] = best_threshold

    y_pred = (y_score_avg >= best_threshold).astype(int)
    metrics["f1"] = f1_scores[best_threshold_index]
    metrics["precision"] = precision_score(y_true, y_pred)
    metrics["recall"] = recall_score(y_true, y_pred)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    print_results(dataset_name, model_name, metrics, None, "anomaly_detection", level="graph")

def load_results(folder, mode, include_negative_source=False):
    predicted_links = load_pickle_file(f"{folder}/{mode}_predicted_links.pkl")
    actual_links = load_pickle_file(f"{folder}/{mode}_actual_links.pkl")
    non_exist_links = load_pickle_file(f"{folder}/non_exist_links.pkl")

    if include_negative_source:
        negative_source_path = f"{folder}/non_exist_links_by_negative_source.pkl"
        if os.path.exists(negative_source_path):
            non_exist_links_by_negative_source = load_pickle_file(
                negative_source_path
            )
        else:
            non_exist_links_by_negative_source = {}
        return (
            predicted_links,
            actual_links,
            non_exist_links,
            non_exist_links_by_negative_source,
        )

    return predicted_links, actual_links, non_exist_links

def check_weird_predictions(predicted_links, actual_links, non_exist_links, names, output_file=None):
    # find index of existing links with lowest scores and non-existing links with highest scores
    dataset_name, model_name = names
    predicted_links_sorted = sorted(predicted_links, key=lambda x: x[2])
    non_exist_links_sorted = sorted(non_exist_links, key=lambda x: x[2], reverse=True)
    weird_existing_links = predicted_links_sorted[:100]
    good_existing_links = predicted_links_sorted[-100:]
    weird_non_existing_links = non_exist_links_sorted[:100]
    good_non_existing_links = non_exist_links_sorted[-100:]

    # open csv files node_features.csv and find the features of these weird predictions and save them in a text file
    node_features_path = f"{BASE}/processed_data/{dataset_name}/node_features.csv"
    node_features = pd.read_csv(node_features_path)

    
    # node_features.csv has 2 columns: object_type and path. The node ids correspond to the row number in the csv file. We will save the features of the weird predictions in a text file with the following format:


    output_folder = f"experiments/{dataset_name}/{model_name.lower()}/weird_predictions"
    create_folder(output_folder)
    with open(os.path.join(output_folder, output_file or "weird_predictions.txt"), 'w') as f:
        f.write("Weird Existing Links (lowest scores):\n")
        for src, dst, score, ts in weird_existing_links:
            src_features = node_features.iloc[src].to_dict()
            dst_features = node_features.iloc[dst].to_dict()
            f.write(f"  {src} -> {dst} at {ts} with score {score:.4f}\n")
            f.write(f"    Source Features: {src_features}\n")
            f.write(f"    Destination Features: {dst_features}\n")
        f.write("\nGood Existing Links (highest scores):\n")
        for src, dst, score, ts in good_existing_links:
            src_features = node_features.iloc[src].to_dict()
            dst_features = node_features.iloc[dst].to_dict()
            f.write(f"  {src} -> {dst} at {ts} with score {score:.4f}\n")
            f.write(f"    Source Features: {src_features}\n")
            f.write(f"    Destination Features: {dst_features}\n")
        f.write("\nWeird Non-Existing Links (highest scores):\n")
        for src, dst, score, ts in weird_non_existing_links:
            src_features = node_features.iloc[src].to_dict()
            dst_features = node_features.iloc[dst].to_dict()
            f.write(f"  {src} -> {dst} at {ts} with score {score:.4f}\n")
            f.write(f"    Source Features: {src_features}\n")
            f.write(f"    Destination Features: {dst_features}\n")
        f.write("\nGood Non-Existing Links (lowest scores):\n")
        for src, dst, score, ts in good_non_existing_links:
            src_features = node_features.iloc[src].to_dict()
            dst_features = node_features.iloc[dst].to_dict()
            f.write(f"  {src} -> {dst} at {ts} with score {score:.4f}\n")
            f.write(f"    Source Features: {src_features}\n")
            f.write(f"    Destination Features: {dst_features}\n")
    print(f"Weird predictions saved to {os.path.join(output_folder, output_file or 'weird_predictions.txt')}")


def main(args):
    output_folder = f"experiments/{args.dataset_name}/{args.model_name.lower()}/ad_results"
    create_folder(output_folder)

    logging.basicConfig(
        filename=f"{output_folder}/ad_results.log",
        level=logging.INFO,
        format="%(message)s",
    )

    val_score_folder = f"{BASE}/val_result_data/{args.dataset_name}/{args.model_name}"
    (
        val_predicted_links,
        val_actual_links,
        val_non_exist_links,
        val_non_exist_links_by_negative_source,
    ) = load_results(val_score_folder, "val", include_negative_source=True)

    if args.calibration:
        cal_score_folder = f"{BASE}/cal_result_data/{args.dataset_name}/{args.model_name}"
        (cal_predicted_links, cal_actual_links, cal_non_exist_links) = load_results(
            cal_score_folder, "cal"
        )

    test_score_folder = f"{BASE}/test_result_data/{args.dataset_name}/{args.model_name}"
    (test_predicted_links, test_actual_links, test_non_exist_links) = load_results(
        test_score_folder, "test"
    )

    if args.calibration:
        conf_evaluator = ConformalForecastingEvaluator(cal_predicted_links, 
                                        cal_actual_links, 
                                        cal_non_exist_links, 
                                        args.miscoverage_level, 
                                        method=args.calibration_method, 
                                        dataset_name=args.dataset_name, 
                                        model_name=args.model_name)
    else:
        conf_evaluator = None


    link_prediciton_metrics(
        val_predicted_links,
        val_actual_links,
        val_non_exist_links,
        conf_evaluator,
        names=(args.dataset_name, args.model_name),
    )

    for negative_name, negative_non_exist_links in val_non_exist_links_by_negative_source.items():
        safe_negative_name = negative_name.replace(".", "_").replace(" ", "_")
        link_prediciton_metrics(
            val_predicted_links,
            val_actual_links,
            negative_non_exist_links,
            conf_evaluator,
            names=(args.dataset_name, args.model_name),
            mode=f"link_prediction_{safe_negative_name}",
            level=negative_name,
        )

    anomaly_detection_edge_level_metrics(
        test_predicted_links,
        test_actual_links,
        test_non_exist_links,
        conf_evaluator,
        names=(args.dataset_name, args.model_name),
    )

    anomaly_detection_graph_level_metrics(test_predicted_links, test_actual_links, test_non_exist_links, None, names=(args.dataset_name, args.model_name))

    check_weird_predictions(val_predicted_links, val_actual_links, val_non_exist_links, names=(args.dataset_name, args.model_name), output_file="weird_predictions_val.txt")
    check_weird_predictions(test_predicted_links, test_actual_links, test_non_exist_links, names=(args.dataset_name, args.model_name), output_file="weird_predictions_test.txt")

