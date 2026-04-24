import logging
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
)
from collections import defaultdict

from utils.utils import create_folder, load_pickle_file, BASE

# Global variables for validation data
_validation_thresholds = {}
_roc_data = {}


def group_links_by_timestamp(predicted_links, actual_links):
    """Group links by timestamp for evaluation."""
    timestamp_data = defaultdict(lambda: {"predicted": [], "actual": []})

    for src, dst, score, timestamp in predicted_links:
        timestamp_data[timestamp]["predicted"].append((src, dst, score))

    for src, dst, label, timestamp in actual_links:
        timestamp_data[timestamp]["actual"].append((src, dst, label))

    return dict(timestamp_data)


def calculate_timestamp_score(links, method="min"):
    """
    Calculate anomaly score for a timestamp (higher = more anomalous).

    Args:
        links: List of (src, dst, score) tuples
        method: Scoring method ('min' or 'bottom_1_percent')

    Returns:
        float: Timestamp anomaly score
    """
    if not links:
        return 1.0

    scores = [score for _, _, score in links]

    if method == "min":
        return  min(scores)
    elif method == "bottom_1_percent":
        num_bottom = max(1, int(len(scores) * 0.01))
        return  np.mean(sorted(scores)[:num_bottom])
    else:
        raise ValueError(f"Unknown method: {method}")


def is_timestamp_anomalous(links):
    """Check if timestamp contains any anomalous link."""
    return any(label == 1 for _, _, label in links)


def calculate_validation_thresholds(predicted_links, actual_links, methods):
    """Calculate optimal thresholds from validation data."""
    timestamp_data = group_links_by_timestamp(predicted_links, actual_links)
    thresholds = {}

    for method in methods:
        method_scores = [
            calculate_timestamp_score(data["predicted"], method)
            for data in timestamp_data.values()
        ]

        if method_scores:
            if method == "min":
                thresholds[f"{method}_min"] = min(method_scores)
                thresholds[f"{method}_mean"] = np.mean(method_scores)
            elif method == "bottom_1_percent":
                thresholds[f"{method}_min"] = min(method_scores)
                thresholds[f"{method}_mean"] = np.mean(method_scores)

    return thresholds


def calculate_link_statistics(predicted_links, actual_links, non_exist_links):
    """Calculate score statistics by link type."""
    score_map = {(src, dst, ts): score for src, dst, score, ts in predicted_links}

    def get_stats(scores):
        return {
            "count": len(scores),
            "mean": np.mean(scores) if scores else 0,
            "std": np.std(scores) if scores else 0,
        }

    benign_scores = []
    anomaly_scores = []

    for src, dst, label, ts in actual_links:
        if (src, dst, ts) in score_map:
            score = score_map[(src, dst, ts)]
            (benign_scores if label == 0 else anomaly_scores).append(score)

    non_exist_scores = [score for _, _, score, _ in non_exist_links]

    return {
        "benign": get_stats(benign_scores),
        "anomaly": get_stats(anomaly_scores),
        "non_exist": get_stats(non_exist_scores),
    }


def evaluate_timestamp_detection(predicted_links, actual_links, method, threshold):
    """Evaluate timestamp-level anomaly detection."""
    timestamp_data = group_links_by_timestamp(predicted_links, actual_links)

    scores, labels, predictions = [], [], []

    for data in timestamp_data.values():
        score = calculate_timestamp_score(data["predicted"], method)
        label = int(is_timestamp_anomalous(data["actual"]))
        prediction = int(score >= threshold)

        scores.append(score)
        labels.append(label)
        predictions.append(prediction)

    # Handle confusion matrix
    cm = confusion_matrix(labels, predictions)
    if cm.shape == (1, 1):
        tn = fp = fn = tp = 0
        if labels[0] == 0:
            tn = cm[0, 0]
        else:
            tp = cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()

    total = tp + tn + fp + fn
    return {
        "cm": {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)},
        "metrics": {
            "accuracy": (tp + tn) / total if total > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        },
        "summary": {"total": len(labels), "anomalous": sum(labels)},
        "method": method,
        "threshold": threshold,
    }


def create_threshold_distribution_plots(predicted_links, actual_links, names):
    """Create distribution plots with thresholds for both methods."""
    global _validation_thresholds

    dataset_name, model_name, temperature = names
    experiment_folder = f"experiments/{dataset_name}/{model_name}"
    create_folder(experiment_folder)

    timestamp_data = group_links_by_timestamp(predicted_links, actual_links)
    methods = ["min", "bottom_1_percent"]

    for method in methods:
        scores = []
        colors = []

        # Collect scores and their corresponding colors
        for data in timestamp_data.values():
            score = calculate_timestamp_score(data["predicted"], method)
            is_anomalous = is_timestamp_anomalous(data["actual"])

            scores.append(score)
            colors.append("red" if is_anomalous else "blue")

        if not scores:
            continue

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Create fine histogram bins
        num_bins = min(200, max(50, int(len(scores) * 2)))
        bins = np.linspace(min(scores), max(scores), num_bins)

        # Separate anomalous and benign scores
        anomalous_scores = [
            score for score, color in zip(scores, colors) if color == "red"
        ]
        benign_scores = [
            score for score, color in zip(scores, colors) if color == "blue"
        ]

        # Plot histograms with fine bins
        plt.hist(
            benign_scores,
            bins=bins,
            alpha=0.7,
            color="blue",
            label=f"Benign ({len(benign_scores)})",
            edgecolor="none",
            linewidth=0,
        )
        plt.hist(
            anomalous_scores,
            bins=bins,
            alpha=0.7,
            color="red",
            label=f"Anomalous ({len(anomalous_scores)})",
            edgecolor="none",
            linewidth=0,
        )

        # Add threshold lines
        threshold_min_key = f"{method}_min"
        threshold_mean_key = f"{method}_mean"

        if threshold_min_key in _validation_thresholds:
            plt.axvline(
                _validation_thresholds[threshold_min_key],
                color="darkgreen",
                linestyle="-",
                linewidth=2,
                label=f"Threshold Min: {_validation_thresholds[threshold_min_key]:.6f}",
            )

        if threshold_mean_key in _validation_thresholds:
            plt.axvline(
                _validation_thresholds[threshold_mean_key],
                color="lightgreen",
                linestyle="--",
                linewidth=2,
                label=f"Threshold Mean: {_validation_thresholds[threshold_mean_key]:.6f}",
            )

        plt.xlabel("Score Values", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(
            f"Score Distribution - {method.replace('_', ' ').title()} Method\nDataset: {names}",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot
        create_folder(f"{experiment_folder}/pdf_anom")
        filename = f"{experiment_folder}/pdf_anom/distribution_{method}_{names}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"  Distribution plot saved: {filename}")


def save_combined_roc_curve(names):
    """Save combined ROC curves for all methods."""
    global _roc_data

    if not _roc_data:
        return

    dataset_name, model_name, temperature = names
    experiment_folder = f"experiments/{dataset_name}/{model_name}"
    create_folder(experiment_folder)

    plt.figure(figsize=(10, 8))
    colors = ["darkorange", "darkgreen", "darkblue", "darkred", "purple"]

    for i, (method, data) in enumerate(_roc_data.items()):
        plt.plot(
            data["fpr"],
            data["tpr"],
            color=colors[i % len(colors)],
            lw=2,
            label=f"{method.replace('_', ' ').title()} (AUC = {data['roc_auc']:.3f})",
        )

    plt.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--", label="Random (AUC = 0.500)")

    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(
        "ROC Curves Comparison - Timestamp Anomaly Detection",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    create_folder(f"{experiment_folder}/courbes_auc")
    filename = f"{experiment_folder}/courbes_auc/roc_timestamp_comparison_{names}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"\n  Combined ROC curve saved: {filename}")
    _roc_data.clear()


def save_timestamp_details(predicted_links, actual_links, method, names):
    """Save detailed timestamp information to JSON."""
    timestamp_data = group_links_by_timestamp(predicted_links, actual_links)

    dataset_name, model_name, temperature = names
    experiment_folder = f"experiments/{dataset_name}/{model_name}"
    create_folder(experiment_folder)

    details = []
    for timestamp, data in timestamp_data.items():
        benign_count = sum(1 for _, _, label in data["actual"] if label == 0)
        anomaly_count = sum(1 for _, _, label in data["actual"] if label == 1)

        scores = [score for _, _, score in data["predicted"]]
        if scores:
            confidence_mean = np.mean(scores)
            confidence_min = min(scores)
            num_bottom = max(1, int(len(scores) * 0.01))
            bottom_1_percent_mean = np.mean(sorted(scores)[:num_bottom])
        else:
            confidence_mean = confidence_min = bottom_1_percent_mean = None

        details.append(
            {
                "timestamp": timestamp,
                "nb_links_benign": benign_count,
                "nb_links_anomaly": anomaly_count,
                "is_suspicious": anomaly_count > 0,
                "total_predicted_links": len(data["predicted"]),
                "confidence_mean": confidence_mean,
                "confidence_min": confidence_min,
                "confidence_bottom_1_percent": bottom_1_percent_mean,
                "method_score": calculate_timestamp_score(data["predicted"], method)
                if data["predicted"]
                else None,
            }
        )

    create_folder(f"{experiment_folder}/json")
    filename = f"{experiment_folder}/json/timestamp_details_{method}_{names}.json"

    with open(filename, "w") as f:
        json.dump(
            {
                "method": method,
                "dataset": names,
                "total_timestamps": len(details),
                "suspicious_timestamps": sum(1 for d in details if d["is_suspicious"]),
                "details": details,
            },
            f,
            indent=2,
        )

    logging.info(f"  Timestamp details saved: {filename}")


def print_results(results):
    """Print evaluation results."""
    cm = results["cm"]
    metrics = results["metrics"]

    logging.info(
        f"\n--- RESULTS (Threshold: {results['threshold']:.4f}, Method: {results['method']}) ---"
    )
    logging.info(
        f"Confusion Matrix: TP={cm['TP']}, FN={cm['FN']}, FP={cm['FP']}, TN={cm['TN']}"
    )
    logging.info(
        f"Metrics: Acc={metrics['accuracy']:.3f}, Prec={metrics['precision']:.3f}, Rec={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
    )
    logging.info(
        f"Summary: {results['summary']['total']} timestamps ({results['summary']['anomalous']} anomalous)"
    )


def print_link_stats(stats):
    """Print link score statistics."""
    logging.info("\n--- LINK SCORE STATISTICS ---")
    for link_type, data in stats.items():
        logging.info(
            f"{link_type.capitalize()} links: Count={data['count']}, Mean={data['mean']:.4f}, Std={data['std']:.4f}"
        )


def timestamp_anomaly_result(
    predicted_links,
    actual_links,
    non_exist_links,
    validate,
    names=None,
    methods=["min", "bottom_1_percent"],
):
    """
    Main function for timestamp-based anomaly detection evaluation.

    Args:
        predicted_links: List of (src, dst, score, timestamp) tuples
        actual_links: List of (src, dst, label, timestamp) tuples
        non_exist_links: List of non-existing links with scores
        validate: Boolean flag for validation vs test mode
        names: Dataset name for file naming
        methods: List of scoring methods to evaluate
    """
    global _validation_thresholds

    logging.info(f"\n{'=' * 80}")
    logging.info(
        f"TIMESTAMP ANOMALY DETECTION - {'VALIDATE' if validate else 'TEST'} {names}"
    )
    logging.info(f"{'=' * 80}")

    # Calculate and print link statistics
    link_stats = calculate_link_statistics(
        predicted_links, actual_links, non_exist_links
    )
    print_link_stats(link_stats)

    logging.info("Logic: Timestamps with >=1 anomalous link are anomalous")

    # Create distribution plots
    create_threshold_distribution_plots(predicted_links, actual_links, names)
    calculate_edge_level_metrics(predicted_links, actual_links, names)

    # Evaluate each method
    for method in methods:
        logging.info(f"\n{'=' * 50}")
        logging.info(f"METHOD: {method.upper()}")
        logging.info(f"{'=' * 50}")

        roc_auc, avg_precision, best_th, best_f1_score = calculate_global_metrics(
            predicted_links, actual_links, method, names
        )
        save_timestamp_details(predicted_links, actual_links, method, names)
        _validation_thresholds[method] = best_th

        logging.info("\n--- THRESHOLD EVALUATION ---")
        results = evaluate_timestamp_detection(
            predicted_links, actual_links, method, best_th
        )

        print_results(results)

        # Save combined ROC curves
        save_combined_roc_curve(names)


def calculate_global_metrics(predicted_links, actual_links, method, names):
    """Calculate ROC AUC and Average Precision globally (without threshold)."""
    global _roc_data

    # Suppress matplotlib warnings
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    timestamp_data = group_links_by_timestamp(predicted_links, actual_links)

    scores = [
        calculate_timestamp_score(data["predicted"], method)
        for data in timestamp_data.values()
    ]
    labels = [
        int(is_timestamp_anomalous(data["actual"])) for data in timestamp_data.values()
    ]

    if len(set(labels)) <= 1:
        logging.info(
            f"\nGlobal metrics (method={method}): Cannot calculate (only one class)"
        )
        logging.info(f"  Timestamps: {len(labels)}")
        return None, None

    roc_auc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    fpr, tpr, threshold = roc_curve(labels, scores)

    # find threshold that maximize f1_score
    tp = tpr * labels.count(1)
    fp = fpr * labels.count(0)
    fn = (1 - tpr) * labels.count(1)
    f1_score = 2 * tp / (2 * tp + fp + fn)

    best_th_idx = np.argmax(f1_score)
    best_th = threshold[best_th_idx]
    best_f1_score = f1_score[best_th_idx]

    logging.info(f"\nGlobal metrics (method={method}):")
    logging.info(f"  ROC AUC: {roc_auc:.3f}")
    logging.info(f"  Average Precision: {avg_precision:.3f}")
    logging.info(f"  F1-score: {best_f1_score}, maximized with threshold = {best_th}")
    logging.info(
        f"  Timestamps: {len(labels)} (normal: {labels.count(0)}, anomalous: {labels.count(1)})"
    )

    return roc_auc, avg_precision, best_th, best_f1_score


def calculate_edge_level_metrics(predicted_links, actual_links, names):
    score_map = {(src, dst, ts): 1 - score for src, dst, score, ts in predicted_links}

    scores = []
    labels = []

    for src, dst, label, ts in actual_links:
        if (src, dst, ts) in score_map:
            scores.append(score_map[(src, dst, ts)])
            labels.append(label)

    roc_auc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    fpr, tpr, threshold = roc_curve(labels, scores)

    # find threshold that maximize f1_score
    tp = tpr * labels.count(1)
    fp = fpr * labels.count(0)
    fn = (1 - tpr) * labels.count(1)
    tn = len(labels) - tp - fp - fn
    f1_score = 2 * tp / (2 * tp + fp + fn)

    best_th_idx = np.argmax(f1_score)
    best_th = threshold[best_th_idx]
    best_f1_score = f1_score[best_th_idx]
    best_tp = tp[best_th_idx]
    best_fp = fp[best_th_idx]
    best_fn = fn[best_th_idx]
    best_tn = tn[best_th_idx]

    logging.info("\Edge metrics :")
    logging.info(f"  ROC AUC: {roc_auc:.6f}")
    logging.info(f"  Average Precision: {avg_precision:.6f}")
    logging.info(
        f"  F1-score: {best_f1_score:.6f}, maximized with threshold = {best_th:.6f}"
    )
    logging.info(f"  TP: {best_tp}, FP: {best_fp}, FN: {best_fn}, TN: {best_tn}")

    logging.info(
        f"  Edges: {len(labels)} (normal: {labels.count(0)}, anomalous: {labels.count(1)})"
    )

    return roc_auc, avg_precision, best_th, best_f1_score


def main(args):
    output_folder = f"experiments/{args.dataset_name}/{args.model_name}/ad_results"
    create_folder(output_folder)

    logging.basicConfig(
        filename=f"{output_folder}/ad_results.log",
        level=logging.INFO,
        format="%(message)s",
    )

    test_score_folder = f"{BASE}/test_result_data/{args.dataset_name}/{args.model_name}"

    test_predicted_links = load_pickle_file(
        f"{test_score_folder}/test_predicted_links.pkl"
    )
    test_actual_links = load_pickle_file(f"{test_score_folder}/test_actual_links.pkl")
    non_exist_links = load_pickle_file(f"{test_score_folder}/non_exist_links.pkl")

    timestamp_anomaly_result(
        test_predicted_links,
        test_actual_links,
        non_exist_links,
        validate=False,
        names=(args.dataset_name, args.model_name, args.temperature),
    )
