import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor, threshold=0.5):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts_scores = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    predicts_labels = [1 if p > threshold else 0 for p in predicts_scores]

    accuracy = accuracy_score(y_true=labels, y_pred=predicts_labels)
    average_precision = average_precision_score(y_true=labels, y_score=predicts_scores)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts_scores)

    return {"accuracy": accuracy, 'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}
