from torchmetrics import Accuracy, Precision, Recall, MetricCollection


def get_metrics(num_classes: int):

    accuracy_metrics = get_accuracy_metric(num_classes=num_classes)
    precision_metrics = get_precision_metric(num_classes=num_classes)
    recall_metrics = get_recall_metric(num_classes=num_classes)

    return MetricCollection({**accuracy_metrics, **precision_metrics, **recall_metrics})


def get_accuracy_metric(num_classes: int):
    return {
        "accuracy_weighted": Accuracy(
            task="multiclass", num_classes=num_classes, average="weighted", top_k=1
        ),
        "accuracy_macro": Accuracy(
            task="multiclass", num_classes=num_classes, average="macro", top_k=1
        ),
        "accuracy_micro": Accuracy(
            task="multiclass", num_classes=num_classes, average="micro", top_k=1
        ),
    }


def get_precision_metric(num_classes: int):
    return {
        "precision_weighted": Precision(
            task="multiclass", num_classes=num_classes, average="weighted", top_k=1
        ),
        "precision_macro": Precision(
            task="multiclass", num_classes=num_classes, average="macro", top_k=1
        ),
        "precision_micro": Precision(
            task="multiclass", num_classes=num_classes, average="micro", top_k=1
        ),
    }


def get_recall_metric(num_classes: int):
    return {
        "recall_weighted": Recall(
            task="multiclass", num_classes=num_classes, average="weighted", top_k=1
        ),
        "recall_macro": Recall(
            task="multiclass", num_classes=num_classes, average="macro", top_k=1
        ),
        "recall_micro": Recall(
            task="multiclass", num_classes=num_classes, average="micro", top_k=1
        ),
    }
