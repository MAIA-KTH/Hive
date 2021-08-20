val_key_metric = "Mean_Dice"

val_metrics = [
    {
        "metric_name": "Mean_Dice",
        "class_import": "monai.handlers",
        "class_name": "MeanDice",
        "class_params": {"include_background": False},
    },
    {
        "metric_name": "val_mean_acc",
        "class_import": "ignite.metrics",
        "reduce_y_label": True,
        "class_name": "Accuracy",
    },
    {
        "metric_name": "val_recall",
        "class_import": "ignite.metrics",
        "reduce_y_label": True,
        "class_name": "Recall",
    },
    {
        "metric_name": "val_precision",
        "class_import": "ignite.metrics",
        "reduce_y_label": True,
        "class_name": "Precision",
    },
]

val_cm_metrics = [
    {
        "metric_name": "val_dice",
        "class_import": "ignite.metrics.confusion_matrix",
        "class_name": "DiceCoefficient",
    },
    {
        "metric_name": "val_jaccard",
        "class_import": "ignite.metrics.confusion_matrix",
        "class_name": "JaccardIndex",
    },
]

MeanDice_DaRPJ = (val_metrics, val_cm_metrics, val_key_metric)
