val_metrics = {
    "Mean_Dice": {
        "class_import": "monai.handlers",
        "class_name": "MeanDice",
        "class_params": {"include_background": False},
    },
    "val_mean_acc": {
        "class_import": "ignite.metrics",
        "reduce_y_label": True,
        "class_name": "Accuracy",
    },
    "val_recall": {
        "class_import": "ignite.metrics",
        "reduce_y_label": True,
        "class_name": "Recall",
    },
    "val_precision": {
        "class_import": "ignite.metrics",
        "reduce_y_label": True,
        "class_name": "Precision",
    },
    "val_dice": {
        "class_import": "ignite.metrics.confusion_matrix",
        "class_name": "DiceCoefficient",
    },
    "val_jaccard": {
        "class_import": "ignite.metrics.confusion_matrix",
        "class_name": "JaccardIndex",
    },
}
