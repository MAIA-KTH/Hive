DEFAULT_BAR_CONFIGS = {
    "Dice": {"colors": ["red", "orange", "green"]},
    "Hausdorff Distance": {"colors": ["green", "orange", "red"]},
    "Hausdorff Distance 95": {"colors": ["green", "orange", "red"]},
    "Avg. Surface Distance": {"colors": ["green", "orange", "red"]},
    "Precision": {"colors": ["red", "orange", "green"]},
    "Recall": {"colors": ["red", "orange", "green"]},
    "Accuracy": {"colors": ["red", "orange", "green"]},
    "Specificity": {"colors": ["red", "orange", "green"]},
    "Jaccard": {"colors": ["red", "orange", "green"]},
    "Relative Volumetric Difference": {"colors": ["red", "orange", "green"]},
}

DEFAULT_METRIC_UNITS = {
    "Dice": "",
    "Hausdorff Distance": "[ mm ]",
    "Avg. Surface Distance": "[ mm ]",
    "Hausdorff Distance 95": "[ mm ]",
    "Precision": "",
    "Recall": "",
    "Accuracy": "",
    "Specificity": "",
    "Jaccard": "",
    "Relative Volumetric Difference": "[ % ]",
}
SECTIONS = ["testing", "validation"]

RESULTS_SECTIONS = ["testing", "validation", "experiment"]

METRICS_FOLDER_NAME = "metrics_DF"
