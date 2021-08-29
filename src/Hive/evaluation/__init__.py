DEFAULT_BAR_CONFIGS = {
    "Dice": {"thresholds": [0.9, 0.95], "colors": ["red", "orange", "green"], "min_value": 0.0, "max_value": 1.0},
    "Hausdorff Distance": {"thresholds": [40, 80], "colors": ["green", "orange", "red"], "min_value": 0.0},
    "Hausdorff Distance 95": {"thresholds": [40, 80], "colors": ["green", "orange", "red"], "min_value": 0.0},
    "Avg. Surface Distance": {"thresholds": [5, 10], "colors": ["green", "orange", "red"], "min_value": 0.0},
    "Precision": {"thresholds": [0.9, 0.95], "colors": ["red", "orange", "green"], "min_value": 0.0, "max_value": 1.0},
    "Recall": {"thresholds": [0.9, 0.95], "colors": ["red", "orange", "green"], "min_value": 0.0, "max_value": 1.0},
    "Accuracy": {"thresholds": [0.9, 0.95], "colors": ["red", "orange", "green"], "min_value": 0.0, "max_value": 1.0},
    "Specificity": {"thresholds": [0.9, 0.95], "colors": ["red", "orange", "green"], "min_value": 0.0,
                    "max_value": 1.0},
    "Jaccard": {"thresholds": [0.9, 0.95], "colors": ["red", "orange", "green"], "min_value": 0.0, "max_value": 1.0},
    "Relative Volumetric Difference": {
        "thresholds": [0.9, 0.95],
        "colors": ["red", "orange", "green"],
        "min_value": 0.0,
        "max_value": 1.0,
    },
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
