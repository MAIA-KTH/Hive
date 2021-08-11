import os
from collections import OrderedDict
from multiprocessing import Pool
from typing import List, Dict, Union, Any

import numpy as np
from k8s_DP.evaluation.metrics import compute_confusion_matrix, METRICS
from k8s_DP.utils.file_utils import subfiles
from k8s_DP.utils.log_utils import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

DEFAULT_METRICS = [
    "Dice",
    "Accuracy",
    "Jaccard",
    "Recall",
    "Precision",
    "False Positive Rate",
    "False Omission Rate",
    "Hausdorff Distance",
    "Hausdorff Distance 95",
    "Avg. Surface Distance",
    "Avg. Symmetric Surface Distance",
]


def compute_metrics_for_case(
        gt_filename: str, pred_filename: str, labels: List[str], metrics: List[str] = DEFAULT_METRICS
) -> Dict[str, Union[str, Dict]]:
    r"""
    Computes given metrics for the specified subject and labels. The subject is defined by the *ground truth image* and the
    *predicted image*. The ``metrics`` to compute are specified in a list, alongside the ``labels`` to consider.
    Returns a dictionary containing ``gt_filename``, ``pred_filename`` and, for each label a sub-dictionary
    specifying the metric scores:
        .. math::
            label_i: \{ Metric_i : score_i, Metric_j : score_j, ... \}

    Parameters
    ----------
    gt_filename : str
        Ground truth filepath. The image voxels should contain the class indices.
    pred_filename : str
        Prediction filepath. The image voxels should contain the class indices.
    labels : List[str]
        list of strings, indicating for which labels the confusion matrix is computed.
    metrics : List[str]
        list of strings, specifying which metrics to compute. Defaults to [ ``"Dice"``, ``"Accuracy"``, ``"Jaccard"``,
        ``"Recall"``, ``"Precision"``, ``"False Positive Rate"``, ``"False Omission Rate"``, ``"Hausdorff Distance"``,
         ``"Hausdorff Distance 95"``, ``"Avg. Surface Distance"``, ``"Avg. Symmetric Surface Distance"`` ].

    Returns
    -------
    Dict[str, Union[str, Dict]]
        Dictionary including ``gt_filename``, ``pred_filename`` and, for each label, a dictionary containing the metric
       scores.
    """
    cm_class_map = compute_confusion_matrix(gt_filename, pred_filename, labels)

    metrics_dict = {}  # type: Dict[str, Union[str, Dict]]

    for c in labels:
        metrics_dict[c] = {}
        metrics_dict[c]["True Positives"] = cm_class_map[c]["tp"]
        metrics_dict[c]["False Positives"] = cm_class_map[c]["fp"]
        metrics_dict[c]["True Negatives"] = cm_class_map[c]["tn"]
        metrics_dict[c]["False Negatives"] = cm_class_map[c]["fn"]
        metrics_dict[c]["Total Positives Test"] = cm_class_map[c]["pred_p"]
        metrics_dict[c]["Total Positives Reference"] = cm_class_map[c]["test_p"]

    for metric_name in metrics:
        if metric_name in METRICS:
            metric_result = METRICS[metric_name](cm_class_map)
            for c in labels:
                metrics_dict[c][metric_name] = metric_result[c]

    metrics_dict["reference"] = gt_filename
    metrics_dict["test"] = pred_filename

    return metrics_dict


def compute_metrics_for_folder(
        gt_folder: str,
        pred_folder: str,
        labels: List[str],
        file_suffix: str,
        metrics: List[str] = DEFAULT_METRICS,
        num_threads: int = None,
) -> List[Dict[str, Any]]:
    """
    Computes given metrics for the specified subjects and labels. The subjects are defined by the *ground truth folder*
    and the *predicted folder*. Only subjects with a correspondence in both the folder are considered and evaluated.
    The ``metrics`` to compute are specified in a list, alongside the ``labels`` to consider.
    Returns a list of dictionaries, one per subject, each containing ``gt_filename``, ``pred_filename`` and, for each
    label a sub-dictionary specifying the metric scores.
    The single elements in the list are obtained from :func:`compute_metrics_for_case`

    Parameters
    ----------
    gt_folder : str
        Ground truth folder path.
    pred_folder : str
        Prediction folder path.
    file_suffix : str
        File extension for the images to be evaluated. Example: ``".nii.gz"``
    labels : List[str]
        list of strings, indicating for which labels the confusion matrix is computed.
    metrics : List[str]
        list of strings, specifying which metrics to compute. Defaults to [ ``"Dice"``, ``"Accuracy"``, ``"Jaccard"``,
        ``"Recall"``, ``"Precision"``, ``"False Positive Rate"``, ``"False Omission Rate"``, ``"Hausdorff Distance"``,
         ``"Hausdorff Distance 95"``, ``"Avg. Surface Distance"``, ``"Avg. Symmetric Surface Distance"`` ].
    num_threads : int
        number of threads to use in multiprocessing ( Default: N_THREADS )

    Returns
    -------
    List[Dict[str, Any]]
        list of dictionaries, one per subject, each containing ``gt_filename``, ``pred_filename`` and, for each
        label a sub-dictionary specifying the metric scores.
    """
    gt_files = subfiles(gt_folder, join=False, suffix=file_suffix)
    pred_files = subfiles(pred_folder, join=False, suffix=file_suffix)

    if num_threads is None:
        try:
            num_threads = int(os.environ["N_THREADS"])
        except KeyError:
            logger.warning("N_THREADS is not set as environment variable. Using Default [1]")
            num_threads = 1

    pool = Pool(num_threads)
    evaluated_cases = []
    for gt_filepath in gt_files:
        if gt_filepath in pred_files:
            if not os.path.isfile(os.path.join(gt_folder, gt_filepath)):
                logger.warning("{} does not exist".format(os.path.join(gt_folder, gt_filepath)))
                continue
            if not os.path.isfile(os.path.join(pred_folder, gt_filepath)):
                logger.warning("{} does not exist".format(os.path.join(pred_folder, gt_filepath)))

            evaluated_cases.append(
                pool.starmap_async(
                    compute_metrics_for_case,
                    ((os.path.join(gt_folder, gt_filepath), os.path.join(pred_folder, gt_filepath), labels, metrics),),
                )
            )
        else:
            logger.warning("{} cannot be found in {}".format(gt_filepath, pred_folder))
    all_metrics = [i.get()[0] for i in tqdm(evaluated_cases)]
    return all_metrics


def order_scores_with_means(all_res: List[Dict[str, Any]]) -> Dict[str, Union[List, Dict]]:
    """
    Takes a list of dictionaries ``"all_res"`` ( from :func:`compute_metrics_for_folder` ) and returns a dictionary
    containing two keys:
        .. math::
            all: all\_res \n
            mean: \mu(all\_res)
    Where :math:`\mu(all\_res)` is the average result for each label and each metric.

    Parameters
    ----------
    all_res : List[Dict[str, Any]]
        List of dictionaries, one per subject, each with a sub-dictionary including the metric scores for each label.

    Returns
    -------
    Dict[str, Union[List, Dict]]
        Dictionary containing ```"all_res"`` and the average score for each metric and each label.
    """  # noqa: W605
    all_scores = OrderedDict()  # type: Dict[str, Union[List, Dict]]
    all_scores["all"] = []
    all_scores["mean"] = OrderedDict()

    for i in range(len(all_res)):
        all_scores["all"].append(all_res[i])

        # append score list for mean
        for label, score_dict in all_res[i].items():
            if label in ("test", "reference"):
                continue
            if label not in all_scores["mean"]:
                all_scores["mean"][label] = OrderedDict()
            for score, value in score_dict.items():
                if score not in all_scores["mean"][label]:
                    all_scores["mean"][label][score] = []
                all_scores["mean"][label][score].append(value)

    for label in all_scores["mean"]:
        for score in all_scores["mean"][label]:
            all_scores["mean"][label][score] = float(np.nanmean(all_scores["mean"][label][score]))
    return all_scores
