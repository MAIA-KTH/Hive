import hashlib
import json
import os
from collections import OrderedDict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Union, Any

import numpy as np
from tqdm import tqdm

from Hive.evaluation.metrics import compute_confusion_matrix, METRICS
from Hive.utils.file_utils import subfiles, subfolders
from Hive.utils.log_utils import get_logger, DEBUG

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
        gt_filename: str,
        pred_filename: str,
        labels: List[str],
        prediction_suffix: str,
        metrics: List[str] = DEFAULT_METRICS,
):
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
    prediction_suffix : str
        filename suffix used to save the JSON summary accordingly. Example: ``"XYZ_post.nii.gz"`` generates
        ``"summary_post.json"``
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

    metrics_dict = {}  # type: Dict[str,Any]

    for c in labels:
        if (cm_class_map[c]["tp"] + cm_class_map[c]["fp"]) == 0:
            logger.warning("Class {} in file {} has no positive values".format(c, pred_filename))
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

    with open(Path(pred_filename).parent.joinpath("summary{}.json".format(prediction_suffix)), "w") as outfile:
        json.dump(metrics_dict, outfile)

    return


def compute_metrics_for_folder(
        gt_folder: str,
        pred_folder: str,
        labels: List[str],
        file_suffix: str,
        metrics: List[str] = DEFAULT_METRICS,
        num_threads: int = None,
        prediction_suffix: str = "",
):
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
    prediction_suffix : str
        Name suffix for the prediction files to be considered for the evaluation. Can be None. Example: ``"_pred"``

    Returns
    -------
    List[Dict[str, Any]]
        list of dictionaries, one per subject, each containing ``gt_filename``, ``pred_filename`` and, for each
        label a sub-dictionary specifying the metric scores.
    """
    if prediction_suffix is None:
        prediction_suffix = ""

    gt_files = subfiles(gt_folder, join=False, suffix=file_suffix)

    pred_subfolders = subfolders(pred_folder, join=True)
    pred_files = [
        subfiles(pred_subfolder, join=True, suffix=prediction_suffix + file_suffix)[0] for pred_subfolder in
        pred_subfolders
    ]

    if num_threads is None:
        try:
            num_threads = int(os.environ["N_THREADS"])
        except KeyError:
            logger.warning("N_THREADS is not set as environment variable. Using Default [1]")
            num_threads = 1

    pool = Pool(num_threads)
    evaluated_cases = []
    for pred_filepath in pred_files:
        gt_filepath = str(Path(pred_filepath).name).replace(prediction_suffix, "")
        if gt_filepath in gt_files:
            if not Path(gt_folder).joinpath(gt_filepath).is_file():
                logger.warning("{} does not exist".format(Path(gt_folder).joinpath(gt_filepath)))
                continue

            if Path(pred_filepath).parent.joinpath("summary{}.json".format(prediction_suffix)).is_file():
                continue
            evaluated_cases.append(
                (
                    str(Path(gt_folder).joinpath(gt_filepath)),
                    pred_filepath,
                    labels,
                    prediction_suffix,
                    metrics,
                ),
            )

        else:
            logger.log(DEBUG, "{} cannot be found in {}".format(gt_filepath, gt_folder))

    res = pool.starmap_async(compute_metrics_for_case, evaluated_cases)
    pool.close()

    [res.get() for _ in tqdm(evaluated_cases, desc="Prediction Evaluation")]
    pool.join()

    return


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
    all_scores = OrderedDict()  # type: Dict[str, Any]
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


def load_json_summaries(pred_folder: str, prediction_suffix: str, file_suffix: str) -> List[Dict]:
    """
    Load the single JSON summaries, one per each case, and create a list containing all of them.

    Parameters
    ----------
    pred_folder : str
        folder for the prediction files.
    prediction_suffix : str
        filename suffix to be considered in the evaluation, for example when choosing post-processed volumes
        ( Example: ``"_post"`` ).
    file_suffix : str
        filename extension ( Example: ``"nii.gz"`` ).

    Returns
    -------
    List[Dict]
        List of metric dictionaries for each case and each label.
    """
    all_res = []
    if prediction_suffix is None:
        prediction_suffix = ""

    pred_subfolders = subfolders(pred_folder, join=True)
    pred_files = [
        subfiles(pred_subfolder, join=True, suffix=prediction_suffix + file_suffix)[0] for pred_subfolder in
        pred_subfolders
    ]

    for pred_filepath in pred_files:
        json_summary_path = Path(pred_filepath).parent.joinpath("summary{}.json".format(prediction_suffix))
        if json_summary_path.is_file():
            with open(json_summary_path) as json_file:
                summary = json.load(json_file)
                all_res.append(summary)

    return all_res


def compute_metrics_and_save_json(
        config_dict: Dict[str, Any], ground_truth_folder: str, prediction_folder: str, prediction_suffix: str = ""
):
    """
    Given a ground truth folder and a prediction folder, computes the evaluation metrics ans store the results in JSON format.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        configuration dictionary with needed parameters.
    ground_truth_folder : str
        folder containing ground truth files.
    prediction_folder : str
        folder containing prediction files.
    prediction_suffix : str
        filename suffix to be considered in the evaluation, for example when choosing post-processed volumes
        ( Example: ``"post"`` ).
    """
    file_suffix = config_dict["FileExtension"]
    label_dict = config_dict["label_dict"]
    label_dict.pop("0", None)

    labels = list(label_dict.keys())

    if prediction_suffix != "":
        prediction_suffix = "_" + prediction_suffix

    compute_metrics_for_folder(ground_truth_folder, prediction_folder, labels, file_suffix,
                               prediction_suffix=prediction_suffix)

    all_res = load_json_summaries(prediction_folder, prediction_suffix, file_suffix)
    all_scores = order_scores_with_means(all_res)

    json_dict = OrderedDict()
    json_dict["name"] = config_dict["DatasetName"]
    timestamp = datetime.today()
    json_dict["timestamp"] = str(timestamp)
    json_dict["task"] = "Task" + config_dict["Task_ID"] + "_" + config_dict["Task_Name"]
    json_dict["results"] = all_scores
    json_dict["id"] = hashlib.md5(json.dumps(json_dict).encode("utf-8")).hexdigest()[:12]
    with open(Path(prediction_folder).joinpath("summary{}.json".format(prediction_suffix)), "w") as outfile:
        json.dump(json_dict, outfile)

    return
