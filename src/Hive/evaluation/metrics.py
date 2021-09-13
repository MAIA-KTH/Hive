from typing import List, Dict, Union, Any

import SimpleITK as sitk
import numpy as np
from medpy.metric.binary import __surface_distances
from monai.transforms import LoadImaged
from nptyping import NDArray

EPSILON = 1e-9


def accuracy(cm_map: Dict[str, Any]) -> Dict[str, float]:
    r"""
    Computes **Accuracy** for each label specified in ``cm_map``.

        .. math::
            Accuracy = \frac{TP + TN}{ FP + FN + TP + TN}

    Parameters
    ----------
    cm_map : Dict[str, Dict[str, float]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each label to its respective **Accuracy** value.

    """
    accuracy_map = {}
    for c in list(cm_map.keys()):
        try:
            acc = (cm_map[c]["tp"] + cm_map[c]["tn"]) / (
                    cm_map[c]["tp"] + cm_map[c]["tn"] + cm_map[c]["fp"] + cm_map[c]["fn"] + EPSILON)
            accuracy_map[c] = acc
        except TypeError:
            continue
    return accuracy_map


def dice(cm_map: Dict[str, Any]) -> Dict[str, float]:
    r"""
    Computes **Dice - F1 Score** for each label specified in ``cm_map``.

        .. math::
            Dice = \frac{2 * TP}{ 2 * TP + FP + FN}

    Parameters
    ----------
    cm_map : Dict[str, Dict[str, float]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each label to its respective **Dice** value.

    """
    dice_map = {}
    for c in list(cm_map.keys()):
        try:
            dice_val = (2 * cm_map[c]["tp"]) / (2 * cm_map[c]["tp"] + cm_map[c]["fp"] + cm_map[c]["fn"] + EPSILON)
            dice_map[c] = dice_val
        except TypeError:
            continue
    return dice_map


def jaccard(cm_map: Dict[str, Any]) -> Dict[str, float]:
    r"""
    Computes **Jaccard coefficient** for each label specified in ``cm_map``.

        .. math::
            Jaccard = \frac{TP}{ FP + FN + TP }

    Parameters
    ----------
    cm_map : Dict[str, Dict[str, float]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each label to its respective **Jaccard** coefficient.

    """
    jaccard_map = {}
    for c in list(cm_map.keys()):
        try:
            jac = (cm_map[c]["tp"]) / (cm_map[c]["tp"] + cm_map[c]["fp"] + cm_map[c]["fn"] + EPSILON)
            jaccard_map[c] = jac
        except TypeError:
            continue
    return jaccard_map


def precision(cm_map: Dict[str, Any]) -> Dict[str, float]:
    r"""
    Computes **Precision** for each label specified in ``cm_map``.

        .. math::
            Precision = \frac{TP}{ FP + TP }

    Parameters
    ----------
    cm_map : Dict[str, Dict[str, float]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each label to its respective **Precision** value.

    """
    precision_map = {}
    for c in list(cm_map.keys()):
        try:
            prec = (cm_map[c]["tp"]) / (cm_map[c]["tp"] + cm_map[c]["fp"] + EPSILON)
            precision_map[c] = prec
        except TypeError:
            continue
    return precision_map


def recall(cm_map: Dict[str, Any]) -> Dict[str, float]:
    r"""
    Computes **Recall** for each label specified in ``cm_map``.

        .. math::
            Recall = \frac{TP }{ FN + TP }

    Parameters
    ----------
    cm_map : Dict[str, Dict[str, float]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each label to its respective **Recall** value.

    """
    recall_map = {}
    for c in list(cm_map.keys()):
        try:
            rec = (cm_map[c]["tp"]) / (cm_map[c]["tp"] + cm_map[c]["fn"] + EPSILON)
            recall_map[c] = rec
        except TypeError:
            continue
    return recall_map


def fpr(cm_map: Dict[str, Any]) -> Dict[str, float]:
    r"""
    Computes **False Positive Rate** for each label specified in ``cm_map``.

        .. math::
            False Positive Rate = \frac{FP}{ FP + TN}

    Parameters
    ----------
    cm_map : Dict[str, Dict[str, float]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each label to its respective **False Positive Rate**.

    """
    fpr_map = {}
    for c in list(cm_map.keys()):
        try:
            fpr_val = (cm_map[c]["fp"]) / (cm_map[c]["tn"] + cm_map[c]["fp"] + EPSILON)
            fpr_map[c] = fpr_val
        except TypeError:
            continue
    return fpr_map


def compute_distance_metrics(cm_map: Dict[str, Any]):
    """
    Computes distance metrics for each label and append them in the cm map

    Parameters
    ----------
    cm_map : Dict[str, Dict[str, float]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------

    """
    gt_itk_image = sitk.ReadImage(cm_map["reference"])
    pred_itk_image = sitk.ReadImage(cm_map["test"])
    gt_data = sitk.GetArrayFromImage(gt_itk_image)
    pred_data = sitk.GetArrayFromImage(pred_itk_image)

    for c in list(cm_map.keys()):
        try:
            gt_data_one_hot = (gt_data == int(c)).astype(np.uint8)
            pred_data_one_hot = (pred_data == int(c)).astype(np.uint8)

            hd1 = __surface_distances(pred_data_one_hot, gt_data_one_hot, np.array(gt_itk_image.GetSpacing())[::-1])
            hd2 = __surface_distances(gt_data_one_hot, pred_data_one_hot, np.array(gt_itk_image.GetSpacing())[::-1])
            hd_val = max(hd1.max(), hd2.max())
            hd95_val = np.percentile(np.hstack((hd1, hd2)), 95)
            asd_val = hd1.mean()
            assd_val = np.mean((asd_val, hd2.mean()))
            cm_map[c]['hd'] = hd_val
            cm_map[c]['hd95'] = hd95_val
            cm_map[c]['asd'] = asd_val
            cm_map[c]['assd'] = assd_val

        except ValueError:
            continue
    return


def fomr(cm_map: Dict[str, Any]) -> Dict[str, float]:
    r"""
    Computes **False Omission Rate** for each label specified in ``cm_map``.

        .. math::
            False Omission Rate = \frac{FN}{ FN + TN }

    Parameters
    ----------
    cm_map : Dict[str, Dict[str, float]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each label to its respective **False Omission Rate**.

    """
    fomr_map = {}
    for c in list(cm_map.keys()):
        try:
            fomr_val = (cm_map[c]["fn"]) / (cm_map[c]["tn"] + cm_map[c]["fn"] + EPSILON)
            fomr_map[c] = fomr_val
        except TypeError:
            continue
    return fomr_map


def hd(cm_map: Dict[str, Any]) -> Dict[str, float]:
    r"""
    Computes **Hausdorff Distance** for each label specified in ``cm_map``.

    Parameters
    ----------
    cm_map : Dict[str, Union[str, Dict[str, float]]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each label to its respective **Hausdorff Distance**.

    """
    hd_map = {}
    for c in list(cm_map.keys()):
        try:
            c_label = int(c)
            hd_map[c] = cm_map[c]['hd']
        except ValueError:
            continue
    return hd_map


def hd95(cm_map: Dict[str, Any]) -> Dict[str, float]:
    r"""
    Computes **Hausdorff Distance 95 Quantile** for each label specified in ``cm_map``.

    Parameters
    ----------
    cm_map : Dict[str, Union[str, Dict[str, float]]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each label to its respective **Hausdorff Distance 95 Quantile**.

    """

    hd95_map = {}
    for c in list(cm_map.keys()):
        try:
            c_label = int(c)
            hd95_map[c] = cm_map[c]['hd95']
        except ValueError:
            continue
    return hd95_map


def asd(cm_map: Dict[str, Any]) -> Dict[str, float]:
    r"""
    Computes **Average Surface Distance** for each label specified in ``cm_map``.

    Parameters
    ----------
    cm_map : Dict[str, Union[str, Dict[str, float]]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each label to its respective **Average Surface Distance**.

    """
    asd_map = {}
    for c in list(cm_map.keys()):
        try:
            c_label = int(c)
            asd_map[c] = cm_map[c]['asd']
        except ValueError:
            continue
    return asd_map


def assd(cm_map: Dict[str, Any]) -> Dict[str, float]:
    r"""
    Computes **Average Symmetric Surface Distance** for each label specified in ``cm_map``.

    Parameters
    ----------
    cm_map : Dict[str, Union[str, Dict[str, float]]]
        Confusion matrix map, mapping each label to its confusion matrix values, as described in :func:`compute_confusion_matrix`.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each label to its respective **Average Symmetric Surface Distance**.

    """
    assd_map = {}
    for c in list(cm_map.keys()):
        try:
            c_label = int(c)
            assd_map[c] = cm_map[c]['assd']
        except ValueError:
            continue
    return assd_map


METRICS = {
    "Accuracy": accuracy,
    "Dice": dice,
    "Jaccard": jaccard,
    "Precision": precision,
    "Recall": recall,
    "False Positive Rate": fpr,
    "Hausdorff Distance": hd,
    "Hausdorff Distance 95": hd95,
    "Avg. Surface Distance": asd,
    "Avg. Symmetric Surface Distance": assd,
    "False Omission Rate": fomr,
}


def _confusion_matrix(gt_data: NDArray[(Any,), int], pred_data: NDArray[(Any,), int], labels: List[str]) -> Dict[
    str, Any]:
    """

    Parameters
    ----------
    gt_data : NDArray[(Any,), int]
        flat Numpy array containing the ground truth data to be evaluated. Each value indicates the class
    pred_data : NDArray[(Any,), int]
        flat Numpy array containing the prediction data to be evaluated. Each value indicates the class
    labels : List[str]
        list of strings, indicating for which labels the confusion matrix is computed.
        Example: [``"1"``, ``"2"``, ``"3"``]

    Returns
    -------
    Dict[str, Any]
        Map containing Confusion Matrix values for each label.
    """
    cm = {}  # type: Dict[str, Any]
    for label in labels:
        cm[label] = {}
        cm[label]["tp"] = int(((pred_data == int(label)) * (gt_data == int(label))).sum())
        cm[label]["fp"] = int(((pred_data == int(label)) * (gt_data != int(label))).sum())
        cm[label]["tn"] = int(((pred_data != int(label)) * (gt_data != int(label))).sum())
        cm[label]["fn"] = int(((pred_data != int(label)) * (gt_data == int(label))).sum())

    return cm


def compute_confusion_matrix(gt_filename: str, pred_filename: str, labels: List[str],
                             include_distance_metrics: bool = True) -> Dict[str, Any]:
    r"""
    Compute the Confusion matrix given the reference file, the prediction file and a list of labels to evaluate.
    For each label, a dictionary including the amount of True Positives, True Negatives, False Positives
    and False Negatives is returned:
            .. math::
                label_i: \{ tp : True Positives_i,  tn : True Negatives_i, \\
                            fp : False Positives_i, fn : False Negatives_i \}

    Parameters
    ----------
    gt_filename : str
        Ground truth filepath. The image voxels should contain the class indices.
    pred_filename : str
        Prediction filepath. The image voxels should contain the class indices.
    labels : List[str]
        list of strings, indicating for which labels the confusion matrix is computed.
        Example: [``"1"``, ``"2"``, ``"3"``]
    include_distance_metrics : bool
        Include distance metrics in the output map. Defaults to True

    Returns
    -------
    Dict[str, Union[str, Dict]]
       Dictionary including ``gt_filename``, ``pred_filename`` and, for each label, a dictionary containing the confusion
       matrix values.
    """

    gt_data = LoadImaged(keys=['image'])({"image": gt_filename})['image'].flatten()
    pred_data = LoadImaged(keys=['image'])({"image": pred_filename})['image'].flatten()

    cm_map = _confusion_matrix(gt_data, pred_data, labels)

    cm_map["reference"] = gt_filename
    cm_map["test"] = pred_filename
    for c in labels:
        cm_map[c]["test_p"] = np.sum((gt_data == int(c)).astype(np.uint8), dtype=np.float64)
        cm_map[c]["pred_p"] = np.sum((pred_data == int(c)).astype(np.uint8), dtype=np.float64)

    if include_distance_metrics:
        compute_distance_metrics(cm_map)

    return cm_map
