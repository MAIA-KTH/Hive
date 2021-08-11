from typing import List, Dict, Union

import SimpleITK as sitk
import numpy as np
from medpy import metric
from sklearn.metrics import confusion_matrix


def accuracy(cm_map: Dict[str, Dict[str, float]]) -> Dict[str, float]:
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
                        cm_map[c]["tp"] + cm_map[c]["tn"] + cm_map[c]["fp"] + cm_map[c]["fn"])
            accuracy_map[c] = acc
        except TypeError:
            continue
    return accuracy_map


def dice(cm_map: Dict[str, Dict[str, float]]) -> Dict[str, float]:
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
            dice_val = (2 * cm_map[c]["tp"]) / (2 * cm_map[c]["tp"] + cm_map[c]["fp"] + cm_map[c]["fn"])
            dice_map[c] = dice_val
        except TypeError:
            continue
    return dice_map


def jaccard(cm_map: Dict[str, Dict[str, float]]) -> Dict[str, float]:
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
            jac = (cm_map[c]["tp"]) / (cm_map[c]["tp"] + cm_map[c]["fp"] + cm_map[c]["fn"])
            jaccard_map[c] = jac
        except TypeError:
            continue
    return jaccard_map


def precision(cm_map: Dict[str, Dict[str, float]]) -> Dict[str, float]:
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
            prec = (cm_map[c]["tp"]) / (cm_map[c]["tp"] + cm_map[c]["fp"])
            precision_map[c] = prec
        except TypeError:
            continue
    return precision_map


def recall(cm_map: Dict[str, Dict[str, float]]) -> Dict[str, float]:
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
            rec = (cm_map[c]["tp"]) / (cm_map[c]["tp"] + cm_map[c]["fn"])
            recall_map[c] = rec
        except TypeError:
            continue
    return recall_map


def fpr(cm_map: Dict[str, Dict[str, float]]) -> Dict[str, float]:
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
            fpr_val = (cm_map[c]["fp"]) / (cm_map[c]["tn"] + cm_map[c]["fp"])
            fpr_map[c] = fpr_val
        except TypeError:
            continue
    return fpr_map


def fomr(cm_map: Dict[str, Dict[str, float]]) -> Dict[str, float]:
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
            fomr_val = (cm_map[c]["fn"]) / (cm_map[c]["tn"] + cm_map[c]["fn"])
            fomr_map[c] = fomr_val
        except TypeError:
            continue
    return fomr_map


def hd(cm_map: Dict[str, Union[str, Dict[str, float]]]) -> Dict[str, float]:
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
    gt_itk_image = sitk.ReadImage(cm_map["reference"])
    pred_itk_image = sitk.ReadImage(cm_map["test"])
    gt_data = sitk.GetArrayFromImage(gt_itk_image)
    pred_data = sitk.GetArrayFromImage(pred_itk_image)
    hd_map = {}
    for c in list(cm_map.keys()):
        try:
            gt_data_one_hot = (gt_data == int(c)).astype(np.uint8)
            pred_data_one_hot = (pred_data == int(c)).astype(np.uint8)
            hd_val = metric.hd(pred_data_one_hot, gt_data_one_hot, np.array(gt_itk_image.GetSpacing())[::-1])
            hd_map[c] = hd_val
        except ValueError:
            continue
    return hd_map


def hd95(cm_map: Dict[str, Union[str, Dict[str, float]]]) -> Dict[str, float]:
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
    gt_itk_image = sitk.ReadImage(cm_map["reference"])
    pred_itk_image = sitk.ReadImage(cm_map["test"])
    gt_data = sitk.GetArrayFromImage(gt_itk_image)
    pred_data = sitk.GetArrayFromImage(pred_itk_image)
    hd95_map = {}
    for c in list(cm_map.keys()):
        try:
            gt_data_one_hot = (gt_data == int(c)).astype(np.uint8)
            pred_data_one_hot = (pred_data == int(c)).astype(np.uint8)
            hd95_val = metric.hd95(pred_data_one_hot, gt_data_one_hot, np.array(gt_itk_image.GetSpacing())[::-1])
            hd95_map[c] = hd95_val
        except ValueError:
            continue
    return hd95_map


def asd(cm_map: Dict[str, Union[str, Dict[str, float]]]) -> Dict[str, float]:
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
    gt_itk_image = sitk.ReadImage(cm_map["reference"])
    pred_itk_image = sitk.ReadImage(cm_map["test"])
    gt_data = sitk.GetArrayFromImage(gt_itk_image)
    pred_data = sitk.GetArrayFromImage(pred_itk_image)
    asd_map = {}
    for c in list(cm_map.keys()):
        try:
            gt_data_one_hot = (gt_data == int(c)).astype(np.uint8)
            pred_data_one_hot = (pred_data == int(c)).astype(np.uint8)
            asd_val = metric.asd(pred_data_one_hot, gt_data_one_hot, np.array(gt_itk_image.GetSpacing())[::-1])
            asd_map[c] = asd_val
        except ValueError:
            continue
    return asd_map


def assd(cm_map: Dict[str, Union[str, Dict[str, float]]]) -> Dict[str, float]:
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
    gt_itk_image = sitk.ReadImage(cm_map["reference"])
    pred_itk_image = sitk.ReadImage(cm_map["test"])
    gt_data = sitk.GetArrayFromImage(gt_itk_image)
    pred_data = sitk.GetArrayFromImage(pred_itk_image)
    assd_map = {}
    for c in list(cm_map.keys()):
        try:
            gt_data_one_hot = (gt_data == int(c)).astype(np.uint8)
            pred_data_one_hot = (pred_data == int(c)).astype(np.uint8)
            assd_val = metric.assd(pred_data_one_hot, gt_data_one_hot, np.array(gt_itk_image.GetSpacing())[::-1])
            assd_map[c] = assd_val
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


def compute_confusion_matrix(gt_filename: str, pred_filename: str, labels: List[str]) -> Dict[str, Union[str, Dict]]:
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

    Returns
    -------
    Dict[str, Union[str, Dict]]
       Dictionary including ``gt_filename``, ``pred_filename`` and, for each label, a dictionary containing the confusion
       matrix values.
    """

    gt_itk_image = sitk.ReadImage(gt_filename)
    pred_itk_image = sitk.ReadImage(pred_filename)
    gt_data = sitk.GetArrayFromImage(gt_itk_image)
    pred_data = sitk.GetArrayFromImage(pred_itk_image)

    cm = confusion_matrix(gt_data.flatten(), pred_data.flatten())
    pop = np.sum(cm)
    cm_class_map = {"reference": gt_filename, "test": pred_filename}  # type: Dict[str, Union[str, Dict]]

    for c in labels:
        cm_single_class_map = {}
        tp = cm[int(c), int(c)]
        fp = np.sum(cm[:, int(c)]) - tp
        fn = np.sum(cm[int(c), :]) - tp
        tn = pop - (fn + tp + fp)
        cm_single_class_map["tp"] = tp
        cm_single_class_map["fp"] = fp
        cm_single_class_map["fn"] = fn
        cm_single_class_map["tn"] = tn
        cm_single_class_map["test_p"] = np.sum((gt_data.flatten() == int(c)).astype(np.uint8), dtype=np.float64)
        cm_single_class_map["pred_p"] = np.sum((pred_data.flatten() == int(c)).astype(np.uint8), dtype=np.float64)
        cm_class_map[c] = cm_single_class_map
    return cm_class_map
