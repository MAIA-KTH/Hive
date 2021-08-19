import copy
import json
import os
import time

import SimpleITK as sitk
import numpy as np
import pandas as pd
import plotly.express as px
from k8s_DP.utils.file_utils import subfiles
from medpy import metric
from sklearn.metrics import confusion_matrix

COMPOSED_METRICS = {
    "Specificity": {
        "Base_Metrics": ["False Positive Rate"],
        "Function": lambda x: 1 - x,
    },
    "Fowlkes–Mallows index": {"Base_Metrics": ["Precision", "Recall"], "Function": lambda x, y: np.sqrt(np.multiply(x, y))},
    "Relative Volumetric Difference": {
        "Base_Metrics": ["Total Positives Reference", "Total Positives Test"],
        "Function": lambda x, y: np.divide(np.subtract(y, x), x),
    },
    "Segmented Volume": {"Base_Metrics": ["Total Positives Test"], "Function": lambda x: x * (0.7 * 0.7 * 0.5) / 1000000},
}


def find_file_from_pattern(folder, pattern, file_extension):
    files = subfiles(folder, prefix=pattern[: -len(file_extension)], suffix=file_extension)
    return files[0]


def test_read_metric_list():
    fold = 0
    config_file = "/home/LungLobeSeg/nnUNet_results/LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json"
    with open(config_file) as config_file:
        config_dict = json.load(config_file)
    label_dict = config_dict["label_dict"]
    label_dict.pop("0", None)
    full_task_name = "Task" + config_dict["Task_ID"] + "_" + config_dict["Task_Name"]
    fold_folder = "fold_" + str(fold)
    summary_filepath = os.path.join(
        config_dict["results_folder"],
        "nnUNet",
        config_dict["TRAINING_CONFIGURATION"],
        full_task_name,
        config_dict["TRAINER_CLASS_NAME"] + "__" + config_dict["TRAINER_PLAN"],
        fold_folder,
        "validation_raw_postprocessed",
        "summary.json",
    )
    with open(summary_filepath) as json_file:
        data = json.load(json_file)
    metric_list = list(data["results"]["all"][0][list(label_dict.keys())[0]].keys())
    return metric_list


def test_read_metrics():
    pd.options.display.float_format = "{:.2}".format
    metrics = [
        "Accuracy",
        "Avg. Surface Distance",
        "Avg. Symmetric Surface Distance",
        "Dice",
        "False Discovery Rate",
        "False Negative Rate",
        "False Omission Rate",
        "False Positive Rate",
        "Hausdorff Distance",
        "Hausdorff Distance 95",
        "Jaccard",
        "Negative Predictive Value",
        "Precision",
        "Recall",
        "Total Positives Reference",
        "Total Positives Test",
        "True Negative Rate",
    ]

    additional_columns = {"test": "Prediction File", "reference": "Ground Truth File"}
    n_folds = 1  # 5
    config_file = "C:/Users/simon/Desktop/LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json"
    section = "testing"  # 'validation'
    metric_name = "Segmented Volume"
    composed_metric = None
    if metric_name not in metrics:
        if metric_name in COMPOSED_METRICS:
            composed_metric = metric_name

    with open(config_file) as config_file:
        config_dict = json.load(config_file)
    full_task_name = "Task" + config_dict["Task_ID"] + "_" + config_dict["Task_Name"]
    images_folder_path = os.path.join(config_dict["base_folder"], "nnUNet_raw_data", full_task_name, "imagesTs")  # imagesTr
    label_dict = config_dict["label_dict"]
    label_dict.pop("0", None)

    df = pd.DataFrame()
    subj_id = 0
    for fold in range(n_folds):
        fold_folder = "fold_" + str(fold)
        parent_folder = os.path.join(fold_folder, "validation_raw_postprocessed")  # "predictionsTs"
        summary_filepath = os.path.join(
            config_dict["results_folder"],
            "nnUNet",
            config_dict["TRAINING_CONFIGURATION"],
            full_task_name,
            config_dict["TRAINER_CLASS_NAME"] + "__" + config_dict["TRAINER_PLAN"],
            parent_folder,
            "summary.json",
        )

        with open(summary_filepath) as json_file:
            data = json.load(json_file)
        for i in [x for x in range(len(data["results"]["all"]))]:
            df_temp = pd.DataFrame(data["results"]["all"][i])
            column_selection = list(label_dict.keys())
            column_rename = copy.deepcopy(label_dict)
            for additional_column in list(additional_columns.keys()):
                column_selection.append(additional_column)
                column_rename[additional_column] = additional_columns[additional_column]

            if composed_metric is not None:
                base_metric = COMPOSED_METRICS[composed_metric]["Base_Metrics"][0]
                df_single_temp = (
                    df_temp[column_selection].loc[[base_metric]].rename(columns=column_rename, index={base_metric: str(subj_id)})
                )
                base_metrics = [df_temp[list(label_dict.keys())].loc[["Total Positives Test"]]]
                df_composed_temp = COMPOSED_METRICS[composed_metric]["Function"](*base_metrics)
                df_single_temp[[label_dict[key] for key in label_dict]] = df_composed_temp.values
            else:
                df_single_temp = (
                    df_temp[column_selection].loc[[metric_name]].rename(columns=column_rename, index={metric_name: str(subj_id)})
                )

            volume_file = find_file_from_pattern(
                images_folder_path, os.path.basename(df_single_temp["Ground Truth File"][0]), config_dict["FileExtension"]
            )

            if os.path.isfile(volume_file):
                df_single_temp["Volume File"] = volume_file
            else:
                df_single_temp["Volume File"] = ""

            df_single_temp["Section"] = "Testing"  # 'Fold {}'.format(fold)
            df = df.append(df_single_temp)
            subj_id = subj_id + 1

    save_stats = True
    os.makedirs(os.path.join(config_dict["results_folder"], "metrics_DF", section), exist_ok=True)
    if save_stats:
        df_aggregate = pd.DataFrame(zip(df.mean(), df.std()), columns=["Mean", "SD"], index=label_dict).rename(index=label_dict)
        df_aggregate.to_pickle(
            os.path.join(config_dict["results_folder"], "metrics_DF", section, "{}_stats.pkl".format(metric_name))
        )

    if composed_metric is not None:
        metric_name = composed_metric
    df_flat = df[[label_dict[key] for key in label_dict]].stack()
    df_flat = pd.DataFrame(df_flat)
    df_flat.reset_index(inplace=True)
    df_flat.columns = ["Subject", "Label", metric_name]

    df.to_pickle(os.path.join(config_dict["results_folder"], "metrics_DF", section, "{}_table.pkl".format(metric_name)))
    df_flat.to_pickle(os.path.join(config_dict["results_folder"], "metrics_DF", section, "{}_flat.pkl".format(metric_name)))


def test_display_plotly():
    section = "testing"
    metric_name = "Segmented Volume"
    measurement_unit = "[L]"
    results_folder = "C:/Users/simon/Desktop"
    df_flat = pd.read_pickle(os.path.join(results_folder, "metrics_DF", section, "{}_flat.pkl".format(metric_name)))
    fig = px.box(
        df_flat,
        x="Label",
        y=metric_name,
        color="Label",
        labels={
            metric_name: metric_name + " " + measurement_unit,
        },
        title="{} Set, {}".format(section.capitalize(), metric_name),
    )
    fig.show()
    fig = px.histogram(
        df_flat,
        x=metric_name,
        color="Label",
        labels={
            metric_name: metric_name + " " + measurement_unit,
        },
        title="{} Set, {}".format(section.capitalize(), metric_name),
    )
    fig.show()


def compute_distance_metrics_from_medpy(gt_filename, pred_filename, c):
    gt_itk_image = sitk.ReadImage(gt_filename)
    pred_itk_image = sitk.ReadImage(pred_filename)
    gt_data = sitk.GetArrayFromImage(gt_itk_image)
    pred_data = sitk.GetArrayFromImage(pred_itk_image)

    gt_data_one_hot = (gt_data == int(c)).astype(np.uint8)
    pred_data_one_hot = (pred_data == int(c)).astype(np.uint8)
    asd = metric.asd(pred_data_one_hot, gt_data_one_hot, np.array(gt_itk_image.GetSpacing())[::-1])
    hd95 = metric.hd95(pred_data_one_hot, gt_data_one_hot, np.array(gt_itk_image.GetSpacing())[::-1])
    hd_val = metric.hd(pred_data_one_hot, gt_data_one_hot, np.array(gt_itk_image.GetSpacing())[::-1])
    assd = metric.assd(pred_data_one_hot, gt_data_one_hot, np.array(gt_itk_image.GetSpacing())[::-1])

    dist_metrics = {
        "Avg. Surface Distance": asd,
        "Avg. Symmetric Surface Distance": assd,
        "Hausdorff Distance": hd_val,
        "Hausdorff Distance 95": hd95,
    }

    return dist_metrics


def compute_distance_metrics_from_sitk(gt_filename, pred_filename, c):
    gt_itk_image = sitk.ReadImage(gt_filename)
    pred_itk_image = sitk.ReadImage(pred_filename)
    gt_data = sitk.GetArrayFromImage(gt_itk_image)
    pred_data = sitk.GetArrayFromImage(pred_itk_image)

    gt_data_one_hot = (gt_data == int(c)).astype(np.uint8)
    pred_data_one_hot = (pred_data == int(c)).astype(np.uint8)

    gt_one_hot_itk_image = sitk.GetImageFromArray(gt_data_one_hot)
    gt_one_hot_itk_image.CopyInformation(gt_itk_image)
    pred_one_hot_itk_image = sitk.GetImageFromArray(pred_data_one_hot)
    pred_one_hot_itk_image.CopyInformation(pred_itk_image)

    signed_distance_map = sitk.SignedMaurerDistanceMap(
        gt_one_hot_itk_image > 0.5, squaredDistance=False, useImageSpacing=True
    )  # It need to be adapted.
    ref_distance_map = sitk.Abs(signed_distance_map)
    ref_surface = sitk.LabelContour(gt_one_hot_itk_image > 0.5, fullyConnected=True)
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(ref_surface > 0.5)
    num_ref_surface_pixels = int(statistics_image_filter.GetSum())

    signed_distance_map_pred = sitk.SignedMaurerDistanceMap(
        pred_one_hot_itk_image > 0.5, squaredDistance=False, useImageSpacing=True
    )
    seg_distance_map = sitk.Abs(signed_distance_map_pred)
    seg_surface = sitk.LabelContour(pred_one_hot_itk_image > 0.5, fullyConnected=True)
    statistics_image_filter.Execute(seg_surface > 0.5)
    num_seg_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distance_map = ref_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)
    ref2seg_distance_map = seg_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_seg_surface_pixels - len(seg2ref_distances)))

    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_ref_surface_pixels - len(ref2seg_distances)))  #

    all_surface_distances = seg2ref_distances + ref2seg_distances
    hd_val = np.max(all_surface_distances)
    hd95 = np.quantile(all_surface_distances, 0.95)
    assd = np.mean(all_surface_distances)
    asd = np.mean(seg2ref_distances)

    dist_metrics = {
        "Avg. Surface Distance": asd,
        "Avg. Symmetric Surface Distance": assd,
        "Hausdorff Distance": hd_val,
        "Hausdorff Distance 95": hd95,
    }

    return dist_metrics


def compute_confusion_matrix(gt_filename, pred_filename, classes):
    gt_itk_image = sitk.ReadImage(gt_filename)
    pred_itk_image = sitk.ReadImage(pred_filename)
    gt_data = sitk.GetArrayFromImage(gt_itk_image)
    pred_data = sitk.GetArrayFromImage(pred_itk_image)

    cm = confusion_matrix(gt_data.flatten(), pred_data.flatten())
    pop = np.sum(cm)
    cm_class_map = {"reference": gt_filename, "test": pred_filename}

    for c in classes:
        cm_single_class_map = {}
        tp = cm[int(c), int(c)]
        fp = np.sum(cm[:, int(c)]) - tp
        fn = np.sum(cm[int(c), :]) - tp
        tn = pop - (fn + tp + fp)
        cm_single_class_map["tp"] = tp
        cm_single_class_map["fp"] = fp
        cm_single_class_map["fn"] = fn
        cm_single_class_map["tn"] = tn
        cm_class_map[c] = cm_single_class_map
    return cm_class_map


def test_compare_distance_metrics_fn():
    gt_filename = (
        "/home/LungLobeSeg/nnUNet_results/nnUNet/3d_fullres/Task100_LungLobeSeg_3D_Single_Modality"
        "/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis"
        "/1_3_6_1_4_1_14519_5_2_1_6279_6001_146429221666426688999739595820.nii.gz"
    )
    pred_filename = (
        "/home/LungLobeSeg/nnUNet_results/nnUNet/3d_fullres/Task100_LungLobeSeg_3D_Single_Modality"
        "/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw_postprocessed"
        "/1_3_6_1_4_1_14519_5_2_1_6279_6001_146429221666426688999739595820.nii.gz"
    )
    c = "1"

    start = time.time()
    dist_metrics_medpy = compute_distance_metrics_from_medpy(gt_filename, pred_filename, c)
    end = time.time()
    exec_time_medpy = end - start
    start = time.time()
    dist_metrics_sitk = compute_distance_metrics_from_sitk(gt_filename, pred_filename, c)
    end = time.time()
    exec_time_sitk = end - start
    print("Execution time: ", exec_time_medpy, exec_time_sitk)
    print("HD: ", dist_metrics_medpy["Hausdorff Distance"], dist_metrics_sitk["Hausdorff Distance"])
    print("HD95: ", dist_metrics_medpy["Hausdorff Distance 95"], dist_metrics_sitk["Hausdorff Distance 95"])
    print("ASD: ", dist_metrics_medpy["Avg. Surface Distance"], dist_metrics_sitk["Avg. Surface Distance"])
    print(
        "ASSD: ", dist_metrics_medpy["Avg. Symmetric Surface Distance"], dist_metrics_sitk["Avg. Symmetric Surface " "Distance"]
    )


def accuracy(cm_map):
    accuracy_map = {}
    for c in list(cm_map.keys()):
        try:
            acc = (cm_map[c]["tp"] + cm_map[c]["tn"]) / (cm_map[c]["tp"] + cm_map[c]["tn"] + cm_map[c]["fp"] + cm_map[c]["fn"])
            accuracy_map[c] = acc
        except TypeError:
            continue
    return accuracy_map


def dice(cm_map):
    dice_map = {}
    for c in list(cm_map.keys()):
        try:
            dice_val = (2 * cm_map[c]["tp"]) / (2 * cm_map[c]["tp"] + cm_map[c]["fp"] + cm_map[c]["fn"])
            dice_map[c] = dice_val
        except TypeError:
            continue
    return dice_map


def jaccard(cm_map):
    jaccard_map = {}
    for c in list(cm_map.keys()):
        try:
            jac = (cm_map[c]["tp"]) / (cm_map[c]["tp"] + cm_map[c]["fp"] + cm_map[c]["fn"])
            jaccard_map[c] = jac
        except TypeError:
            continue
    return jaccard_map


def precision(cm_map):
    precision_map = {}
    for c in list(cm_map.keys()):
        try:
            prec = (cm_map[c]["tp"]) / (cm_map[c]["tp"] + cm_map[c]["fp"])
            precision_map[c] = prec
        except TypeError:
            continue
    return precision_map


def recall(cm_map):
    recall_map = {}
    for c in list(cm_map.keys()):
        try:
            rec = (cm_map[c]["tp"]) / (cm_map[c]["tp"] + cm_map[c]["fn"])
            recall_map[c] = rec
        except TypeError:
            continue
    return recall_map


def fpr(cm_map):
    fpr_map = {}
    for c in list(cm_map.keys()):
        try:
            fpr_val = (cm_map[c]["fp"]) / (cm_map[c]["tn"] + cm_map[c]["fp"])
            fpr_map[c] = fpr_val
        except TypeError:
            continue
    return fpr_map


def hd(cm_map):
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


def test_compute_metrics_sitk():
    gt_filename = (
        "/home/LungLobeSeg/nnUNet_results/nnUNet/3d_fullres/Task100_LungLobeSeg_3D_Single_Modality"
        "/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis"
        "/1_3_6_1_4_1_14519_5_2_1_6279_6001_146429221666426688999739595820.nii.gz"
    )
    pred_filename = (
        "/home/LungLobeSeg/nnUNet_results/nnUNet/3d_fullres/Task100_LungLobeSeg_3D_Single_Modality"
        "/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw_postprocessed"
        "/1_3_6_1_4_1_14519_5_2_1_6279_6001_146429221666426688999739595820.nii.gz"
    )
    classes = ["1", "2", "3", "4", "5"]

    cm_class_map = compute_confusion_matrix(gt_filename, pred_filename, classes)

    metrics_map = {"Accuracy": accuracy, "Dice": dice}

    distance_metrics = ["Hausdorff Distance", "Hausdorff Distance 95", "Avg. Surface Distance", "Avg. Symmetric Surface Distance"]

    metrics_dict = {}

    metrics_list = ["Accuracy", "Dice", "Hausdorff Distance"]
    required_distance_metrics = [i for i in distance_metrics if i in metrics_list]

    for c in classes:
        metrics_dict[c] = {}

        if len(required_distance_metrics) > 0:
            dist_metrics = compute_distance_metrics_from_sitk(gt_filename, pred_filename, c)

            for distance_metric in required_distance_metrics:
                metrics_dict[c][distance_metric] = dist_metrics[distance_metric]

    for metric_name in metrics_list:
        if metric_name in metrics_map:
            metric_result = metrics_map[metric_name](cm_class_map)
            for c in classes:
                metrics_dict[c][metric_name] = metric_result[c]
    metrics_dict["reference"] = gt_filename
    metrics_dict["test"] = pred_filename

    return metrics_dict


def test_compute_metrics_medpy():
    gt_filename = (
        "/home/LungLobeSeg/nnUNet_results/nnUNet/3d_fullres/Task100_LungLobeSeg_3D_Single_Modality"
        "/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis"
        "/1_3_6_1_4_1_14519_5_2_1_6279_6001_146429221666426688999739595820.nii.gz"
    )
    pred_filename = (
        "/home/LungLobeSeg/nnUNet_results/nnUNet/3d_fullres/Task100_LungLobeSeg_3D_Single_Modality"
        "/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw_postprocessed"
        "/1_3_6_1_4_1_14519_5_2_1_6279_6001_146429221666426688999739595820.nii.gz"
    )
    classes = ["1"]

    cm_class_map = compute_confusion_matrix(gt_filename, pred_filename, classes)

    metrics_map = {"Accuracy": accuracy, "Dice": dice, "Hausdorff Distance": hd}

    metrics_dict = {}

    metrics_list = ["Accuracy", "Dice", "Hausdorff Distance"]

    for c in classes:
        metrics_dict[c] = {}

    for metric_name in metrics_list:
        if metric_name in metrics_map:
            metric_result = metrics_map[metric_name](cm_class_map)
            for c in classes:
                metrics_dict[c][metric_name] = metric_result[c]
    metrics_dict["reference"] = gt_filename
    metrics_dict["test"] = pred_filename

    return metrics_dict
