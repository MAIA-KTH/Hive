#!/usr/bin/env python

import json
import pickle
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from Hive.evaluation.detection.coco import COCOMetric
from Hive.evaluation.detection.froc import FROCMetric
from Hive.evaluation.detection.hist import PredictionHistogram
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args

DESC = dedent(
    """
    Script to perform Object Detection (COCO and FROC Metrics) and Segmentation (Dice score) evaluation on nnDetection experiments.
    
    By specifying  the ``class-file`` and ``classes`` parameters, the class-wise analysis of the metrics is performed.
    
    An Excel Spreadsheet and several PNG plots (representing FROC curves and Histogram Analysis), are produced as output.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file /PATH/TO/CONFIG_FILE.json --output-dir /OUTPUT/PATH
        {filename} --config-file /PATH/TO/CONFIG_FILE.json --output-dir /OUTPUT/PATH --n-fold 0
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)


def iou_filter(image_dict: Dict[int, Dict[str, np.ndarray]], iou_idx: List[int],
               filter_keys: Sequence[str] = ('dtMatches', 'gtMatches', 'dtIgnore')):
    """
    This functions can be used to filter specific IoU values from the results
    to make sure that the correct IoUs are passed to metric

    Parameters
    ----------
    image_dict : dict
        dictionary containin :param:`filter_keys` which contains IoUs in the first dimension
    iou_idx : List[int]
        indices of IoU values to filter from keys
    filter_keys : tuple, optional
        keys to filter, by default ('dtMatches', 'gtMatches', 'dtIgnore')

    Returns
    -------
    dict
        filtered dictionary
    """
    iou_idx = list(iou_idx)
    filtered = {}
    for cls_key, cls_item in image_dict.items():
        filtered[cls_key] = {key: item[iou_idx] if key in filter_keys else item
                             for key, item in cls_item.items()}
    return filtered


def get_unique_iou_thresholds(metric):
    """
    Compute unique set of iou thresholds
    """
    iou_thresholds = [_i for _i in metric.get_iou_thresholds()]
    iou_thresholds = list(set(iou_thresholds))
    iou_thresholds.sort()
    return iou_thresholds


def get_indices_of_iou(metric):
    """
    Find indices of iou thresholds for each metric
    """

    return [get_unique_iou_thresholds(metric).index(th) for th in metric.get_iou_thresholds()]


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiments variables.",
    )

    pars.add_argument(
        "--run-fold",
        type=str,
        default="-1",
        required=False,
        help="Index indicating which fold to run. Default: ``0``. If set to ``-1``, runs the metric evaluation on the "
             "consolidated predictions.",
    )

    pars.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Folder path where to save the Excel spreadsheet and the PNG plots containing the evaluation metrics.",
    )

    pars.add_argument(
        "--class-file",
        type=str,
        required=False,
        help="Optional JSON file, containing a dictionary where each Subject ID is stored with the corresponding "
             "Subject Class."
             "If not specified, all the subjects will be analysed.",
    )

    pars.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=False,
        help="Optional list of Subject Classes, to be considered for the class-wise result analysis.",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()

    arguments, unknown_arguments = parser.parse_known_args()
    args = vars(arguments)

    logger = get_logger(  # NOQA: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(args),
    )

    config_file = args["config_file"]

    with open(config_file) as json_file:
        data = json.load(json_file)

    output_dir = args["output_dir"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args["class_file"] is not None and args["classes"] is not None:
        class_file = args["class_file"]
        patient_classes = args["classes"]
        with open(class_file, "rb") as file:
            patients_classes_dict = json.load(file)
    else:
        patient_classes = None

    if args["run_fold"] != "-1":
        results_folder = Path(data["results_folder"]).joinpath("Task{}_{}".format(data["Task_ID"], data["Task_Name"]),
                                                               "RetinaUNetV001_D3V001_3d",
                                                               "fold{}".format(args["run_fold"]),
                                                               "val_results"
                                                               )

    else:
        results_folder = Path(data["results_folder"]).joinpath(
            "Task{}_{}".format(data["Task_ID"], data["Task_Name"]),
            "RetinaUNetV001_D3V001_3d",
            "consolidated",
            "val_results"
        )

    boxes_metrics_file = str(Path(results_folder).joinpath("results_boxes_per_case.pkl"))

    boxes_ids_file = str(Path(results_folder).joinpath("results_boxes_per_case_IDs.json"))

    seg_file = str(Path(results_folder).joinpath("results_seg_per_case.json"))

    with open(seg_file, "rb") as file:
        patients_seg_dict = json.load(file)

    df_seg = pd.DataFrame()
    for patient in patients_seg_dict:
        df_seg = df_seg.append({"Subject": patient, "Metric": "Dice", "Score": patients_seg_dict[patient]},
                               ignore_index=True)

    with open(boxes_metrics_file, "rb") as f:
        boxes_metrics = pickle.load(f)
    with open(boxes_ids_file, "rb") as file:
        boxes_ids = json.load(file)

    iou_thresholds = np.arange(0.1, 1.0, 0.1)
    iou_range = (0.1, 0.5, 0.05)
    per_class = True
    verbose = False
    classes = [label for label in data["label_dict"]]

    coco = COCOMetric(classes,
                      iou_list=iou_thresholds,
                      iou_range=iou_range,
                      max_detection=(100,),
                      per_class=per_class,
                      verbose=verbose,
                      )

    writer = pd.ExcelWriter(
        Path(output_dir).joinpath("{}.xlsx".format("Task{}_{}".format(data["Task_ID"], data["Task_Name"]))),
        engine='openpyxl')

    if patient_classes is None:
        froc = FROCMetric(classes,
                          iou_thresholds=iou_thresholds,
                          fpi_thresholds=(1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8),
                          per_class=per_class,
                          verbose=verbose,
                          save_dir=Path(output_dir)
                          )

        histo = PredictionHistogram(classes=classes,
                                    save_dir=Path(output_dir),
                                    iou_thresholds=(0.1, 0.5),
                                    )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filtered_boxes_metrics = []
        for boxes_metric, id in zip(boxes_metrics, boxes_ids):
            filtered_boxes_metrics.append(boxes_metric)

        iou_filtered_results = [iou_filter(boxes_metric, iou_idx=get_indices_of_iou(froc)) for boxes_metric in
                                filtered_boxes_metrics]
        scores, curves = froc.compute(iou_filtered_results)

        df = pd.DataFrame()
        for score in scores:
            df = df.append({"Metric": score, "Score": scores[score]}, ignore_index=True)
        iou_filtered_results = [iou_filter(boxes_metric, iou_idx=get_indices_of_iou(coco)) for boxes_metric in
                                filtered_boxes_metrics]
        scores, _ = coco.compute(iou_filtered_results)
        for score in scores:
            df = df.append({"Metric": score, "Score": scores[score]}, ignore_index=True)

        df.to_excel(writer, sheet_name="Object Detection")

        iou_filtered_results = [iou_filter(boxes_metric, iou_idx=get_indices_of_iou(histo)) for boxes_metric in
                                filtered_boxes_metrics]
        _, _ = histo.compute(iou_filtered_results)
    else:
        for class_name in patient_classes:

            froc = FROCMetric(classes,
                              iou_thresholds=iou_thresholds,
                              fpi_thresholds=(1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8),
                              per_class=per_class,
                              verbose=verbose,
                              save_dir=Path(output_dir).joinpath(class_name)
                              )

            histo = PredictionHistogram(classes=classes,
                                        save_dir=Path(output_dir).joinpath(class_name),
                                        iou_thresholds=(0.1, 0.5),
                                        )

            Path(output_dir).joinpath(class_name).mkdir(parents=True, exist_ok=True)
            filtered_boxes_metrics = []
            for boxes_metric, id in zip(boxes_metrics, boxes_ids):
                if patients_classes_dict[id] == class_name:
                    filtered_boxes_metrics.append(boxes_metric)
                # if class_name == "POSITIVE":
                #    filtered_boxes_metrics.append(boxes_metric)

            iou_filtered_results = [iou_filter(boxes_metric, iou_idx=get_indices_of_iou(froc)) for boxes_metric in
                                    filtered_boxes_metrics]
            scores, curves = froc.compute(iou_filtered_results)

            df = pd.DataFrame()
            for score in scores:
                df = df.append({"Metric": score, "Score": scores[score]}, ignore_index=True)
            iou_filtered_results = [iou_filter(boxes_metric, iou_idx=get_indices_of_iou(coco)) for boxes_metric in
                                    filtered_boxes_metrics]
            scores, _ = coco.compute(iou_filtered_results)
            for score in scores:
                df = df.append({"Metric": score, "Score": scores[score]}, ignore_index=True)

            df.to_excel(writer, sheet_name="Object Detection-{}".format(class_name))

            iou_filtered_results = [iou_filter(boxes_metric, iou_idx=get_indices_of_iou(histo)) for boxes_metric in
                                    filtered_boxes_metrics]
            _, _ = histo.compute(iou_filtered_results)

    df_seg.to_excel(writer, sheet_name="Segmentation")

    writer.close()


if __name__ == "__main__":
    main()
