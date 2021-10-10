#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import monai
from Hive.evaluation.evaluator import compute_metrics_and_save_json
from Hive.monai.engines.evaluator import Hive2DTo3DEvaluator
from Hive.monai.engines.trainer import HiveSupervisedTrainer, get_id_volume_map
from Hive.monai.transforms.model_transforms import post_processing_25D_transform, pre_processing_25D_transform
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, str2bool
from monai.data import list_data_collate
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureType,
)
from monai.transforms import (
    AddChanneld,
    NormalizeIntensityd,
    ToTensord,
    Resized,
)

DESC = dedent(
    """
    Run 2.5D Training, based on the configuration settings. 
    It is possible to set ``run-evaluation-only`` to skip the training and run the fold validation only.
    With ``compute-metrics`` it is possible to set if to compute metric scores on the predictions and save the results in
    a JSON file.
    """  # noqa: W291
)

EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file /path/to/config_file.json 
        {filename} --config-file /path/to/config_file.json --run-evaluation-only true
        {filename} --config-file /path/to/config_file.json --compute-metrics false
    """.format(  # noqa: W291
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiment settings ",
    )

    pars.add_argument(
        "--run-evaluation-only",
        type=str2bool,
        required=False,
        default="no",
        help="Run only evaluation step. Default: No ",
    )

    pars.add_argument(
        "--compute-metrics",
        type=str2bool,
        required=False,
        default="yes",
        help="Compute validation metrics and save results to JSON file. Default: Yes ",
    )
    add_verbosity_options_to_argparser(pars)
    return pars


def main():
    parser = get_arg_parser()

    args = vars(parser.parse_args())

    logger = get_logger(  # NOQA: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(args),
    )

    run_evaluation_only = args["run_evaluation_only"]
    compute_metrics = args["compute_metrics"]

    with open(args["config_file"]) as json_file:
        config_dict = json.load(json_file)

    orientations = list(config_dict["slice_size_2d"].keys())
    results_folder = config_dict["results_folder"]
    prediction_suffix = config_dict["post_processing_suffix"]
    gt_folder = str(
        Path(config_dict["base_folder"]).joinpath("Task" + config_dict["task_ID"] + "_" + config_dict["task_name"], "labelsTr")
    )
    n_folds = config_dict["n_folds"]

    for fold in range(n_folds):
        if not run_evaluation_only:
            for orientation in orientations:
                run_2d_training_for_fold_and_orientation(fold, orientation, config_dict)
        run_25d_evaluation_for_fold(fold, config_dict)
        if compute_metrics:
            compute_metrics_and_save_json(
                config_dict,
                gt_folder,
                str(Path(results_folder).joinpath("fold_{}".format(fold), "predictions")),
                prediction_suffix=prediction_suffix,
            )


if __name__ == "__main__":
    main()


def run_25d_evaluation_for_fold(fold, config_dict):
    orientations = list(config_dict["slice_size_2d"].keys())
    preprocess_folder = config_dict["preprocessing_folder"]
    results_folder = config_dict["results_folder"]

    evaluator = Hive2DTo3DEvaluator(
        str(Path(results_folder).joinpath("fold_{}".format(fold), config_dict["predictions_folder_name"])),
        config_dict,
        pre_processing_25D_transform,
        post_processing_25D_transform,
    )

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    val_filename = Path(preprocess_folder).joinpath(
        orientations[0], "data", "dataset_fold_{}_validation_{}.json".format(fold, orientations[0])
    )

    with open(val_filename) as json_val_file:
        val_files = json.load(json_val_file)["validation_3D"]

    val_ds = monai.data.Dataset(data=val_files)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    evaluator.run(val_loader)


def run_2d_training_for_fold_and_orientation(fold, orientation, config_dict):
    logger = get_logger(__name__)
    preprocess_folder = config_dict["preprocessing_folder"]
    results_folder = config_dict["results_folder"]

    try:
        n_workers = int(os.environ["N_THREADS"])
    except KeyError:
        logger.warning("N_THREADS is not set as environment variable. Using Default [1]")
        n_workers = 1

    monai.config.print_config()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_filename = str(
        Path(preprocess_folder).joinpath(orientation, "data", "dataset_fold_{}_training_{}.json".format(fold, orientation))
    )
    val_filename = str(
        Path(preprocess_folder).joinpath(orientation, "data", "dataset_fold_{}_validation_{}.json".format(fold, orientation))
    )

    fold_results_path = str(Path(results_folder).joinpath("fold_{}".format(fold), orientation))
    model_checkpoint_path = str(Path(fold_results_path).joinpath("checkpoints"))
    tb_log_path = str(
        Path(results_folder).joinpath("runs", "{}_fold_{}_{}".format(config_dict["Experiment Name"], fold, orientation))
    )
    val_key_path = str(Path(fold_results_path).joinpath("val_key_metric"))
    tb_image_log_path = str(Path(fold_results_path).joinpath("volume_checkpoints"))

    with open(train_filename) as json_train_file:
        train_files = json.load(json_train_file)["training_2D"]

    with open(val_filename) as json_val_file:
        val_files = json.load(json_val_file)["validation_2D"]

    in_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            NormalizeIntensityd(
                keys=["image"], subtrahend=config_dict["norm_stats"]["mean"], divisor=config_dict["norm_stats"]["sd"]
            ),
            Resized(keys=["image", "label"], spatial_size=config_dict["slice_size_2d"][orientation], mode=("area", "nearest")),
            ToTensord(keys=["image", "label"]),
        ]
    )

    train_ds = monai.data.Dataset(data=train_files, transform=in_transforms)
    val_ds = monai.data.Dataset(data=val_files, transform=in_transforms)

    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=config_dict["batch_size"],
        shuffle=True,
        num_workers=n_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
    )

    val_loader = monai.data.DataLoader(
        val_ds, batch_size=config_dict["batch_size"], num_workers=n_workers, collate_fn=list_data_collate
    )

    trainer = HiveSupervisedTrainer(config_dict, fold_results_path)
    trainer.set_orientation(orientation)
    trainer.prepare_trainer_event_handlers(True, model_checkpoints_path=model_checkpoint_path, tb_log_path=tb_log_path)

    post_pred = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(threshold_values=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=len(config_dict["label_dict"].keys()))])

    trainer.create_evaluator(
        val_loader, post_pred, post_label, config_dict["validation_metrics"], config_dict["validation_key_metric"]
    )

    volume_id_map = None
    if "3D_validation_idx" in config_dict:
        with open(val_filename) as json_val_file:
            val_3D_files = json.load(json_val_file)["validation_3D"]

        volume_id_map = get_id_volume_map(
            config_dict["3D_validation_idx"], val_3D_files, val_files, orientation, config_dict["FileExtension"]
        )

    trainer.prepare_evaluator_event_handlers(
        val_key_path=val_key_path, tb_image_log_path=tb_image_log_path, tb_log_path=tb_log_path, volume_id_map=volume_id_map
    )

    trainer.run_training(train_loader)
