import datetime
import importlib
import json
import logging
import math
from os import PathLike
from pathlib import Path
from typing import Dict, Any, List, Union, Callable

import monai.networks.utils
import torch
import torch.nn as nn
import torch.optim
from Hive.monai.config.default_metrics import val_metrics
from Hive.utils.file_utils import subfiles
from ignite.engine import _prepare_batch, Engine
from ignite.engine.events import State
from ignite.metrics.confusion_matrix import ConfusionMatrix
from monai.handlers import CheckpointLoader
from torch.utils.tensorboard import SummaryWriter


def save_final_state_summary(results_folder_path: Union[str, PathLike], trainer_state: State, evaluator: Engine):
    """
    Save a JSON training summary, containing information from the trainer and the evaluator engine.

    Parameters
    ----------
    results_folder_path : Union[str, PathLike]
        Result folder path where to save the JSON summary.
    trainer_state : State
        Ignite Engine state for the trainer.
    evaluator : Engine
        Ignite Engine for the evaluator.
    """
    summary = {
        "Epochs": trainer_state.epoch,
        "Max Epochs": trainer_state.max_epochs,
        "Training Time": str(datetime.timedelta(seconds=trainer_state.times["COMPLETED"])).split(sep=".")[0],
        "Validation_key_metric_series": evaluator.state.val_key_metric_list,
        "Validation Metrics": state_metrics_to_dict(evaluator),
    }

    with open(Path(results_folder_path).joinpath("training_summary.json"), "w") as fp:
        json.dump(summary, fp)


def prepare_batch(batch: Dict[str, Any], device: str = None, non_blocking: bool = False):
    return _prepare_batch((batch["image"], batch["label"]), device, non_blocking)


def reduce_y_label(x: List[List[torch.Tensor]]) -> (List[torch.Tensor], List[torch.Tensor]):
    """
    Reduces y_label along the channel dimension, from One-Hot representation to a class label format.
    Accepts as input [``y_pred``, ``y_label``], where *y_pred* and *y_label*  are in **BCHW[D]** format.
    Parameters
    ----------
    x : List[List[torch.Tensor]]
        List of [``y_pred``, ``y_label``], where *y_pred* and *y_label*  are in **BCHW[D]** format.

    Returns
    -------
    (List[torch.Tensor], List[torch.Tensor])
        (y_pred, y_label), with y_label reduced along channel dimension.
    """
    y_label = x[1]
    y_pred = x[0]
    y = [torch.argmax(y_single, dim=0) for y_single in y_label]

    return y_pred, y


def binary_one_hot_and_reduce_label(x: List[List[torch.Tensor]]) -> (List[torch.Tensor], List[torch.Tensor]):
    """
    Reduces y_label along the channel dimension, from One-Hot representation to a class label format.
    y_pred is expected to be in binary format, thus it is converted in One-Hot format (2 classes).
    Accepts as input [``y_pred``, ``y_label``], where *y_pred* and *y_label*  are in **B1HW[D]** format.

    Parameters
    ----------
    x : List[List[torch.Tensor]]
        List of [``y_pred``, ``y_label``], where *y_pred* and *y_label*  are in **B1HW[D]** format.

    Returns
    -------
    (List[torch.Tensor], List[torch.Tensor])
        (y_pred, y_label), with y_label reduced along channel dimension and y_pred is in One-Hot format (from binary).
    """
    y_label = x[1]
    y_pred = x[0]

    y_pred = [monai.networks.utils.one_hot(y_pred_single, 2, dim=0) for y_pred_single in y_pred]
    y = [torch.squeeze(y_single, dim=0).long() for y_single in y_label]

    return y_pred, y


def create_validation_metric_dict(val_metric_list: List[str], n_classes: int) -> Dict[str, Callable]:
    """
    Creates and return a dictionary of Callable metric functions, to be attached to the Evaluator engine for evaluation.

    Parameters
    ----------
    val_metric_list : List[str]
        List of metrics to include
    n_classes : int
        Number of output classes, to be used when computing Confusion Matrix.

    Returns
    -------
    Dict[str, Callable]
        Dictionary of Callable metric functions, to be attached to the Evaluator engine.
    """
    validation_metrics = {}

    for val_metric in val_metric_list:
        metric_name = val_metric
        val_metric = val_metrics[val_metric]
        super_class = importlib.import_module(val_metric["class_import"])
        if hasattr(super_class, val_metric["class_name"]):

            if not val_metric.get("class_params"):
                val_metric["class_params"] = {}

            if "reduce_y_label" in val_metric and val_metric["reduce_y_label"]:
                if n_classes == 2:
                    val_metric["class_params"]["output_transform"] = binary_one_hot_and_reduce_label
                else:
                    val_metric["class_params"]["output_transform"] = reduce_y_label

            if val_metric["class_import"] == "ignite.metrics.confusion_matrix":
                if n_classes == 2:
                    cm = ConfusionMatrix(num_classes=n_classes, output_transform=binary_one_hot_and_reduce_label)
                else:
                    cm = ConfusionMatrix(num_classes=n_classes, output_transform=reduce_y_label)
                val_metric["class_params"]["cm"] = cm

            validation_metrics[metric_name] = getattr(super_class, val_metric["class_name"])(**val_metric["class_params"])

    return validation_metrics


def epoch_score_function(engine: Engine) -> float:
    """
    Returns the Epoch key metric score:
        .. math::
            Score[epcoh_i] = \frac{biased\_score[epoch_i]}{1-\alpha^{epoch_i}}
    Where:
        .. math::
            biased_score[epoch_i] = \alpha*biased_score [epoch_{i-1}]+(1-\alpha)*key\_metric\_score
    Parameters
    ----------
    engine: Engine
        Ignite Engine for the trainer.

    Returns
    -------
    float
        Epoch key metric score.
    """  # noqa: W605
    if engine.state.val_key_metric_biased is None:
        engine.state.val_key_metric_biased = 0

    engine.state.val_key_metric_biased = (engine.state.val_key_metric_alpha * engine.state.val_key_metric_biased) + (
        1 - engine.state.val_key_metric_alpha
    ) * engine.state.metrics[engine.state.key_metric]
    bias_weight = 1
    if engine.state.val_key_metric_alpha < 1:
        bias_weight = 1 - math.pow(engine.state.val_key_metric_alpha, engine.state.trainer.state.epoch)

    engine.state.val_key_metric_list.append(engine.state.val_key_metric_biased / bias_weight)
    return engine.state.val_key_metric_biased / bias_weight


def epoch_writer(engine: Engine, writer: SummaryWriter):
    """
    Execute epoch level event write operation based on Ignite engine.state data.
    Default is to write the values from Ignite state.metrics dict.

    Parameters
    ----------
    engine : Engine
        Ignite Engine, it can be a trainer, validator or evaluator.
    writer : SummaryWriter
        TensorBoard writer, created in TensorBoardHandler.
    """
    current_epoch = engine.state.trainer.state.epoch
    summary_dict = engine.state.metrics
    label_dict = engine.state.label_dict

    for name, value in summary_dict.items():

        if type(value) is torch.Tensor:
            for index, class_value in enumerate(value):
                if index == 0:
                    continue
                writer.add_scalar(name + "_{}".format(label_dict[str(index)]), value[index], current_epoch)
        else:
            writer.add_scalar(name, value, current_epoch)
    writer.flush()


def state_metrics_to_string(engine: Engine) -> str:
    """
    Convert values from engine.state.metrics as a string.

    Parameters
    ----------
    engine: Engine
        Ignite Engine.
    Returns
    -------
    str
        String containing Trainer state metrics info.
    """
    prints_dict = engine.state.metrics

    label_dict = engine.state.label_dict

    out_str = ""
    for name in sorted(prints_dict):
        value = prints_dict[name]
        if isinstance(value, torch.Tensor):
            for class_id in range(len(value)):
                if class_id == 0:
                    continue
                out_str += "{}: {:.4f} \n".format("{}_{}".format(name, label_dict[str(class_id)]), value[class_id])
        else:
            out_str += "{}: {:.4f} \n".format(name, value)

    return out_str


def state_metrics_to_dict(engine: Engine) -> Dict[str, float]:
    """
    Convert values from engine.state.metrics as a dictionary of float values, indicating the metric scores.

    Parameters
    ----------
     engine: Engine
        Ignite Engine.
    Returns
    -------
    Dict[str, float]
        Dictionary containing Trainer state metric scores as values.

    """
    prints_dict = engine.state.metrics

    label_dict = engine.state.label_dict

    out_dict = {}
    for name in sorted(prints_dict):
        value = prints_dict[name]
        if isinstance(value, torch.Tensor):
            for class_id in range(len(value)):
                if class_id == 0:
                    continue
                out_dict["{}_{}".format(name, label_dict[str(class_id)])] = float(value[class_id])
        else:
            out_dict[name] = value

    return out_dict


def epoch_printer(engine: Engine):
    """
    Callable to print epoch metric scores at info level on the "evaluator" logger.

    Parameters
    ----------
    engine : Engine
        Ignite Engine.
    """
    current_epoch = engine.state.trainer.state.epoch
    out_str = f"Epoch[{current_epoch}] Metrics \n"
    out_str += state_metrics_to_string(engine)
    logging.getLogger("evaluator").info(out_str)


def reload_checkpoint(checkpoint_folder: Union[str, PathLike], net: nn.Module, opt: torch.optim, trainer: Engine) -> int:
    """
    Callable to be attached to an Ignite engine to reload any saved checkpoint in the checkpoint folder.
    Returns the epoch number from where to restore the state.

    Parameters
    ----------
    checkpoint_folder : Union[str, PathLike]
        Folder where to search checkpoint files.
    net : nn.Module
        Network to restore the state.
    opt : torch.optim
        Optimizer to restore the state.
    trainer: Engine
        Ignite engine to attach in order to call the reloading function.

    Returns
    -------
    int
        Epoch number from where the state is restored.
    """
    resume_epoch = 0
    prefix = "net_checkpoint_"
    suffix = ".pt"
    checkpoint_files = subfiles(checkpoint_folder, join=False, prefix=prefix, suffix=suffix)
    if len(checkpoint_files) > 0:
        for checkpoint_file in checkpoint_files:
            temp_resume_epoch = int(checkpoint_file.replace(prefix, "").replace(suffix, ""))
            if temp_resume_epoch > resume_epoch:
                resume_epoch = temp_resume_epoch
        checkpoint_filepath = str(Path(checkpoint_folder).joinpath(prefix + "{}".format(resume_epoch) + suffix))

        CheckpointLoader(load_path=checkpoint_filepath, load_dict={"net": net, "opt": opt}).attach(trainer)

    return resume_epoch
