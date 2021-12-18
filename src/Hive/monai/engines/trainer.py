import importlib
import logging
import pickle
from os import PathLike
from pathlib import Path
from typing import Dict, Any, Union, List

import monai
import numpy as np
import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import create_supervised_trainer, Events, create_supervised_evaluator, Engine
from ignite.handlers import ModelCheckpoint
from monai.data import decollate_batch
from monai.handlers import TensorBoardStatsHandler, StatsHandler, LrScheduleHandler
from monai.transforms import LoadImaged, Compose, Transform
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Hive.monai.engines.utils import (
    prepare_batch,
    create_validation_metric_dict,
    epoch_writer,
    epoch_printer,
    epoch_score_function,
    reload_checkpoint,
    save_final_state_summary,
)
from Hive.monai.handlers import Hive2Dto3DTensorBoardImageHandler
from Hive.monai.transforms import OrientToRAId
from Hive.monai.utils.email_utils import send_email


def get_id_volume_map(
    volume_indexes: List[int],
    val_3D_files: List[Dict],
    val_2D_files: List[Dict],
    orientation: str,
    file_extension: str = ".nii.gz",
) -> Dict[str, Any]:
    """
    Generates and returns as Map of Volume IDs with values corresponding to the volume metadata dictionary and a list of
    the corresponding 2D slices to stack along the specified orientation, in order to generate the 3D volume from the 2D slices.

    Parameters
    ----------
    volume_indexes : List[int]
        List of indexes for the 3D files to be included.
    val_3D_files : List[Dict]
        Input to the 3D Dataset, containing a List of Dict for the 3D volumes.
    val_2D_files : List[Dict]
        Input to the 2D Dataset, containing a List of Dict for the 2D volumes.
    orientation : str
        Given orientation to create the 3D volume from the 2D stack.
    file_extension : str
        Filename extension. Defaults to ``nii.gz``.

    Returns
    -------
    Dict[str, Any]
        Map of volume IDs anc corresponding metadata and list of 2D slices.
    """
    volume_ids = [Path(val_3D_files[volume_index]["label"]).name[: -len(file_extension)] for volume_index in volume_indexes]

    val_list = list(val_3D_files[i] for i in volume_indexes)

    val_3D_transform = Compose([LoadImaged(keys=["label"]), OrientToRAId(keys=["label"], slicing_axes=orientation)])
    val_data = val_3D_transform(val_list)

    volume_id_map = {}
    for idx, volume_id in enumerate(volume_ids):
        selected_indexes = []
        for index, val_file in enumerate(val_2D_files):
            if volume_id in val_file["label"]:
                selected_indexes.append(index)
        val_data[idx]["label_meta_dict"]["indexes"] = selected_indexes
        volume_id_map[volume_id] = val_data[idx]["label_meta_dict"]
    return volume_id_map


class HiveSupervisedTrainer:
    """
    Hive Supervised Trainer Engine. The engine receives network configurations and parameters, experiment settings and
    validation metrics and automatically creates a supervised trainer and optionally an evaluator. The engines can be attached
    to some Event Handlers to print and save the training state and the computed validation metrics.
    """

    def __init__(self, config_dict: Dict[str, Any], training_result_folder: Union[str, PathLike]):
        """
        Initialize the Supervised Evaluator Engine from the configuration dictionary. A decorator to reload the saved checkpoints
        and resume the training is also created.
        The training and evaluation Events are stored in the *training_results_folder*.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration settings for the Trainer and the Evaluator Engines.
        training_result_folder : Union[str, PathLike]
            Folder path where to save any training and evluation event.
        """
        self.config_dict = config_dict
        self.epoch_iterations = 250
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_result_folder = training_result_folder
        self.net = self._load_net()
        self.loss = self._load_loss()
        self.optimizer = self._load_optim(self.net)
        self.lr_scheduler = self._load_lr_scheduler()
        self.checkpoint_path = None
        self.trainer = create_supervised_trainer(
            self.net,
            self.optimizer,
            self.loss,
            self.device,
            False,
            prepare_batch=prepare_batch,
        )
        self.resume_epoch = 0

        self.train_loss_MA = []
        self.train_epoch_loss_MA = []
        self.train_loss_MA_alpha = 0.93
        self.train_loss_MA_eps = 5e-4
        self.patience = 30
        self.patience_counter = 0

        @self.trainer.on(Events.STARTED)
        def resume_training(engine: Engine):
            engine.state.iteration = self.resume_epoch * self.epoch_iterations  # len(engine.state.dataloader)
            engine.state.epoch = self.resume_epoch

            if self.checkpoint_path is not None and self.resume_epoch != 0:
                output = open(Path(self.checkpoint_path).joinpath("checkpoint_{}.pkl".format(self.resume_epoch)), "rb")
                checkpoint_dict = pickle.load(output)
                output.close()
                self.train_epoch_loss_MA = checkpoint_dict["Train_loss_MA"]
                self.patience_counter = checkpoint_dict["Patience_lvl"]

        @self.trainer.on(Events.ITERATION_COMPLETED)
        def update_MA_loss(engine: Engine):
            if len(self.train_loss_MA) == 0:
                self.train_loss_MA.append(engine.state.output)
            else:
                self.train_loss_MA.append(
                    self.train_loss_MA_alpha * self.train_loss_MA[-1] + (1 - self.train_loss_MA_alpha) * engine.state.output
                )

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def check_patience(engine: Engine):
            average_loss_epoch = np.mean(self.train_loss_MA[-self.epoch_iterations :])

            if len(self.train_epoch_loss_MA) == 0:
                self.train_epoch_loss_MA.append(average_loss_epoch)
            else:
                self.train_epoch_loss_MA.append(
                    self.train_loss_MA_alpha * self.train_epoch_loss_MA[-1] + (1 - self.train_loss_MA_alpha) * average_loss_epoch
                )
                if np.abs(self.train_epoch_loss_MA[-1] - self.train_epoch_loss_MA[-2]) < self.train_loss_MA_eps:
                    self.patience_counter = self.patience_counter + 1
                else:
                    self.patience_counter = 0
                logging.getLogger("trainer").info("Train Loss MA: {}".format(self.train_epoch_loss_MA[-1]))
                logging.getLogger("trainer").info("Patience: {}/{}".format(self.patience_counter, self.patience))
                if self.patience_counter >= self.patience:
                    print("Patience limit reached!")
                    self.lr_scheduler.step(self.train_epoch_loss_MA[-1])

            if self.checkpoint_path is not None:
                checkpoint_dict = {"Train_loss_MA": self.train_epoch_loss_MA, "Patience_lvl": self.patience_counter}
                output = open(Path(self.checkpoint_path).joinpath("checkpoint_{}.pkl".format(self.trainer.state.epoch)), "wb")
                pickle.dump(checkpoint_dict, output)
                output.close()

        self.evaluator = None
        self.key_score_name = None
        self.swap_HW = False
        self.orientation = None

    def set_swap_HW(self, swap_HW: bool):
        """
        Set Flag to Swap HW dimensions at TensorBoard GIF creation Handler call.

        Parameters
        ----------
        swap_HW : bool
            Flag to swap HW dimensions.
        """
        self.swap_HW = swap_HW

    def set_orientation(self, orientation: str):
        """
        Set training orientation as class parameter.

        Parameters
        ----------
        orientation : str
            Training orientation.
        """
        self.orientation = orientation

    def _resume_training(self):
        self.resume_epoch = int(
            reload_checkpoint(
                str(Path(self.training_result_folder).joinpath("checkpoints")),
                self.net,
                self.optimizer,
                self.lr_scheduler,
                self.trainer,
            )
        )
        logging.getLogger("trainer").info("Resuming training at epoch {}".format(self.resume_epoch + 1))

    def _load_net(self):
        if "class_import" in self.config_dict["net_config"]:
            import_class = importlib.import_module(self.config_dict["net_config"]["class_import"])
        else:
            import_class = monai.networks.nets
        if hasattr(import_class, self.config_dict["net_config"]["class_name"]):

            net = getattr(import_class, self.config_dict["net_config"]["class_name"])(
                **self.config_dict["net_config"]["class_params"]
            ).to(self.device)
            return net
        else:
            raise AttributeError("{} has no attribute {}".format(import_class, self.config_dict["net_config"]["class_name"]))

    def _load_loss(self):
        if "class_import" in self.config_dict["loss_config"]:
            import_class = importlib.import_module(self.config_dict["loss_config"]["class_import"])
        else:
            import_class = monai.losses
        if hasattr(import_class, self.config_dict["loss_config"]["class_name"]):
            loss = getattr(import_class, self.config_dict["loss_config"]["class_name"])(
                **self.config_dict["loss_config"]["class_params"]
            )
            return loss
        else:
            raise AttributeError("{} has no attribute {}".format(import_class, self.config_dict["loss_config"]["class_name"]))

    def _load_optim(self, net):
        if "class_import" in self.config_dict["optim_config"]:
            import_class = importlib.import_module(self.config_dict["optim_config"]["class_import"])
        else:
            import_class = torch.optim

        if hasattr(import_class, self.config_dict["optim_config"]["class_name"]):
            opt = getattr(import_class, self.config_dict["optim_config"]["class_name"])(
                net.parameters(), **self.config_dict["optim_config"]["class_params"]
            )
            return opt
        else:
            raise AttributeError("{} has no attribute {}".format(import_class, self.config_dict["optim_config"]["class_name"]))

    def _load_lr_scheduler(self):
        if "lr_scheduler_config" not in self.config_dict:
            return None

        if "class_import" in self.config_dict["lr_scheduler_config"]:
            import_class = importlib.import_module(self.config_dict["lr_scheduler_config"]["class_import"])
        else:
            import_class = torch.optim.lr_scheduler

        if hasattr(import_class, self.config_dict["lr_scheduler_config"]["class_name"]):
            lr_scheduler = getattr(import_class, self.config_dict["lr_scheduler_config"]["class_name"])(
                self.optimizer, **self.config_dict["lr_scheduler_config"]["class_params"]
            )
            return lr_scheduler
        else:
            raise AttributeError(
                "{} has no attribute {}".format(import_class, self.config_dict["lr_scheduler_config"]["class_name"])
            )

    def prepare_trainer_event_handlers(
        self,
        progress_bar: bool = True,
        model_checkpoints_path: Union[str, PathLike] = None,
        tb_log_path: Union[str, PathLike] = None,
    ):
        """
        Creates and attach training event to the trainer engine. If *progress_bar* is True, the training state
        (epoch, iteration, loss, ETA) is printed using a tqdm progress bar. If set to False, the standard Engine logger
        is used to print the training state.
        If *model_checkpoint_path* is set, the network and optimizer state are saved as ModelCheckpoint at the end of each epoch.
        If *tb_log_path* is set, the loss values is written in a TensorBoard Summary Writer after each iteration.

        Parameters
        ----------
        progress_bar : bool
            Flag to set/unset the tqdm progress bar
        model_checkpoints_path : Union[str, PathLike]
            Folder where to save the model checkpoints.
        tb_log_path : Union[str, PathLike]
            Folder where to save the TensorBoard summary log.
        """
        if self.lr_scheduler is not None:
            lr_scheduler_handler = LrScheduleHandler(self.lr_scheduler, step_transform=lambda x: self.train_epoch_loss_MA[-1])
            lr_scheduler_handler.attach(self.trainer)
        if model_checkpoints_path is not None:
            checkpoint_handler = ModelCheckpoint(
                model_checkpoints_path,
                "net",
                n_saved=1,
                require_empty=False,
                global_step_transform=lambda x, y: self.trainer.state.epoch,
            )
            self.checkpoint_path = model_checkpoints_path
            save_dict = {"net": self.net, "opt": self.optimizer}
            if self.lr_scheduler is not None:
                save_dict["lr_scheduler"] = self.lr_scheduler
            self.trainer.add_event_handler(
                event_name=Events.EPOCH_COMPLETED,
                handler=checkpoint_handler,
                to_save=save_dict,
            )

        if progress_bar:
            pbar = ProgressBar()
            pbar.attach(self.trainer, output_transform=lambda x: {"loss": x})
        else:
            train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
            train_stats_handler.attach(self.trainer)

        if tb_log_path is not None:
            train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=tb_log_path, output_transform=lambda x: x)
            train_tensorboard_stats_handler.attach(self.trainer)

            writer = SummaryWriter(tb_log_path)
            input_shape = [1, len(self.config_dict["Modalities"].keys())]
            if "patch_size" in self.config_dict:
                _ = [input_shape.append(dim_size) for dim_size in self.config_dict["patch_size"]]
            elif "slice_size_2d" in self.config_dict:
                _ = [input_shape.append(dim_size) for dim_size in self.config_dict["slice_size_2d"][self.orientation]]
            writer.add_graph(self.net, torch.zeros(tuple(input_shape)).to(self.device))
            writer.close()

    def create_evaluator(
        self,
        val_loader: DataLoader,
        post_pred_transform: Transform,
        post_label_transform: Transform,
        val_metric_list: List[str],
        val_key_metric: str,
    ):
        """
        Creates an Engine evaluator and attaches it to the trainer. A list of metrics are used to perform the training evaluation,
        while the **key metric** is used to assess the best epoch.

        Parameters
        ----------
        val_loader : DataLoader
            torch DataLoader with the data used to create and run the evaluation step.
        post_pred_transform : Transform
            Post-processing Transform to apply to the prediction output.
        post_label_transform : Transform
            Post-processing Transform to apply to the label output.
        val_metric_list : List[str]
            List of metrics to compute when running the evaluation step.
        val_key_metric : str
            Metric name to be considered as the key metric.
        """
        validation_metrics = create_validation_metric_dict(val_metric_list, len(self.config_dict["label_dict"].keys()))
        self.evaluator = create_supervised_evaluator(
            self.net,
            validation_metrics,
            self.device,
            True,
            output_transform=lambda x, y, y_pred: (
                [post_pred_transform(i) for i in decollate_batch(y_pred)],
                [post_label_transform(i) for i in decollate_batch(y)],
            ),
            prepare_batch=prepare_batch,
        )
        self.evaluator.state.trainer = self.trainer
        self.evaluator.state.key_metric = val_key_metric
        self.evaluator.state.label_dict = self.config_dict["label_dict"]
        if not self.config_dict.get("val_key_metric_alpha"):
            self.evaluator.state.val_key_metric_alpha = 0.0
        else:
            self.evaluator.state.val_key_metric_alpha = self.config_dict["val_key_metric_alpha"]

        self.evaluator.state.val_key_metric_biased = None
        self.key_score_name = val_key_metric
        self.evaluator.state.val_key_metric_list = []

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def run_validation(engine):
            self.evaluator.run(val_loader)

    def prepare_evaluator_event_handlers(
        self,
        val_key_path: Union[str, PathLike] = None,
        tb_image_log_path: Union[str, PathLike] = None,
        tb_log_path: Union[str, PathLike] = None,
        volume_id_map: Dict[str, Any] = None,
    ):
        """
        Creates and attach evaluation event to the evaluator engine.
        If *val_key_path* is set, the best network and optimizer state ( according to the key metric) are saved at the
        end of the best epoch.
        If *tb_log_path* is set, the evaluation metric values are written in a TensorBoard Summary Writer after running
        the evaluation step.

        Parameters
        ----------
        val_key_path : Union[str, PathLike]
            Folder where to save the best epoch Model Checkpoint.
        tb_image_log_path: Union[str, PathLike]
            Folder path where to save the volumes to be monitored during validation.
        volume_id_map : Dict[str, Any]
            Dictionary listing the volume IDs to monitor and saved after every epoch validation run. The dictionary contains a
            metadata dictionary and a list of 2D slices to be stacked for each ID case.
        tb_log_path : Union[str, PathLike]
            Folder where to save the TensorBoard evaluation metric logs.
        """
        if self.evaluator is None:
            raise AttributeError("{} has no evaluator".format(self.__class__.__name__))

        val_stats_handler = StatsHandler(
            epoch_print_logger=epoch_printer,
            name="evaluator",
            output_transform=lambda x: None,
            global_epoch_transform=lambda x: self.trainer.state.epoch,
        )
        val_stats_handler.attach(self.evaluator)

        if tb_log_path is not None:
            val_tensorboard_stats_handler = TensorBoardStatsHandler(
                log_dir=tb_log_path,
                output_transform=lambda x: None,
                global_epoch_transform=lambda x: self.trainer.state.epoch,
                epoch_event_writer=epoch_writer,
            )

            val_tensorboard_stats_handler.attach(self.evaluator)

            if volume_id_map is not None:
                output_dir = None
                if tb_image_log_path is not None:
                    output_dir = tb_image_log_path
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                val_tensorboard_image_handler = Hive2Dto3DTensorBoardImageHandler(
                    volume_id_map=volume_id_map,
                    output_dir=output_dir,
                    log_dir=tb_log_path,
                    swapHW=self.swap_HW,
                    orientation=self.orientation,
                    batch_transform=lambda x: (x["image"], x["label"]),
                    output_transform=lambda x: x[0],
                    epoch_level=False,
                    max_channels=3,
                    max_frames=64,
                    global_iter_transform=lambda x: self.trainer.state.epoch,
                )
                val_tensorboard_image_handler.attach(self.evaluator)

        if val_key_path is not None:
            checkpoint_handler_val = ModelCheckpoint(
                val_key_path,
                "net",
                score_name=self.key_score_name,
                score_function=epoch_score_function,
                n_saved=1,
                require_empty=False,
            )

            save_dict = {"net": self.net, "opt": self.optimizer}
            if self.lr_scheduler is not None:
                save_dict["lr_scheduler"] = self.lr_scheduler
            self.evaluator.add_event_handler(
                event_name=Events.EPOCH_COMPLETED,
                handler=checkpoint_handler_val,
                to_save=save_dict,
            )

    def run_training(self, train_loader: DataLoader):
        """
        Start the training, after checking if any Checkpoint was previously stored.
        At the end of the training, a JSON training summary is also saved.

        Parameters
        ----------
        train_loader : DataLoader
             torch DataLoader with the data used to run the training.
        """
        self._resume_training()
        state = self.trainer.run(train_loader, self.config_dict["train_epochs"], self.epoch_iterations)
        save_final_state_summary(self.training_result_folder, state, self.evaluator)
        send_email()  # TODO AS Handler ?
