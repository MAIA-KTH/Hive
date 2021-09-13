from os import PathLike
from pathlib import Path
from typing import Dict, Any, Union
from typing import Optional, Callable

import torch
from Hive.monai import ORIENTATION_MAP
from Hive.monai.data import HiveNiftiSaver
from ignite.engine import Events, Engine
from monai.handlers import TensorBoardImageHandler
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard.writer import SummaryWriter


class Hive2Dto3DTensorBoardImageHandler(TensorBoardImageHandler):
    """
    Hive2Dto3DTensorBoardImageHandler is an Ignite Event handler that can generate GIF images to visualize 3D volumes from
    2D training data. The GIF for Image, Label and Prediction volumes are generated and saved as TensorBoard summary after each
    validation epoch step. The volumes to be considered are specified in **volume_id_map**, where a list of 2D slices are
    included, in order to stack them and create the GIFs.
    If **output_dir** is provided, the 3D stacked volumes are also saved as NIFTI volumes after each epoch validation step.

    """

    def __init__(
        self,
        volume_id_map: Dict[str, Any],
        orientation: str,
        output_dir: Union[str, PathLike] = None,
        summary_writer: Optional[SummaryWriter] = None,
        log_dir: str = "./runs",
        interval: int = 1,
        swapHW: bool = False,
        gif_step: int = 10,
        epoch_level: bool = True,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        global_iter_transform: Callable = lambda x: x,
        index: int = 0,
        max_channels: int = 1,
        max_frames: int = 64,
    ) -> None:
        """

        Parameters
        ----------
        volume_id_map : Dict[str, Any]
            Dictionary containing metadata and 2D slices list for each volume ID to stack.
        orientation : str
            Stacking orientation for the 2D slices.
        output_dir : Union[str, PathLike]
            Folder path where to save the NIFTI 3D volumes.
        summary_writer : Optional[SummaryWriter]
            user can specify TensorBoard SummaryWriter, default to create a new writer.
        log_dir : str
            if using default SummaryWriter, write logs to this directory, default is `./runs`.
        interval : int
            plot content from engine.state every N epochs or every N iterations, default is 1.
        swapHW : bool
            Flag to swap Height and Width axes.
        gif_step : int
            Interpolation step to generate the GIFs from the 2D stacked slices. Defaults to 10.
        epoch_level : bool
            plot content from engine.state every N epochs or N iterations. `True` is epoch level,
                `False` is iteration level.
        batch_transform : Callable
            a callable that is used to extract `image` and `label` from `ignite.engine.state.batch`,
                then construct `(image, label)` pair. for example: if `ignite.engine.state.batch` is `{"image": xxx,
                "label": xxx, "other": xxx}`, `batch_transform` can be `lambda x: (x["image"], x["label"])`.
                will use the result to plot image from `result[0][index]` and plot label from `result[1][index]`.
        output_transform : Callable
            a callable that is used to extract the `predictions` data from
                `ignite.engine.state.output`, will use the result to plot output from `result[index]`.
        global_iter_transform : Callable
            a callable that is used to customize global step number for TensorBoard.
                For example, in evaluation, the evaluator engine needs to know current epoch from trainer.
        index : int
            plot which element in a data batch, default is the first element.
        max_channels : int
            number of channels to plot.
        max_frames : int
            number of frames for 2D-t plot.
        """
        super().__init__(summary_writer=summary_writer, log_dir=log_dir)
        self.interval = interval
        self.epoch_level = epoch_level
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.global_iter_transform = global_iter_transform
        self.index = index
        self.max_frames = max_frames
        self.max_channels = max_channels

        self.volume_id_map = volume_id_map
        self.image = {}
        self.label = {}
        self.prediction = {}
        self.save_volume = False
        self.output_dir = None
        if output_dir is not None:
            self.output_dir = output_dir
            self.save_volume = True
        for volume_id in volume_id_map.keys():
            self.image[volume_id] = []
            self.label[volume_id] = []
            self.prediction[volume_id] = []
        self.swap_HW = swapHW
        self.orientation = orientation
        self.batch_size = 0
        self.gif_step = gif_step

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if engine.state.iteration == 1:
            self.batch_size = self.batch_transform(engine.state.batch)[0].shape[0]
        iteration_indexes = range(
            (engine.state.iteration - 1) * self.batch_size,
            (engine.state.iteration - 1) * self.batch_size + self.batch_transform(engine.state.batch)[0].shape[0],
        )

        for volume_id in self.volume_id_map.keys():
            indexes = [
                iteration_index
                for iteration_index in range(0, self.batch_transform(engine.state.batch)[0].shape[0])
                if iteration_indexes[iteration_index] in self.label_dict[volume_id]["indexes"]
            ]

            if len(indexes) == 0:
                continue
            step = self.global_iter_transform(engine.state.epoch if self.epoch_level else engine.state.iteration)

            self.image[volume_id].append(self.batch_transform(engine.state.batch)[0][indexes, :, :, :])
            self.label[volume_id].append(self.batch_transform(engine.state.batch)[1][indexes, :, :, :])
            self.prediction[volume_id].append(torch.stack(self.output_transform(engine.state.output))[indexes, :, :, :])
            if iteration_indexes[indexes[-1]] == self.volume_id_map[volume_id]["indexes"][-1]:
                self.image[volume_id] = torch.cat(self.image[volume_id]).permute(1, 2, 3, 0)
                depth = self.image[volume_id].shape[-1]
                self.label[volume_id] = torch.cat(self.label[volume_id]).permute(1, 2, 3, 0)
                self.prediction[volume_id] = torch.cat(self.prediction[volume_id]).permute(1, 2, 3, 0)

                if self.save_volume:
                    output_saver = HiveNiftiSaver(
                        output_dir=self.output_dir,
                    )
                    if self.swap_HW:
                        self.image[volume_id] = torch.swapaxes(self.image[volume_id], 1, 2)
                        self.label[volume_id] = torch.swapaxes(self.label[volume_id], 1, 2)

                    prediction = self.prediction[volume_id].clone()
                    prediction = prediction.permute(0, 3, 1, 2)
                    orientation_index = ORIENTATION_MAP[self.orientation]
                    axis_index = self.volume_id_map[volume_id]["axis_orientation"].index(orientation_index)
                    prediction = torch.transpose(prediction, 1, axis_index + 1)
                    prediction = torch.unsqueeze(prediction, dim=0)

                    prediction = torch.nn.functional.interpolate(
                        prediction, size=tuple(self.volume_id_map[volume_id]["spatial_shape"].tolist()), mode="trilinear"
                    )
                    output_saver.save_with_path(
                        prediction[0],
                        path=str(Path(self.output_dir).joinpath("{}_{}.nii.gz".format(volume_id, step))),
                        meta_data=self.volume_id_map[volume_id],
                    )

                self.image[volume_id] = self.image[volume_id][:, :, :, range(0, depth, self.gif_step)]
                self.label[volume_id] = self.label[volume_id][:, :, :, range(0, depth, self.gif_step)]
                self.prediction[volume_id] = self.prediction[volume_id][:, :, :, range(0, depth, self.gif_step)]
                if self.prediction[volume_id].shape[0] > 1:
                    self.prediction[volume_id] = torch.argmax(self.prediction[volume_id], dim=0, keepdim=True)

                plot_2d_or_3d_image(
                    self.image[volume_id][None],
                    step,
                    self._writer,
                    0,
                    self.max_channels,
                    self.max_frames,
                    "image_" + volume_id,
                )
                plot_2d_or_3d_image(
                    # add batch dim and plot the first item
                    self.label[volume_id][None],
                    step,
                    self._writer,
                    0,
                    self.max_channels,
                    self.max_frames,
                    "label_" + volume_id,
                )
                plot_2d_or_3d_image(
                    # add batch dim and plot the first item
                    self.prediction[volume_id][None],
                    step,
                    self._writer,
                    0,
                    self.max_channels,
                    self.max_frames,
                    "prediction_" + volume_id,
                )
                self.image[volume_id] = []
                self.label[volume_id] = []
                self.prediction[volume_id] = []
