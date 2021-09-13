from typing import List, Any

import torch
from monai.transforms import Transform


class ToListMemorySaver(Transform):
    """
    Append a CPU copy of the input Tensor to a specified list.
    """

    def __init__(self, output_list: List[Any]) -> None:
        """

        Parameters
        ----------
        output_list : List[Any]
            List where to append the input Tensors.
        """
        self.output_list = output_list

    def __call__(
            self,
            img: torch.Tensor,
    ) -> None:
        """

        Parameters
        ----------
        img : torch.Tensor
             the input tensor data to append.
        """
        self.output_list.append(img.cpu())


class ThresholdForegroundLabels(Transform):
    """
    Threshold foreground labels with the given *threshold_value* and the background labels with *1 - threshold_value*.
    Expects an input Tensor in One-Hot format [B, C, [H, W], D], with the logits values representing the predicted values.
    """

    def __init__(self, foreground_labels: List[int], threshold_value: float = 0.5) -> None:
        """

        Parameters
        ----------
        foreground_labels : List[int]
            List of labels to be considered as foreground
        threshold_value : float
            Value used as threshold for the foreground classes.
        """
        self.foreground_labels = foreground_labels
        self.threshold = threshold_value

    def __call__(
            self,
            img: torch.Tensor,
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        img : torch.Tensor
            Input Tensor in One-Hot format [B, C, [H, W], D], with the logits values representing the predicted values.
        Returns
        -------
        torch.Tensor
            Tensor in One-Hot format with the thresholded classes.
        """
        labels = list(range(img.shape[1]))
        background_labels = [label for label in labels if label not in self.foreground_labels]
        img[:, background_labels, ...] = (img[:, background_labels, ...] >= (1 - self.threshold)).float()
        img[:, self.foreground_labels, ...] = (img[:, self.foreground_labels, ...] >= self.threshold).float()
        return img.float()
