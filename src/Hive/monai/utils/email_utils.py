import datetime
import math
import os
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os import PathLike
from pathlib import Path
from typing import Union, Dict, Any, List

import matplotlib.pyplot as plt
from ignite.engine import Engine
from ignite.engine.events import State
from json2html import *  # noqa: F403


def send_email(
    output_folder: Union[str, PathLike],
    receiver_email: str,
    project: str,
    experiment: str,
    run: str = "",
    fold: int = None,
    tb_log_path: Union[str, PathLike] = None,
    trainer_state: Union[Dict[str, Any], State] = None,
    evaluator: Engine = None,
):
    """
    After completing a fold training, sends an e-mail with training summary information to the specified receiver.

    Parameters
    ----------
    output_folder : Union[str, PathLike]
        Folder where the optional e-mail attachments are temporarily stored.
    receiver_email : str
        E-mail address where to send the training summary.
    project : str
        Project name, as in *config_file["DatasetName"]*.
    experiment : str
        Experiment name, as in *config_file["Experiment Name"]*.
    run : str
        Optional run name. Example: ´axial´, ´coronal´ or ´sagittal´.
    fold : int
        Fold number
    tb_log_path : Union[str, PathLike]
        Optional tensorboard log path, where to find GIF logs.
    trainer_state : Union[Dict[str, Any], State]
        Trainer state, can be either an **ignite.engine State** or a dictionary, containing the required key,value pairs.
    evaluator : Engine
        Optional **ignite.engine**, containing validation metrics.
    """
    sender_email = os.environ["email_account"]
    message = MIMEMultipart()
    message["Subject"] = "{} {} Training, fold {}, completed".format(run, experiment, fold)
    message["From"] = sender_email
    message["To"] = receiver_email

    if isinstance(trainer_state, dict):
        val_key_metric_name = trainer_state["key_metric"].replace("_", " ")
        val_key_metric_series = trainer_state["val_key_metric_list"]
        val_metrics = {"Evaluation Metric": trainer_state["val_key_metric_list"][-1]}
        val_key_metric_alpha = trainer_state["val_key_metric_alpha"]
        epoch = trainer_state["epoch"]
        time = str(datetime.timedelta(seconds=trainer_state["COMPLETED"])).split(sep=".")[0]
    elif isinstance(trainer_state, State):
        val_key_metric_name = trainer_state.key_metric.replace("_", " ")
        val_key_metric_series = evaluator.state.val_key_metric_list
        from Hive.monai.engines.utils import state_metrics_to_dict

        val_metrics = state_metrics_to_dict(evaluator)
        val_key_metric_alpha = trainer_state.val_key_metric_alpha
        epoch = trainer_state.epoch
        time = str(datetime.timedelta(seconds=trainer_state.times["COMPLETED"])).split(sep=".")[0]
    else:
        raise ValueError

    time_hm = time.split(":")

    filenames = []
    create_scatter_plot_for_validation_metric(
        val_key_metric_series,
        "Validation Key Metric",
        val_key_metric_name,
        str(Path(output_folder).joinpath("val_key_metric.png")),
        [val_key_metric_alpha],
    )
    filenames.append(str(Path(output_folder).joinpath("val_key_metric.png")))

    if tb_log_path is not None:
        from Hive.monai.utils.tensorboard_io_utils import create_PNG_from_TB_event

        create_PNG_from_TB_event(tb_log_path, str(Path(output_folder).joinpath("prediction.gif")))
        filenames.append(str(Path(output_folder).joinpath("prediction.gif")))

    html = """\
    <html>
        <head></head>
        <body>
            <h3>The {}, {} {} training is now completed at epoch {} in fold {}.</h3>
            <h3>Total Training time {} h {} m.</h3>

            {}
        </body>
    </html>
    """.format(
        project, run, experiment, epoch, fold, time_hm[0], time_hm[1], json2html.convert(json=val_metrics)  # noqa: F405
    )

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(html, "html")

    message.attach(part1)

    for filename in filenames:
        if Path(filename).is_file():
            with open(filename, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {filename}",
                )
                message.attach(part)

    port = 465  # For SSL
    password = os.environ["email_password"]

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

    for filename in filenames:
        os.remove(filename)


def create_scatter_plot_for_validation_metric(
    series_values: List[float],
    title: str,
    metric_name: str,
    output_file: Union[str, PathLike],
    smooth_factors: List[float] = None,
):
    """
    Creates a Scatter plot from a sequence of values, optionally including smoothed sequences.

    Parameters
    ----------
    series_values : List[float]
        Value sequence
    title : str
        Plot title.
    metric_name : str
        Metric name, used in the legend.
    output_file : Union[str, PathLike]
        File path for the scatter plot.
    smooth_factors : List[float]
        Optional smooth factors, to include the corresponding smoothed sequences.
    """
    if smooth_factors is None:
        smooth_factors = []
    plt.plot(series_values, label=metric_name)

    for smooth_factor in smooth_factors:
        smooth_values = smooth_series_with_alpha(series_values, smooth_factor)
        plt.plot(smooth_values, label=metric_name + "  = {}".format(smooth_factor))
    plt.legend()
    plt.ylabel(metric_name)
    plt.xlabel("Epochs")
    plt.title(title)

    plt.savefig(output_file)


def smooth_series_with_alpha(series_values: List[float], alpha: float) -> List[float]:
    """
    Given a sequence of numbers, return the smooth sequence.

    Parameters
    ----------
    series_values : List[float]
        Value sequence
    alpha : float
        Smoothing factor

    Returns
    -------
    List[float]
        Smoothed sequence.
    """
    smooth_values = []
    val_biased = 0.0

    for idx, value in enumerate(series_values):
        val_biased = alpha * val_biased + (1 - alpha) * value
        bias_weight = 1 - math.pow(alpha, (idx + 1))
        smooth_values.append(val_biased / bias_weight)

    return smooth_values
