import json
from os import PathLike
from pathlib import Path
from typing import Dict, Any, List, Callable, Literal, Optional, Union

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
from pandas import DataFrame
from plotly.graph_objects import Figure

from Hive.evaluation import DEFAULT_METRIC_UNITS, DEFAULT_BAR_CONFIGS, METRICS_FOLDER_NAME
from Hive.evaluation.io_metric_results import read_dataframe

BAR_AGGREGATORS = ["min", "max", "mean"]


def get_heatmap(
    df_flat: DataFrame,
    metric_name: str,
    metric_measurement_unit: str,
    section: Literal["testing", "validation", "experiment", "project"],
    bar_configs: Dict[str, Any],
    plot_title: str,
    **kwargs
) -> Figure:
    metric_ID = metric_name
    if section == "experiment":
        facet_col = "Section"
        facet_row = None
    elif section == "project":
        facet_col = "Section"
        facet_row = "Experiment"
        metric_ID = "Metric_Score"
    else:
        facet_col = None
        facet_row = None

    colors = "Inferno"
    if bar_configs is not None:

        if "colors" in bar_configs:
            colors = bar_configs["colors"]

    label_list = list(set(df_flat["Label"].values))
    label_list = list(df_flat["Label"].values[: len(label_list)])

    heatmap_list = []
    subplot_titles = []

    facet_col_list = [None]
    facet_row_list = [None]

    if facet_col is not None:
        facet_col_list = list(set(df_flat[facet_col].values))
        subplot_titles = facet_col_list
        if facet_row is not None:
            facet_row_list = list(set(df_flat[facet_row].values))
            subplot_titles = []
            for facet_row_value in facet_row_list:
                for facet_col_value in facet_col_list:
                    subplot_titles.append(facet_row_value + ", " + facet_col_value)

    for facet_row_value in facet_row_list:
        if facet_row_value is not None:
            df = df_flat[df_flat[facet_row] == facet_row_value]
        else:
            df = df_flat
        if facet_col is not None:
            subject_IDs = df[facet_col].values.reshape((int(df.shape[0] / len(label_list)), len(label_list))).T[0]
        else:
            subject_IDs = df[metric_ID].values.reshape((int(df.shape[0] / len(label_list)), len(label_list))).T[0]

        for facet_col_value in facet_col_list:

            if facet_col_value is not None:
                subject_IDs_for_value = [i for i, val in enumerate(subject_IDs) if val == facet_col_value]
            else:
                subject_IDs_for_value = [i for i, val in enumerate(subject_IDs)]

            if facet_col_value is not None:
                if facet_row_value is not None:
                    df = df_flat[(df_flat[facet_col] == facet_col_value) & (df_flat[facet_row] == facet_row_value)]

                else:
                    df = df_flat[(df_flat[facet_col] == facet_col_value)]

            df = df[metric_ID].values.reshape((int(df.shape[0] / len(label_list)), len(label_list))).T

            fig_heatmap = go.Heatmap(
                z=df,
                y=label_list,
                x=subject_IDs_for_value,
                coloraxis="coloraxis",
                hovertemplate="Subject: %{x}<br>Label: %{y}<br>"
                + metric_name
                + " "
                + metric_measurement_unit
                + ": %{z}<extra></extra>",
            )
            heatmap_list.append(fig_heatmap)

    fig = plotly.subplots.make_subplots(rows=len(facet_row_list), cols=len(facet_col_list), subplot_titles=subplot_titles)

    it = 0
    for i, facet_row_value in enumerate(facet_row_list):
        for j, facet_col_value in enumerate(facet_col_list):
            fig.append_trace(heatmap_list[it], i + 1, j + 1)
            it += 1
            fig.update_xaxes(title_text="Subject", row=i + 1, col=j + 1)
            fig.update_yaxes(title_text="Label", row=i + 1, col=j + 1)

    fig.update_layout(title=plot_title, coloraxis={"colorscale": colors})

    return fig


def get_phase_table(phase_json_file):
    df = pd.DataFrame(columns=["Subject", "Phase"])
    with open(phase_json_file) as json_file:
        phase_dict = json.load(json_file)

    for key in phase_dict:
        df = df.append({"Subject": key, "Phase": phase_dict[key]}, ignore_index=True)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Subject", "Phase"], fill_color="paleturquoise", align="left"),
                cells=dict(values=[df.Subject, df.Phase], fill_color="lavender", align="left"),
            )
        ]
    )
    fig.update_layout(title="Subjects Breathing Phase")
    return fig


def get_plotly_histo(
    df_flat: DataFrame,
    metric_name: str,
    metric_measurement_unit: str,
    section: Literal["testing", "validation", "experiment", "project"],
    plot_title: str,
    **kwargs
) -> Figure:
    """
    Creates and returns a ```Plotly`` :py:class`px.histogram`, according to the given DataFrame, metric and section.

    Parameters
    ----------
    df_flat : DataFrame
        Pandas DataFrame used to generate the plot.
    metric_name : str
        Specified metric for the plot.
    metric_measurement_unit : str
        Metric measurement unit, to be appended in the plot labels.
    section : Literal['testing', 'validation', 'experiment', 'project']
        Specified section for the plot.
    plot_title : str
        Plot title.
    kwargs

    Returns
    -------
    Figure
        ```Plotly`` :py:class`px.histogram` for the given metric and section.
    """
    x_value = metric_name
    if section == "experiment":
        color_value = "Section"
        facet = None
    elif section == "project":
        color_value = "Experiment"
        facet = "Section"
        x_value = "Metric_Score"
    else:
        color_value = "Label"
        facet = None

    fig_histo = px.histogram(
        df_flat,
        x=x_value,
        color=color_value,
        facet_row=facet,
        labels={
            x_value: metric_name + " " + metric_measurement_unit,
        },
        title=plot_title,
    )

    return fig_histo


def get_plotly_average_bar(
    df_flat: DataFrame,
    metric_name: str,
    metric_measurement_unit: str,
    section: Literal["testing", "validation", "experiment", "project"],
    plot_title: str,
    bar_configs: Dict[str, Any],
    aggregator: str,
    **kwargs
) -> Figure:
    """
    Creates and returns a ```Plotly`` :py:class`px.bar`, according to the given DataFrame, metric and section.
    The metric is aggregated according to **aggr**.

    Parameters
    ----------
    df_flat : DataFrame
        Pandas DataFrame used to generate the plot.
    metric_name : str
        Specified metric for the plot.
    metric_measurement_unit : str
        Metric measurement unit, to be appended in the plot labels.
    section : Literal['testing', 'validation', 'experiment', 'project']
        Specified section for the plot.
    plot_title : str
        Plot title.
    bar_configs: Dict [str, Any]
        Configuration dictionary used to configure the bar colorscale.
    aggregator : str
        Metric aggregator to create the bar plot. Examples: [```mean``, ```max``, ```min``].
    kwargs

    Returns
    -------
    Figure
        ```Plotly`` :py:class`px.bar` for the given metric, aggregator and section.
    """
    x_value = metric_name
    if section == "experiment":
        y_value = "Section"
        facet_row = "Label"
        facet_col = None
    elif section == "project":
        y_value = "Experiment"
        facet_row = "Label"
        facet_col = "Section"
        x_value = "Metric_Score"
    else:
        y_value = "Label"
        facet_row = None
        facet_col = None

    key_cols = ["Label", "Section", "Experiment"]

    df_flat = df_flat.groupby(key_cols).agg(aggregator).reset_index()

    text = []
    for index, row in df_flat.iterrows():
        text.append("{:.3}".format(row[x_value]))

    colors = "Inferno"
    if bar_configs is not None:

        if "colors" in bar_configs:
            colors = bar_configs["colors"]

    fig_bar = px.bar(
        df_flat,
        y=y_value,
        x=x_value,
        orientation="h",
        color=x_value,
        facet_row=facet_row,
        facet_col=facet_col,
        hover_data=key_cols,
        barmode="group",
        color_continuous_scale=colors,
        text=text,
        labels={
            x_value: metric_name + " " + metric_measurement_unit,
        },
        title=plot_title,
    )

    return fig_bar


def get_plotly_boxplot(
    df_flat: DataFrame,
    metric_name: str,
    metric_measurement_unit: str,
    section: Literal["testing", "validation", "experiment", "project"],
    plot_title: str,
    **kwargs
):
    """
    Creates and returns a ```Plotly`` :py:class`px.box`, according to the given DataFrame, metric and section.

    Parameters
    ----------
    df_flat : DataFrame
        Pandas DataFrame used to generate the plot.
    metric_name : str
        Specified metric for the plot.
    metric_measurement_unit : str
        Metric measurement unit, to be appended in the plot labels.
    section : str
        Specified section for the plot.
    plot_title : str
        Plot title.
    kwargs

    Returns
    -------
    Figure
        ```Plotly`` :py:class`px.box` for the given metric and section.
    """
    y_value = metric_name
    if section == "experiment":
        color_value = "Section"
        facet = None
    elif section == "project":
        color_value = "Experiment"
        facet = "Section"
        y_value = "Metric_Score"
    else:
        color_value = "Label"
        facet = None

    fig_boxplot = px.box(
        df_flat,
        x="Label",
        y=y_value,
        facet_row=facet,
        color=color_value,
        labels={
            y_value: metric_name + " " + metric_measurement_unit,
        },
        title=plot_title,
    )

    return fig_boxplot


PLOTS = {
    "boxplot": get_plotly_boxplot,
    "bar": get_plotly_average_bar,
    "histo": get_plotly_histo,
    "heatmap": get_heatmap,
}  # type: Dict[str, Callable]


def get_plot_title(
    main_title: str,
    section: Literal["testing", "validation", "experiment", "project"],
    metric: str,
    aggr: Optional[Literal["max", "min", "mean"]] = None,
) -> str:
    """
    Compose and return the plot title, according to the specified metric and section.
    The title is composed as follows:
        .. math::
            main\_title, [ + Section Set, ] [ + aggregator ] + metric

    Parameters
    ----------
    main_title : str
        Initial strings for the title, to be prepended.
    section : Literal['testing', 'validation', 'experiment', 'project']
        Section for the title.
    metric: str
        Metric for the title.
    aggr: Optional[Literal['max', 'min', 'mean']]
        Optional metric aggregator. Examples: [```mean``, ```max``, ```min``].

    Returns
    -------
    str
        Plot title for the given section and metric.
    """  # noqa: W605
    section_dataset = ""
    if section in ["validation", "testing"]:
        section_dataset = " {} Set,".format(section.capitalize())
    if aggr is not None:
        aggr = " " + aggr
    else:
        aggr = ""
    title = "{},{}{} {}".format(main_title.replace("_", " "), section_dataset, aggr, metric)
    return title


def create_plots(
    config_dict: Dict[str, Any],
    df_paths: Dict[str, str],
    metrics: List[str],
    plot_title: str,
    sections: List[Literal["testing", "validation", "experiment"]],
) -> Dict[str, Figure]:
    """
    Creates and returns ``Plotly`` :py:class:`plotly.graph_objects.Figure`, according to the specified sections and metrics.
    The Pandas Dataframes used to create the plots are loaded from *df_paths*.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary, used to retrieve the metrics plot configurations.
    df_paths : Dict[str, str]
        Pandas DataFrame filepath dictionary, used to load the DataFrames.
    metrics : List[str]
        List of metrics to create ``Plotly`` plots.
    plot_title : str
        String from where to compose the plot title, as described in :py:`get_plot_title`.
    sections : List[Literal['testing', 'validation', 'experiment']]
        Sections to load and create plots.
    Returns
    -------
    Dict[str, Figure]
        Map of created ```Plotly`` figures.
    """
    plot_dict = {}
    for metric in metrics:
        for section in sections:
            df_flat = read_dataframe(df_paths["{}_flat_{}".format(metric, section)], sheet_name="Flat")
            bar_configs = None
            measurement_unit = ""

            if metric in DEFAULT_BAR_CONFIGS:
                bar_configs = DEFAULT_BAR_CONFIGS[metric]

            if metric in DEFAULT_METRIC_UNITS:
                measurement_unit = DEFAULT_METRIC_UNITS[metric]

            if "Metrics_save_configs" in config_dict and "Metrics_dict" in config_dict["Metrics_save_configs"]:
                metrics_dict = config_dict["Metrics_save_configs"]["Metrics_dict"]
                if metrics_dict is not None and isinstance(metrics_dict, dict) and "m_unit" in metrics_dict[metric]:
                    measurement_unit = metrics_dict[metric]["m_unit"]

                if (
                    isinstance(config_dict["Metrics_save_configs"]["Metrics_dict"], dict)
                    and metric in config_dict["Metrics_save_configs"]["Metrics_dict"]
                ):
                    if "bar_config" in config_dict["Metrics_save_configs"]["Metrics_dict"][metric]:
                        bar_configs = config_dict["Metrics_save_configs"]["Metrics_dict"][metric]["bar_config"]

            title = get_plot_title(plot_title, section, metric)

            args = {
                "df_flat": df_flat,
                "metric_name": metric,
                "metric_measurement_unit": measurement_unit,
                "section": section,
                "bar_configs": bar_configs,
            }

            for plot in PLOTS:
                if plot == "bar":
                    for aggr in BAR_AGGREGATORS:
                        args["aggregator"] = aggr
                        args["plot_title"] = get_plot_title(plot_title, section, metric, aggr)
                        fig = PLOTS[plot](**args)
                        plot_dict["{}_{}_{}_{}".format(aggr, metric, section, plot)] = fig
                else:
                    args["plot_title"] = title
                    fig = PLOTS[plot](**args)
                    plot_dict["{}_{}_{}".format(metric, section, plot)] = fig

    return plot_dict


def create_plots_for_project(
    config_dict: Dict[str, Any],
    df_paths: Dict[str, str],
    metrics: List[str],
    plot_title: str,
    subsection: Optional[Literal["Validation", "Testing"]] = None,
) -> Dict[str, Figure]:
    """
    Creates and returns ``Plotly`` :py:class:`plotly.graph_objects.Figure` for the project, according to the specified metrics.
    The Pandas Dataframes used to create the plots are loaded from *df_paths*.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary, used to retrieve the metrics plot configurations.
    df_paths : Dict[str, str]
        Pandas DataFrame filepath dictionary, used to load the DataFrames.
    metrics : List[str]
        List of metrics to create ``Plotly`` plots.
    plot_title : str
        String from where to compose the plot title, as described in :py:`get_plot_title`.
    subsection : Optional[Literal['Validation','Testing']]
        If set, the metric DataFrame is also filtered according to the specific section, ```Testing`` or ``Validation``].

    Returns
    -------
    Dict[str, Figure]
        Map of created ```Plotly`` figures.
    """
    plot_dict = {}
    project_df_flat = read_dataframe(df_paths["{}_flat".format(config_dict["ProjectName"])], sheet_name="Flat")
    section = "project"
    for metric in metrics:
        if subsection is None:
            df_flat = project_df_flat[project_df_flat.Metric.eq(metric)]
        else:
            df_flat = project_df_flat[project_df_flat.Metric.eq(metric) & project_df_flat.Section.eq(subsection)]
        bar_configs = None
        measurement_unit = ""

        if metric in DEFAULT_BAR_CONFIGS:
            bar_configs = DEFAULT_BAR_CONFIGS[metric]

        if metric in DEFAULT_METRIC_UNITS:
            measurement_unit = DEFAULT_METRIC_UNITS[metric]

        if "Metrics_save_configs" in config_dict and "Metrics_dict" in config_dict["Metrics_save_configs"]:
            metrics_dict = config_dict["Metrics_save_configs"]["Metrics_dict"]
            if metrics_dict is not None and isinstance(metrics_dict, dict) and "m_unit" in metrics_dict[metric]:
                measurement_unit = metrics_dict[metric]["m_unit"]

            if (
                isinstance(config_dict["Metrics_save_configs"]["Metrics_dict"], dict)
                and metric in config_dict["Metrics_save_configs"]["Metrics_dict"]
            ):
                if "bar_config" in config_dict["Metrics_save_configs"]["Metrics_dict"][metric]:
                    bar_configs = config_dict["Metrics_save_configs"]["Metrics_dict"][metric]["bar_config"]

        title = get_plot_title(plot_title, section, metric)

        args = {
            "df_flat": df_flat,
            "metric_name": metric,
            "metric_measurement_unit": measurement_unit,
            "section": section,
            "bar_configs": bar_configs,
        }

        for plot in PLOTS:
            if plot == "bar":
                for aggr in BAR_AGGREGATORS:
                    args["aggregator"] = aggr
                    args["plot_title"] = get_plot_title(plot_title, section, metric, aggr)
                    fig = PLOTS[plot](**args)
                    plot_dict["{}_{}_{}_{}".format(aggr, metric, section, plot)] = fig
            else:
                args["plot_title"] = title
                fig = PLOTS[plot](**args)
                plot_dict["{}_{}_{}".format(metric, section, plot)] = fig

    return plot_dict


SAVE_PLOT_DICT = {"png": "write_image", "json": "write_json", "html": "write_html"}


def save_plots(
    results_folder: Union[str, PathLike],
    plot_dict: Dict[str, Figure],
    metrics: List[str],
    sections: List[Literal["testing", "validation", "experiment", "project"]],
    file_format: Literal["png", "html", "json"] = "png",
):
    """
    Save the ```Plotly`` **Figure** stored in *plot_dict*, according to the specified file format. Accepted formats are:
    ```png``, ```html`` and ``json``.

    Parameters
    ----------
    results_folder : str
        Results folder where to save the plots.
    plot_dict : Dict[str, Figure]
        Map of available ```Plotly`` figures.
    metrics : List[str]
        List of metrics to save as plots.
    sections : List[Literal['testing', 'validation', 'experiment', 'project']]
        Sections to save plots.
    file_format : Literal['png', 'html', 'json']
        File format to save the plots.
    """
    if file_format not in list(SAVE_PLOT_DICT.keys()):
        raise ValueError("Invalid file format. Expected one of: %s" % list(SAVE_PLOT_DICT.keys()))

    for metric in metrics:
        for section in sections:
            Path(results_folder).joinpath(METRICS_FOLDER_NAME, section, metric).mkdir(parents=True, exist_ok=True)
            for plot in PLOTS:
                if plot == "bar":
                    for aggr in BAR_AGGREGATORS:
                        getattr(plot_dict["{}_{}_{}_{}".format(aggr, metric, section, plot)], SAVE_PLOT_DICT[file_format])(
                            str(
                                Path(results_folder).joinpath(
                                    METRICS_FOLDER_NAME, section, metric, "{}_{}_{}.{}".format(aggr, metric, plot, file_format)
                                )
                            )
                        )

                else:
                    getattr(plot_dict["{}_{}_{}".format(metric, section, plot)], SAVE_PLOT_DICT[file_format])(
                        str(
                            Path(results_folder).joinpath(
                                METRICS_FOLDER_NAME, section, metric, "{}_{}.{}".format(metric, plot, file_format)
                            )
                        )
                    )
