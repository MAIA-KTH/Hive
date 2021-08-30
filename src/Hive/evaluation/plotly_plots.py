from pathlib import Path
from typing import Dict, Any, List, Callable

import plotly.express as px
from pandas import DataFrame
from plotly.graph_objects import Figure

from Hive.evaluation import DEFAULT_METRIC_UNITS, DEFAULT_BAR_CONFIGS, METRICS_FOLDER_NAME
from Hive.evaluation.io_metric_results import read_dataframe

BAR_AGGREGATORS = ["min", "max", "mean"]


def get_plotly_histo(
        df_flat: DataFrame, metric_name: str, metric_measurement_unit: str, section: str, plot_title: str, **kwargs
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
    section : str
        Specified section for the plot.
    plot_title : str
        Plot title.
    kwargs

    Returns
    -------
    Figure
        ```Plotly`` :py:class`px.histogram` for the given metric and section.
    """
    if section == "experiment":
        color_value = "Section"
        facet = None
    elif section == "project":
        color_value = "Experiment"
        facet = "Section"
    else:
        color_value = "Label"
        facet = None

    fig_histo = px.histogram(
        df_flat,
        x=metric_name,
        color=color_value,
        facet_row=facet,
        labels={
            metric_name: metric_name + " " + metric_measurement_unit,
        },
        title=plot_title,
    )

    return fig_histo


def get_plotly_average_bar(
        df_flat: DataFrame,
        metric_name: str,
        metric_measurement_unit: str,
        section: str,
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
    section : str
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

    if section == "experiment":
        y_value = "Section"
        facet_row = "Label"
        facet_col = None
    elif section == "project":
        y_value = "Experiment"
        facet_row = "Label"
        facet_col = "Section"
    else:
        y_value = "Label"
        facet_row = None
        facet_col = None

    key_cols = ["Label", "Section", "Experiment"]

    df_flat = df_flat.groupby(key_cols).agg(aggregator).reset_index()

    text = []
    for index, row in df_flat.iterrows():
        text.append("{:.3}".format(row[metric_name]))

    colors = "Inferno"
    if bar_configs is not None:

        if "colors" in bar_configs:
            colors = bar_configs["colors"]

    fig_bar = px.bar(
        df_flat,
        y=y_value,
        x=metric_name,
        orientation="h",
        color=metric_name,
        facet_row=facet_row,
        facet_col=facet_col,
        barmode="group",
        color_continuous_scale=colors,
        text=text,
        labels={
            metric_name: metric_name + " " + metric_measurement_unit,
        },
        title=plot_title,
    )

    return fig_bar


def get_plotly_boxplot(
        df_flat: DataFrame, metric_name: str, metric_measurement_unit: str, section: str, plot_title: str, **kwargs
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

    if section == "experiment":
        color_value = "Section"
        facet = None
    elif section == "project":
        color_value = "Experiment"
        facet = "Section"
    else:
        color_value = "Label"
        facet = None

    fig_boxplot = px.box(
        df_flat,
        x="Label",
        y=metric_name,
        facet_row=facet,
        color=color_value,
        labels={
            metric_name: metric_name + " " + metric_measurement_unit,
        },
        title=plot_title,
    )

    return fig_boxplot


PLOTS = {"boxplot": get_plotly_boxplot, "bar": get_plotly_average_bar,
         "histo": get_plotly_histo}  # type: Dict[str, Callable]


def get_plot_title(main_title: str, section: str, metric: str, aggr="") -> str:
    """
    Compose and return the plot title, according to the specified metric and section.
    The title is composed as follows:
        .. math::
            main_title, [ + Section Set, ] [ + aggregator ] + metric

    Parameters
    ----------
    main_title : str
        Initial strings for the title, to be prepended.
    section : str
        Section for the title.
    metric: str
        Metric for the title.
    aggr: str
        Optional metric aggregator. Examples: [```mean``, ```max``, ```min``].

    Returns
    -------
    str
        Plot title for the given section and metric.
    """
    section_dataset = ""
    if section in ["validation", "testing"]:
        section_dataset = " {} Set,".format(section.capitalize())
    if aggr != "":
        aggr = " " + aggr
    title = "{},{}{} {}".format(main_title.replace("_", " "), section_dataset, aggr, metric)
    return title


def create_plots(
        config_dict: Dict[str, Any],
        df_paths: Dict[str, str],
        metrics: List[str],
        plot_title: str,
        sections: List[str],
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
    sections : List[str]
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

            if "Metrics_save_configs" in config_dict and "Metrics_dict" in config_dict["Metrics_save_configs"]:
                metrics_dict = config_dict["Metrics_save_configs"]["Metrics_dict"]
                if metrics_dict is not None and isinstance(metrics_dict, dict) and "m_unit" in metrics_dict[metric]:
                    measurement_unit = metrics_dict[metric]["m_unit"]
                else:
                    if metric in DEFAULT_METRIC_UNITS:
                        measurement_unit = DEFAULT_METRIC_UNITS[metric]

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
                "plot_title": title,
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
                    fig = PLOTS[plot](**args)
                    plot_dict["{}_{}_{}".format(metric, section, plot)] = fig

    return plot_dict


SAVE_PLOT_DICT = {"png": "write_image", "json": "write_json", "html": "write_html"}


def save_plots(
        results_folder: str, plot_dict: Dict[str, Figure], metrics: List[str], sections: List[str],
        file_format: str = "png"
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
    sections : List[str]
        Sections to save plots.
    file_format : str
        File format to save the plots.
    """
    if file_format not in list(SAVE_PLOT_DICT.keys()):
        raise ValueError("Invalid file format. Expected one of: %s" % list(SAVE_PLOT_DICT.keys()))

    for metric in metrics:
        for section in sections:
            for plot in PLOTS:
                if plot == "bar":
                    for aggr in BAR_AGGREGATORS:
                        getattr(plot_dict["{}_{}_{}_{}".format(aggr, metric, section, plot)],
                                SAVE_PLOT_DICT[file_format])(
                            str(
                                Path(results_folder).joinpath(
                                    METRICS_FOLDER_NAME, section, metric,
                                    "{}_{}_{}.{}".format(aggr, metric, plot, file_format)
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
