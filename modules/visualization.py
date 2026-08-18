import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def create_histogram(
    df,
    column,
    bins=30
):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        df[column]
    ):
        raise ValueError(
            "Histogram requires a numerical column."
        )

    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        marginal="box",
        title=f"Distribution of {column}"
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title=column,
        yaxis_title="Frequency"
    )

    return fig


def create_boxplot(
    df,
    column,
    category=None
):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        df[column]
    ):
        raise ValueError(
            "Box plot requires a numerical column."
        )

    if (
        category is not None
        and category in df.columns
    ):

        fig = px.box(
            df,
            x=category,
            y=column,
            points="outliers",
            title=f"{column} by {category}"
        )

    else:

        fig = px.box(
            df,
            y=column,
            points="outliers",
            title=f"Box Plot of {column}"
        )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    return fig


def create_scatter_plot(
    df,
    x_column,
    y_column,
    color_column=None,
    size_column=None
):
    if x_column not in df.columns:
        raise ValueError(
            f"Column '{x_column}' does not exist."
        )

    if y_column not in df.columns:
        raise ValueError(
            f"Column '{y_column}' does not exist."
        )

    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=(
            color_column
            if color_column in df.columns
            else None
        ),
        size=(
            size_column
            if size_column in df.columns
            else None
        ),
        trendline="ols",
        title=f"{y_column} vs {x_column}"
    )

    fig.update_layout(
        template="plotly_white",
        height=550
    )

    return fig


def create_bar_chart(
    df,
    x_column,
    y_column=None,
    aggregation="count",
    orientation="v"
):
    if x_column not in df.columns:
        raise ValueError(
            f"Column '{x_column}' does not exist."
        )

    if y_column is not None:

        if y_column not in df.columns:
            raise ValueError(
                f"Column '{y_column}' does not exist."
            )

        if aggregation == "sum":

            grouped = (
                df.groupby(
                    x_column,
                    dropna=False
                )[y_column]
                .sum()
                .reset_index()
            )

        elif aggregation == "mean":

            grouped = (
                df.groupby(
                    x_column,
                    dropna=False
                )[y_column]
                .mean()
                .reset_index()
            )

        elif aggregation == "median":

            grouped = (
                df.groupby(
                    x_column,
                    dropna=False
                )[y_column]
                .median()
                .reset_index()
            )

        elif aggregation == "max":

            grouped = (
                df.groupby(
                    x_column,
                    dropna=False
                )[y_column]
                .max()
                .reset_index()
            )

        elif aggregation == "min":

            grouped = (
                df.groupby(
                    x_column,
                    dropna=False
                )[y_column]
                .min()
                .reset_index()
            )

        else:

            grouped = (
                df[x_column]
                .value_counts(
                    dropna=False
                )
                .reset_index()
            )

            grouped.columns = [
                x_column,
                "Count"
            ]

            y_column = "Count"

    else:

        grouped = (
            df[x_column]
            .value_counts(
                dropna=False
            )
            .reset_index()
        )

        grouped.columns = [
            x_column,
            "Count"
        ]

        y_column = "Count"


    if orientation == "h":

        fig = px.bar(
            grouped,
            x=y_column,
            y=x_column,
            orientation="h",
            title=f"{y_column} by {x_column}"
        )

    else:

        fig = px.bar(
            grouped,
            x=x_column,
            y=y_column,
            title=f"{y_column} by {x_column}"
        )


    fig.update_layout(
        template="plotly_white",
        height=500
    )

    return fig


def create_line_chart(
    df,
    x_column,
    y_column,
    color_column=None
):
    if x_column not in df.columns:
        raise ValueError(
            f"Column '{x_column}' does not exist."
        )

    if y_column not in df.columns:
        raise ValueError(
            f"Column '{y_column}' does not exist."
        )

    chart_df = df.copy()

    try:

        if pd.api.types.is_object_dtype(
            chart_df[x_column]
        ):

            converted = pd.to_datetime(
                chart_df[x_column],
                errors="coerce"
            )

            if converted.notna().sum() > 0:

                chart_df[x_column] = converted

    except Exception:

        pass


    chart_df = (
        chart_df
        .sort_values(
            by=x_column
        )
    )


    fig = px.line(
        chart_df,
        x=x_column,
        y=y_column,
        color=(
            color_column
            if color_column in chart_df.columns
            else None
        ),
        markers=True,
        title=f"{y_column} over {x_column}"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    return fig


def create_heatmap(
    df,
    columns=None,
    method="pearson"
):
    if columns is None:

        columns = (
            df.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

    valid_columns = [
        column
        for column in columns
        if (
            column in df.columns
            and
            pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]

    if len(valid_columns) < 2:

        raise ValueError(
            "At least two numerical columns are required for a correlation heatmap."
        )

    correlation = (
        df[valid_columns]
        .corr(
            method=method
        )
    )

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap"
    )

    fig.update_layout(
        height=650
    )

    return fig


def create_pie_chart(
    df,
    column,
    top_n=10
):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    counts = (
        df[column]
        .value_counts(
            dropna=False
        )
        .head(top_n)
        .reset_index()
    )

    counts.columns = [
        column,
        "Count"
    ]

    fig = px.pie(
        counts,
        names=column,
        values="Count",
        title=f"Distribution of {column}"
    )

    fig.update_layout(
        height=500
    )

    return fig


def create_area_chart(
    df,
    x_column,
    y_column,
    color_column=None
):
    if x_column not in df.columns:
        raise ValueError(
            f"Column '{x_column}' does not exist."
        )

    if y_column not in df.columns:
        raise ValueError(
            f"Column '{y_column}' does not exist."
        )

    fig = px.area(
        df,
        x=x_column,
        y=y_column,
        color=(
            color_column
            if color_column in df.columns
            else None
        ),
        title=f"{y_column} over {x_column}"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    return fig


def create_violin_plot(
    df,
    column,
    category=None
):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        df[column]
    ):
        raise ValueError(
            "Violin plot requires a numerical column."
        )

    if (
        category is not None
        and category in df.columns
    ):

        fig = px.violin(
            df,
            x=category,
            y=column,
            box=True,
            points="outliers",
            title=f"{column} distribution by {category}"
        )

    else:

        fig = px.violin(
            df,
            y=column,
            box=True,
            points="outliers",
            title=f"Distribution of {column}"
        )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    return fig


def create_pairwise_scatter(
    df,
    columns=None
):
    if columns is None:

        columns = (
            df.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

    valid_columns = [
        column
        for column in columns
        if (
            column in df.columns
            and
            pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]

    if len(valid_columns) < 2:

        raise ValueError(
            "At least two numerical columns are required."
        )

    fig = px.scatter_matrix(
        df,
        dimensions=valid_columns,
        title="Numerical Feature Relationships"
    )

    fig.update_layout(
        height=800
    )

    return fig