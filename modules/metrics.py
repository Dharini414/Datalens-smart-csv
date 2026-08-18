import pandas as pd
import numpy as np


def calculate_dataset_metrics(df):

    if df is None:

        return {}

    rows = len(df)

    columns = len(df.columns)

    total_cells = (
        rows * columns
    )

    missing_values = int(
        df.isna()
        .sum()
        .sum()
    )

    duplicate_rows = int(
        df.duplicated()
        .sum()
    )

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    datetime_columns = (
        df.select_dtypes(
            include=[
                "datetime",
                "datetimetz"
            ]
        )
        .columns
        .tolist()
    )

    non_missing_cells = (
        total_cells
        - missing_values
    )

    completeness = (
        non_missing_cells
        / total_cells
        * 100
        if total_cells > 0
        else 0
    )

    duplicate_percentage = (
        duplicate_rows
        / rows
        * 100
        if rows > 0
        else 0
    )

    unique_rows = (
        rows - duplicate_rows
    )

    uniqueness = (
        unique_rows
        / rows
        * 100
        if rows > 0
        else 0
    )

    memory_usage = (
        df.memory_usage(
            deep=True
        )
        .sum()
    )

    memory_mb = (
        memory_usage
        / (
            1024 * 1024
        )
    )

    return {

        "Rows": rows,

        "Columns": columns,

        "Total Cells": total_cells,

        "Missing Values": missing_values,

        "Duplicate Rows": duplicate_rows,

        "Numerical Columns": len(
            numeric_columns
        ),

        "Categorical Columns": len(
            categorical_columns
        ),

        "Datetime Columns": len(
            datetime_columns
        ),

        "Completeness %": round(
            completeness,
            2
        ),

        "Uniqueness %": round(
            uniqueness,
            2
        ),

        "Duplicate %": round(
            duplicate_percentage,
            2
        ),

        "Memory Usage MB": round(
            memory_mb,
            2
        )
    }


def get_column_metrics(
    df
):

    results = []

    for column in df.columns:

        series = df[column]

        missing = int(
            series.isna()
            .sum()
        )

        non_missing = (
            len(series)
            - missing
        )

        unique = int(
            series.nunique(
                dropna=True
            )
        )

        missing_percentage = (
            missing
            / len(series)
            * 100
            if len(series) > 0
            else 0
        )

        uniqueness_percentage = (
            unique
            / non_missing
            * 100
            if non_missing > 0
            else 0
        )

        results.append({

            "Column": column,

            "Data Type": str(
                series.dtype
            ),

            "Rows": len(series),

            "Non-Null": non_missing,

            "Missing": missing,

            "Missing %": round(
                missing_percentage,
                2
            ),

            "Unique Values": unique,

            "Unique %": round(
                uniqueness_percentage,
                2
            )
        })

    return pd.DataFrame(
        results
    )


def get_numeric_metrics(
    df
):

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    results = []

    for column in numeric_columns:

        series = (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            .dropna()
        )

        if series.empty:

            continue

        results.append({

            "Column": column,

            "Count": int(
                series.count()
            ),

            "Mean": float(
                series.mean()
            ),

            "Median": float(
                series.median()
            ),

            "Std Dev": float(
                series.std()
            ),

            "Variance": float(
                series.var()
            ),

            "Minimum": float(
                series.min()
            ),

            "Maximum": float(
                series.max()
            ),

            "Range": float(
                series.max()
                - series.min()
            ),

            "Q1": float(
                series.quantile(
                    0.25
                )
            ),

            "Q3": float(
                series.quantile(
                    0.75
                )
            ),

            "IQR": float(
                series.quantile(
                    0.75
                )
                -
                series.quantile(
                    0.25
                )
            ),

            "Skewness": float(
                series.skew()
            ),

            "Kurtosis": float(
                series.kurt()
            )
        })

    return pd.DataFrame(
        results
    )


def compare_datasets(
    original_df,
    cleaned_df
):

    original_metrics = (
        calculate_dataset_metrics(
            original_df
        )
    )

    cleaned_metrics = (
        calculate_dataset_metrics(
            cleaned_df
        )
    )

    metrics = [
        "Rows",
        "Columns",
        "Missing Values",
        "Duplicate Rows",
        "Completeness %",
        "Uniqueness %"
    ]

    results = []

    for metric in metrics:

        original_value = (
            original_metrics
            .get(
                metric,
                0
            )
        )

        cleaned_value = (
            cleaned_metrics
            .get(
                metric,
                0
            )
        )

        results.append({

            "Metric": metric,

            "Original": original_value,

            "Cleaned": cleaned_value,

            "Change": (
                cleaned_value
                - original_value
            )
        })

    return pd.DataFrame(
        results
    )