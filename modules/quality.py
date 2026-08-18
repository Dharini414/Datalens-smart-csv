import pandas as pd
import numpy as np


def get_basic_metrics(df):
    """
    Calculate basic dataset metrics.
    """

    rows = len(df)
    columns = len(df.columns)

    total_cells = rows * columns

    missing_cells = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        ).columns
    )

    unique_cells = 0

    for column in df.columns:

        unique_cells += (
            df[column]
            .nunique(
                dropna=True
            )
        )

    completeness = (
        (
            total_cells - missing_cells
        )
        / total_cells
        * 100
        if total_cells > 0
        else 100
    )

    duplicate_rate = (
        duplicate_rows
        / rows
        * 100
        if rows > 0
        else 0
    )

    return {
        "rows": rows,
        "columns": columns,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_rate": (
            round(
                missing_cells
                / total_cells
                * 100,
                2
            )
            if total_cells > 0
            else 0
        ),
        "completeness": round(
            completeness,
            2
        ),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": round(
            duplicate_rate,
            2
        ),
        "unique_cells": unique_cells,
        "numerical_columns": len(
            numeric_columns
        ),
        "categorical_columns": len(
            categorical_columns
        )
    }


def calculate_uniqueness_score(df):
    """
    Estimate how unique the dataset values are.

    Higher score means fewer repeated values
    across columns.
    """

    if df.empty or len(df.columns) == 0:

        return 100.0

    scores = []

    for column in df.columns:

        non_null = (
            df[column]
            .dropna()
        )

        if len(non_null) == 0:

            continue

        unique_ratio = (
            non_null.nunique()
            / len(non_null)
        )

        scores.append(
            unique_ratio * 100
        )

    if not scores:

        return 100.0

    return round(
        np.mean(scores),
        2
    )


def calculate_duplicate_score(df):
    """
    Calculate a score based on duplicate rows.

    100 = no duplicates
    0 = all rows are duplicates
    """

    rows = len(df)

    if rows == 0:

        return 100.0

    duplicate_rate = (
        df.duplicated().sum()
        / rows
        * 100
    )

    return round(
        max(
            0,
            100 - duplicate_rate
        ),
        2
    )


def calculate_completeness_score(df):
    """
    Calculate percentage of non-missing cells.
    """

    total_cells = (
        len(df)
        * len(df.columns)
    )

    if total_cells == 0:

        return 100.0

    missing_cells = (
        df.isna()
        .sum()
        .sum()
    )

    score = (
        total_cells - missing_cells
    ) / total_cells * 100

    return round(
        score,
        2
    )


def calculate_consistency_score(df):
    """
    Estimate consistency based on
    problematic values.

    Checks:
    - Infinite values
    - Mixed data types in object columns
    - Empty strings
    """

    if df.empty:

        return 100.0

    penalties = 0

    total_columns = len(
        df.columns
    )

    if total_columns == 0:

        return 100.0

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    infinite_values = 0

    if len(numeric_columns) > 0:

        infinite_values = np.isinf(
            df[numeric_columns]
            .select_dtypes(
                include="number"
            )
            .to_numpy()
        ).sum()

    if infinite_values > 0:

        penalties += min(
            infinite_values
            / max(len(df), 1)
            * 10,
            20
        )

    empty_string_count = 0

    object_columns = (
        df.select_dtypes(
            include=[
                "object",
                "string"
            ]
        ).columns
    )

    for column in object_columns:

        empty_string_count += (
            df[column]
            .astype("string")
            .str.strip()
            .eq("")
            .sum()
        )

    if empty_string_count > 0:

        penalties += min(
            empty_string_count
            / max(len(df), 1)
            * 10,
            20
        )

    return round(
        max(
            0,
            100 - penalties
        ),
        2
    )


def calculate_outlier_rate(
    df,
    method="IQR"
):
    """
    Estimate the percentage of rows
    containing numerical outliers.

    Uses the IQR method internally.
    """

    numerical_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    if len(numerical_columns) == 0:

        return 0.0

    if len(df) == 0:

        return 0.0

    outlier_rows = set()

    for column in numerical_columns:

        series = df[column]

        if series.dropna().empty:

            continue

        q1 = series.quantile(
            0.25
        )

        q3 = series.quantile(
            0.75
        )

        iqr = q3 - q1

        if iqr == 0:

            continue

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        mask = (
            (series < lower_bound)
            |
            (series > upper_bound)
        )

        outlier_rows.update(
            df.index[mask]
        )

    return round(
        len(outlier_rows)
        / len(df)
        * 100,
        2
    )


def calculate_outlier_score(df):
    """
    Convert outlier rate into a score.

    Lower outlier rate gives a higher score.
    """

    outlier_rate = (
        calculate_outlier_rate(df)
    )

    return round(
        max(
            0,
            100 - outlier_rate
        ),
        2
    )


def calculate_quality_score(
    df,
    include_outliers=True
):
    """
    Calculate the overall dataset quality score.

    Components:
    - Completeness
    - Duplicate quality
    - Consistency
    - Outlier quality
    """

    completeness = (
        calculate_completeness_score(df)
    )

    duplicate_score = (
        calculate_duplicate_score(df)
    )

    consistency = (
        calculate_consistency_score(df)
    )

    outlier_score = (
        calculate_outlier_score(df)
        if include_outliers
        else 100
    )

    if include_outliers:

        score = (
            completeness * 0.35
            + duplicate_score * 0.20
            + consistency * 0.20
            + outlier_score * 0.25
        )

    else:

        score = (
            completeness * 0.45
            + duplicate_score * 0.25
            + consistency * 0.30
        )

    return round(
        score,
        2
    )


def get_quality_report(
    df,
    include_outliers=True
):
    """
    Generate a complete quality report.
    """

    metrics = get_basic_metrics(
        df
    )

    completeness = (
        calculate_completeness_score(df)
    )

    uniqueness = (
        calculate_uniqueness_score(df)
    )

    duplicate_score = (
        calculate_duplicate_score(df)
    )

    consistency = (
        calculate_consistency_score(df)
    )

    outlier_rate = (
        calculate_outlier_rate(df)
        if include_outliers
        else 0
    )

    outlier_score = (
        calculate_outlier_score(df)
        if include_outliers
        else 100
    )

    overall_score = (
        calculate_quality_score(
            df,
            include_outliers
        )
    )

    return {
        "metrics": metrics,
        "completeness_score": completeness,
        "uniqueness_score": uniqueness,
        "duplicate_score": duplicate_score,
        "consistency_score": consistency,
        "outlier_rate": outlier_rate,
        "outlier_score": outlier_score,
        "overall_score": overall_score
    }


def compare_quality(
    original_df,
    cleaned_df
):
    """
    Compare data quality before and after
    cleaning.
    """

    original = get_quality_report(
        original_df
    )

    cleaned = get_quality_report(
        cleaned_df
    )

    comparison = []

    metrics = [
        (
            "Overall Quality",
            original["overall_score"],
            cleaned["overall_score"]
        ),
        (
            "Completeness",
            original["completeness_score"],
            cleaned["completeness_score"]
        ),
        (
            "Uniqueness",
            original["uniqueness_score"],
            cleaned["uniqueness_score"]
        ),
        (
            "Duplicate Quality",
            original["duplicate_score"],
            cleaned["duplicate_score"]
        ),
        (
            "Consistency",
            original["consistency_score"],
            cleaned["consistency_score"]
        ),
        (
            "Outlier Quality",
            original["outlier_score"],
            cleaned["outlier_score"]
        )
    ]

    for metric, before, after in metrics:

        comparison.append({
            "Metric": metric,
            "Before": round(
                before,
                2
            ),
            "After": round(
                after,
                2
            ),
            "Change": round(
                after - before,
                2
            )
        })

    return pd.DataFrame(
        comparison
    )


def get_cleaning_impact(
    original_df,
    cleaned_df
):
    """
    Calculate the practical impact of
    cleaning operations.
    """

    original_missing = int(
        original_df.isna()
        .sum()
        .sum()
    )

    cleaned_missing = int(
        cleaned_df.isna()
        .sum()
        .sum()
    )

    original_duplicates = int(
        original_df
        .duplicated()
        .sum()
    )

    cleaned_duplicates = int(
        cleaned_df
        .duplicated()
        .sum()
    )

    return {
        "rows_removed": (
            len(original_df)
            - len(cleaned_df)
        ),

        "columns_removed": (
            len(original_df.columns)
            - len(cleaned_df.columns)
        ),

        "missing_values_removed": max(
            0,
            original_missing
            - cleaned_missing
        ),

        "duplicates_removed": max(
            0,
            original_duplicates
            - cleaned_duplicates
        ),

        "original_quality": (
            calculate_quality_score(
                original_df
            )
        ),

        "cleaned_quality": (
            calculate_quality_score(
                cleaned_df
            )
        ),

        "quality_improvement": round(
            calculate_quality_score(
                cleaned_df
            )
            - calculate_quality_score(
                original_df
            ),
            2
        )
    }