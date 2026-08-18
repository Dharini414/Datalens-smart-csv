import pandas as pd
import numpy as np


def get_descriptive_statistics(
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

    if not columns:

        return pd.DataFrame()

    valid_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    if not valid_columns:

        return pd.DataFrame()

    statistics = (
        df[valid_columns]
        .describe()
        .T
        .reset_index()
    )

    statistics = statistics.rename(
        columns={
            "index": "Column"
        }
    )

    statistics["missing"] = [
        int(
            df[column]
            .isna()
            .sum()
        )
        for column in valid_columns
    ]

    statistics["missing_%"] = [
        round(
            df[column]
            .isna()
            .mean()
            * 100,
            2
        )
        for column in valid_columns
    ]

    statistics["unique"] = [
        int(
            df[column]
            .nunique(
                dropna=True
            )
        )
        for column in valid_columns
    ]

    return statistics


def get_distribution_summary(
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

    results = []

    for column in columns:

        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

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
            "Mean": round(
                float(
                    series.mean()
                ),
                4
            ),
            "Median": round(
                float(
                    series.median()
                ),
                4
            ),
            "Std Dev": round(
                float(
                    series.std()
                ),
                4
            ),
            "Variance": round(
                float(
                    series.var()
                ),
                4
            ),
            "Minimum": round(
                float(
                    series.min()
                ),
                4
            ),
            "Maximum": round(
                float(
                    series.max()
                ),
                4
            ),
            "Skewness": round(
                float(
                    series.skew()
                ),
                4
            ),
            "Kurtosis": round(
                float(
                    series.kurt()
                ),
                4
            )
        })

    return pd.DataFrame(
        results
    )


def get_categorical_summary(
    df,
    columns=None,
    top_n=10
):
    if columns is None:

        columns = (
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

    results = []

    for column in columns:

        if column not in df.columns:
            continue

        series = df[column]

        value_counts = (
            series
            .value_counts(
                dropna=False
            )
            .head(top_n)
        )

        top_value = None
        top_frequency = 0

        if not value_counts.empty:

            top_value = str(
                value_counts.index[0]
            )

            top_frequency = int(
                value_counts.iloc[0]
            )

        results.append({
            "Column": column,
            "Unique Values": int(
                series.nunique(
                    dropna=True
                )
            ),
            "Missing": int(
                series.isna().sum()
            ),
            "Missing %": round(
                float(
                    series.isna().mean()
                    * 100
                ),
                2
            ),
            "Top Value": top_value,
            "Top Frequency": top_frequency
        })

    return pd.DataFrame(
        results
    )


def get_correlation_matrix(
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

        return pd.DataFrame()

    return (
        df[valid_columns]
        .corr(
            method=method
        )
    )


def get_strong_correlations(
    df,
    threshold=0.7,
    columns=None
):
    correlation_matrix = (
        get_correlation_matrix(
            df,
            columns
        )
    )

    if correlation_matrix.empty:

        return pd.DataFrame(
            columns=[
                "Column 1",
                "Column 2",
                "Correlation"
            ]
        )

    results = []

    columns = (
        correlation_matrix
        .columns
        .tolist()
    )

    for i in range(
        len(columns)
    ):

        for j in range(
            i + 1,
            len(columns)
        ):

            value = (
                correlation_matrix
                .iloc[i, j]
            )

            if pd.isna(value):
                continue

            if abs(value) >= threshold:

                results.append({
                    "Column 1": columns[i],
                    "Column 2": columns[j],
                    "Correlation": round(
                        float(value),
                        4
                    ),
                    "Strength": (
                        "Strong Positive"
                        if value > 0
                        else "Strong Negative"
                    )
                })

    return pd.DataFrame(
        results
    )


def get_dataset_overview(
    df
):
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

    total_cells = (
        len(df)
        * len(df.columns)
    )

    missing_cells = int(
        df.isna()
        .sum()
        .sum()
    )

    completeness = (
        (
            total_cells
            - missing_cells
        )
        / total_cells
        * 100
        if total_cells > 0
        else 0
    )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": len(
            numeric_columns
        ),
        "categorical_columns": len(
            categorical_columns
        ),
        "datetime_columns": len(
            datetime_columns
        ),
        "missing_values": missing_cells,
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "completeness": round(
            completeness,
            2
        )
    }