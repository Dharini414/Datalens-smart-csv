import pandas as pd


def detect_datetime_columns(df):
    """
    Detect columns that contain datetime values.
    """

    datetime_columns = []

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        if pd.api.types.is_object_dtype(series):

            sample = series.dropna().head(100)

            if sample.empty:
                continue

            converted = pd.to_datetime(
                sample,
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            if valid_ratio >= 0.80:
                datetime_columns.append(column)

    return datetime_columns


def detect_column_types(df):
    """
    Automatically classify dataset columns.
    """

    datetime_columns = detect_datetime_columns(df)

    numerical_columns = []
    categorical_columns = []
    boolean_columns = []
    other_columns = []

    for column in df.columns:

        series = df[column]

        if column in datetime_columns:

            continue

        if pd.api.types.is_bool_dtype(series):

            boolean_columns.append(column)

        elif pd.api.types.is_numeric_dtype(series):

            numerical_columns.append(column)

        elif (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):

            categorical_columns.append(column)

        else:

            other_columns.append(column)

    return {
        "numerical": numerical_columns,
        "categorical": categorical_columns,
        "datetime": datetime_columns,
        "boolean": boolean_columns,
        "other": other_columns
    }


def get_column_profile(df):
    """
    Generate detailed information for every column.
    """

    column_types = detect_column_types(df)

    profile = []

    for column in df.columns:

        series = df[column]

        missing_count = int(
            series.isna().sum()
        )

        missing_percentage = (
            missing_count / len(df) * 100
            if len(df) > 0
            else 0
        )

        unique_count = int(
            series.nunique(dropna=True)
        )

        unique_percentage = (
            unique_count / len(df) * 100
            if len(df) > 0
            else 0
        )

        if column in column_types["numerical"]:

            data_type = "Numerical"

        elif column in column_types["categorical"]:

            data_type = "Categorical"

        elif column in column_types["datetime"]:

            data_type = "Datetime"

        elif column in column_types["boolean"]:

            data_type = "Boolean"

        else:

            data_type = "Other"

        profile.append({
            "Column": column,
            "Type": data_type,
            "Pandas Type": str(series.dtype),
            "Missing": missing_count,
            "Missing %": round(
                missing_percentage,
                2
            ),
            "Unique": unique_count,
            "Unique %": round(
                unique_percentage,
                2
            )
        })

    return pd.DataFrame(profile)


def get_dataset_profile(df):
    """
    Generate high-level dataset information.
    """

    column_types = detect_column_types(df)

    rows = len(df)
    columns = len(df.columns)

    total_cells = rows * columns

    missing_cells = int(
        df.isna().sum().sum()
    )

    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells > 0
        else 0
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(
            dropna=False
        ) <= 1
    ]

    high_cardinality_columns = [
        column
        for column in df.columns
        if (
            rows > 0
            and (
                df[column].nunique(
                    dropna=True
                ) / rows
            ) >= 0.90
        )
    ]

    memory_usage = (
        df.memory_usage(
            deep=True
        ).sum()
        / (1024 ** 2)
    )

    return {
        "rows": rows,
        "columns": columns,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percentage": round(
            missing_percentage,
            2
        ),
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "memory_usage_mb": round(
            memory_usage,
            2
        ),
        "numerical_columns": column_types[
            "numerical"
        ],
        "categorical_columns": column_types[
            "categorical"
        ],
        "datetime_columns": column_types[
            "datetime"
        ],
        "boolean_columns": column_types[
            "boolean"
        ],
        "other_columns": column_types[
            "other"
        ]
    }


def calculate_health_score(profile):
    """
    Calculate a simple dataset quality score.
    """

    score = 100

    missing_penalty = min(
        profile["missing_percentage"] * 0.5,
        25
    )

    duplicate_percentage = (
        profile["duplicate_rows"]
        / profile["rows"]
        * 100
        if profile["rows"] > 0
        else 0
    )

    duplicate_penalty = min(
        duplicate_percentage * 0.5,
        15
    )

    constant_penalty = min(
        len(
            profile["constant_columns"]
        ) * 5,
        15
    )

    score -= missing_penalty
    score -= duplicate_penalty
    score -= constant_penalty

    return max(
        0,
        min(
            100,
            round(score)
        )
    )