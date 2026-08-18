import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_iqr_outliers(
    df,
    columns=None,
    multiplier=1.5
):
    result = df.copy()

    if columns is None:
        columns = (
            result
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )

    outlier_columns = []

    for column in columns:

        if column not in result.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            result[column]
        ):
            continue

        series = result[column]

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = (
            q1 - multiplier * iqr
        )

        upper_bound = (
            q3 + multiplier * iqr
        )

        column_flag = (
            (series < lower_bound)
            |
            (series > upper_bound)
        )

        flag_name = f"{column}_IQR_Outlier"

        result[flag_name] = (
            column_flag
            .fillna(False)
        )

        outlier_columns.append(
            flag_name
        )

    if outlier_columns:

        result["IQR_Outlier"] = (
            result[outlier_columns]
            .any(axis=1)
        )

    else:

        result["IQR_Outlier"] = False

    return result


def detect_zscore_outliers(
    df,
    columns=None,
    threshold=3.0
):
    result = df.copy()

    if columns is None:
        columns = (
            result
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )

    outlier_columns = []

    for column in columns:

        if column not in result.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            result[column]
        ):
            continue

        series = result[column]

        mean = series.mean()
        std = series.std()

        if pd.isna(std) or std == 0:
            continue

        z_score = (
            (series - mean)
            / std
        )

        flag_name = f"{column}_ZScore_Outlier"

        result[flag_name] = (
            z_score.abs()
            > threshold
        ).fillna(False)

        outlier_columns.append(
            flag_name
        )

    if outlier_columns:

        result["ZScore_Outlier"] = (
            result[outlier_columns]
            .any(axis=1)
        )

    else:

        result["ZScore_Outlier"] = False

    return result


def detect_isolation_forest_outliers(
    df,
    columns=None,
    contamination=0.05,
    random_state=42
):
    result = df.copy()

    if columns is None:
        columns = (
            result
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )

    valid_columns = []

    for column in columns:

        if column not in result.columns:
            continue

        if pd.api.types.is_numeric_dtype(
            result[column]
        ):
            valid_columns.append(
                column
            )

    if not valid_columns:

        result["IsolationForest_Outlier"] = False

        return result

    model_data = (
        result[valid_columns]
        .copy()
    )

    model_data = (
        model_data
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    model_data = (
        model_data
        .fillna(
            model_data.median()
        )
    )

    if model_data.empty:

        result["IsolationForest_Outlier"] = False

        return result

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        model_data
    )

    contamination = float(
        np.clip(
            contamination,
            0.001,
            0.5
        )
    )

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200
    )

    predictions = model.fit_predict(
        scaled_data
    )

    result["IsolationForest_Outlier"] = (
        predictions == -1
    )

    return result


def get_outlier_summary(
    df,
    flag_column=None
):
    if flag_column is not None:

        if flag_column not in df.columns:

            raise ValueError(
                f"Column '{flag_column}' does not exist."
            )

        mask = (
            df[flag_column]
            .astype(bool)
        )

        return {
            "outliers": int(
                mask.sum()
            ),
            "percentage": round(
                float(
                    mask.mean() * 100
                ),
                2
            ),
            "total_rows": len(df)
        }


    flag_columns = [
        column
        for column in df.columns
        if (
            "outlier" in column.lower()
            or "anomaly" in column.lower()
        )
    ]

    summary = []

    for column in flag_columns:

        mask = (
            df[column]
            .astype(bool)
        )

        summary.append({
            "Method": column,
            "Outliers": int(
                mask.sum()
            ),
            "Percentage": round(
                float(
                    mask.mean() * 100
                ),
                2
            )
        })

    return pd.DataFrame(
        summary
    )