from __future__ import annotations

"""Utilities for converting row-wise FEV datasets into Chronos long format."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_FEV_DATASET_REPO = "autogluon/fev_datasets"


def infer_fev_value_columns(
    row: Mapping[str, Any],
    *,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
) -> list[str]:
    """
    Infer array-valued data columns from a single FEV row.

    FEV datasets store one row per time series. The `timestamp` column contains a
    full time index and the remaining sequence-valued columns contain the series
    payload, such as `target` or multivariate target dimensions.
    """
    value_columns: list[str] = []
    for column, value in row.items():
        if column in {id_column, timestamp_column}:
            continue
        if _is_sequence_like(value):
            value_columns.append(column)

    if not value_columns:
        raise ValueError(
            "Could not infer any sequence-valued FEV columns. "
            f"Expected at least one array column besides {id_column!r} and {timestamp_column!r}."
        )

    return value_columns


def infer_fev_target_columns(value_columns: Sequence[str]) -> list[str]:
    """
    Choose Chronos target columns from the available FEV value columns.

    If the dataset exposes a canonical `target` column, we keep inference
    univariate by default. Otherwise, all value columns are treated as targets.
    """
    if "target" in value_columns:
        return ["target"]
    return list(value_columns)


def fev_rows_to_long_dataframe(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    value_columns: str | Sequence[str] | None = None,
) -> tuple["pd.DataFrame", list[str]]:
    """
    Convert row-wise FEV records into a Chronos-compatible long dataframe.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for FEV dataframe conversion. "
            "Please install it with `pip install pandas`."
        ) from exc

    if len(rows) == 0:
        raise ValueError("Cannot convert an empty FEV dataset.")

    first_row = rows[0]
    resolved_value_columns = _normalize_value_columns(
        value_columns=value_columns,
        row=first_row,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    frames = [
        _fev_row_to_frame(
            row=row,
            id_column=id_column,
            timestamp_column=timestamp_column,
            value_columns=resolved_value_columns,
        )
        for row in rows
    ]

    df = pd.concat(frames, ignore_index=True)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    return df, resolved_value_columns


def load_fev_long_dataframe(
    dataset_name: str,
    *,
    split: str = "train",
    repo_id: str = DEFAULT_FEV_DATASET_REPO,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    value_columns: str | Sequence[str] | None = None,
    max_series: int | None = None,
) -> tuple["pd.DataFrame", list[str]]:
    """
    Load an FEV dataset from Hugging Face and convert it to long format.

    Parameters
    ----------
    dataset_name
        Hugging Face config name inside `autogluon/fev_datasets`, for example
        `epf_de` or `ett_small_15min`.
    split
        Dataset split to load, usually `train`, `validation`, or `test`.
    repo_id
        Dataset repository on Hugging Face Hub.
    value_columns
        Sequence-valued columns to include in the long dataframe. When omitted,
        all sequence-valued columns except `id` and `timestamp` are used.
    max_series
        Optional cap on the number of time series rows loaded from the dataset.
    """
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(
            "The Hugging Face `datasets` package is required for loading FEV datasets. "
            "Please install it with `pip install datasets pyarrow`."
        ) from exc

    dataset = datasets.load_dataset(repo_id, name=dataset_name, split=split)
    if max_series is not None:
        if max_series < 1:
            raise ValueError(f"max_series must be >= 1, got {max_series}")
        dataset = dataset.select(range(min(len(dataset), max_series)))

    if len(dataset) == 0:
        raise ValueError(f"Loaded empty dataset for {repo_id}:{dataset_name} split={split!r}")

    return fev_rows_to_long_dataframe(
        dataset,
        id_column=id_column,
        timestamp_column=timestamp_column,
        value_columns=value_columns,
    )


def _fev_row_to_frame(
    *,
    row: Mapping[str, Any],
    id_column: str,
    timestamp_column: str,
    value_columns: Sequence[str],
):
    import pandas as pd

    timestamps = _to_python_list(row[timestamp_column])
    series_length = len(timestamps)

    frame_data: dict[str, Any] = {
        id_column: [row[id_column]] * series_length,
        timestamp_column: timestamps,
    }

    for column in value_columns:
        values = _to_python_list(row[column])
        if len(values) != series_length:
            raise ValueError(
                f"Column {column!r} has length {len(values)} but {timestamp_column!r} has length {series_length} "
                f"for series {row[id_column]!r}."
            )
        frame_data[column] = values

    return pd.DataFrame(frame_data)


def _normalize_value_columns(
    *,
    value_columns: str | Sequence[str] | None,
    row: Mapping[str, Any],
    id_column: str,
    timestamp_column: str,
) -> list[str]:
    if value_columns is None:
        return infer_fev_value_columns(row, id_column=id_column, timestamp_column=timestamp_column)

    if isinstance(value_columns, str):
        resolved = [value_columns]
    else:
        resolved = list(value_columns)

    missing_columns = [column for column in resolved if column not in row]
    if missing_columns:
        raise ValueError(f"Requested value columns are missing from the FEV row: {missing_columns}")

    return resolved


def _is_sequence_like(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Sequence):
        return True
    return hasattr(value, "shape") and getattr(value, "ndim", 0) == 1


def _to_python_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(value)
