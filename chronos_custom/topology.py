"""Helpers that define which time series may share information in Chronos-2."""

import torch


def compute_group_ids(num_series: int) -> torch.Tensor:
    """Assign Chronos-2 group ids for the current prediction batch.

    Chronos-2 uses ``group_ids`` as a mask for cross-series attention:

    - series that share the same group id can attend to one another
    - series in different groups remain isolated

    This project uses a deliberately simple topology for experiments: split the
    batch into fixed-size neighborhoods of five series. That makes it easy to
    compare "local sharing" against the two default Chronos extremes:

    - ``cross_learning=False``: every series gets its own group id
    - ``cross_learning=True``: every series shares one global group id

    Example with ``group_size = 5``::

        series 0-4   -> group 0
        series 5-9   -> group 1
        series 10-14 -> group 2

    Parameters
    ----------
    num_series
        Number of time series in the batch passed to ``predict_df``.

    Returns
    -------
    torch.Tensor
        A one-dimensional ``LongTensor`` of shape ``(num_series,)`` containing
        the group assignment for each series in batch order.
    """
    group_size = 5
    return torch.arange(num_series, dtype=torch.long) // group_size
