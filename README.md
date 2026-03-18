# chronos-custom

Minimal extensions to [Chronos-2](https://github.com/amazon-science/chronos-forecasting) that add two things on top of the official pipeline:

1. **Custom group topology** — control which series can attend to each other in the encoder by injecting your own `group_ids` before inference, instead of relying on Chronos-2's defaults.
2. **Attention capture** — record temporal (per-series token) and cross-series attention weights during prediction for inspection and analysis.

A lightweight FEV dataset loader is also included for quickly pulling benchmark datasets from Hugging Face.

---

## How Chronos-2 grouping works

Chronos-2's encoder uses `group_ids` to decide cross-series attention. Series with the same group id can exchange information; series in different groups are isolated from each other:

| Chronos-2 mode | Behaviour |
|---|---|
| `cross_learning=False` | Every series gets its own group — no cross-series attention |
| `cross_learning=True` | All series share one group — full global attention |
| **chronos-custom** | Series share within configurable fixed-size neighborhoods |

The default topology in this library splits the batch into neighborhoods of 5:

```
series  0–4  → group 0
series  5–9  → group 1
series 10–14 → group 2
...
```

This is a middle ground useful for experiments: series within a neighborhood share context, but series far apart in the batch remain isolated.

---

## Installation

**Core library** (requires `torch` and `chronos`):

```bash
pip install git+https://github.com/you/chronos-custom.git
```

**With FEV loader** (adds `pandas`, `datasets`, `pyarrow`):

```bash
pip install "chronos-custom[fev] @ git+https://github.com/you/chronos-custom.git"
```

**Editable install from a local clone:**

```bash
git clone https://github.com/you/chronos-custom.git
cd chronos-custom
pip install -e ".[fev]"
```

---

## Quick start

### 1. Load the pipeline

```python
from chronos_custom import CustomChronosPipeline

pipeline = CustomChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
)
```

### 2. Forecast with custom group topology

Prepare a long-format DataFrame with columns `id`, `timestamp`, and `target`, then call `predict_df`:

```python
pipeline.reset_attentions()

predictions = pipeline.predict_df(
    df,
    prediction_length=24,
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)
```

By default `use_custom_group_ids=True`, so the pipeline automatically computes neighborhoods of 5 from the batch size. To fall back to Chronos-2's built-in grouping:

```python
pipeline.use_custom_group_ids = False
```

### 3. Inspect attention weights

After `predict_df()`, the attention tensors are stored on the pipeline instance:

```python
# list with one entry per autoregressive step
# each entry is a list of tensors, one per encoder layer
# temporal attention shape:  (batch, heads, query_tokens, key_tokens)
pipeline.time_attentions

# cross-series attention shape: (tokens, heads, query_series, key_series)
pipeline.group_attentions
```

Always call `reset_attentions()` before a new prediction to avoid accumulating stale tensors from a previous run.

---

## Computing group IDs directly

You can call `compute_group_ids` outside the pipeline to inspect or customise the topology:

```python
from chronos_custom import compute_group_ids

group_ids = compute_group_ids(num_series=12)
# tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])

group_ids = compute_group_ids(num_series=9)
# tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
```

To use a completely custom topology, pass a pre-built tensor directly to `_predict_batch`, or subclass `CustomChronosPipeline` and override `_predict_batch` to supply your own `group_ids`.

---

## FEV dataset loader

Load any dataset from the [autogluon/fev_datasets](https://huggingface.co/datasets/autogluon/fev_datasets) collection and convert it to the long format expected by `predict_df`:

```python
from chronos_custom import load_fev_long_dataframe, infer_fev_target_columns

df, value_columns = load_fev_long_dataframe(
    "LOOP_SEATTLE_1H",   # FEV config name
    split="train",
    max_series=50,
)

target_col = infer_fev_target_columns(value_columns)[0]

print(df.head())
#   id           timestamp     target
#    0 2015-01-01 00:00:00  62.189457
#    0 2015-01-01 01:00:00  62.927940
```

If you already have rows loaded (e.g. from a custom Hugging Face dataset), use the lower-level converter:

```python
from chronos_custom import fev_rows_to_long_dataframe

df, value_columns = fev_rows_to_long_dataframe(rows)
```

---

## End-to-end example

```python
from chronos_custom import (
    CustomChronosPipeline,
    load_fev_long_dataframe,
    infer_fev_target_columns,
)

# Load data
df, value_columns = load_fev_long_dataframe("LOOP_SEATTLE_1H", split="train", max_series=10)
target_col = infer_fev_target_columns(value_columns)[0]

# Load pipeline
pipeline = CustomChronosPipeline.from_pretrained("amazon/chronos-2")
pipeline.reset_attentions()

# Forecast
predictions = pipeline.predict_df(
    df,
    prediction_length=24,
    id_column="id",
    timestamp_column="timestamp",
    target=target_col,
)

# Inspect cross-series attention (layer 0, averaged over tokens and heads)
g_tensor = pipeline.group_attentions[0][0]   # (tokens, heads, N, N)
group_mat = g_tensor.mean(dim=(0, 1))        # (N, N)
print(group_mat)
```

---

## Public API

```python
from chronos_custom import (
    CustomChronosPipeline,        # extended Chronos2Pipeline
    compute_group_ids,            # assign group ids for a batch of N series
    load_fev_long_dataframe,      # load an FEV dataset from HuggingFace → long df
    fev_rows_to_long_dataframe,   # convert already-loaded rows → long df
    infer_fev_value_columns,      # detect sequence-valued columns in an FEV row
    infer_fev_target_columns,     # pick target columns from value columns
    DEFAULT_FEV_DATASET_REPO,     # "autogluon/fev_datasets"
)
```

---

## Requirements

| Dependency | Required | Notes |
|---|---|---|
| `torch` | Yes | Tensor operations |
| `chronos-forecasting` | Yes | Base `Chronos2Pipeline` |
| `pandas` | For FEV loader | Install with `[fev]` extra |
| `datasets` | For FEV loader | Install with `[fev]` extra |
| `pyarrow` | For FEV loader | Install with `[fev]` extra |

Python >= 3.9 required.
