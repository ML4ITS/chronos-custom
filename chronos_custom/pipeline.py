"""Custom Chronos-2 pipeline extensions used throughout the notebooks.

The goal of this module is to keep the customization surface area very small:

1. Replace Chronos-2's automatically generated ``group_ids`` with a
   project-specific topology before the parent pipeline expands them for
   long-horizon forecasting.
2. Capture temporal and cross-series attention weights during inference so
   notebook experiments can inspect how the model routed information.

Everything else is delegated to ``Chronos2Pipeline`` to stay aligned with the
official implementation.
"""

import torch
from chronos import Chronos2Pipeline

from .topology import compute_group_ids


class CustomChronosPipeline(Chronos2Pipeline):
    """Chronos-2 pipeline with project-specific grouping and attention capture.

    The class intentionally overrides only two internal prediction hooks:

    ``_predict_batch``
        Injects group assignments produced by :func:`compute_group_ids` before
        the parent pipeline performs its quantile and horizon expansion. This is
        the earliest safe point to control which series are allowed to exchange
        information through Chronos-2's cross-series attention.

    ``_predict_step``
        Enables ``output_attentions=True`` on the underlying model call and
        stores the resulting temporal and group attention tensors on the
        instance for later inspection.

    Typical usage
    -------------
    ``pipeline = CustomChronosPipeline.from_pretrained(...)``
    ``pipeline.use_custom_group_ids = True``
    ``pipeline.reset_attentions()``
    ``pred_df = pipeline.predict_df(...)``

    After ``predict_df()``, attention tensors are available as:

    ``pipeline.time_attentions``
        Temporal self-attention per autoregressive prediction step.

    ``pipeline.group_attentions``
        Cross-series attention per autoregressive prediction step.

    Each list entry corresponds to one autoregressive step and contains one
    tensor per encoder layer with shape ``(batch, heads, query_len, key_len)``.
    """

    def __init__(self, *args, use_custom_group_ids: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_custom_group_ids = use_custom_group_ids
        self.time_attentions: list = []
        self.group_attentions: list = []

    def reset_attentions(self) -> None:
        """Clear cached attention tensors from any previous prediction call."""
        self.time_attentions.clear()
        self.group_attentions.clear()

    # ------------------------------------------------------------------
    # Override 1: inject custom group_ids
    # Runs before long-horizon expansion so the topology is applied at
    # the original batch level, not the expanded quantile level.
    # ------------------------------------------------------------------

    def _predict_batch(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor,
        unrolled_quantiles_tensor: torch.Tensor,
        prediction_length: int,
        max_output_patches: int,
        target_idx_ranges: list,
    ) -> list:
        """Run one batched prediction pass with optional custom grouping.

        Parameters are defined by the parent ``Chronos2Pipeline`` implementation.
        The only behavior change here is that we can replace the incoming
        ``group_ids`` tensor with a topology generated from the current batch
        size before delegating back to the base class.
        """
        if self.use_custom_group_ids:
            num_series = context.shape[0]
            group_ids = compute_group_ids(num_series).to(context.device)
        
        print(f"Using group_ids:\n{group_ids}")

        return super()._predict_batch(
            context=context,
            group_ids=group_ids,
            future_covariates=future_covariates,
            unrolled_quantiles_tensor=unrolled_quantiles_tensor,
            prediction_length=prediction_length,
            max_output_patches=max_output_patches,
            target_idx_ranges=target_idx_ranges,
        )

    # ------------------------------------------------------------------
    # Override 2: capture temporal + group attention weights
    # Called by _predict_batch for each autoregressive step.
    # group_ids here are already expanded by the parent's long-horizon
    # logic, so we forward them as-is to the model.
    # ------------------------------------------------------------------

    def _predict_step(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor | None,
        num_output_patches: int,
    ) -> torch.Tensor:
        """Run a single autoregressive prediction step and store attentions.

        By the time this hook is invoked, the parent pipeline has already
        expanded the batch for the current horizon/quantile configuration, so we
        simply forward the tensors and record the extra attention outputs.
        """
        output = self.model(
            context=context,
            group_ids=group_ids,
            future_covariates=future_covariates,
            num_output_patches=num_output_patches,
            output_attentions=True,
        )

        if output.enc_time_self_attn_weights is not None:
            self.time_attentions.append(output.enc_time_self_attn_weights)
        if output.enc_group_self_attn_weights is not None:
            self.group_attentions.append(output.enc_group_self_attn_weights)

        return output.quantile_preds.to(context)
