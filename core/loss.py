# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# exclude extremely large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def pad_gt_layers_with_last(flow_gt, valid, num_layers):
    """
    Pad ground-truth layers to match the number of predicted layers.

    Args:
        flow_gt:
            [B, 2, H, W] or [B, K, 2, H, W]

        valid:
            [B, H, W], [B, K, H, W], or [B, K, 1, H, W]

        num_layers:
            Number of predicted layers L.

    Returns:
        flow_gt:
            [B, L, 2, H, W]

        valid:
            [B, L, H, W]
    """
    if flow_gt.ndim == 4:
        flow_gt = flow_gt[:, None]

    if valid.ndim == 3:
        valid = valid[:, None]

    if valid.ndim == 5:
        if valid.shape[2] != 1:
            raise ValueError(
                f"Expected valid shape [B, K, 1, H, W], but got {valid.shape}"
            )
        valid = valid[:, :, 0]

    K = flow_gt.shape[1]

    if K > num_layers:
        flow_gt = flow_gt[:, :num_layers]
        valid = valid[:, :num_layers]

    elif K < num_layers:
        repeat_count = num_layers - K

        last_flow = flow_gt[:, -1:].repeat(
            1,
            repeat_count,
            1,
            1,
            1,
        )

        last_valid = valid[:, -1:].repeat(
            1,
            repeat_count,
            1,
            1,
        )

        flow_gt = torch.cat(
            [flow_gt, last_flow],
            dim=1,
        )

        valid = torch.cat(
            [valid, last_valid],
            dim=1,
        )

    return flow_gt, valid


def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """
    Loss function defined over sequence of flow predictions.

    This function supports both:

    Single-layer FlowSeek:
        output['nf'][i]: [B, 2, H, W]
        flow_gt:         [B, 2, H, W]
        valid:           [B, H, W]

    Multi-layer FlowSeek:
        output['nf'][i]: [B, L, 2, H, W]
        flow_gt:         [B, K, 2, H, W] or [B, 2, H, W]
        valid:           [B, K, H, W] or [B, H, W]
    """
    n_predictions = len(output['flow'])
    flow_loss = 0.0

    first_nf = output['nf'][0]

    # ------------------------------------------------------------
    # Case 1: original single-layer FlowSeek
    # ------------------------------------------------------------
    if first_nf.ndim == 4:
        mag = torch.sum(
            flow_gt ** 2,
            dim=1,
        ).sqrt()

        valid = (valid >= 0.5) & (mag < max_flow)

        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)

            loss_i = output['nf'][i]

            final_mask = (
                (~torch.isnan(loss_i.detach()))
                & (~torch.isinf(loss_i.detach()))
                & valid[:, None]
            )

            denom = final_mask.sum().clamp(min=1)

            flow_loss += i_weight * (
                (final_mask * loss_i).sum() / denom
            )

        return flow_loss

    # ------------------------------------------------------------
    # Case 2: multi-layer FlowSeek
    # ------------------------------------------------------------
    if first_nf.ndim != 5:
        raise ValueError(
            f"Unexpected output['nf'][0] shape: {first_nf.shape}"
        )

    num_layers = first_nf.shape[1]

    flow_gt, valid = pad_gt_layers_with_last(
        flow_gt,
        valid,
        num_layers,
    )

    # flow_gt: [B, L, 2, H, W]
    # valid:   [B, L, H, W]
    mag = torch.sum(
        flow_gt ** 2,
        dim=2,
    ).sqrt()

    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        # loss_i: [B, L, 2, H, W]
        loss_i = output['nf'][i]

        final_mask = (
            (~torch.isnan(loss_i.detach()))
            & (~torch.isinf(loss_i.detach()))
            & valid[:, :, None]
        )

        denom = final_mask.sum().clamp(min=1)

        flow_loss += i_weight * (
            (final_mask * loss_i).sum() / denom
        )

    return flow_loss