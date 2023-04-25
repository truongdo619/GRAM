#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: collate_functions.py

import torch
from typing import List

def collate_to_max_length(batch: List[List[torch.Tensor]], ignore_index=-100, pad_id=0) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    pad_output = torch.full([batch_size, max_length], pad_id, dtype=batch[0][0].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][0]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)
    
    for field_idx in range(1, 6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    if len(batch[0]) == 8:
        output.append(torch.stack([x[-2] for x in batch]))
        output.append(torch.stack([x[-1] for x in batch]))
    else:
        output.append(torch.stack([x[-1] for x in batch]))
    return output



def collate_to_max_length_with_lb_emb(batch: List[List[torch.Tensor]], ignore_index=-100, pad_id=0) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    pad_output = torch.full([batch_size, max_length], pad_id, dtype=batch[0][0].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][0]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)
    
    for field_idx in range(1, 6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)


    output.append(torch.stack([x[-3] for x in batch]))
    # output.append(torch.stack([x[-2] for x in batch]))
    pad_output = torch.full([batch_size, 1, batch[0][-2].shape[-1]], 0, dtype=batch[0][-2].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][-2]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    output.append(torch.stack([x[-1] for x in batch]))

    return output