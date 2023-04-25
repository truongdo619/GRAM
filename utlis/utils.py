import argparse
import torch

def get_parser() -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--max_keep_ckpt", default=1, type=int, help="the number of keeping ckpt max.")
    parser.add_argument("--bert_config_dir", type=str, required=True, help="bert config dir")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--max_length", type=int, default=128, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--seed", default=0, type=int, help="set random seed for reproducing results.")
    return parser

# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# last update: xiaoya li
# issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/1868
# set for trainer: https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
#   from pytorch_lightning import Trainer, seed_everything
#   seed_everything(42)
#   sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
#   model = Model()
#   trainer = Trainer(deterministic=True)

import random
import torch
import numpy as np
from pytorch_lightning import seed_everything

def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    return attention_mask.eq(0)

def label_masks_based_on_grammars(tokens, tmp_start_pred, grammars, label2id_dict):
    # Find the parent entity
    tmp_tokens = tokens[:tmp_start_pred]
    context = "".join(tmp_tokens).replace("Ġ", " ")
    words = context.split(" ")
    parent_entity = "ROOT"
    count_close_brackets = 0
    for word in reversed(words):
        if word.startswith("]"):
            count_close_brackets += 1
            
        if word.startswith("["):
            if count_close_brackets == 0:
                parent_entity = word[1:]
                break
            else:
                count_close_brackets -= 1
    
    # Create entity mask
    if parent_entity not in grammars:
        return torch.ones((1, len(label2id_dict))).to("cuda:0")

    entity_mask = torch.zeros((1, len(label2id_dict))).to("cuda:0")
    entity_candidates = grammars[parent_entity] + ["[EOP]"]
    for candidate in entity_candidates:
        if "UNSUPPORTED" not in candidate:
            entity_mask[0, label2id_dict[candidate]] = 1
    return entity_mask


def label_masks_based_on_grammars_v2(tokens, tmp_start_pred, grammars, label2id_dict):
    # Find the parent entity
    tmp_tokens = tokens[:tmp_start_pred]
    context = "".join(tmp_tokens).replace("Ġ", " ")
    words = context.split(" ")
    parent_entity = "ROOT"
    count_close_brackets = 0
    for word in reversed(words):
        if word.startswith("]"):
            count_close_brackets += 1
            
        if word.startswith("["):
            if count_close_brackets == 0:
                parent_entity = word[1:]
                break
            else:
                count_close_brackets -= 1
    
    # Create entity mask
    if parent_entity not in grammars:
        return torch.ones((1, len(label2id_dict))).to("cuda:0"), ""

    entity_mask = torch.zeros((1, len(label2id_dict))).to("cuda:0")
    entity_candidates = grammars[parent_entity] + ["[EOP]"]
    label_text = ""
    for candidate in entity_candidates:
        if "UNSUPPORTED" not in candidate:
            entity_mask[0, label2id_dict[candidate]] = 1
            label_text += " " + candidate
    label_text = label_text.strip()
    return entity_mask, label_text


def label_masks_based_on_grammars_with_embedding(tokens, tmp_start_pred, grammars, label2id_dict):
    # Find the parent entity
    tmp_tokens = tokens[:tmp_start_pred]
    context = "".join(tmp_tokens).replace("Ġ", " ")
    words = context.split(" ")
    parent_entity = "ROOT"
    count_close_brackets = 0
    for word in reversed(words):
        if word.startswith("]"):
            count_close_brackets += 1
            
        if word.startswith("["):
            if count_close_brackets == 0:
                parent_entity = word[1:]
                break
            else:
                count_close_brackets -= 1
    
    # Create entity mask
    if parent_entity not in grammars:
        return torch.ones((len(label2id_dict))).to("cuda:0").unsqueeze(0).unsqueeze(0)

    label_mask = torch.zeros((len(label2id_dict)), dtype=torch.bool).cuda()
    entity_candidates = grammars[parent_entity] + ["[EOP]"]
    for candidate in entity_candidates:
        if "UNSUPPORTED" not in candidate:
            label_mask[label2id_dict[candidate]] = 1
    # +2 for <s> and </s>
    label_mask = label_mask.unsqueeze(0)
    return label_mask.unsqueeze(0)

def find_valid_span_range(tokens):
    for idx, token in enumerate(tokens):
        if  token == '</s>' or token == '<s>' or token.startswith("[") or token.startswith("]") or token.startswith("Ġ[") or token.startswith("Ġ]"):
            continue
        else:
            start = idx
            break

    for idx in range(len(tokens)-1, -1, -1):
        token = tokens[idx]
        if  token == '</s>' or token == '<s>' or token.startswith("[") or token.startswith("]") or token.startswith("Ġ[") or token.startswith("Ġ]"):
            continue
        else:
            end = idx
            break
    return (start, end)


if __name__ == '__main__':
    # without this line, x would be different in every execution.
    set_random_seed(0)

    x = np.random.random()
    print(x)


