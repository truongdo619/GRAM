#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: model_config.py

from transformers import BertConfig, RobertaConfig

class GRAMConfig(RobertaConfig):
    def __init__(self, **kwargs):
        super(RobertaQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)
        self.classifier_intermediate_hidden_size = kwargs.get("classifier_intermediate_hidden_size", 1024)
        self.classifier_act_func = kwargs.get("classifier_act_func", "gelu")
        self.loss_weight = kwargs.get("weight_pos_span", (1.0, 1.0, 1.0))
        self.ignore_index = kwargs.get("ignore_index", -100)
        self.possible_labels_span = kwargs.get("possible_labels_span", 4)
        self.num_label_types = kwargs.get("num_slot_type_labels", 60)
        self.label_emb_attn_num_heads = kwargs.get("label_emb_attn_num_heads", 4)
        self.label_emb_attn_dropout = kwargs.get("label_emb_attn_dropout", 0.2)
