#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.modules import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import  RobertaModel, RobertaPreTrainedModel
from model.classifier import MultiNonLinearClassifier

class GRAMModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super(GRAMModel, self).__init__(config)
        self.roberta = RobertaModel(config)
        
        self.start_outputs = MultiNonLinearClassifier(config.hidden_size, 1, config.mrc_dropout,
                                                       intermediate_hidden_size=config.classifier_intermediate_hidden_size)
        self.end_outputs = MultiNonLinearClassifier(config.hidden_size, 1, config.mrc_dropout,
                                                       intermediate_hidden_size=config.classifier_intermediate_hidden_size)
        self.label_outputs = MultiNonLinearClassifier(config.hidden_size * 2, config.num_label_types, config.mrc_dropout,
                                                       intermediate_hidden_size=config.classifier_intermediate_hidden_size)
        self.label_emb_attn = nn.MultiheadAttention(config.hidden_size, config.label_emb_attn_num_heads, dropout=config.label_emb_attn_dropout)

        self.label_emb_attn_num_heads = config.label_emb_attn_num_heads
        self.label_embedding = nn.Embedding(config.num_label_types, config.hidden_size)
        self.bce_loss = BCEWithLogitsLoss(reduction="none")
        self.ce_loss = CrossEntropyLoss(ignore_index=config.ignore_index)
        self.loss_weight = config.loss_weight
        self.num_label_types = config.num_label_types
        self.init_weights()


    def compute_loss(self, start_logit, end_logit, label_logit,
                     start_label, end_label, label_type, start_label_mask, end_label_mask, label_mask):
        """
        Compute the loss for the span prediction task.
        """
        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()

        # Start loss
        start_loss = self.bce_loss(start_logit.view(-1), start_label.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()

        # End loss
        end_loss = self.bce_loss(end_logit.view(-1), end_label.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()

        # Label loss
        label_logit = label_logit.view(-1, self.num_label_types)
        label_loss = self.ce_loss(label_logit, label_type.view(-1))

        total_loss = self.loss_weight[0] * start_loss + self.loss_weight[1] * end_loss + self.loss_weight[2] * label_loss
        return total_loss


    def forward(self, input_ids, token_type_ids, attention_mask, start_label_mask, end_label_mask, label_mask, start_labels, end_labels, label_types, input_label_seq_tensor):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [bz, seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [bz, seq_len]
            attention_mask: attention mask, tensor of shape [bz, seq_len]
            start_label_mask: start label mask, tensor of shape [bz, seq_len]
            end_label_mask: end label mask, tensor of shape [bz, seq_len]
            start_labels: start label, tensor of shape [bz, seq_len]
            end_labels: end label, tensor of shape [bz, seq_len]
            match_labels: match label, tensor of shape [bz, seq_len]
            type_labels_ids: type label, tensor of shape [bz, seq_len]
        Returns:
            start_logits: start/non-start probs of shape [bz, seq_len]
            end_logits: end/non-end probs of shape [bz, seq_len]
            span_logits: start-end-match probs of shape [bz, seq_len, seq_len]
            total_loss: total loss of shape [bz]
        """
        roberta_outputs = self.roberta(input_ids, token_type_ids=token_type_ids, \
                                        attention_mask=attention_mask, output_hidden_states=True)

        start_logit = self.start_outputs(roberta_outputs.hidden_states[-1]).squeeze(-1)  # [batch, seq_len]
        end_logit = self.end_outputs(roberta_outputs.hidden_states[-2]).squeeze(-1)  # [batch, seq_len]
        
        # Label attention
        cls_embedding = roberta_outputs.last_hidden_state[:, :1, :].transpose(1, 0)  # [1, batch, hidden]
        label_embs = self.label_embedding(input_label_seq_tensor).transpose(1, 0)
        label_mask = label_mask == False
        label_mask = torch.repeat_interleave(label_mask, self.label_emb_attn_num_heads, dim=0)
        attn_label_last_hidden_state = self.label_emb_attn(cls_embedding, label_embs, label_embs, attn_mask=label_mask)[0]

        # Label logit caculating
        label_embedding = F.relu(torch.cat((cls_embedding, attn_label_last_hidden_state), dim=2)) # residual connection
        label_embedding = label_embedding.transpose(1, 0).squeeze(dim=1) # [batch, 2 * hidden]
        label_logit = self.label_outputs(label_embedding)  # [batch, num_label_types]

        total_loss = self.compute_loss(start_logit=start_logit,
                                            end_logit=end_logit,
                                            label_logit=label_logit,
                                            start_label=start_labels,
                                            end_label=end_labels,
                                            label_type=label_types,
                                            start_label_mask=start_label_mask,
                                            end_label_mask=end_label_mask,
                                            label_mask=label_mask
                                        )

        return (start_logit, end_logit, label_logit), total_loss


    def predict_start_end(self, input_ids, token_type_ids, attention_mask):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [bz, seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [bz, seq_len]
            attention_mask: attention mask, tensor of shape [bz, seq_len]
            start_label_mask: start label mask, tensor of shape [bz, seq_len]
            end_label_mask: end label mask, tensor of shape [bz, seq_len]
            start_labels: start label, tensor of shape [bz, seq_len]
            end_labels: end label, tensor of shape [bz, seq_len]
            match_labels: match label, tensor of shape [bz, seq_len]
            type_labels_ids: type label, tensor of shape [bz, seq_len]
        Returns:
            start_logits: start/non-start probs of shape [bz, seq_len]
            end_logits: end/non-end probs of shape [bz, seq_len]
            span_logits: start-end-match probs of shape [bz, seq_len, seq_len]
            total_loss: total loss of shape [bz]
        """
        roberta_outputs = self.roberta(input_ids, token_type_ids=token_type_ids, \
                                        attention_mask=attention_mask, output_hidden_states=True)
        
        start_logit = self.start_outputs(roberta_outputs.hidden_states[-1]).squeeze(-1)  # [batch, seq_len]
        end_logit = self.end_outputs(roberta_outputs.hidden_states[-2]).squeeze(-1)  # [batch, seq_len]
        
        return start_logit, end_logit, roberta_outputs


    def predict_label(self, roberta_outputs, input_label_seq_tensor, label_mask):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [bz, seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [bz, seq_len]
            attention_mask: attention mask, tensor of shape [bz, seq_len]
            start_label_mask: start label mask, tensor of shape [bz, seq_len]
            end_label_mask: end label mask, tensor of shape [bz, seq_len]
            start_labels: start label, tensor of shape [bz, seq_len]
            end_labels: end label, tensor of shape [bz, seq_len]
            match_labels: match label, tensor of shape [bz, seq_len]
            type_labels_ids: type label, tensor of shape [bz, seq_len]
        Returns:
            start_logits: start/non-start probs of shape [bz, seq_len]
            end_logits: end/non-end probs of shape [bz, seq_len]
            span_logits: start-end-match probs of shape [bz, seq_len, seq_len]
            total_loss: total loss of shape [bz]
        """
        # Label attention
        cls_embedding = roberta_outputs.last_hidden_state[:, :1, :].transpose(1, 0)  # [1, batch, hidden]
        label_embs = self.label_embedding(input_label_seq_tensor).transpose(1, 0)
        label_mask = label_mask == False
        label_mask = torch.repeat_interleave(label_mask, self.label_emb_attn_num_heads, dim=0)
        attn_label_last_hidden_state = self.label_emb_attn(cls_embedding, label_embs, label_embs, attn_mask=label_mask)[0]

        # Label logit caculating
        label_embedding = F.relu(torch.cat((cls_embedding, attn_label_last_hidden_state), dim=2)) # residual connection
        label_embedding = label_embedding.transpose(1, 0).squeeze(dim=1) # [batch, 2 * hidden]
        label_logit = self.label_outputs(label_embedding)  # [batch, num_label_types]
        return label_logit

