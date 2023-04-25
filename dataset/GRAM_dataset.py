#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

class GRAMDataset(Dataset):
    """
    GRAM Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
    """
    def __init__(self, json_path, label2id_path, grammar_path, tokenizer: AutoTokenizer, max_length: int = 512):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = json.load(open(label2id_path, encoding="utf-8"))
        self.label2id["[EOP]"] = len(self.label2id.values())
        self.grammars = json.load(open(grammar_path, encoding="utf-8"))
        self.processed_data = self.process_data(self.all_data)

    @staticmethod
    def update_pos(start_positions, end_positions, cursor):
        new_start_positions = [start+1 if start >= cursor else start for start in start_positions ]
        new_end_positions = [end+1 if end >= cursor else end for end in end_positions ]
        return new_start_positions, new_end_positions
    
    @staticmethod
    def find_parent_entity(words):
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
        return parent_entity
        
    def process_data(self, data):
        data_dict = {}
        print("Processing data...")
        for item in tqdm(data):
            context = item["context"]
            start_positions = item["start_position"]
            end_positions = item["end_position"]
            labels = item["label_type"]
            entity_levels = item["entity_level"]
            cur_words = context.split()
            num_entities = len(start_positions)
            for idx in range(num_entities):
                if entity_levels[idx] not in data_dict:
                    data_dict[entity_levels[idx]] = []

                parent_entity = self.find_parent_entity(cur_words[:start_positions[idx]])
                data_dict[entity_levels[idx]].append(self.create_training_tensors(cur_words, start_positions[idx], \
                                                    end_positions[idx], labels[idx], parent_entity))

                # Update context and postions
                cur_words.insert(end_positions[idx]+1, f"]")
                cur_words.insert(start_positions[idx], f"[{labels[idx]}")
                start_positions, end_positions = self.update_pos(start_positions, end_positions, start_positions[idx])
                start_positions, end_positions = self.update_pos(start_positions, end_positions, end_positions[idx]+1)

            if entity_levels[-1] + 1 not in data_dict:
                data_dict[entity_levels[-1] + 1] = []
            data_dict[entity_levels[-1] + 1].append(self.create_training_tensors(cur_words, -1, -1, "[EOP]", None))
            # print(data_dict[entity_levels[-1] + 1])

        result = []
        for value in data_dict.values():
            result += value
        return result

    def create_training_tensors(self, words, start, end, label, parent_entity):

         # query = "Find the intents and slots of the sentence."
        context = " ".join(words)
        start_positions = [start] if start != -1 else []
        end_positions = [end] if end != -1 else []

        words = context.split()
        start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
        end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        sample_tokens = self.tokenizer(context, return_token_type_ids=True, return_offsets_mapping=True)
        tokens = sample_tokens.input_ids
        type_ids = sample_tokens.token_type_ids
        offsets = sample_tokens.offset_mapping

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            if token_start not in origin_offset2token_idx_start:
                origin_offset2token_idx_start[token_start] = token_idx
            if token_end not in origin_offset2token_idx_end:
                origin_offset2token_idx_end[token_end] = token_idx

        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]

        # print(tokens)
        # print(self.tokenizer.convert_ids_to_tokens(tokens))
        # print(new_start_positions)
        # print(new_end_positions)
        label_mask = [
            (0 if offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]        

        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        # the start/end position must be whole word
        for token_idx in range(len(tokens)):
            current_word_idx = sample_tokens.words()[token_idx]
            next_word_idx = sample_tokens.words()[token_idx+1] if token_idx+1 < len(tokens) else None
            prev_word_idx = sample_tokens.words()[token_idx-1] if token_idx-1 > 0 else None
            if prev_word_idx is not None and current_word_idx == prev_word_idx:
                start_label_mask[token_idx] = 0
            if next_word_idx is not None and current_word_idx == next_word_idx:
                end_label_mask[token_idx] = 0

        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(label_mask) == len(tokens)
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]

        # make sure last token is </s>
        sep_token = 2 # SEP token id
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0

        # Label types
        label_type = [self.label2id[label]]

        # Start, end at [CLS] for [EOP] entity
        start_label_mask[0] = 1
        end_label_mask[0] = 1
        if label == "[EOP]":
            start_labels[0] = 1
            end_labels[0] = 1

        # Label mask based on grammars
        label_mask = [0] * len(self.label2id)
        label_mask[-1] = 1 # Unmask [EOP] entity
        if parent_entity is not None:
            entity_candidates = self.grammars[parent_entity]
            for candidate in entity_candidates:
                if "UNSUPPORTED" not in candidate:
                    label_mask[self.label2id[candidate]] = 1
        
        label_mask = torch.BoolTensor(label_mask).unsqueeze(0)
        input_label_seq_tensor = range(len(self.label2id))

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            torch.LongTensor(label_type),
            label_mask,
            torch.LongTensor(input_label_seq_tensor)
        ]

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
        """
        return self.processed_data[item]



class GRAMInferDataset(Dataset):
    """
    GRAM Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
    """
    def __init__(self, json_path, label2id_path, tokenizer: AutoTokenizer, max_length: int = 512):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = json.load(open(label2id_path, encoding="utf-8"))
        self.label2id["[EOP]"] = len(self.label2id.values())
        self.processed_data = self.process_data(self.all_data)

    @staticmethod
    def update_pos(start_positions, end_positions, cursor):
        new_start_positions = [start+1 if start >= cursor else start for start in start_positions ]
        new_end_positions = [end+1 if end >= cursor else end for end in end_positions ]
        return new_start_positions, new_end_positions
        
    def process_data(self, data):
        result = []
        print("Processing data...")
        for item in tqdm(data):
            context = item["context"]
            start_positions = item["start_position"]
            end_positions = item["end_position"]
            labels = item["label_type"]
            cur_words = context.split()
            num_entities = len(start_positions)
            cur_item = {"tokens": [], "starts": [], "ends": [], "entities": []}
            for idx in range(num_entities):
                if idx == 0:
                    cur_item["tokens"] = " ".join(cur_words)
                    cur_item["org_label"] = item["org_label"]
                start, end = self.new_start_end_pos(" ".join(cur_words), start_positions[idx], end_positions[idx])
                cur_item["starts"].append(start)
                cur_item["ends"].append(end)
                cur_item["entities"].append(self.label2id[labels[idx]])

                # Update context and postions
                cur_words.insert(end_positions[idx]+1, f"]")
                cur_words.insert(start_positions[idx], f"[{labels[idx]}")
                start_positions, end_positions = self.update_pos(start_positions, end_positions, start_positions[idx])
                start_positions, end_positions = self.update_pos(start_positions, end_positions, end_positions[idx]+1)

            result.append(cur_item)
        return result

    def new_start_end_pos(self, context, start, end):
         # query = "Find the intents and slots of the sentence."
        
        start_positions = [start] if start != -1 else []
        end_positions = [end] if end != -1 else []

        words = context.split()
        start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
        end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        sample_tokens = self.tokenizer(context, return_token_type_ids=True, return_offsets_mapping=True)
        tokens = sample_tokens.input_ids
        offsets = sample_tokens.offset_mapping

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            if token_start not in origin_offset2token_idx_start:
                origin_offset2token_idx_start[token_start] = token_idx
            if token_end not in origin_offset2token_idx_end:
                origin_offset2token_idx_end[token_end] = token_idx

        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]
        return new_start_positions[0], new_end_positions[0]

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
        """
        return self.processed_data[item]