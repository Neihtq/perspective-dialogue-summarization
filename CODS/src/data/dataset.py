'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''


import re
import json
import string
import debugger

import torch
import torch.nn.functional as F

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from transformers import BertTokenizer


class DialogueFunctionalPrediction:
    """
    for general usage of loading dialogues
    """

    def __init__(self, tokenizer, filepath='data/train.json', bert_path='bert-base-uncased', max_single_turn_length=30):
        
        with open(filepath) as f:
            data = json.load(f)
        
        self.data = []
        trig_count = 0
        for d in data:
            for i, text in enumerate(d["clean_dialog"]):
                local_d = {
                    "id": d["id"]+":turn{}".format(i),
                    "text": text,
                    "label_p1": d["label_p1"][i],
                    "label_p2": d["label_p2"][i],
                }
                self.data.append(local_d)   
                
                if d["label_p1"][i] == 1 or d["label_p2"][i] == 1:
                    trig_count += 1
        
        self.max_single_turn_length = max_single_turn_length
        self.tokenizer = tokenizer #BertTokenizer.from_pretrained(bert_path) 
        
        print("[INFO] Positive Ratio: {:.4f}".format(trig_count/len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        _turn_plain = self.data[index]['text']

        PAD = self.tokenizer.convert_tokens_to_ids('[PAD]')  # 0
        SEP = self.tokenizer.convert_tokens_to_ids('[SEP]')  # 102
        CLS = self.tokenizer.convert_tokens_to_ids('[CLS]')  # 101
        local_max_num_of_turns = None
        
        _turn = self.tokenizer.encode(_turn_plain, add_special_tokens=True)
        if len(_turn) > self.max_single_turn_length - 1:
            _turn = _turn[:self.max_single_turn_length - 1]
            _turn = _turn + [SEP]
        else:
            _turn += [PAD] * (self.max_single_turn_length - len(_turn))

        assert len(_turn) == self.max_single_turn_length

        return {
            'id': self.data[index]['id'],
            "document_plain": _turn_plain,
            'document': torch.LongTensor(_turn),
            'label_p1': self.data[index]['label_p1'], # torch.LongTensor(self.data[index]['label'])
            'label_p2': self.data[index]['label_p2']
        }
    
    
class DialogueSegmentation:
    """
    for general usage of loading dialogues
    """

    def __init__(self, tokenizer, filepath='data/train.json', bert_path='bert-base-uncased',
                 max_sequence_length=512, max_single_turn_length=30, max_num_of_turns=20):
        self.data = []
        with open(filepath) as f:
            data = json.load(f)

        for d in data:
            if "segment_label_1" in d.keys() or "segment_label_2" in d.keys():
                d['id'] = d['fname']
                self.data.append(d)
            
        self.max_sequence_length = max_sequence_length
        self.max_single_turn_length = max_single_turn_length
        self.max_num_of_turns = max_num_of_turns
        self.tokenizer = tokenizer #BertTokenizer.from_pretrained(bert_path) # bert?

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        clean_dialog = self.data[index]['clean_dialog'] 
        segment_label_1, segment_label_2 = self.data[index]['segment_label_1'], self.data[index]['segment_label_2']

        PAD = self.tokenizer.convert_tokens_to_ids('[PAD]')  # 0
        SEP = self.tokenizer.convert_tokens_to_ids('[SEP]')  # 102
        CLS = self.tokenizer.convert_tokens_to_ids('[CLS]')  # 101
        local_max_num_of_turns = None

        # Pad functional_dialogs with PAD
        input_ids = []
        clean_dialog_turns = []
        input_ids_len = 0
        for text in clean_dialog:
            _turn = self.tokenizer.encode(text, add_special_tokens=True)
            if len(_turn) > self.max_single_turn_length - 1:
                _turn = _turn[:self.max_single_turn_length - 1]
                _turn = _turn + [SEP]
            if len(clean_dialog_turns) < self.max_num_of_turns:
                input_ids.extend(_turn)
                clean_dialog_turns.append(_turn)

  
        if len(input_ids) < self.max_sequence_length:
            input_ids += [PAD] * (self.max_sequence_length - len(input_ids))
        else:
            # If longer than maximum length, then need to keep track of the left turns (smaller than max_num_of_turn)
            input_ids = input_ids[: self.max_sequence_length - 1]
            input_ids += [SEP]
            local_max_num_of_turns = input_ids.count(CLS)

        assert len(input_ids) == self.max_sequence_length

        if local_max_num_of_turns:
            segment_label_1 = segment_label_1[:local_max_num_of_turns]
            segment_label_2 = segment_label_2[:local_max_num_of_turns]
            clean_dialog_turns = clean_dialog_turns[:local_max_num_of_turns]

        # Pad module_index with -1
        def pad_module_index(segment_label):
            if len(segment_label) < self.max_num_of_turns:
                segment_label += [-1] * (self.max_num_of_turns - len(segment_label))
            else:
                segment_label = segment_label[:self.max_num_of_turns]
            return segment_label

        segment_label_1 = pad_module_index(segment_label_1)
        segment_label_2 = pad_module_index(segment_label_2)
        
        assert len([s for s in segment_label_1 if s != -1]) == input_ids.count(CLS)
        assert len([s for s in segment_label_2 if s != -1]) == input_ids.count(CLS)
        
        return {'id':self.data[index]['id'],
                'document': torch.LongTensor(input_ids),
                'segment_label_1': torch.LongTensor(segment_label_1),
                'segment_label_2': torch.LongTensor(segment_label_2)
            }
    
    
class InputExample:
    def __init__(
        self, ID, context, 
        summary_p1, summary_p2,
        func_turn_label_p1, func_turn_label_p2,
        key_phrases_p1, key_phrases_p2,
        module_index
    ):
        self.ID = ID
        self.context = context
        self.summary_p1, self.summary_p2 = summary_p1, summary_p2
        self.func_turn_label_p1, self.func_turn_label_p2 = func_turn_label_p1, func_turn_label_p2 
        self.key_phrases_p1, self.key_phrases_p2 = key_phrases_p1, key_phrases_p2
        self.module_index = module_index



class InputFeatures:
    def __init__(self,
                 ID, example_index,
                 source_ids, source_mask, source_len,
                 target_ids_p1, target_labels_p1, target_len_p1,
                 target_ids_p2, target_labels_p2, target_len_p2,
                 func_turn_label_p1, func_turn_label_p2):
        self.ID = ID
        self.example_index = example_index
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.source_len = source_len

        self.target_ids_p1 = target_ids_p1
        self.target_labels_p1 = target_labels_p1
        self.target_len_p1 = target_len_p1


        self.target_ids_p2 = target_ids_p2
        self.target_labels_p2 = target_labels_p2
        self.target_len_p2 = target_len_p2

        self.func_turn_label_p1 = func_turn_label_p1
        self.func_turn_label_p2 = func_turn_label_p2


class CDataset(torch.utils.data.Dataset):
    def __init__(self, features, is_train):
        self.is_train = is_train
        self.length = len(features)
        self.ID = [f.ID for f in features]
        self.all_example_indices = torch.tensor([f.example_index for f in features], dtype=torch.long)
        self.all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        self.all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
        self.all_source_len = torch.tensor([f.source_len for f in features], dtype=torch.long)

        self.all_target_ids_p1 = torch.tensor([f.target_ids_p1 for f in features], dtype=torch.long)
        self.all_target_labels_p1 = torch.tensor([f.target_labels_p1 for f in features], dtype=torch.long)
        self.all_target_len_p1 = torch.tensor([f.target_len_p1 for f in features], dtype=torch.long)
        self.all_func_label_p1 = torch.tensor([f.func_turn_label_p1 for f in features], dtype=torch.long)

        self.all_target_ids_p2 = torch.tensor([f.target_ids_p2 for f in features], dtype=torch.long)
        self.all_target_labels_p2 = torch.tensor([f.target_labels_p2 for f in features], dtype=torch.long)
        self.all_target_len_p2 = torch.tensor([f.target_len_p2 for f in features], dtype=torch.long)
        self.all_func_label_p2 = torch.tensor([f.func_turn_label_p2 for f in features], dtype=torch.long)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = {
            "ID":               self.ID[idx],
            "example_indices":  self.all_example_indices[idx],
            "source_ids":       self.all_source_ids[idx],
            "source_mask":      self.all_source_mask[idx],
            "source_len":       self.all_source_len[idx]
        }

        if self.is_train:
                data["target_labels_p1"] = self.all_target_labels_p1[idx]
                data["target_ids_p1"]   = self.all_target_ids_p1[idx]
                data["target_len_p1"]   = self.all_target_len_p1[idx]
                data["func_label_p1"]   = self.all_func_label_p1[idx]
                
                data["target_labels_p2"] = self.all_target_labels_p2[idx]
                data["target_ids_p2"]   = self.all_target_ids_p2[idx]
                data["target_len_p2"]   = self.all_target_len_p2[idx]
                data["func_label_p2"]   = self.all_func_label_p2[idx]

        return data