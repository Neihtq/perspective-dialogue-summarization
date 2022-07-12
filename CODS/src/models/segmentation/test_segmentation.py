import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel
import debugger

import json
import os
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, AutoTokenizer
from sklearn.metrics import f1_score
from src.utils.utils import compute_accuracy 
from tqdm import tqdm
global inf
inf = 1e10

PATH = 'save/mini_dialogsum_TEST/'
if not os.path.exists(PATH):
    os.makedirs(PATH)

class SegmentPredictor(nn.Module):
    def __init__(self, bert_path='bert-base-uncased', num_labels=2):
        super().__init__()
        self.model = BertModel.from_pretrained(bert_path)
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)
        self.num_labels = num_labels

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, evaluate=False):
        document = data['document'].to(self.device)
        segment_label = data['segment_label'].to(self.device)
        output = self.model(document)
        all_seq_hs = output[0]  # batch_size, seq_len, hd_dim

        sent_repr_mat = []
        turn_nums = [(item == 101).sum().cpu().item() for item in document]
        print(document.shape, turn_nums)
        max_turn_num = max(turn_nums)
        for i in range(all_seq_hs.size(0)):
            sent_repr = all_seq_hs[i][document[i] == 101]  # [num_of_turns, hd_dim]
            print(sent_repr.shape)
            sent_repr = torch.cat(
                [sent_repr, torch.zeros(max_turn_num - turn_nums[i], sent_repr.size(1)).to(self.device)], 0)
            sent_repr_mat.append(sent_repr)
            segment_label[i][turn_nums[i]:] = -1
        sent_repr_mat = torch.stack(sent_repr_mat, 0)  # [batch_size, max_turn_num, hd_dim]
        segment_label = segment_label[:, :max_turn_num]
        prediction_logits = self.classifier(sent_repr_mat)
        loss = F.cross_entropy(prediction_logits.reshape(-1, self.num_labels), segment_label.reshape(-1), ignore_index=-1,
                               reduction='mean')

        # For prediction_logits
        if evaluate:
            batch_size = prediction_logits.size(0)
            eval_prediction_logits = []
            for i in range(batch_size):
                eval_prediction_logits.append(prediction_logits[i][:turn_nums[i], :])
            # eval_prediction_logits = torch.cat(eval_prediction_logits, 0)
            return loss, eval_prediction_logits, segment_label
        else:
            return loss


class DialogueSegmentation:
    """
    for general usage of loading dialogues
    """
    def __init__(self, tokenizer, filepath='data/train.json', bert_path='bert-base-uncased',
                 max_sequence_length=512, max_single_turn_length=30, max_num_of_turns=20):
        self.data = []
        with open(filepath) as f:
            data = json.load(f)
            #print(self.data[0])
        for d in data:
            if "segment_label" in d.keys():
                self.data.append(d)
            
        self.max_sequence_length = max_sequence_length
        self.max_single_turn_length = max_single_turn_length
        self.max_num_of_turns = max_num_of_turns
        self.tokenizer = tokenizer #BertTokenizer.from_pretrained(bert_path) # bert?

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        clean_dialog, segment_label = self.data[index]['clean_dialog'], self.data[index]['segment_label']

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
            segment_label = segment_label[:local_max_num_of_turns]
            clean_dialog_turns = clean_dialog_turns[:local_max_num_of_turns]

        # Pad module_index with -1
        if len(segment_label) < self.max_num_of_turns:
            segment_label += [-1] * (self.max_num_of_turns - len(segment_label))
        else:
            segment_label = segment_label[:self.max_num_of_turns]
        
        assert len([s for s in segment_label if s != -1]) == input_ids.count(CLS)
        
        return {'id':self.data[index]['id'],
                'document': torch.LongTensor(input_ids),
                'segment_label': torch.LongTensor(segment_label)}


def get_optimzier(model):
    weight_decay = 0.0
    adam_epsilon = 1e-6
    learning_rate = 1E-5
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon, betas=(0., 0.999))


def dump_segment(base_model, evaluate_dataloader):
    print("[INFO] Dump Segment Label...")
    data_type = 'pred_test_new' 
    base_model.eval()
    eval_losses = []
    eval_predictions = []
    eval_gtruths = []
    out_data = {}
    softmax_layer = nn.Softmax(-1)
    for eval_data in tqdm(evaluate_dataloader):
        eval_loss, eval_prediction_logits, segment_index = base_model(eval_data, evaluate=True)
        eval_prediction_logits_cat = torch.cat(eval_prediction_logits, 0)
        eval_probs = [softmax_layer(logit) for logit in eval_prediction_logits]
        eval_prediction = eval_prediction_logits_cat.topk(1)[1].squeeze()
        eval_losses.append(eval_loss.item())
        eval_predictions.extend(eval_prediction.cpu().tolist())
        # Filter -1 in segment_index
        segment_index = segment_index.cpu().tolist()
        for item in segment_index:
            eval_gtruths.extend(list(filter((-1).__ne__, item)))
        
        for i, _id in enumerate(eval_data["id"]):
            
            segment_prob, segment_label = [], []
            for segi, idx in enumerate(segment_index[i]):
                if idx != -1:
                    segment_label.append(idx)
                    segment_prob.append(eval_probs[i][segi, 1].item())
            
            out_data[_id] = {
                "segment_label": segment_label,
                "segment_prob": segment_prob
            }
    
    first_key = list(out_data.keys())[0]
    acc = compute_accuracy(eval_predictions, eval_gtruths)
    f1 = f1_score(y_pred=eval_predictions, y_true=eval_gtruths, average="weighted")
    print("ACC: {}, F1-MA: {}".format(acc, f1))

    with open(os.path.join(PATH, "{}.json".format(data_type)), "w") as fout:
        json.dump(out_data, fout, indent=4)
    
    with open(os.path.join(PATH, "{}_log.txt".format(data_type)), "w") as fout:
        fout.write("ACC: {}, F1-MA: {}".format(acc, f1))

    return out_data

def run():
    model_file = "bert-base-uncased"
    data_dir = 'data/dialogsum/mini_clean/test.json'
    batch_size = 4
    tokenizer = AutoTokenizer.from_pretrained(model_file)
    test_dataset = DialogueSegmentation(tokenizer, filepath=data_dir,
                bert_path=model_file, max_sequence_length=512,
                max_single_turn_length=30,
                max_num_of_turns=30)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    predictor = SegmentPredictor(bert_path=model_file, num_labels=2)

    load_path = 'save/train_segment_predictor/pytorch.bin'  
    print("[INFO] Loading model from", load_path)
    predictor.load_state_dict(torch.load(load_path))

    dump_segment(predictor, test_dataloader)


if __name__ == '__main__':
    run()