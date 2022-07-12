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
from model import SegmentPredictor
from solver import stack_eval_preds, get_gtruths, get_segment_label_and_probs, get_eval_metrics
from src.data.dataset import DialogueSegmentation


def dump_segment(base_model, evaluate_dataloader):
    print("[INFO] Dump Segment Label...")
    
    base_model.eval()
    eval_predictions_1, eval_predictions_2 = [], []
    eval_gtruths_1, eval_gtruths_2 = [], []
    eval_losses, out_data = [], {}
    for eval_data in tqdm(evaluate_dataloader):
        eval_loss, eval_prediction_logits_1, eval_prediction_logits_2, segment_index_1, segment_index_2 = base_model(eval_data, evaluate=True)

        eval_probs_1 = stack_eval_preds(eval_prediction_logits_1, eval_predictions_1, concat_zero=True, get_probs=True)
        eval_probs_2 = stack_eval_preds(eval_prediction_logits_2, eval_predictions_2, concat_zero=True, get_probs=True)

        get_gtruths(segment_index_1, eval_gtruths_1)
        get_gtruths(segment_index_2, eval_gtruths_2)
        
        eval_losses.append(eval_loss.item())
        segment_index_1 = segment_index_1.cpu().tolist()
        segment_index_2 = segment_index_2.cpu().tolist()
        for i, _id in enumerate(eval_data["id"]):
            segment_prob_1, segment_label_1 = [], []
            get_segment_label_and_probs(segment_index_1, eval_probs_1, i, segment_label_1, segment_prob_1)
            segment_prob_2, segment_label_2 = [], []
            get_segment_label_and_probs(segment_index_2, eval_probs_2, i, segment_label_2, segment_prob_2)

            out_data[_id] = {
                "segment_label_1": segment_label_1,
                "segment_prob_1": segment_prob_1,
                "segment_label_2": segment_label_2,
                "segment_prob_2": segment_prob_2
            }
    
    acc, f1 = get_eval_metrics([eval_predictions_1, eval_predictions_2], [eval_gtruths_1, eval_gtruths_2])    
    print("ACC: {}, F1-MA: {}".format(acc, f1))


def run():
    model_file = "bert-base-uncased"
    data_dir = 'data/mini_dialogsum/test.json'
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