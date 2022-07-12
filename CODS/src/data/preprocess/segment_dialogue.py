'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import rouge
import time
import json
import nltk
import os
from tqdm import tqdm
import debugger
nltk.download('punkt')

sep_token = '</s>'
    
rouge_evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                        max_n=4,
                        limit_length=True,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=False,
                        apply_best=True,
                        alpha=0.5, # Default F1_score
                        weight_factor=1.2,
                        stemming=True)


def calculate_similarity(pred, summary):
    try:
        rouge_score = rouge_evaluator.get_scores([pred], [summary])
    except:
        print("pred", pred)
        print("summary", summary)
        return 0
    return rouge_score["rouge-1"]['f']


def segment_dial(dialog, summary):
    #sum_list = [s.strip()+"." for s in summary.strip().split(".") if (s.strip() not in ["", "."] and len(s.strip()) > 5)]
    sum_list = [s.strip()+"." for s in summary.split("<EOS>") if (s.strip() not in ["", "."] and len(s.strip()) > 5)]
    # print("[SUMMARY]", sum_list)
    if len(sum_list) == 0:
        print("[ERROR] summary without a '.':", summary)
        return 0, None, None, None
    elif len(sum_list) == 1:
        return 1, [dialog], [0] * len(dialog), sum_list
    elif len(dialog) < len(sum_list):
        #return 1, [dialog], [0] * len(dialog), [" ".join(sum_list).replace(".", "")]
        return 1, [dialog], [0] * len(dialog), [" ".join(sum_list).replace("<EOS>", "")]
    else:
        seg_dial, seg_dial_idx = [], []
        dial_curser = 0
        #print(len(dialog))
        for si, sum_item in enumerate(sum_list):
            max_score = (dial_curser, 0.0) 
            #print(dial_curser, len(dialog)-len(sum_list)+si+1)
            #print(list(range(dial_curser, len(dialog)-(len(sum_list)-si)+1, 1)))
            for di in range(dial_curser, len(dialog)-(len(sum_list)-si)+1, 1):
                context = " ".join(dialog[dial_curser:di+1])
                score = calculate_similarity(context, sum_item)
                if score > max_score[1]:
                    max_score = (di, score)
            #print(max_score)
            if si == len(sum_list) - 1:
                seg_dial.append(list(dialog[dial_curser:]))
                #seg_dial_idx.append(max_score[0])
            else:
                seg_dial.append(list(dialog[dial_curser:max_score[0]+1])) 
                dial_curser = max_score[0] + 1
                seg_dial_idx.append(max_score[0])
        seg_label = [1 if i in seg_dial_idx else 0 for i in range(len(dialog))]
        return 1, seg_dial, seg_label, sum_list

def process_dialogue(file_path, d_type):
    sources, targets = [], []
    data = json.load(open(file_path, "r"))
    data_counter = 0
    for di, d in enumerate(tqdm(data)):
        #summary = d["summary"].replace("\015", "").replace("\n", "")
        summary_1 = d["summary_p1"].replace("\015", "").replace("\n", "")
        summary_2 = d["summary_p2"].replace("\015", "").replace("\n", "")
        d["clean_dialog"] = [item for item in d["clean_dialog"] if item.strip() != ""]
        #flag, seg_dial, seg_label, sum_list = segment_dial(d["clean_dialog"], summary)
        flag_1, seg_dial_1, seg_label_1, sum_list_1 = segment_dial(d["clean_dialog"], summary_1)
        flag_2, seg_dial_2, seg_label_2, sum_list_2 = segment_dial(d["clean_dialog"], summary_2)
        
        if not flag_1 and not flag_2: continue
        
        d["segment_dialog_1"] = list(seg_dial_1)
        d["segment_label_1"] = list(seg_label_1)
        d["sum_list_1"] = list(sum_list_1)

        d["segment_dialog_2"] = list(seg_dial_2)
        d["segment_label_2"] = list(seg_label_2)
        d["sum_list_2"] = list(sum_list_2)
        data_counter += 1
        
    with open(file_path, "w") as fout:
        json.dump(data, fout, indent=4)

if __name__ == "__main__":
    st = time.time()

    data_path = '../../../data/processed/new_dialogsum/new_clean_data/'
    process_dialogue(os.path.join(data_path, "test.json"), 'test')
    process_dialogue(os.path.join(data_path, "eval.json"), 'val')
    process_dialogue(os.path.join(data_path, "train.json"), 'train')
    #process_dialogue("../SAMsum/clean_data/test.json", "test")
    #process_dialogue("../SAMsum/clean_data/eval.json", "val")
    #process_dialogue("../SAMsum/clean_data/train.json", "train")
    print(time.time() - st)
