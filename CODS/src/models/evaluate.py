import os
import json
import rouge
import torch

from tqdm import tqdm
from difflib import SequenceMatcher
from torch.utils.data import DataLoader, SequentialSampler

from src.data.dataset import CDataset
from src.utils.constants import CPU, DEVICE, SUM_TOKEN, TEST_DIALOGSUM_PATH 
from src.utils.data_preprocess import load_examples, convert_examples_to_features

evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                    max_n=4,
                    limit_length=True,
                    length_limit=100,
                    length_limit_type='words',
                    apply_avg=False,
                    apply_best=True,
                    alpha=0.5, # Default F1_score
                    weight_factor=1.2,
                    stemming=True)


def evaluate_rouge(dev_examples, pred):
    true_sum_arr_p1 = [d.summary_p1 for d in dev_examples]
    true_sum_arr_p2 = [d.summary_p2 for d in dev_examples]
    pred_sum_arr_p1 = [pred[d.ID]['Person1'] for d in dev_examples]
    pred_sum_arr_p2 = [pred[d.ID]['Person2'] for d in dev_examples]

    assert len(true_sum_arr_p1) == len(pred_sum_arr_p1)
    assert len(true_sum_arr_p2) == len(pred_sum_arr_p2)

    scores_1 = evaluator.get_scores(pred_sum_arr_p1, true_sum_arr_p1)
    scores_2 = evaluator.get_scores(pred_sum_arr_p2, true_sum_arr_p2)
    
    scores = {}
    for key in scores_1:
        scores[key] = {}
        for metric in scores_1[key]:
            scores[key][metric] = (scores_1[key][metric] + scores_2[key][metric]) / 2

    return scores


def get_eval_dataloader(dev_features, dev_batch_size):
    eval_data = CDataset(dev_features, is_train=False)
    eval_sampler = SequentialSampler(eval_data)
    return DataLoader(eval_data, sampler=eval_sampler, batch_size=dev_batch_size, num_workers=10)


def get_eval_batch_data(batch):
    example_indices = batch["example_indices"].tolist()
    batch_source_max_len = batch["source_len"].max().item()

    source_ids = batch["source_ids"][:, :batch_source_max_len].to(DEVICE)
    source_mask = batch["source_mask"][:, :batch_source_max_len].to(DEVICE)
    item = {
        "ID": batch["ID"],
        "example_indices":example_indices,
        "source_ids": source_ids,
        "source_mask": source_mask,
    }
    return item


def gen_keyphrase_summary(answer):
    if SUM_TOKEN in answer:
        answer_split = answer.split(SUM_TOKEN)
        keyphrase, summary = answer_split[0], answer_split[1]
    else:
        summary = " ".join(answer.strip().split(" ")[-50:])
        keyphrase = " ".join(answer.strip().split(" ")[:-50])
        #print("[WARNING] No special token [{}] found in the output...".format(SUM_TOKEN))
        # print(appr_answer)

    return keyphrase, summary
    

def predict(args, model, dev_data):
    dev_examples, dev_features = dev_data
    eval_dataloader = get_eval_dataloader(dev_features, args.eval_batch_size)

    model.eval()

    pred = {} #[None] * len(dev_examples)
    pred_kp = {}

    for bi, batch in tqdm(enumerate(eval_dataloader), desc="Generating", total=len(eval_dataloader)):
        item_dict = get_eval_batch_data(batch)
        IDs, example_indices, source_ids, source_mask = item_dict["ID"], item_dict["example_indices"], item_dict["source_ids"], item_dict["source_mask"]

        with torch.no_grad():
            # encoder_outputs, source_mask = self.encode(source_ids, source_mask)
            decoding_p1, decoding_p2 = model.module.generate(
                source_ids, source_mask,
                num_beams=args.beam,
                max_length=args.test_target_max_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=True
            )

        decoding_p1 = decoding_p1.to(CPU)
        decoding_p2 = decoding_p2.to(CPU)

        for i in range(len(example_indices)):
            key = IDs[i] if isinstance(IDs[i], str) else IDs[i].item()
            if key in pred.keys():
                continue
            
            answer_p1 = model.module.tokenizer.decode(decoding_p1[i].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            answer_p2 = model.module.tokenizer.decode(decoding_p2[i].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False) 
            if args.gen_keyphrase_summary:
                keyphrase_p1, summary_p1 = gen_keyphrase_summary(answer_p1) 
                keyphrase_p2, summary_p2 = gen_keyphrase_summary(answer_p2) 
                
                pred_kp[key] ={'Person1': keyphrase_p1.strip(), 'Person2': keyphrase_p2.strip()}
                pred[key] ={'Person1': summary_p1.strip(), 'Person2': summary_p2.strip()}
            else:
                pred[key] ={'Person1': answer_p1.strip(), 'Person2': answer_p2.strip()}
        
    model.train()
    return pred, pred_kp


def evaluate(args, model, dev_data = None, source="dev", dump_pred=False):
    if dev_data is None:
        file_path = args.dev_file_path if source == "dev" else args.test_file_path
        dev_examples = load_examples(args, file_path)
        dev_features = convert_examples_to_features(args, model.module.config, model.module.tokenizer, dev_examples)
        print("[INFO] Testing file_path:", file_path)
    else:
        dev_examples, dev_features = dev_data
    
    pred, pred_kp = predict(args, model, (dev_examples, dev_features))
    scores = evaluate_rouge(dev_examples, pred)
    
    print("\n\n")
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        print('{}:\t{}: {:5.4f}\t{}: {:5.4f}\t{}: {:5.4f}'.format(metric, 'P', 100.0 * results['p'], 'R', 100.0 * results['r'], 'F1', 100.0 * results['f']))
    print("\n\n")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if dump_pred:
        with open(os.path.join(args.output_dir, "{}.pred.summary{}.json".format(source, args.add_name)), "w") as fout:
            json.dump(pred, fout, indent=4)
    
        with open(os.path.join(args.output_dir, "{}.predkp.summary{}.json".format(source, args.add_name)), "w") as fout:
            json.dump(pred_kp, fout, indent=4)

    return scores


def run_test(args, model):
    print("[INFO] Start generate test summary...") 
    scores = evaluate(args, model, source="test", dump_pred=args.dump_pred)
    with open(os.path.join(args.output_dir, "test.metrics{}".format(args.add_name)), "w") as fout:
        json.dump(scores, fout, indent=4)

     # Combine separate segments into one single summary for evaluation
    if args.do_segment:
        print("[INFO] Combine separate segments into one single summary...")
        
        with open(TEST_DIALOGSUM_PATH, "r") as f:
            f_gold = json.load(f)

        summaries_p1 = {}
        summaries_p2 = {}
        for di, data in enumerate(f_gold):
            summaries_p1[data["id"]] = data["Person1"].strip().replace("\n", " ").replace("\015", "")
            summaries_p2[data["id"]] = data["Person2"].strip().replace("\n", " ").replace("\015", "")

        with open(os.path.join(args.output_dir, "test.pred.summary{}.json".format(args.add_name)), "r") as f:
            f_pred_bart_single = json.load(f)

        update_pred = {}
        for item_id, item in f_pred_bart_single.items():
            ID, turn = item_id.rsplit("_", 1)
            if ID not in update_pred.keys():
                update_pred[ID] = {}
            update_pred[ID][turn] = item
        
        final_full_pred = {}
        for ID, value in update_pred.items():
            final_full_pred[ID] = {'Person1': [], 'Person2': []}
            for t in range(len(value)):
                if t == 0:
                    final_full_pred[ID].append(value[str(t)])
                else:
                    similar_score_max_p1 = max([SequenceMatcher(None, s, value[str(t)]['Person1']).ratio() for s in final_full_pred[ID]])
                    if similar_score_max_p1 < 0.75:
                        final_full_pred[ID]['Person1'].append(value[str(t)])

                    similar_score_max_p2 = max([SequenceMatcher(None, s, value[str(t)]['Person2']).ratio() for s in final_full_pred[ID]])
                    if similar_score_max_p2 < 0.75:
                        final_full_pred[ID]['Person2'].append(value[str(t)])
        
        with open(os.path.join(args.output_dir, "test.pred.summary_full{}".format(args.add_name)), "w") as fout:
            json.dump(final_full_pred, fout, indent=4)

        test_prediction = {'Person1': [], 'Person2': []}
        test_summaries = {'Person1': [], 'Person2': []}
        sample_count = 0
        for ID, value in final_full_pred.items():
            sample_count += 1
            test_summaries['Person1'].append(summaries_p1[ID])
            test_summaries['Person2'].append(summaries_p2[ID])
            test_prediction['Person1'].append(" ".join(value['Person1']))
            test_prediction['Person2'].append(" ".join(value['Person2']))
        
        print("Testing {} samples...".format(sample_count))
        
        scores_1 = evaluator.get_scores(test_prediction['Person1'], test_summaries['Person1'])
        scores_2 = evaluator.get_scores(test_prediction['Person2'], test_summaries['Person2'])

        scores = {}
        for key in scores_1:
            scores[key] = {}
            for metric in scores_1[key]:
                scores[key][metric] = (scores_1[key][metric] + scores_2[key][metric]) / 2

        with open(os.path.join(args.output_dir, "test.metrics_full{}".format(args.add_name)), "w") as fout:
            json.dump(scores, fout, indent=4)
        
        print("\n\n")
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            print('{}:\t{}: {:5.4f}\t{}: {:5.4f}\t{}: {:5.4f}'.format(metric, 'P', 100.0 * results['p'], 'R', 100.0 * results['r'], 'F1', 100.0 * results['f']))
        print("\n\n")
        
