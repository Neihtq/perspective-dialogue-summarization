import os
import json

from hf_train_bart import SequenceMatcher
from hf_train_bart import Bart

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
save_path = './save/dialoguesum_4'
model_path = os.path.join(save_path, 'pytorch.bin')

dialoguesum_path = './data/dialogsum/'
clean_path = os.path.join(dialoguesum_path, 'clean_data')
test_path = os.path.join(clean_path, 'test.json')

def create_sample(dest_path, n=20):
    model = Bart()
    model.args.oracle_functurn_context = False
    model.args.ctrl_nb_summary_sent = 0
    model.args.source_max_len = 512
    model.args.target_max_len = 50
    model.args.eval_batch_size = 8
    model.args.no_repeat_ngram_size = 0
    model.args.beam = 4

    summarys = {}
    f_gold = json.load(open("./data/dialogsum/clean_data/test.json", "r"))
    for di, data in enumerate(f_gold):
        summarys[data["id"]] = data["summary"].strip().replace("\n", " ").replace("\015", "")
    
    f_pred_bart_single = json.load(open(os.path.join(model.args.output_dir, "test.pred.summary{}.json".format(model.args.add_name)), "r"))
    update_pred = {}
    for item_id, item in f_pred_bart_single.items():
        ID, turn = item_id.rsplit("#", 1)
        if ID not in update_pred.keys():
            update_pred[ID] = {}
        update_pred[ID][turn] = item

    final_full_pred = {}
    for ID, value in update_pred.items():
        final_full_pred[ID] = []
        for t in range(len(value)):
            if t == 0:
                final_full_pred[ID].append(value[str(t)])
            else:
                similar_score_max = max([SequenceMatcher(None, s, value[str(t)]).ratio() for s in final_full_pred[ID]])
                if similar_score_max < 0.75:
                    final_full_pred[ID].append(value[str(t)])

    test_summary = []
    test_prediction = []
    for ID, value in final_full_pred.items():
        test_summary.append(summarys[ID])
        test_prediction.append(" ".join(value))

    with open(dest_path, 'w') as file:
        for i in range(n):
            file.write(test_prediction[i] + '\n')


if __name__ == '__main__':
    create_sample('CODS_summary_samples.txt')