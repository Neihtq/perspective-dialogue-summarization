import os
import re
import json
import torch

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.models.model import ModelWrapper
from src.utils.similarity import accumulate_predictions, similar
from src.utils.data_preprocess import load_examples, convert_examples_to_features
from src.models.evaluate import evaluate, predict, get_eval_batch_data, get_eval_dataloader, evaluate_rouge

test_path = os.path.join('data', 'processed', 'new_dialogsum_clean_data', 'test.json')
batch_size = 128
class DummyArgs:
    do_segment = True
    output_dir='models/penalty_salesforce_5'
    use_pred_segment = False
    test_file_path = test_path
    oracle_functurn_context = False
    source_max_len = 512
    gen_keyphrase_summary = True
    target_max_len = 50
    add_module_loss = False
    add_functurn_loss = False
    eval_batch_size = batch_size
    no_repeat_ngram_size = 0
    beam = 4
    test_target_max_len = 50
    distributed = False


def get_pred_similarity(examples, pred, sent_embedder):
    preds_p1, preds_p2 = accumulate_predictions(examples, pred)
    cosine_scores = similar(preds_p1, preds_p2, sent_embedder)
    sim = 0
    for i in range(cosine_scores.shape[0]):
        sim += cosine_scores[i, i].item()

    sim /= cosine_scores.shape[0]
    
    return sim


def initialize_model(args):
    model_name = "Salesforce/bart-large-xsum-samsum"
    params = {
        'model_name': model_name,
        'load_path': None,
        'add_module_loss': None,
        'add_functurn_loss': None
    }

    model = ModelWrapper(args, params)

    return model


def load_data(args, model):
    file_path = args.test_file_path
    dev_examples = load_examples(args, file_path)
    dev_features = convert_examples_to_features(args, model.module.config, model.module.tokenizer, dev_examples)

    return (dev_examples, dev_features)


def main():
    checkpoints_dir = os.path.join('models', 'penalty_salesforce_5', 'checkpoints')
    sent_embed = SentenceTransformer('all-MiniLM-L6-v2').cuda()
    args = DummyArgs()
    model = initialize_model(args)
    model.cuda()
    model.eval()
    similarities = {}
    dev_data = load_data(args, model)
    for file in tqdm(os.listdir(checkpoints_dir)):
        checkpoint = os.path.join(checkpoints_dir, file)
        epoch = int(re.findall('[0-9]+', file)[0])
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
        pred, _ = predict(args, model, dev_data)

        f1_score = evaluate_rouge(dev_data[0], pred)['rouge-1']['f']
        cosine_score = get_pred_similarity(dev_data[0], pred, sent_embed) 
        similarities[epoch] = [f1_score, cosine_score]

    with open('f1_cos_test_metrics.json', 'w') as json_file:
        json.dump(similarities, json_file, indent=4)


if __name__ == '__main__':
    main()
