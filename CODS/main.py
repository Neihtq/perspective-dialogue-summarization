import os
import json
import argparse
import torch

from torch import equal
from src.models.evaluate import run_test
from src.models.training import run_training
from src.models.model import SummarizerModel, ModelWrapper

from src.utils.constants import DEVICE

parser = argparse.ArgumentParser()
# Gerenal
parser.add_argument("--do_train", action="store_true", help = 'do model training')
parser.add_argument("--seed", default=42, type=int, help="Random seed")

# Input/Output
parser.add_argument('--train_file_path', type = str, default = './data/dialogsum/clean_data/train.json',
    help = 'Training data path')
parser.add_argument('--dev_file_path', type = str, default = './data/dialogsum/clean_data/eval.json',
    help = 'Validation data path')
parser.add_argument('--test_file_path', type = str, default = './data/dialogsum/clean_data/test.json',
    help = 'Test data path')
parser.add_argument('--load_path', type = str, default = None, help = 'Load trained model file')
parser.add_argument('--output_dir', type = str, required = True, help = 'output saving directory')
parser.add_argument("--gen_keyphrase_summary", action="store_true",
    help="for decoding, first generate keyphrase then generate summary")
parser.add_argument("--oracle_functurn_context", action="store_true", 
    help="For oracle study, using functional turns as input")
parser.add_argument("--do_segment", action="store_true",
    help="train and evaluate with segmented dialogues")
parser.add_argument("--use_pred_segment", action="store_true",
    help="for inference, using the predicted dialogue segmentation")
parser.add_argument("--add_coref_input", action="store_true",
    help="appending coreference text to the dialogue text.")
parser.add_argument("--ctrl_nb_summary_sent",  default=0, type=int, 
    help="for inference, controlling number of summary sentences by dialogue segmentation")

# Modeling
parser.add_argument("--model_name", type=str, default='facebook/bart-large-xsum', help="BART model")
parser.add_argument("--source_max_len", default=512, type=int, help="Max len of source")
parser.add_argument("--target_max_len", default=50, type=int, help="Max len of target")
parser.add_argument("--test_target_max_len", default=50, type=int, help="Max len of target")
parser.add_argument("--beam", default=4, type=int, help="Beam size")
parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs", default=300, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--max_grad_norm', required=False, default=1.0, type=float,
    help='gradient clipping for Max gradient norm.')
parser.add_argument("--patience", default=10, type=int, help="number of validation checking for earlystopping")
parser.add_argument("--no_repeat_ngram_size", default=0, type=int,
    help="for decoding, give penalty to repeated ngram")
parser.add_argument("--penalty-term", default=10, type=float, help="penalization term for similar encodings/generations")
parser.add_argument("--distributed", action="store_true", help="Flag for enabling Multi GPU training")


# Objectives
parser.add_argument("--add_functurn_loss", action="store_true",
    help="Add additional training loss (encoder side) to predict 0/1 for functional turns")
parser.add_argument("--add_module_loss", action="store_true",
    help="Add additional training loss (encoder side) to predict 7-way for modules")
parser.add_argument("--weight_addition_loss", default=1.0, type=float, help="weights to combine additional losses")
parser.add_argument("--k", default=5, type=int, help="Number of folds for K-Fold Cross Validation")
parser.add_argument("--k-fold-cross-validation", action="store_true", help="Whether to use K-Fold Cross Validation")

# Saving/Logging
parser.add_argument("--validation_timing", default=2500, type=int, help="Check dev score after every N updates")
parser.add_argument("--wandb", action="store_true", help="use weight and bias to monitor experiments")
parser.add_argument("--dump_pred", default=1, type=int, help="for inference, dumping prediction files")
parser.add_argument("--add_name", default="", type=str,
    help="for inference, appending string to the prediction file name")

args = parser.parse_args()

if __name__ == '__main__':
    if args.do_train:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        else:
            print("[WARNING] {} exists...".format(args.output_dir))
            overwrite = input('Overwrite? [y/n]')
            if overwrite.lower() == 'n':
                exit(1)
        model = run_training(args)
    elif args.load_path == '':
        print("[ERROR] No trained model specified...")
        exit(1)
    else:
        params = {
            'model_name': args.model_name,
            'load_path': args.load_path,
            'add_module_loss': args.add_module_loss,
            'add_functurn_loss': args.add_functurn_loss
        }
        model = SummarizerModel(params)
        #model = ModelWrapper(args, params)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'full_train_pytorch.bin')))
        model.to(DEVICE)

    print('[INFO] Start generate test summaries')
    run_test(args, model)
