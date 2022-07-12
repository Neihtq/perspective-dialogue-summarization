
'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import os
import time
import json
import random
import wandb
import nltk
import torch
import logging
import pandas as pd
import numpy as np
import torch.nn.functional as F

from sys import exit
from tqdm import tqdm, trange
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, BartForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, RandomSampler

nltk.download('punkt')

from src.data.dataset import CDataset
from src.utils.constants import DEVICE
from src.models.evaluate import evaluate
from src.models.model import ModelWrapper, SummarizerModel
from src.utils.data_preprocess import load_examples, convert_examples_to_features

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def init_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(args.seed)


def get_train_dataloader(train_features, train_batch_size):
    train_data = CDataset(train_features, is_train=True)
    train_sampler = RandomSampler(train_data)
    return DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size, num_workers=10)


def get_train_batch_data(batch):
    batch_source_max_len = batch["source_len"].max().item()
    source_ids = batch["source_ids"][:, :batch_source_max_len].to(DEVICE)
    source_mask = batch["source_mask"][:, :batch_source_max_len].to(DEVICE)

    batch_target_max_len_p1 = batch["target_len_p1"].max().item()
    target_ids_p1 = batch["target_ids_p1"][:, :batch_target_max_len_p1].to(DEVICE)
    target_labels_p1 = batch["target_labels_p1"][:, :batch_target_max_len_p1].contiguous().to(DEVICE)

    batch_target_max_len_p2 = batch["target_len_p2"].max().item()
    target_ids_p2 = batch["target_ids_p2"][:, :batch_target_max_len_p2].to(DEVICE)
    target_labels_p2 = batch["target_labels_p2"][:, :batch_target_max_len_p2].contiguous().to(DEVICE)

    item = {
        "ID": batch["ID"],
        "source_ids": source_ids,
        "source_mask": source_mask,

        "target_ids_p1": target_ids_p1,
        "target_labels_p1": target_labels_p1,
        "func_label_p1": batch["func_label_p1"],

        "target_ids_p2": target_ids_p2,
        "target_labels_p2": target_labels_p2,
        "func_label_p2": batch["func_label_p2"]
    }

    return item


def save_checkpoint(epoch, path, loss, model, optimizer):
    folder = os.path.join(path, 'checkpoints')
    if not os.path.exists(folder):
        os.makedirs(folder)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, os.path.join(folder, f'checkpoint_{epoch}.pt'))


def check_accumulation_step(args, step, model, scheduler, num_updates, f_log, dev_data, patience, best_em, epoch, tb):
    if (step + 1) % args.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scheduler.step()
        model.zero_grad()
        num_updates += 1
        val_loss = validate(args, model, dev_data, epoch, tb)
        em = 0
        if num_updates % args.validation_timing == 0:
            results = evaluate(args, model, dev_data)
            em = results['rouge-1']['f']
            
            if args.wandb:
                for r in results:
                    wandb.log({'eval_{}'.format(r): results[r]['f']})
            
            if f_log is not None:
                f_log.write(json.dumps(results))
                #f_log.write(f'Validaton loss epoch {epoch}: {results}')
                f_log.write('num_updates: {}\n'.format(num_updates))
                f_log.flush()

            #if results < best_em:
            if em > best_em:
                #best_em = results
                best_em = em
                patience = 0
                model.module.save(args.output_dir, "best_pytorch.bin")
            else:
                patience += 1
                print("[INFO] patience {}/{}".format(patience, args.patience))
    
    return patience, best_em, num_updates, em, val_loss #results 


def validate(args, model, dev_data, epoch, tb):
    dev_examples, dev_features = dev_data
    val_dataloader = get_train_dataloader(dev_features, args.eval_batch_size)
    val_loss = 0 
    model.eval()
    with torch.no_grad():
        for _, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validating"):
            val_loss += iteration_step(model, batch, args.penalty_term) 

    val_loss /= len(val_dataloader)

    print(f"\nValidation loss epoch {epoch}: {val_loss}")
    #tb.add_scalar(f'Loss/val_loss', val_loss, epoch)

    model.train()

    return val_loss


def iteration_step(model, batch, penalize):
    item_dict = get_train_batch_data(batch)
    source_ids, source_mask  = item_dict["source_ids"], item_dict["source_mask"] 
    target_ids_p1, target_labels_p1, func_labels_p1 = item_dict["target_ids_p1"], item_dict["target_labels_p1"], item_dict["func_label_p1"]
    target_ids_p2, target_labels_p2, func_labels_p2 = item_dict["target_ids_p2"], item_dict["target_labels_p2"], item_dict["func_label_p2"]
    
    outputs_p1, outputs_p2, encoder_hidden_state_p1, encoder_hidden_state_p2 = model(
        source_ids, source_mask, 
        target_ids_p1=target_ids_p1, target_ids_p2=target_ids_p2,
        target_labels_p1=target_labels_p1, target_labels_p2=target_labels_p2
    )
#    outputs_p1 = model.generator_forward(input_ids=None,
#                attention_mask=source_mask,
#                encoder_outputs=encoder_outputs_p1,
#                decoder_input_ids=target_ids_p1,
#                labels=target_labels_p1)
#    outputs_p2 = model.generator_forward(input_ids=None,
#                attention_mask=source_mask,
#                encoder_outputs=encoder_outputs_p2,
#                decoder_input_ids=target_ids_p2,
#                labels=target_labels_p2)

    sim = cosine_similarity(encoder_hidden_state_p1, encoder_hidden_state_p2).mean()
    loss = max(outputs_p1[0].mean(), outputs_p2[0].mean()) + (penalize*sim)

    return loss


def cosine_similarity(tensor_1, tensor_2):
    sim = torch.nn.functional.cosine_similarity(tensor_1, tensor_2)
    norm_sim = (sim + 1) / 2
    return norm_sim


def get_loss_functurn(args, model, batch_size, source_ids, encoder_outputs_p1, encoder_outputs_p2, func_labels_p1, func_labels_p2, train_loss_tracker_func, step, loss):
    def calc_loss_functurn(func_labels, encoder_outputs):
        sent_repr_mat = []
        turn_nums = [(item == model.generator.model.config.bos_token_id).sum().cpu().item() for item in source_ids]
        max_turn_num = max(turn_nums)
        for i in range(batch_size):
            sent_repr = encoder_outputs[i][source_ids[i] == model.generator.model.config.bos_token_id]  # [num_of_turns, hd_dim]
            sent_repr = torch.cat(
                [sent_repr, torch.zeros(max_turn_num - turn_nums[i], sent_repr.size(1)).to(DEVICE)], 0)
            sent_repr_mat.append(sent_repr)
            func_labels[i][turn_nums[i]:] = -1
        sent_repr_mat = torch.stack(sent_repr_mat, 0)  # [batch_size, max_turn_num, hd_dim]

        func_labels = func_labels[:, :max_turn_num]
        prediction_logits = model.classifier(sent_repr_mat)
        loss_functurn = F.cross_entropy(prediction_logits.reshape(-1, prediction_logits.size(-1)), \
                                        func_labels.reshape(-1), ignore_index=-1, reduction='mean')
        return loss_functurn

    loss_functurn = max(calc_loss_functurn(encoder_outputs_p1, func_labels_p1) + calc_loss_functurn(encoder_outputs_p2, func_labels_p2))
    
    loss += loss_functurn * args.weight_addition_loss
    
    train_loss_tracker_func.append(loss_functurn.item())
    if args.wandb and step % 50 == 0:
        wandb.log({'avg_training_loss_functurn': np.mean(train_loss_tracker_func)})

    return loss


def train(model, args, train_examples, dev_examples, suffix=''):
    init_seed(args)
    train_features = convert_examples_to_features(args, model.module.config, model.module.tokenizer, train_examples)
    dev_features = convert_examples_to_features(args, model.module.config, model.module.tokenizer, dev_examples)
    dev_data = (dev_examples, dev_features)

    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    num_train_steps = int(len(train_features) / train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    train_dataloader = get_train_dataloader(train_features, train_batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = num_train_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    tb = SummaryWriter() 

    for param in model.module.generator.parameters():
        param.requires_grad = False

    model.to(DEVICE)
    #model = nn.DataParallel(model).to(DEVICE)
    model.zero_grad()
    model.train()
    
    num_updates = 0
    best_em = 0 
    patience = 0

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    start = time.time()
    with open(os.path.join(args.output_dir, "{}.log".format(args.model_name.replace("/", "-") + suffix)), 'w') as f_log:
        train_loss_tracker = []
        N = len(train_dataloader)
        running_steps = 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            running_loss = 0
            for step, batch in tqdm(enumerate(train_dataloader), total=N, desc="Iteration"):
                loss = iteration_step(model, batch, args.penalty_term)
                
                running_steps += 1
                running_loss += loss.item()

                tb.add_scalar('Loss (per step)', loss, running_steps)
                if (step + 1) % 50 == 0:
                    print(f"loss: {loss.item()} [step: {step}]")
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
            running_loss /= N
            train_loss_tracker.append(running_loss)
            print(f"epoch {epoch} avg. loss: {running_loss}\n")

            if not args.k_fold_cross_validation:
                patience, best_em, num_updates, em, val_loss = check_accumulation_step(
                    args, step, model, 
                    scheduler, num_updates, f_log, dev_data,  patience, best_em, epoch, tb
                )
                tb.add_scalars('Loss', {
                    'train_loss': running_loss,
                    'val_loss': val_loss}, epoch)
                #tb.add_scalar('Loss/train_loss', running_loss, epoch)
                tb.add_scalar('Eval ROUGE F1 Score', em, epoch)

            if patience > args.patience:
                print("[INFO] Ran out of patience...")
                break
            
            if epoch % 2:
                save_checkpoint(epoch, args.output_dir, running_loss, model, optimizer)

        print('[INFO] Completed all epochs.')
        end = time.time()
        content = f'Duration: {end - start} s'
        print(content)
        f_log.write('\n' + content)

    model.module.save(args.output_dir, name='full_train_pytorch.bin')


def run_training(args):
    params = {
        'model_name': args.model_name,
        'load_path': args.load_path,
        'add_module_loss': args.add_module_loss,
        'add_functurn_loss': args.add_functurn_loss,
    }

    train_examples = load_examples(args, args.train_file_path)
    dev_examples = load_examples(args, args.dev_file_path)
    if args.k_fold_cross_validation:
        total_examples = np.array(train_examples + dev_examples)
        kfold = KFold(args.k, True, 1)
        
        print(f"[INFO] Start training with {args.k} folds.")
        fold_n, final_scores, names = 1, [], []
        for train_fold, val_fold in kfold.split(total_examples):
            #model = SummarizerModel(params)
            model = ModelWrapper(args, params)
            print(f'[INFO] Train with fold #{fold_n}.')
            train(model, args, total_examples[train_fold], total_examples[val_fold], f"_fold_{fold_n}")
            dev_features = convert_examples_to_features(args, model.config, model.tokenizer, dev_examples)
            dev_data = (dev_examples, dev_features)
            scores = evaluate(args, model, dev_data)[r'rouge-1']['f']
            final_scores.append(scores)
            dest = os.path.join(args.output_dir, "pytorch_fold_{fold_n}.bin")
            model.save(args.output_dir, "pytorch_fold_{fold_n}.bin")
            names.append(dest)
            fold_n += 1
        
        avg_score = sum(final_scores)/ len(final_scores)
        print("Avg Score Rouge-1 F1:", avg_score)
    else:
        model = ModelWrapper(args, params)
        #model = SummarizerModel(params)
        train(model, args, train_examples, dev_examples)

    weight_encoder_p2_k_proj = model.module.encoder_p2.self_attn.k_proj.weight
    weight_encoder_p1_k_proj = model.module.encoder_p1.self_attn.k_proj.weight
    print('Heads having equal weights:', torch.equal(weight_encoder_p1_k_proj, weight_encoder_p2_k_proj))
    #model.module.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch.bin')))
    print('[INFO] Finished training.')

    return model

