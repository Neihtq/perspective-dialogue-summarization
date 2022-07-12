#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import os
import sys
import math
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from rouge import Rouge

from fairseq import (
    checkpoint_utils, distributed_utils, metrics, options, progress_bar, tasks, utils
)
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import StopwatchMeter
from fairseq.models.bart.hub_interface import BARTHubInterface


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    logger.info('training on {} GPUs'.format(args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')

    print(args.multi_views)
    epoch = 1
    while (
        lr > args.min_lr
        and (
            epoch_itr.epoch < max_epoch
            # allow resuming training from the final checkpoint
            or epoch_itr._next_epoch_itr is not None
        )
        and trainer.get_num_updates() < max_update
    ):
        
        # train for one epoch
        train(args, trainer, task, epoch_itr, epoch)
        epoch += 1
        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
         
        epoch_itr = trainer.get_train_iterator(
            epoch_itr.epoch,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(args, 'data', '')),
        )

       # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
        
        bart = BARTHubInterface(args, task, trainer.model).cuda()
        #print(bart.device)
        del trainer
        torch.cuda.empty_cache()
        bart.eval()
        count = 1
        bsz = 8

        print("Test on val set: ")
        val_trans = '../data/dialogsum/DialogSum_Data/val_dialogsum_sent_trans_cons_label_2.source'
        val_c99 = '../data/dialogsum/DialogSum_Data/val_dialogsum_sent_c99_label.source'
        val_hypo = './val_dialogsum_best_multi_attn_'+str(args.lr_weight)+'_.hypo'
        with open(val_trans) as source, open(val_c99) as source2, open(val_hypo, 'wt', encoding='utf-8') as fout:            
            s1 = source.readlines()
            s2 = source2.readlines()
            
            slines = [s1[0].strip()]
            slines2 = [s2[0].strip()]
            
            for i in tqdm(range(1, len(s1))):
                if count % bsz == 0:
                    with torch.no_grad():
                        if args.multi_views:
                            hypotheses_batch = bart.sample(slines, sentences2 = slines2, balance = True, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=3)
                        else:
                            hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=3)
                    for hypothesis in hypotheses_batch:
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                    slines2 = []
                
                slines.append(s1[i].strip())
                slines2.append(s2[i].strip())
            
                count += 1
                
            if slines != []:
                if args.multi_views:
                    hypotheses_batch = bart.sample(slines, sentences2 = slines2, balance = True, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=3)
                else:
                    hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=3)
                #hypotheses_batch = bart.sample(slines, sentences2 = slines2, balance = True, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
        
        hyp_path = './val_dialogsum_best_multi_attn_'+str(args.lr_weight)+'_.hypo'
        ref_path = '../data/dialogsum/DialogSum_Data/val_dialogsum_sent_trans_cons_label_2.target'
        hypothesis = []
        with open(hyp_path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                hypothesis.append(l[:-1])
        
        reference = []
        with open(ref_path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                reference.append(l[:-1])

        rouge = Rouge()
        print("Val", rouge.get_scores(hypothesis, reference, avg = True))
        
        
        trainer = Trainer(args, task, model, criterion)
        print("Test on testing set: ")

        count = 1
        bsz = 8
        test_trans = '../data/dialogsum/DialogSum_Data/test_dialogsum_sent_trans_cons_label_2.source'
        test_c99 = '../data/dialogsum/DialogSum_Data/test_dialogsum_sent_c99_label.source'
        test_hypo = './test_dialogsum_best_multi_attn_'+str(args.lr_weight)+'_.hypo'
        with open(test_trans) as source, open(test_c99) as source2, open(test_hypo, 'wt', encoding='utf-8') as fout:
        #with open('../data/test_sent_trans_cons_label.source') as source, open('../data/test_sent_c99_label.source') as source2, open(test_hypo, 'wt', encoding='utf-8') as fout:
            s1 = source.readlines()
            s2 = source2.readlines()
            
            slines = [s1[0].strip()]
            slines2 = [s2[0].strip()]
            
            for i in tqdm(range(1, len(s1))):
                if count % bsz == 0:
                    with torch.no_grad():
                        if args.multi_views:
                            hypotheses_batch = bart.sample(slines, sentences2 = slines2, balance = True, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=3)
                        else:
                            hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=3)
                    for hypothesis in hypotheses_batch:
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                    slines2 = []
                
                slines.append(s1[i].strip())
                slines2.append(s2[i].strip())
            
                count += 1
                
            if slines != []:
                if args.multi_views:
                    hypotheses_batch = bart.sample(slines, sentences2 = slines2, balance = True, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=3)
                else:
                    hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=3)
                
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
        hyp_path = './test_dialogsum_best_multi_attn_'+str(args.lr_weight)+'_.hypo'
        ref_path = '../data/dialogsum/DialogSum_Data/test_dialogsum_sent_trans_cons_label_2.target'
        hypothesis = []
        with open(hyp_path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                hypothesis.append(l[:-1])
        
        reference = []
        with open(ref_path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                reference.append(l[:-1])

        rouge = Rouge()
        print('Test', rouge.get_scores(hypothesis, reference, avg = True))
        
        # early stop
        if should_stop_early(args, valid_losses[0]):
            logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
            break

        del bart
        torch.cuda.empty_cache()

        trainer = Trainer(args, task, model, criterion)
        extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    train_meter.stop()
    logger.info('done training in {:.1f} seconds'.format(train_meter.sum))


def should_stop_early(args, valid_loss):
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        return should_stop_early.num_runs > args.patience


@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr, epoch):
    """Train the model for one epoch."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch >= args.curriculum),
        #shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch - 1]
        #args.update_freq[epoch_itr.epoch - 1]
        if epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch, no_progress_bar='simple',
    )

    # task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf

    for samples in progress:
        log_output = trainer.train_step(samples)
        num_updates = trainer.get_num_updates()
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(metrics.get_smoothed_values('train'))
        progress.log(stats, tag='train', step=num_updates)

        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(metrics.get_smoothed_values('train'))
    progress.print(stats, tag='train', step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters('train')


def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main(modify_parser=None):
    
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            logger.info('NOTE: you may get faster training with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)

def show_args():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=None)
    with open("/home/tnguyen/dialogue-text-summarization-dokument/args.txt", 'w') as file:
        for k, v in vars(args).items():
            if isinstance(v, str):
                value = '"' + v + '"'
            else:
                value = str(v)
            content = k + ' = ' + value + '\n'
            file.write(content)


if __name__ == '__main__':
    cli_main()
