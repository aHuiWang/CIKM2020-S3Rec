# -*- coding: utf-8 -*-
# @Time    : 2020/11/5 21:11
# @Author  : Hui Wang

import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from trainers import FinetuneTrainer
from models import S3RecModel
from utils import EarlyStopping, get_user_seqs_and_sample, get_item2attribute_json, check_path, set_seed

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='Finetune_sample', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    args.sample_file = args.data_dir + args.data_name + '_sample.txt'
    item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

    user_seq, max_item, sample_seq = \
        get_user_seqs_and_sample(args.data_file, args.sample_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.ckp}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    args.item2attribute = item2attribute

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, test_neg_items=sample_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SASRecDataset(args, user_seq, test_neg_items=sample_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)


    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)


    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=False)

    else:
        pretrained_path = os.path.join(args.output_dir, f'{args.data_name}-epochs-{args.ckp}.pt')
        try:
            trainer.load(pretrained_path)
            print(f'Load Checkpoint From {pretrained_path}!')

        except FileNotFoundError:
            print(f'{pretrained_path} Not Found! The Model is same as SASRec')

        early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=False)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Sample 99 results-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=False)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
main()