from __future__ import absolute_import, division, print_function
from collections import defaultdict

import shutil
import argparse
import logging
import os
import json
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
import wandb
import tqdm

from s2s_ft.modeling import BertForSequenceToSequenceWithPseudoMask, BertForSequenceToSequenceUniLMV1
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertTokenizer

from s2s_ft import utils
from s2s_ft.config import BertForSeq2SeqConfig

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGLEVEL)


MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
}

def training_cpt(args, tokenizer, input_ids, attention_mask,  position_ids, _init_label_emb, num_hiers, reversed_hiers):
    from transformers import BertForMaskedLM
    from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
    label_nums = input_ids.shape[0] - 2

    model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    model = model.train()
    model.cuda()

    init_label_emb = _init_label_emb.float().cuda().requires_grad_()
    torch.save(init_label_emb.cpu(), 'before.pt')

    optimizer_grouped_parameters = [
        {'params': [init_label_emb, ], 'weight_decay': 0.0}
    ]
    cpt_optimizer = AdamW(optimizer_grouped_parameters, lr=args.label_cpt_lr, eps=args.adam_epsilon)

    mask_ratio = 0.15
    bs = args.label_cpt_bsz
    b_input_ids = input_ids.unsqueeze(0).repeat(bs, 1).cuda().long()
    position_ids = position_ids.unsqueeze(0).repeat(bs, 1).cuda().long()

    if args.label_cpt_decodewithpos:
        position_ids[:, 1:-1] += args.max_source_seq_length - 1
        position_ids[:, -1] = args.max_source_seq_length + args.max_target_seq_length - 1
    attention_mask = attention_mask.unsqueeze(0).repeat(bs, 1, 1).cuda().long()
    for step in range(args.label_cpt_steps):
        if args.label_cpt_not_incr_mask_ratio:
            c_mask_ratio = mask_ratio
        else:
            c_mask_ratio = mask_ratio + (step / args.label_cpt_steps) * 0.3
        inputs_embeds = torch.cat([model.bert.embeddings.word_embeddings.weight[tokenizer.cls_token_id].unsqueeze(0),
                                   init_label_emb,
                                   model.bert.embeddings.word_embeddings.weight[tokenizer.sep_token_id].unsqueeze(0),])
        inputs_embeds = inputs_embeds.unsqueeze(0).repeat(bs, 1, 1).cuda()
        mask_tokens = ~torch.bernoulli(torch.ones_like(b_input_ids) * (1 - c_mask_ratio)).bool()
        labels = torch.ones_like(b_input_ids).long() * -100
        # keep cls & sep unmask
        mask_tokens[:, 0] = 0
        mask_tokens[:, -1] = 0
        labels[mask_tokens] = b_input_ids[mask_tokens] - model.bert.embeddings.word_embeddings.num_embeddings
        inputs_embeds[mask_tokens] = model.bert.embeddings.word_embeddings.weight[tokenizer.mask_token_id]
        outputs = model.bert(
            None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        hidden_states = model.cls.predictions.transform(sequence_output)
        prediction_scores = hidden_states @ init_label_emb.T

        if args.label_cpt_use_bce:
            loss_fct = BCEWithLogitsLoss()  # -100 index = padding token
            with torch.no_grad():
                bce_labels = torch.zeros_like(prediction_scores)
                _bce_labels = []
                for b in range(bs):
                    l = labels[b][mask_tokens[b]].tolist()
                    bce_l = bce_labels[b][mask_tokens[b]]
                    c = defaultdict(list)
                    lmap = {}
                    for il in l:
                        if il not in num_hiers:
                            # last labels
                            p = reversed_hiers[il]
                            c[p].append(il)
                            lmap[il] = p
                    for i, il in enumerate(l):
                        if il not in lmap:
                            bce_l[i][il] = 1
                        else:
                            for j in c[lmap[il]]:
                                bce_l[i][j] = 1
                    _bce_labels.append(bce_l)
                bce_labels = torch.cat(_bce_labels, dim=0)
                print(bce_labels.sum())
            masked_lm_loss = loss_fct(prediction_scores[mask_tokens], bce_labels)
        else:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, label_nums), labels.view(-1))

        masked_lm_loss.backward()
        cpt_optimizer.step()
        model.zero_grad()
        init_label_emb.grad = None
        print(f'step {step}', masked_lm_loss.item())
    torch.save(init_label_emb.cpu(), 'after.pt')
    return init_label_emb

def prepare_for_training(args, model, checkpoint_state_dict, amp):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if checkpoint_state_dict:
        optimizer.load_state_dict(checkpoint_state_dict['optimizer'])
        model.load_state_dict(checkpoint_state_dict['model'])

        # then remove optimizer state to make amp happy
        # https://github.com/NVIDIA/apex/issues/480#issuecomment-587154020
        if amp:
            optimizer.state = {}

    if amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        if checkpoint_state_dict:
            amp.load_state_dict(checkpoint_state_dict['amp'])

            # Black Tech from https://github.com/NVIDIA/apex/issues/480#issuecomment-587154020
            # forward, backward, optimizer step, zero_grad
            random_input = {'source_ids': torch.ones(size=(2, 2), device=args.device, dtype=torch.long),
                            'target_ids': torch.ones(size=(2, 2), device=args.device, dtype=torch.long),
                            'label_ids': torch.ones(size=(2, 2), device=args.device, dtype=torch.long),
                            'pseudo_ids': torch.ones(size=(2, 2), device=args.device, dtype=torch.long),
                            'num_source_tokens': torch.zeros(size=(2,), device=args.device, dtype=torch.long),
                            'num_target_tokens': torch.zeros(size=(2,), device=args.device, dtype=torch.long)}
            loss = model(**random_input)
            print("Loss = %f" % loss.cpu().item())
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            model.zero_grad()

            # then load optimizer state_dict again (this time without removing optimizer.state)
            optimizer.load_state_dict(checkpoint_state_dict['optimizer'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    return model, optimizer


def train(args, training_features, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0] and args.log_dir:
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        tb_writer = None

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None

    # model recover
    recover_step = utils.get_max_epoch_model(args.output_dir)

    if recover_step:
        checkpoint_state_dict = utils.get_checkpoint_state_dict(args.output_dir, recover_step)
    else:
        checkpoint_state_dict = None

    model.to(args.device)
    model, optimizer = prepare_for_training(args, model, checkpoint_state_dict, amp=amp)

    per_node_train_batch_size = args.per_gpu_train_batch_size * args.n_gpu * args.gradient_accumulation_steps
    train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    global_step = recover_step if recover_step else 0

    if args.num_training_steps == -1:
        args.num_training_steps = args.num_training_epochs * len(training_features) / train_batch_size

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps, last_epoch=-1)

    if checkpoint_state_dict:
        scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler"])

    train_dataset = utils.Seq2seqDatasetForBert(
        features=training_features, max_source_len=args.max_source_seq_length,
        max_target_len=args.max_target_seq_length, vocab_size=model.bert.embeddings.word_embeddings.num_embeddings,
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id, random_prob=args.random_prob, keep_prob=args.keep_prob,
        offset=train_batch_size * global_step, num_training_instances=train_batch_size * args.num_training_steps,
        source_mask_prob=args.source_mask_prob, target_mask_prob=args.target_mask_prob,
        mask_way=args.mask_way, num_max_mask_token=args.num_max_mask_token,
        soft_label=args.soft_label,
    )


    logger.info("Check dataset:")
    for i in range(5):
        source_ids, target_ids = train_dataset.__getitem__(i)[:2]
        logger.info("Instance-%d" % i)
        logger.info("Source tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(source_ids)))
        if args.soft_label:
            real_target_ids = []
            if type(target_ids) is list:
                target_ids = torch.tensor(target_ids)
            for i in range(target_ids.shape[0]):
                real_target_ids.append(torch.arange(target_ids.shape[-1])[target_ids[i].bool()].tolist())
            for rti in real_target_ids:
                logger.info("Target tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(rti)))
        else:
            logger.info("Target tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(target_ids)))

    logger.info("Mode = %s" % str(model))

    # Train!
    logger.info("  ***** Running training *****  *")
    logger.info("  Num examples = %d", len(training_features))
    logger.info("  Num Epochs = %.2f", len(train_dataset) / len(training_features))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Batch size per node = %d", per_node_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.num_training_steps)

    if args.num_training_steps <= global_step:
        logger.info("Training is done. Please use a new dir or clean this dir!")
    else:
        # The training features are shuffled
        train_sampler = SequentialSampler(train_dataset) \
            if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=per_node_train_batch_size // args.gradient_accumulation_steps,
            collate_fn=utils.batch_list_to_batch_tensors)

        train_iterator = tqdm.tqdm(
            train_dataloader, initial=global_step * args.gradient_accumulation_steps,
            desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=args.local_rank not in [-1, 0])

        model.train()
        model.zero_grad()

        tr_loss, logging_loss = 0.0, 0.0
        best_macro_f1, best_micro_f1 = 0, 0
        best_macro_f1_path, best_micro_f1_path = None, None

        for step, batch in enumerate(train_iterator):
            if global_step > args.num_training_steps:
                break
            batch = tuple(t.to(args.device) for t in batch)
            if args.mask_way == 'v2':
                inputs = {'source_ids': batch[0],
                        'target_ids': batch[1],
                        'label_ids': batch[2],
                        'pseudo_ids': batch[3],
                        'num_source_tokens': batch[4],
                        'num_target_tokens': batch[5]}
            elif args.mask_way == 'v1' or args.mask_way == 'v0':
                inputs = {'source_ids': batch[0],
                        'target_ids': batch[1],
                        'masked_ids': batch[2],
                        'masked_pos': batch[3],
                        'masked_weight': batch[4],
                        'num_source_tokens': batch[5],
                        'num_target_tokens': batch[6]}

            if args.soft_label:
                inputs['label_ids'] = inputs['label_ids'].float()
                inputs['target_ids'] = inputs['target_ids'].float()

            if args.label_cpt_decodewithpos:
                model.target_offset = args.max_source_seq_length
                inputs['target_no_offset'] = True

            loss = model(**inputs)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))
            if args.wandb:
                if (step + 1) % 50 == 0:
                    wandb.log({'train/loss': loss.item()})
                    wandb.log({'train/learning_rate': scheduler.get_lr()[0],
                                   "train/global_step": step})
            else:
                if (step + 1) % 50 == 0:
                    print('train/loss', loss.item())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()


            logging_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("")
                    logger.info(" Step [%d ~ %d]: %.2f", global_step - args.logging_steps, global_step, logging_loss)
                    logging_loss = 0.0

                if args.local_rank in [-1, 0] and args.save_steps > 0 and \
                        (global_step % args.save_steps == 0 or global_step == args.num_training_steps):

                    save_path = os.path.join(args.output_dir, "ckpt-%d" % global_step)
                    os.makedirs(save_path, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(save_path)

                    optim_to_save = {
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": scheduler.state_dict(),
                    }
                    if args.fp16:
                        optim_to_save["amp"] = amp.state_dict()
                    torch.save(optim_to_save, os.path.join(save_path, utils.OPTIM_NAME))
                    logger.info("Saving model checkpoint %d into %s", global_step, save_path)

                    from test import main

                    flags = ['--model_type'     , args.model_type                          ,
                    '--tokenizer_name'         , args.model_name_or_path             ,
                     '--input_file'             , args.valid_file                  ,
                     '--split'                  , 'valid'                         ,
                     '--do_lower_case'          ,
                     '--model_path'             , str(save_path)              ,
                     '--max_seq_length'         , str(args.max_source_seq_length + args.max_target_seq_length) if args.label_cpt_decodewithpos else str(args.max_source_seq_length)             ,
                     '--max_tgt_length'         , str(args.max_target_seq_length)             ,
                     '--batch_size'             , '32'                            ,
                     '--beam_size'              , '1'                             ,
                     '--length_penalty'         , '0'                             ,
                     '--forbid_duplicate_ngrams',
                     '--mode'                   , 's2s'                           ,
                     '--forbid_ignore_word'     , '"."'                           ,
                     '--cached_features_file'   , str(os.path.join(args.output_dir, "cached_features_for_valid.pt")),
                     '--add_vocab_file'         , args.add_vocab_file]

                    if args.softmax_label_only:
                        flags.append('--softmax_label_only')
                    if args.soft_label:
                        flags.append('--soft_label')
                    if args.soft_label_hier_real:
                        flags.append('--soft_label_hier_real_with_train_file')
                        flags.append(args.train_file)
                    if args.label_cpt_decodewithpos:
                        flags.append('--target_no_offset')

                    if args.model_type == 'roberta':
                        del flags[flags.index('--do_lower_case')]

                    out = main(flags)
                    if args.wandb:
                        wandb.log({'eval/macro_f1': out['macro_f1'], 'eval/micro_f1': out['micro_f1']})

                    keep_save_model = False
                    if out['macro_f1'] > best_macro_f1:
                        best_macro_f1 = out['macro_f1']
                        if best_macro_f1_path != best_micro_f1_path and best_macro_f1_path is not None:
                            try:
                                shutil.rmtree(best_macro_f1_path)
                            except:
                                pass
                        best_macro_f1_path = save_path
                        keep_save_model = True

                    if out['micro_f1'] > best_micro_f1:
                        best_micro_f1 = out['micro_f1']
                        if best_micro_f1_path != best_macro_f1_path and best_micro_f1_path is not None:
                            try:
                                shutil.rmtree(best_micro_f1_path)
                            except:
                                pass
                        best_micro_f1_path = save_path
                        keep_save_model = True

                    if not keep_save_model:
                        try:
                            shutil.rmtree(save_path)
                        except:
                            pass
                    print('best micro', best_micro_f1_path, best_micro_f1)
                    print('best macro', best_macro_f1_path, best_macro_f1)

    if args.local_rank in [-1, 0] and tb_writer:
        tb_writer.close()
    return best_macro_f1_path, best_micro_f1_path


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--train_source_file", default=None, type=str, required=True,
    #                     help="Training data contains source")
    # parser.add_argument("--train_target_file", default=None, type=str, required=True,
    #                     help="Training data contains target")
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--valid_file", default=None, type=str, required=True,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--test_file", default=None, type=str,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_source_seq_length", default=464, type=int,
                        help="The maximum total source sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_target_seq_length", default=48, type=int,
                        help="The maximum total target sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--cached_train_features_file", default=None, type=str,
                        help="Cached training features file")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_training_steps", default=-1, type=int,
                        help="set total number of training steps to perform")
    parser.add_argument("--num_training_epochs", default=10, type=int,
                        help="set total number of training epochs to perform (--num_training_steps has higher priority)")
    parser.add_argument("--num_warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--random_prob", default=0.1, type=float,
                        help="prob to random replace a masked token")
    parser.add_argument("--keep_prob", default=0.1, type=float,
                        help="prob to keep no change for a masked token")
    parser.add_argument("--fix_word_embedding", action='store_true',
                        help="Set word embedding no grad when finetuning.")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument('--source_mask_prob', type=float, default=-1.0,
                        help="Probability to mask source sequence in fine-tuning")
    parser.add_argument('--target_mask_prob', type=float, default=0.5,
                        help="Probability to mask target sequence in fine-tuning")
    parser.add_argument('--num_max_mask_token', type=int, default=0,
                        help="The number of the max masked tokens in target sequence")
    parser.add_argument('--mask_way', type=str, default='v2',
                        help="Fine-tuning method (v0: position shift, v1: masked LM, v2: pseudo-masking)")
    parser.add_argument("--lmdb_cache", action='store_true',
                        help="Use LMDB to cache training features")
    parser.add_argument("--lmdb_dtype", type=str, default='h',
                        help="Data type for cached data type for LMDB")

    parser.add_argument("--add_vocab_file", type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--softmax_label_only', action='store_true')

    parser.add_argument('--soft_label', action='store_true')
    parser.add_argument('--soft_label_hier_real', action='store_true')

    parser.add_argument('--one_by_one_label_init_map', type=str, default=None)
    parser.add_argument('--label_cpt', type=str, default=None)
    parser.add_argument('--label_cpt_lr', type=float, default=1e-3)
    parser.add_argument('--label_cpt_steps', type=int, default=500)
    parser.add_argument('--label_cpt_bsz', type=int, default=32)
    parser.add_argument('--label_cpt_not_incr_mask_ratio', action='store_true')
    parser.add_argument('--label_cpt_use_bce', action='store_true')

    parser.add_argument('--label_cpt_decodewithpos', action='store_true')

    parser.add_argument('--random_label_init', action='store_true')

    parser.add_argument('--nyt_only_last_label_init', action='store_true')

    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--only_test_path', type=str, default=None)

    parser.add_argument('--rcv1_expand', type=str, default=None)
    parser.add_argument
    args = parser.parse_args()
    return args


def prepare(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


def get_model_and_tokenizer(args):
    config_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    config = BertForSeq2SeqConfig.from_exist_config(
        config=model_config, label_smoothing=args.label_smoothing,
        fix_word_embedding=args.fix_word_embedding,
        max_position_embeddings=args.max_source_seq_length + args.max_target_seq_length)

    logger.info("Model config for seq2seq: %s", str(config))

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None)

    model_class = \
        BertForSequenceToSequenceWithPseudoMask if args.mask_way == 'v2' \
            else BertForSequenceToSequenceUniLMV1

    logger.info("Construct model %s" % model_class.MODEL_NAME)

    model = model_class.from_pretrained(
        args.model_name_or_path, config=config, model_type=args.model_type,
        reuse_position_embedding=True,
        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.add_vocab_file:
        import pickle
        with open(args.add_vocab_file, 'rb') as f:
            label_map = pickle.load(f)
        label_tokens_start_index  = model.bert.embeddings.word_embeddings.num_embeddings
        labels_key = list(label_map.keys())
        label_name_tensors = []
        max_l = -1
        if args.rcv1_expand:
            rcv1_label_expand = {}
            for i in open(args.rcv1_expand):
                oi = [j for j in i.replace('\n', '').split(' ') if len(j) > 0]
                rcv1_label_expand[oi[3]] = i.split('child-description: ')[-1].lower().replace('\n', '')

        for lk in labels_key:
            if args.one_by_one_label_init_map:
                from collections import defaultdict
                hiera = defaultdict(set)
                _label_dict = {}
                with open(args.one_by_one_label_init_map) as f:
                    _label_dict['Root'] = -1
                    for line in f.readlines():
                        line = line.strip().split('\t')
                        for i in line[1:]:
                            if i not in _label_dict:
                                _label_dict[i] = len(_label_dict) - 1
                            hiera[line[0]].add(i)
                    _label_dict.pop('Root')

                r_hiera = {}
                for i in hiera:
                    for j in list(hiera[i]):
                        r_hiera[j] = i

                def _loop(a):
                    if r_hiera[a] != 'Root':
                        return [a,] + _loop(r_hiera[a])
                    else:
                        return [a]

                one_by_one_label_init_map = {}
                for i in _label_dict:
                    one_by_one_label_init_map[i] = '/'.join(_loop(i)[::-1])
                print(f'map {lk} to {one_by_one_label_init_map[lk]}')
                label_name_tensors.append(tokenizer.encode(one_by_one_label_init_map[lk], add_special_tokens=False))
            elif args.nyt_only_last_label_init:
                print(f'map {lk} to {lk.split("/")[-1]}')
                label_name_tensors.append(tokenizer.encode(lk.split("/")[-1], add_special_tokens=False))
            elif args.rcv1_expand:
                print(f'map {lk} to {rcv1_label_expand[lk]}')
                label_name_tensors.append(tokenizer.encode(rcv1_label_expand[lk], add_special_tokens=False))
            else:
                label_name_tensors.append(tokenizer.encode(lk, add_special_tokens=False))
            max_l = max(len(label_name_tensors[-1]), max_l)
        label_name_tensors = torch.LongTensor([i + [tokenizer.pad_token_id] * (max_l - len(i)) for i in label_name_tensors])

        with torch.no_grad():
            init_label_emb = model.bert.embeddings.word_embeddings(label_name_tensors)
            label_mask = label_name_tensors != tokenizer.pad_token_id
            init_label_emb = (label_mask.unsqueeze(-1) * init_label_emb).sum(1)
        label_tokens = [i for i in range(len(label_map))]
        tokenizer.add_tokens([label_map[label] for label in labels_key])
        #import pdb;pdb.set_trace()
        #labels_embeds = torch.nn.Embedding(len(label_tokens), config.hidden_size).weight.data
        if args.label_cpt:
            # for compare with same seed
            rng_state = torch.get_rng_state()

            from collections import defaultdict
            hiera = defaultdict(set)
            _label_dict = {}
            with open(args.label_cpt) as f:
                _label_dict['Root'] = -1
                for line in f.readlines():
                    line = line.strip().split('\t')
                    for i in line[1:]:
                        if i not in _label_dict:
                            _label_dict[i] = len(_label_dict) - 1
                        hiera[line[0]].add(i)
                _label_dict.pop('Root')
            r_hiera = {}
            for i in hiera:
                for j in list(hiera[i]):
                    r_hiera[j] = i

            def _loop(a):
                if r_hiera[a] != 'Root':
                    return [a,] + _loop(r_hiera[a])
                else:
                    return [a]

            label_class = {}
            for i in _label_dict:
                label_class[i] = len(_loop(i))
            # cls l1 l2 l3 sep
            attention_mask = torch.zeros((len(label_tokens) + 2, len(label_tokens) + 2))
            num_hiers = defaultdict(set)
            reversed_hiers = {}
            for hi in hiera:
                for hj in list(hiera[hi]):
                    def _label_map_f(x):
                        if x == 'Root': return -1
                        return int(label_map[x].replace('[A_', '').replace(']', ''))
                    attention_mask[_label_map_f(hi) + 1][_label_map_f(hj) + 1] = 1
                    num_hiers[_label_map_f(hi) + 1].add(_label_map_f(hj) + 1)
                    reversed_hiers[_label_map_f(hj) + 1] = _label_map_f(hi) + 1
                    if args.label_cpt_use_bce:
                        attention_mask[_label_map_f(hj) + 1][_label_map_f(hi) + 1] = 1
            input_ids = torch.LongTensor(tokenizer.encode(' '.join(label_map.values()).lower()))
            assert len(input_ids) == len(labels_key) + 2
            position_ids = torch.LongTensor([0, ] + [label_class[i] for i in labels_key] + [max(label_class.values()) + 1,])

            init_label_emb = training_cpt(args, tokenizer, input_ids, attention_mask,
                                            position_ids, init_label_emb, num_hiers, reversed_hiers).detach().cpu()

            # for compare with same seed
            torch.set_rng_state(rng_state)
        elif args.random_label_init:
            rng_state = torch.get_rng_state()
            init_label_emb = torch.nn.Embedding(len(label_tokens), config.hidden_size).weight.data
            torch.set_rng_state(rng_state)

        model.bert.embeddings.word_embeddings.weight.data = torch.cat([model.bert.embeddings.word_embeddings.weight.data, init_label_emb], dim=0)
        model.bert.embeddings.word_embeddings.num_embeddings += len(label_tokens)
        model.cls.predictions.bias.data =  torch.cat([model.cls.predictions.bias.data, torch.zeros(len(label_tokens))],
                                                        dim=0)
        vs = config.vocab_size
        config.vocab_size = config.vocab_size + len(label_tokens)
        if args.softmax_label_only:
            model.label_start_index = label_tokens_start_index
    else:
        vs = config.vocab_size

    if args.soft_label:
        model.soft_label = True
        model.mask_token_id = tokenizer.mask_token_id
        model.sep_token_id = tokenizer.sep_token_id
        model.vs = vs

    return model, tokenizer, vs

def test(args, best_macro_f1_path, best_micro_f1_path):
    from test import main
    bout = None
    for i, save_path in enumerate([best_micro_f1_path, best_macro_f1_path]):
        if save_path is None: continue
        flags = ['--model_type'     , args.model_type                          ,
            '--tokenizer_name'         , args.model_name_or_path             ,
            '--input_file'             , args.test_file                  ,
            '--split'                  , 'test'                         ,
            '--do_lower_case'          ,
            '--model_path'             , str(save_path)              ,
            '--max_seq_length'         , str(args.max_source_seq_length + args.max_target_seq_length) if args.label_cpt_decodewithpos else str(args.max_source_seq_length)             ,
            '--max_tgt_length'         , str(args.max_target_seq_length)             ,
            '--batch_size'             , '128'                            ,
            '--beam_size'              , '1'                             ,
            '--length_penalty'         , '0'                             ,
            '--forbid_duplicate_ngrams',
            '--mode'                   , 's2s'                           ,
            '--forbid_ignore_word'     , '"."'                           ,
            '--cached_features_file'   , str(os.path.join(args.output_dir, "cached_features_for_test.pt")),
            '--add_vocab_file'         , args.add_vocab_file]

        if args.softmax_label_only:
            flags.append('--softmax_label_only')
        if args.soft_label:
            flags.append('--soft_label')
        if args.soft_label_hier_real:
            flags.append('--soft_label_hier_real_with_train_file')
            flags.append(args.train_file)
        if args.model_type == 'roberta':
            del flags[flags.index('--do_lower_case')]
        if args.label_cpt_decodewithpos:
            flags.append('--target_no_offset')

        out = main(flags)
        prefix = 'test' + 'micro' if i == 0 else 'macro'
        if args.wandb:
            wandb.log({f'{prefix}/macro_f1': out['macro_f1'], f'{prefix}/micro_f1': out['micro_f1']})
            if bout is None or bout['macro_f1'] < out['macro_f1']:
                bout = out

    if args.wandb and bout:
        prefix = 'test'
        wandb.log({f'{prefix}/macro_f1': bout['macro_f1'], f'{prefix}/micro_f1': bout['micro_f1']})



def main():
    args = get_args()
    prepare(args)
    if args.only_test:
        args.wandb = False
        test(args, args.only_test_path, None)
        exit(0)

    if args.wandb:
        wandb.init(
            project="HBGL",
            name=args.output_dir.split('/')[-1],
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab
    # Load pretrained model and tokenizer
    model, tokenizer, vs = get_model_and_tokenizer(args)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    if args.cached_train_features_file is None:
        if not args.lmdb_cache:
            args.cached_train_features_file = os.path.join(args.output_dir, "cached_features_for_training.pt")
        else:
            args.cached_train_features_file = os.path.join(args.output_dir, "cached_features_for_training_lmdb")

    if args.soft_label:
        args.cached_train_features_file += 'soft_label'
        # args.valid_file = args.valid_file.replace('generated', 'generated_tl')
        #
        if args.soft_label_hier_real:
            hier_labels = None
            for line in open(args.train_file):
                if hier_labels:
                    for i, l in enumerate(json.loads(line)['tgt']):
                        hier_labels[i] |=  set(l)
                else:
                    hier_labels = [set(i) for i in json.loads(line)['tgt']]
            hier_labels = [tokenizer.convert_tokens_to_ids(list([j.lower() for j in i])) for i in hier_labels]

            def to_multi_hot(label):
                _label = torch.zeros(model.config.vocab_size)
                for i in label:
                    _label[i] = 1
                return _label.bool()

            model.hier_labels = [to_multi_hot(i) for i in hier_labels]
            model.soft_label_hier_real = args.soft_label_hier_real


    num_lines = sum(1 for line in open(args.train_file))
    training_features = utils.load_and_cache_examples(
        example_file=args.train_file, tokenizer=tokenizer, local_rank=args.local_rank,
        cached_features_file=args.cached_train_features_file, shuffle=True,
        lmdb_cache=args.lmdb_cache, lmdb_dtype=args.lmdb_dtype,
        soft_label=args.soft_label,
    )

    if args.add_vocab_file:
        for i in training_features:
            for j in i.target_ids:
                if args.soft_label:
                    for ji in j:
                        assert ji >= vs
                else:
                    j >= vs

    best_macro_f1_path, best_micro_f1_path = train(args, training_features, model, tokenizer)
    if args.test_file:
        test(args, best_macro_f1_path, best_micro_f1_path)


if __name__ == "__main__":
    main()
