"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import glob
import logging
import argparse
import math
from tqdm import tqdm
import numpy as np
import torch
import random
import pickle

from s2s_ft.modeling_decoding import BertForSeq2SeqDecoder, BertConfig
from transformers.tokenization_bert import whitespace_tokenize
import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.utils import load_and_cache_examples
from transformers import BertTokenizer

TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
}


class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main(flags=None):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(TOKENIZER_CLASSES.keys()))
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path to config.json for the model.")

    # tokenizer_name
    parser.add_argument("--tokenizer_name", default=None, type=str, required=True,
                        help="tokenizer name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--no_cuda', action='store_true',
                        help="Whether to use CUDA for decoding")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--add_vocab_file", type=str, default=None)
    parser.add_argument("--cached_features_file", type=str, default=None)
    parser.add_argument('--softmax_label_only', action='store_true')
    parser.add_argument('--soft_label', action='store_true')
    parser.add_argument('--soft_label_hier_real_with_train_file', default=None, type=str)

    if flags:
        print(flags)
        args = parser.parse_args(flags)
    else:
        args = parser.parse_args()


    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    else:
        random_seed = random.randint(0, 10000)
        logger.info("Set random seed as: {}".format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.tokenizer_name, do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.add_vocab_file:
        import pickle
        with open(args.add_vocab_file, 'rb') as f:
            label_map = pickle.load(f)
        labels_key = list(label_map.keys())
        # tokenizer.add_special_tokens({'additional_special_tokens': [label_map[label] for label in labels_key]})
        tokenizer.add_tokens([label_map[label] for label in labels_key])
        add_token_num = len(labels_key)

    if args.model_type == "roberta":
        vocab = tokenizer.encoder
    elif args.model_type == "xlm-roberta":
        vocab = {}
        for tk_id in range(len(tokenizer)):
            tk = tokenizer._convert_id_to_token(tk_id)
            vocab[tk] = tk_id
    else:
        vocab = tokenizer.vocab

    if hasattr(tokenizer, 'model_max_length'):
        tokenizer.model_max_length = args.max_seq_length
    elif hasattr(tokenizer, 'max_len'):
        tokenizer.max_len = args.max_seq_length

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.mask_token, tokenizer.sep_token, tokenizer.sep_token])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
    print(args.model_path)
    found_checkpoint_flag = False
    for model_recover_path in glob.glob(args.model_path):
        if not os.path.isdir(model_recover_path):
            continue

        logger.info("***** Recover model: %s *****", model_recover_path)

        config_file = args.config_path if args.config_path else os.path.join(model_recover_path, "config.json")
        logger.info("Read decoding config from: %s" % config_file)
        config = BertConfig.from_json_file(config_file)

        bi_uni_pipeline = []
        bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
            list(vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
            max_tgt_length=args.max_tgt_length, pos_shift=args.pos_shift,
            source_type_id=config.source_type_id, target_type_id=config.target_type_id,
            cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token, pad_token=tokenizer.pad_token))

        found_checkpoint_flag = True
        model = BertForSeq2SeqDecoder.from_pretrained(
            model_recover_path, config=config, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
            length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
            forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
            ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
            max_position_embeddings=args.max_seq_length, pos_shift=args.pos_shift,
        )

        if args.softmax_label_only and args.add_vocab_file:
            label_tokens_start_index = model.bert.embeddings.word_embeddings.num_embeddings - add_token_num
            model.label_start_index = label_tokens_start_index

        if args.soft_label:
            model.soft_label = args.soft_label
            label_tokens_start_index = model.bert.embeddings.word_embeddings.num_embeddings - add_token_num
            model.label_start_index = label_tokens_start_index

            if args.soft_label_hier_real_with_train_file:
                hier_labels = None
                for line in open(args.soft_label_hier_real_with_train_file):
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
                model.soft_label_hier_real = True

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length
        if args.pos_shift:
            max_src_length += 1

        num_lines = sum(1 for line in open(args.input_file))
        if num_lines < 10000:
            to_pred = load_and_cache_examples(
                args.input_file, tokenizer, local_rank=-1,
                cached_features_file=args.cached_features_file, shuffle=False, eval_mode=True)
        else:
            from s2s_ft.utils import load_and_cache_examples_fast
            to_pred = load_and_cache_examples_fast(
                args.input_file, tokenizer, local_rank=-1,
                cached_features_file=args.cached_features_file, shuffle=False, eval_mode=True)

        input_lines = []
        for line in to_pred:
            input_lines.append(tokenizer.convert_ids_to_tokens(line.source_ids)[:max_src_length])
        if args.subset > 0:
            logger.info("Decoding subset: %d", args.subset)
            input_lines = input_lines[:args.subset]

        input_lines = sorted(list(enumerate(input_lines)),
                             key=lambda x: -len(x[1]))
        output_lines = [""] * len(input_lines)
        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        with tqdm(total=total_batch) as pbar:
            batch_count = 0
            first_batch = True
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += args.batch_size
                batch_count += 1
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = model(input_ids, token_type_ids,
                                   position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                    if args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in (tokenizer.sep_token, tokenizer.pad_token):
                                break
                            output_tokens.append(t)
                        if args.model_type == "roberta" or args.model_type == "xlm-roberta":
                            output_sequence = tokenizer.convert_tokens_to_string(output_tokens)
                        else:
                            output_sequence = ' '.join(detokenize(output_tokens))
                        if '\n' in output_sequence:
                            output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                        output_lines[buf_id[i]] = output_sequence
                        if first_batch or batch_count % 50 == 0:
                            logger.info("{} = {}".format(buf_id[i], output_sequence))
                pbar.update(1)
                first_batch = False
        if args.output_file:
            fn_out = args.output_file
        else:
            fn_out = model_recover_path+'.'+args.split
        with open(fn_out, "w", encoding="utf-8") as fout:
            for l in output_lines:
                fout.write(l)
                fout.write("\n")

        import pickle
        from eval import evaluate
        def token_to_id(token):
            token = token.lower()
            try:
                token = int(token.replace('[a_', '').replace(']', ''))
                token = 0 if token >= len(label_map) else token
                return token
            except:
                return 0

        if args.model_type == 'roberta':
            def roberta_token_to_id(token):
                token = token.replace("<s>", '').replace('[A_', ' ').replace(']', ' ').split(' ')
                token = [int(i) for i in token if i != '']
                return token
            predict_labels = [roberta_token_to_id(i) for i in output_lines]
        else:
            predict_labels = [i.replace("\n", '').split(' ') for i in output_lines]
            predict_labels = [list(set([token_to_id(j) for j  in i])) for i in predict_labels]
        with open(args.input_file) as f:
            gd_labels = [json.loads(i)['tgt'] for i in f]
            gd_labels = [[token_to_id(j) for j  in i.split(' ')] for i in gd_labels]

        with open(args.add_vocab_file, 'rb') as f:
            label_map = pickle.load(f)
        id2label = {token_to_id(label_map[k]): k for k in label_map}
        out = evaluate(predict_labels, gd_labels, id2label, as_sample=True)
        del out['full']
        print(out)
        return out

    if not found_checkpoint_flag:
        logger.info("Not found the model checkpoint file!")


if __name__ == "__main__":
    print(main())
