from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os
from collections import OrderedDict
import time
import argparse
import json
import pickle
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import UniVL
from modules.optimization import BertAdam
from modules.beam import Beam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import DSTC_Dialog_DataLoader
from util import get_logger

os.environ['CUDA_VISIBLE_DEVICES']="0,1"
torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='DTGVD (UniVL) on Video Dialog Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", action='store_true', default=True, help="whether to run training")
    parser.add_argument("--do_eval", action='store_true', help="whether to run test")

    parser.add_argument('--train_path', type=str, default="data/train_set4DSTC10-AVSD+reason.json", help='train dialog dataset')
    parser.add_argument('--val_path', type=str, default="data/valid_set4DSTC10-AVSD+reason.json", help='valid dialog dataset')
    parser.add_argument('--test_path', type=str, default='data/test_set4DSTC7-AVSD.json', help="test dialog dataset, choosing from 'DSTC7' and 'DSTC8'")

    parser.add_argument('--duration_path_train', type=str, default="data/duration_Charades_v1_480.csv", help='duration of init videos for train and valid dataset')
    parser.add_argument('--duration_path_test', type=str, default="data/duration_Charades_vu17_test_480.csv", help='duration of init videos for train and valid dataset')
    parser.add_argument("--init_model", type=str, default="data/DTGVD.bin", help="Initial model")
    
    parser.add_argument('--features_path', type=str, default='data/', help='feature path for s3d and vgg')
    parser.add_argument('--grounding_results', type=str, default="data/results_dstc7.pickle", help='grounding results')
    parser.add_argument('--frame_length_file', type=str, default="data/frame_length.json", help='length of each s3d feature')

    parser.add_argument('--vgg', type=str, default=True, help='whether to use vgg features')
    parser.add_argument('--contrastive', type=str, default=True, help='whether to use contrastive selection')
    parser.add_argument('--choice', type=str, default="Best", help="how to select history, choosing from 'Best' and 'Recent'")

    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--max_words', type=int, default=60, help='max words for history')
    parser.add_argument('--max_qa_words', type=int, default=20, help='max words for question and answer')
    parser.add_argument('--max_frames', type=int, default=100, help='max frames for s3d')
    parser.add_argument('--max_frames_vgg', type=int, default=32, help='max frames for vgg')
    parser.add_argument('--n_history', type=int, default=3, help='number of history qas to be selected')

    parser.add_argument('--lr', type=float, default=3e-5, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=32, help='batch size eval')
    parser.add_argument('--save_list', type=list, default=[6,7,8,9], help='which epoch to save models')
    parser.add_argument("--output_dir", default="checkpoint/", type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=2, help='Information display frequence')
    parser.add_argument('--seed', type=int, default=512, help='random seed')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')
    parser.add_argument('--beam_size', type=int, default=6, help='beam search size')
    parser.add_argument('--penalty', type=int, default=0.8, help='penalty for beam search')

    parser.add_argument("--task_type", default="dialog", type=str, help="task type including 'dialog' and 'grounding'")
    parser.add_argument("--datatype", default="DSTC", type=str, help="")

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, help="Visual module")
    parser.add_argument("--cross_model", default="cross-base", type=str, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, help="Decoder module")

    parser.add_argument("--do_lower_case", action='store_true', default=True, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")

    parser.add_argument('--stage_two', action='store_true', default=True, help="Whether training with decoder.")
    parser.add_argument('--device', default = "cuda")
    args = parser.parse_args()

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device(args.device if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = UniVL.from_pretrained(args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args,
                                   use_vgg = args.vgg)

    model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def dataloader_DSTC_train(args, tokenizer):
    DSTC_dataset = DSTC_Dialog_DataLoader(
        args = args,
        tokenizer=tokenizer,
        mode = "Train"
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(DSTC_dataset)
    dataloader = DataLoader(
        DSTC_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(DSTC_dataset), train_sampler

def dataloader_DSTC_test(args, tokenizer):
    DSTC_testset = DSTC_Dialog_DataLoader(
        args = args,
        tokenizer=tokenizer,
        mode="Test"
    )

    test_sampler = SequentialSampler(DSTC_testset)
    dataloader_DSTC = DataLoader(
        DSTC_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    if args.local_rank == 0:
        logger.info('YoucookII validation pairs: {}'.format(len(DSTC_testset)))
    return dataloader_DSTC, len(DSTC_testset)

def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict

def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = UniVL.from_pretrained(args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler,
                global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):

        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        if args.task_type == "dialog":

            input_ids, input_mask, segment_ids, video_rgb, video_vgg, video_mask,vgg_mask, \
            pairs_input_answer_ids, pairs_decoder_mask, pairs_output_answer_ids, \
            question_ids, question_mask, question_segment, \
            video_mask_all = batch

            loss, cross_output = model(input_ids, segment_ids, input_mask, video_rgb, video_vgg, video_mask,vgg_mask,
                                        question_ids, question_mask, question_segment, 
                                        input_answer_ids=pairs_input_answer_ids, decoder_mask=pairs_decoder_mask,
                                        output_answer_ids=pairs_output_answer_ids)

        elif args.task_type == "grounding":

            input_ids, input_mask, segment_ids, video_rgb, video_vgg, video_mask,vgg_mask, \
            pairs_input_answer_ids, pairs_decoder_mask, pairs_output_answer_ids, \
            question_ids, question_mask, question_segment, \
            time_label, useful_length = batch
            time_label = time_label.float()

            loss = model(input_ids, segment_ids, input_mask, video_rgb, video_vgg, video_mask,vgg_mask,
                                        question_ids, question_mask, question_segment,
                                        input_answer_ids=pairs_input_answer_ids, decoder_mask=pairs_decoder_mask,
                                        output_answer_ids=pairs_output_answer_ids, time_label = time_label)

        else:
            raise NotImplementedError


        if n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.contrastive and args.task_type == 'dialog':
            loss.backward(retain_graph = True)
            sim_loss = model(input_ids, segment_ids, input_mask, video_rgb, video_vgg, video_mask_all,vgg_mask,
                                question_ids, question_mask, question_segment,
                                input_answer_ids=pairs_input_answer_ids, decoder_mask=pairs_decoder_mask,
                                output_answer_ids=pairs_output_answer_ids,init_cross_output=cross_output, cal_sim=True)
            if n_gpu > 1:
                sim_loss = sim_loss.mean()
            if args.gradient_accumulation_steps > 1:
                sim_loss = sim_loss / args.gradient_accumulation_steps

            sim_loss.backward()
            total_loss += float(sim_loss)
            show_loss = float(loss + sim_loss)
        else:
            loss.backward()
            show_loss = float(loss)

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            show_loss,
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

# ---------------------------------------->
def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''

    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor

def collate_active_info(input_tuples, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    assert isinstance(input_tuples, tuple)

    sequence_output_rpt, question_output_rpt, rgb_output_rpt, vgg_output_rpt, input_ids_rpt, input_mask_rpt, question_ids_rpt, question_mask_rpt, video_mask_rpt, vgg_mask_rpt = input_tuples

    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

    active_sequence_output_rpt = collect_active_part(sequence_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_question_output_rpt = collect_active_part(question_output_rpt, active_inst_idx, n_prev_active_inst, n_bm) 
    active_rgb_output_rpt = collect_active_part(rgb_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_vgg_output_rpt = collect_active_part(vgg_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    active_input_ids_rpt = collect_active_part(input_ids_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_input_mask_rpt = collect_active_part(input_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_question_ids_rpt = collect_active_part(question_ids_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_question_mask_rpt = collect_active_part(question_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    active_video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_vgg_mask_rpt = collect_active_part(vgg_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    return (active_sequence_output_rpt, active_question_output_rpt, active_rgb_output_rpt, active_vgg_output_rpt, active_input_ids_rpt, active_input_mask_rpt, active_question_ids_rpt, active_question_mask_rpt, active_video_mask_rpt, active_vgg_mask_rpt), \
           active_inst_idx_to_position_map

def beam_decode_step(decoder, inst_dec_beams, len_dec_seq,
                     inst_idx_to_position_map, n_bm, device, input_tuples, decoder_length=None, length = 1, penalty = 1, init=None):

    assert isinstance(input_tuples, tuple)

    ''' Decode and update beam status, and then return active beam idx'''
    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, input_tuples, length = 1, penalty = 1):
        sequence_output_rpt, question_output_rpt, rgb_output_rpt, vgg_output_rpt, input_ids_rpt, input_mask_rpt, question_ids_rpt, question_mask_rpt, video_mask_rpt, vgg_mask_rpt = input_tuples

        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)

        dec_output = decoder(sequence_output_rpt, question_output_rpt, rgb_output_rpt, vgg_output_rpt, input_ids_rpt, input_mask_rpt,
                             question_mask_rpt, video_mask_rpt, vgg_mask_rpt, next_decoder_ids, next_decoder_mask, shaped=True, get_logits=True)

        dec_output = dec_output[:, -1, :]
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)/(length**penalty)

        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        return word_prob 

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None, init=None):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            if decoder_length is None:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], inst_position=inst_position, init=init[inst_idx])
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], inst_position=inst_position, word_length=decoder_length[inst_idx], init = init[inst_idx])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, input_tuples, length = length, penalty = penalty)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                        decoder_length=decoder_length, init = init)

    return active_inst_idx_list

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]
        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps]
    return all_hyp, all_scores
# >----------------------------------------

def eval_epoch(args, model, test_dataloader, tokenizer, device):

    if hasattr(model, 'module'):
        model = model.module.to(device)

    if model._stage_one:
        return 0.

    all_result_lists = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(device, non_blocking=True) for t in batch)

            if args.task_type == "dialog":
                input_ids, input_mask, segment_ids, video_rgb, video_vgg, video_mask,vgg_mask, \
                pairs_input_answer_ids, pairs_decoder_mask, pairs_output_answer_ids, \
                question_ids, question_mask, question_segment, \
                video_mask_all = batch

                sequence_output, question_output, rgb_output, vgg_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask,
                                                                                                                        question_ids, question_segment, question_mask, 
                                                                                                                        video_rgb, video_vgg, video_mask, vgg_mask)
                # -- Repeat data for beam search
                n_bm = args.beam_size
                device = sequence_output.device
                n_inst, len_s, d_h = sequence_output.size()
                _, len_v, v_h = rgb_output.size()
                _, len_a, a_h = vgg_output.size()
                _, len_q, q_h = question_output.size()

                decoder = model.decoder_answer

                # Note: shaped first, then decoder need the parameter shaped=True
                input_ids = input_ids.view(-1, input_ids.shape[-1])
                input_mask = input_mask.view(-1, input_mask.shape[-1])
                video_mask = video_mask.view(-1, video_mask.shape[-1])
                vgg_mask = vgg_mask.view(-1, vgg_mask.shape[-1])

                question_ids = question_ids.view(-1,question_ids.shape[-1])
                question_mask = question_mask.view(-1,question_mask.shape[-1])

                sequence_output_rpt = sequence_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
                rgb_output_rpt = rgb_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
                vgg_output_rpt = vgg_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_a, a_h)
                question_output_rpt = question_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_q, q_h)

                input_ids_rpt = input_ids.repeat(1, n_bm).view(n_inst * n_bm, len_s)
                input_mask_rpt = input_mask.repeat(1, n_bm).view(n_inst * n_bm, len_s)
                video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)
                vgg_mask_rpt = vgg_mask.repeat(1, n_bm).view(n_inst * n_bm, len_a)
                question_ids_rpt = question_ids.repeat(1, n_bm).view(n_inst * n_bm, len_q)
                question_mask_rpt = question_mask.repeat(1, n_bm).view(n_inst * n_bm, len_q)

                # -- Prepare beams
                inst_dec_beams = [Beam(n_bm, device=device, tokenizer=tokenizer) for _ in range(n_inst)]
                # -- Bookkeeping for active or not
                active_inst_idx_list = list(range(n_inst))
                inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
                # -- Decode
                for len_dec_seq in range(1, args.max_words + 1):
                    active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                            len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                            (sequence_output_rpt, question_output_rpt, rgb_output_rpt, vgg_output_rpt, input_ids_rpt, input_mask_rpt, question_ids_rpt, question_mask_rpt, video_mask_rpt,vgg_mask_rpt),
                                                            length = len_dec_seq, penalty = args.penalty, init = question_ids)
                    if not active_inst_idx_list:
                        break  # all instances have finished their path to <EOS>

                    (sequence_output_rpt, question_output_rpt, rgb_output_rpt, vgg_output_rpt, input_ids_rpt, input_mask_rpt, question_ids_rpt, question_mask_rpt, video_mask_rpt, vgg_mask_rpt), \
                    inst_idx_to_position_map = collate_active_info((sequence_output_rpt, question_output_rpt, rgb_output_rpt, vgg_output_rpt, input_ids_rpt, input_mask_rpt, question_ids_rpt, question_mask_rpt, video_mask_rpt,vgg_mask_rpt),
                                                                inst_idx_to_position_map, active_inst_idx_list, n_bm, device)

                batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

                result_list = [batch_hyp[i][0] for i in range(n_inst)]

                for re_idx, re_list in enumerate(result_list):
                    decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                    if "[SEP]" in decode_text_list:
                        SEP_index = decode_text_list.index("[SEP]")
                        decode_text_list = decode_text_list[:SEP_index]
                    if "[PAD]" in decode_text_list:
                        PAD_index = decode_text_list.index("[PAD]")
                        decode_text_list = decode_text_list[:PAD_index]
                    decode_text = ' '.join(decode_text_list)
                    decode_text = decode_text.replace(" ##", "").strip("##").strip()
                    all_result_lists.append(decode_text)
            
            elif args.task_type == "grounding":
                input_ids, input_mask, segment_ids, video_rgb, video_vgg, video_mask, vgg_mask, \
                pairs_input_answer_ids, pairs_decoder_mask, pairs_output_answer_ids, \
                question_ids, question_mask, question_segment, \
                time_label, useful_length = batch

                sequence_output, question_output, rgb_output, vgg_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask,
                                                                                                                        question_ids, question_segment, question_mask, 
                                                                                                                        video_rgb, video_vgg, video_mask, vgg_mask)
                
                grounding_result = model.get_temporal_mask(sequence_output, question_output, rgb_output, vgg_output,
                                                            input_mask, question_mask, video_mask, vgg_mask)

                for i in range(len(grounding_result)):
                    all_result_lists.append(grounding_result[i][args.max_words+args.max_qa_words:args.max_words+args.max_qa_words+useful_length[i]])

            else:
                raise NotImplementedError

    # Save results
    if args.task_type == "dialog":
        out = dict()
        out['dialogs']=[]
        dialog_data = json.load(open(args.test_path, 'r'))
        idx=0
        for dialog in dialog_data['dialogs']:
            item = dict()
            vid = dialog["image_id"]
            item['image_id']=str(vid)
            for d in dialog['dialog']:
                if d['answer']=="__UNDISCLOSED__":
                    item['dialog']=[{'answer':all_result_lists[idx], 'question':d['question']}]
                    idx+=1
            out['dialogs'].append(item)
        out = json.dumps(out)
        file = open(args.output_dir+"result.json", 'w')
        file.write(out)
        file.close()
    elif args.task_type == "grounding":
        file = open(args.output_dir+"grounding_results.pickle", 'wb')
        pickle.dump(all_result_lists, file)
        file.close() 

DATALOADER_DICT = {}
DATALOADER_DICT["DSTC"] = {"train":dataloader_DSTC_train, "val":dataloader_DSTC_test}

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = init_model(args, device, n_gpu, args.local_rank)

    assert args.datatype in DATALOADER_DICT
    if args.do_eval:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
        if args.local_rank == 0:
            logger.info("***** Running test *****")
            logger.info("  Num examples = %d", test_length)
            logger.info("  Batch size = %d", args.batch_size_val)
            logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        global_step = 0
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)

            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                if epoch in args.save_list:
                    save_model(epoch, args, model, type_name="")

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, tokenizer, device)

if __name__ == "__main__":
    main()