# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from modules.until_module import PreTrainedModel, LayerNorm, CrossEn, MILNCELoss, MaxMarginRankingLoss
from modules.module_bert import BertModel, BertConfig
from modules.module_visual import VisualModel, VisualConfig
from modules.module_cross import CrossModel, CrossConfig
from modules.module_decoder import DecoderModel, DecoderConfig

logger = logging.getLogger(__name__)


class UniVLPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs):
        # utilize bert config as base config
        super(UniVLPreTrainedModel, self).__init__(bert_config)
        self.bert_config = bert_config
        self.visual_config = visual_config
        self.cross_config = cross_config
        self.decoder_config = decoder_config

        self.bert = None
        self.visual = None
        self.cross = None
        self.decoder = None

    @classmethod
    def from_pretrained(cls, pretrained_bert_name, visual_model_name, cross_model_name, decoder_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        bert_config, state_dict = BertConfig.get_config(pretrained_bert_name, cache_dir, type_vocab_size, state_dict, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        decoder_config, _ = DecoderConfig.get_config(decoder_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs)

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

class NormalizeVideo(nn.Module):
    def __init__(self, task_config):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(task_config.video_dim)
        self.change_dim_vgg = nn.Linear(128,1024)

    def forward(self, video,mode="video"):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        if mode == "vgg":
            video = self.change_dim_vgg(video)
        video = self.visual_norm2d(video)
        return video

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class DynamicRNN(nn.Module):
    def __init__(self, dim):
        super(DynamicRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, bias=True, batch_first=True,
                            bidirectional=False)

    def forward(self, x, mask):
        out, _ = self.lstm(x)  # (bsz, seq_len, dim)
        mask = mask.type(torch.float32)
        mask = mask.unsqueeze(2)
        out = out * mask
        return out

class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)

class UniVL(UniVLPreTrainedModel):
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, task_config, use_vgg = False):
        super(UniVL, self).__init__(bert_config, visual_config, cross_config, decoder_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words <= decoder_config.max_target_embeddings
        assert self.task_config.max_words + self.task_config.max_qa_words + self.task_config.max_frames + self.task_config.max_frames_vgg <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        if check_attr('stage_two', self.task_config):
            self._stage_one = False
            self._stage_two = self.task_config.stage_two
        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.train_sim_after_cross = False
        if self._stage_one and check_attr('train_sim_after_cross', self.task_config):
            self.train_sim_after_cross = True
            show_log(task_config, "Test retrieval after cross encoder.")

        # Text Encoder ===>
        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "text_num_hidden_layers")
        self.bert = BertModel(bert_config)
        bert_word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        bert_position_embeddings_weight = self.bert.embeddings.position_embeddings.weight
        # <=== End of Text Encoder

        # Video Encoder ===>
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = VisualModel(visual_config)
        visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight
        # <=== End of Video Encoder

        if self._stage_one is False or self.train_sim_after_cross:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers",
                                        self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder

            if self.train_sim_after_cross is False:
                # Decoder ===>
                decoder_config = update_attr("decoder_config", decoder_config, "num_decoder_layers",
                                           self.task_config, "decoder_num_hidden_layers")
                self.decoder = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight)
                # <=== End of Decoder
                
            self.similarity_dense = nn.Linear(bert_config.hidden_size, 1)
            self.decoder_loss_fct = CrossEntropyLoss(ignore_index=-1)
            self.conv1d = nn.Conv1d(in_channels=bert_config.hidden_size, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

            self.start_encoder = DynamicRNN(dim=bert_config.hidden_size)
            self.end_encoder = DynamicRNN(dim=bert_config.hidden_size)
            self.start_block = nn.Sequential(
                Conv1D(in_dim=2 * bert_config.hidden_size, out_dim=bert_config.hidden_size, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(),
                Conv1D(in_dim=bert_config.hidden_size, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
            )
            self.end_block = nn.Sequential(
                Conv1D(in_dim=2 * bert_config.hidden_size, out_dim=bert_config.hidden_size, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(),
                Conv1D(in_dim=bert_config.hidden_size, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.normalize_video = NormalizeVideo(task_config)


        mILNCELoss = MILNCELoss(batch_size=task_config.batch_size // task_config.n_gpu, n_pair=task_config.n_pair, )
        maxMarginRankingLoss = MaxMarginRankingLoss(margin=task_config.margin,
                                                    negative_weighting=task_config.negative_weighting,
                                                    batch_size=task_config.batch_size // task_config.n_gpu,
                                                    n_pair=task_config.n_pair,
                                                    hard_negative_rate=task_config.hard_negative_rate, )

        if task_config.use_mil:
            self.loss_fct = CrossEn() if self._stage_two else mILNCELoss
            self._pretrain_sim_loss_fct = mILNCELoss
        else:
            self.loss_fct = CrossEn() if self._stage_two else maxMarginRankingLoss
            self._pretrain_sim_loss_fct = maxMarginRankingLoss

        self.apply(self.init_weights)
        self.use_vgg = use_vgg

    def forward(self, input_ids, token_type_ids, attention_mask, video_rgb, video_vgg, video_mask, vgg_mask,
                question_ids, question_mask, question_segment, 
                input_answer_ids = None, decoder_mask = None, output_answer_ids = None, time_label = None, init_cross_output = None, cal_sim = False):

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        vgg_mask = vgg_mask.view(-1, vgg_mask.shape[-1])

        video_rgb = self.normalize_video(video_rgb)
        video_vgg = self.normalize_video(video_vgg,mode="vgg")

        question_ids = question_ids.view(-1,question_ids.shape[-1])
        question_mask = question_mask.view(-1,question_mask.shape[-1])
        question_segment = question_segment.view(-1,question_segment.shape[-1])

        if input_answer_ids is not None:
            input_answer_ids = input_answer_ids.view(-1, input_answer_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        if self.training:
            loss = 0.
            if self.task_config.task_type == "dialog":
                if cal_sim:
                    sequence_output, question_output, rgb_output, vgg_output, frgb_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                                                question_ids, question_segment, question_mask, 
                                                                                                video_rgb, video_vgg, video_mask, vgg_mask, shaped=True, contrastive=True)
                    cross_output1, _, _ = self._get_cross_output(sequence_output, question_output, rgb_output, vgg_output, attention_mask, question_mask, video_mask, vgg_mask)
                    cross_output2, _, _ = self._get_cross_output(sequence_output, question_output, frgb_output, vgg_output, attention_mask, question_mask, 1-video_mask, vgg_mask)

                    sim_loss1 = F.mse_loss(cross_output1[:, rgb_output.shape[1],:], init_cross_output[:, rgb_output.shape[1],:])
                    sim_loss2 = 1 - F.mse_loss(cross_output2[:, rgb_output.shape[1],:], init_cross_output[:, rgb_output.shape[1],:])
                    return 0.2 * (sim_loss1 + 0.5 * sim_loss2)

                sequence_output, question_output, rgb_output, vgg_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                                            question_ids, question_segment, question_mask, 
                                                                                            video_rgb, video_vgg, video_mask, vgg_mask, shaped=True)
                decoder_scores, cross_output = self._get_decoder_score(sequence_output, question_output, rgb_output, vgg_output,
                                                                        input_ids, attention_mask, question_mask, video_mask, vgg_mask,
                                                                        input_answer_ids, decoder_mask, shaped=True)
                
                output_answer_ids = output_answer_ids.view(-1, output_answer_ids.shape[-1])
                decoder_loss = self.decoder_loss_fct(decoder_scores.view(-1, self.bert_config.vocab_size), output_answer_ids.view(-1))
                loss += decoder_loss
                return loss, cross_output

            elif self.task_config.task_type == "grounding":
                sequence_output, question_output, rgb_output, vgg_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                                            question_ids, question_segment, question_mask, 
                                                                                            video_rgb, video_vgg, video_mask, vgg_mask, shaped=True)
                mask_score, s, e = self.get_temporal_mask(sequence_output, question_output, rgb_output, vgg_output,
                                                    attention_mask, question_mask, video_mask, vgg_mask)
                loss1 = self.mask_loss(mask_score, time_label)
                loss2 = self.cross_entropy_loss(s, e, time_label)

                return loss1 + 0.2 * loss2

            else:
                raise NotImplementedError

        else:
            return None

    def get_temporal_mask(self, sequence_output, question_output, rgb_output, vgg_output, attention_mask, question_mask, video_mask, vgg_mask):

        cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, question_output, rgb_output, vgg_output, attention_mask, question_mask, video_mask, vgg_mask)

        mask = torch.cat((torch.zeros((cross_output.size(0),self.task_config.max_words+self.task_config.max_qa_words)), torch.ones((cross_output.size(0),self.task_config.max_frames)), torch.zeros((cross_output.size(0),self.task_config.max_frames_vgg))),dim=1).to(cross_output.device)
        
        x = cross_output.transpose(1,2)
        x = self.conv1d(x)
        x = x.transpose(1,2)
        x = x.squeeze(2)
        x = self.mask_logits(x, mask)
        mask_score = nn.Sigmoid()(x)

        cross_output = cross_output * mask_score.unsqueeze(2)
        start_features = self.start_encoder(cross_output, mask)
        end_features = self.end_encoder(start_features, mask)
        start_features = self.start_block(torch.cat([start_features, cross_output], dim=2))  # (batch_size, seq_len, 1)
        end_features = self.end_block(torch.cat([end_features, cross_output], dim=2))
        start_logits = start_features.squeeze(2) # (batch_size, seq_len)
        end_logits = end_features.squeeze(2)

        return mask_score, start_logits, end_logits

    def mask_loss(self, scores, labels):
        labels = torch.cat((torch.zeros((labels.size(0),self.task_config.max_words+self.task_config.max_qa_words)).to(labels.device), labels, torch.zeros((labels.size(0),self.task_config.max_frames_vgg)).to(labels.device)), dim=1)
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location * weights
        mask = torch.cat((torch.zeros((labels.size(0),self.task_config.max_words+self.task_config.max_qa_words)), torch.ones((labels.size(0),self.task_config.max_frames)), torch.zeros((labels.size(0),self.task_config.max_frames_vgg))),dim=1).to(labels.device)
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + 1e-30)
        return loss

    def mask_logits(self, inputs, mask, mask_value=-1e30):
        mask = mask.type(torch.float32)
        return inputs + (1.0 - mask) * mask_value

    def cross_entropy_loss(self,s,e,labels):
        start_labels=torch.zeros((s.size(0))).to(s.device)
        end_labels=torch.zeros((s.size(0))).to(s.device)

        for i in range(s.size(0)):
            for j in range(100):
                if labels[i][j]==1:
                    start_labels[i]=j
                    break
            for j in range(100):
                if labels[i][-j-1]==1:
                    end_labels[i]=99-j
                    break
        start_labels = start_labels.long()
        end_labels = end_labels.long()
        start_loss = nn.CrossEntropyLoss(reduction='mean')(s[:,80:180], start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='mean')(e[:,80:180], end_labels)

        return 0.5 * (start_loss + end_loss)

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, question_ids, question_segment, question_mask, video_rgb, video_vgg, video_mask, vgg_mask, shaped=False, contrastive=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            vgg_mask = vgg_mask.view(-1,vgg_mask.shape[-1])
            video_rgb = self.normalize_video(video_rgb)
            video_vgg = self.normalize_video(video_vgg,mode="vgg")

            question_ids = question_ids.view(-1,question_ids.shape[-1])
            question_mask = question_mask.view(-1,question_mask.shape[-1])
            question_segment = question_segment.view(-1,question_segment.shape[-1])

        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]

        question_layers, _ = self.bert(question_ids, question_segment, question_mask, output_all_encoded_layers=True)
        question_output = question_layers[-1]

        vgg_layers, _ = self.visual(video_vgg, vgg_mask, output_all_encoded_layers=True)
        vgg_output = vgg_layers[-1]

        rgb_layers, _ = self.visual(video_rgb, video_mask, output_all_encoded_layers=True)
        rgb_output = rgb_layers[-1]

        if contrastive:
            frgb_layers, _ = self.visual(video_rgb, 1-video_mask, output_all_encoded_layers=True)
            frgb_output = frgb_layers[-1]
            return sequence_output, question_output, rgb_output, vgg_output, frgb_output
        else:
            return sequence_output, question_output, rgb_output, vgg_output

    def _get_cross_output(self, sequence_output, question_output, rgb_output, vgg_output, attention_mask, question_mask, video_mask, vgg_mask):

        concat_text = torch.cat((sequence_output, question_output),dim=1)
        concat_features = torch.cat((concat_text, rgb_output), dim=1) 

        concat_mask = torch.cat((attention_mask, question_mask),dim=1)
        concat_mask = torch.cat((concat_mask, video_mask), dim=1)

        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        vgg_type_ = torch.ones_like(vgg_mask)
        question_type_ = torch.zeros_like(question_mask)

        concat_type = torch.cat((text_type_, question_type_), dim=1)
        concat_type = torch.cat((concat_type, video_type_), dim=1)
        
        if self.use_vgg:
            concat_features = torch.cat((concat_features, vgg_output), dim=1)
            concat_mask = torch.cat((concat_mask, vgg_mask), dim=1)
            concat_type = torch.cat((concat_type, vgg_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _get_decoder_score(self, sequence_output, question_output, rgb_output, vgg_output, input_ids, attention_mask, question_mask, video_mask, vgg_mask, input_answer_ids, decoder_mask, shaped=False):

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            vgg_mask = vgg_mask.view(-1, vgg_mask.shape[-1])

            input_answer_ids = input_answer_ids.view(-1, input_answer_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, question_output, rgb_output, vgg_output, attention_mask, question_mask, video_mask, vgg_mask)
        decoder_scores = self.decoder(input_answer_ids, encoder_outs=cross_output, answer_mask=decoder_mask, encoder_mask=concat_mask)

        return decoder_scores, cross_output

    def decoder_answer(self, sequence_output, question_output, rgb_output, vgg_output, input_ids, attention_mask, question_mask, video_mask, vgg_mask, input_answer_ids, decoder_mask,
                        shaped=False, get_logits=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            vgg_mask = vgg_mask.view(-1, vgg_mask.shape[-1])

            input_answer_ids = input_answer_ids.view(-1, input_answer_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        decoder_scores, _ = self._get_decoder_score(sequence_output, question_output, rgb_output, vgg_output,
                                                    input_ids, attention_mask, question_mask, video_mask,vgg_mask,
                                                    input_answer_ids, decoder_mask, shaped=True)

        if get_logits:
            return decoder_scores

        _, decoder_scores_result = torch.max(decoder_scores, -1)

        return decoder_scores_result