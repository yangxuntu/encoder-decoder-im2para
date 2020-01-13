#= This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
import copy
import math
import numpy as np
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel

import pdb

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class HierModel(CaptionModel):
    def __init__(self, opt):
        super(HierModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.fs_index = opt.fs_index

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.relu_mod = getattr(opt, 'relu_mod', 'relu')
        self.rela_index = getattr(opt, 'rela_index', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        if self.relu_mod == 'relu':
            self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
            self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))
            self.att_embed = nn.Sequential(*(
                                        ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        elif self.relu_mod == 'leaky_relu':
            self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Dropout(self.drop_prob_lm))
            self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.Dropout(self.drop_prob_lm))
            self.att_embed = nn.Sequential(*(
                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                    (nn.Linear(self.att_feat_size, self.rnn_size),
                     nn.LeakyReLU(0.1, inplace=True),
                     nn.Dropout(self.drop_prob_lm)) +
                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        if self.rela_index:
            self.rela_mod = Rela_Mod(opt)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        if self.rela_index:
            att_feats = self.rela_mod(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        old_state = fc_feats.new_zeros(batch_size, self.rnn_size)
        old_att = fc_feats.new_zeros(batch_size, self.rnn_size)

        # outputs = []
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state, old_state, old_att = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, state, old_state, old_att)
            outputs[:, i] = output
            # outputs.append(output)

        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, old_state, old_att):
        # 'it' contains a word index
        xt = self.embed(it)
        cs_index = it==int(self.fs_index)

        output, state, old_state, old_att = self.core(xt, cs_index, fc_feats, att_feats, p_att_feats, state, old_state, old_att, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state, old_state, old_att

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k+1].expand(*((beam_size,)+att_masks.size()[1:])).contiguous() if att_masks is not None else None

            it = fc_feats.new_zeros([beam_size], dtype=torch.long) # input <bos>
            logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        old_state = fc_feats.new_zeros(batch_size, self.rnn_size)
        old_att = fc_feats.new_zeros(batch_size, self.rnn_size)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        # BEG MODIFIED
        trigrams = [] # will be a list of batch_size dictionaries
        # END MODIFIED

        # seq = []
        # seqLogprobs = []
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it
                # seq.append(it) #seq[t] the input of t+2 time step

                # seqLogprobs.append(sampleLogprobs.view(-1))
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)

            logprobs, state, old_state, old_att = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, state, old_state, old_att)

            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp


            # BEG MODIFIED ----------------------------------------------------

            # Mess with trigrams
            if block_trigrams and t >= 3 and sample_max:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)


                # Length penalty
                #penalty = 1.00
                #formula = penalty**t  # (5 + t)**penalty / (5 + 1)**penalty
                #helper = (torch.ones(logprobs.shape) - (1.0 - formula) * (torch.arange(logprobs.shape[1]).expand(logprobs.shape) <= 1).float()).cuda() 
                #logprobs = logprobs * helper 
                
            # END MODIFIED ----------------------------------------------------

        return seq, seqLogprobs
        # return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.topdown_res = getattr(opt, 'topdown_res', 0)
        self.hier_strategy = getattr(opt, 'hier_strategy', 1)

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # fc, h^4_s-1, c^4_s-1
        self.sen_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_s, \hat v
        self.watt_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 3, opt.rnn_size) # we, fc, h^2_t
        if self.hier_strategy == 2:
            self.word_lstm = nn.LSTMCell(opt.rnn_size, opt.rnn_size) # we, fc, h^2_t
        else:
            self.word_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t
        self.attention = Attention(opt)
        if self.hier_strategy == 1:
            self.wattention = Attention(opt)

    def forward(self, xt, cs_index, fc_feats, att_feats, p_att_feats, state, old_state, old_att = None, att_masks=None):
        """

        :param cs_index: batch_size*1, if the element is 1, then the last sentence is finished
        :param xt:
        :param fc_feats:
        :param att_feats:
        :param p_att_feats:
        :param state:
        :param att_masks:
        :return:
        """
        # state[0][0] means the hidden state h in first lstm
        # state[1][0] means the cell c in first lstm
        # state[0] means hidden state and state[1] means cell, state[0][i] means
        # the i-th layer's hidden state

        ####################sentence level######################
        prev_h = state[0][1]
        att_lstm_input = torch.cat([prev_h, xt, fc_feats], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        sen_lstm_input = torch.cat([att, h_att], 1)
        h_sen, c_sen = self.sen_lstm(sen_lstm_input, (state[0][1], state[1][1]))

        ####################word level######################
        batch_size = fc_feats.size(0)
        for batch_id in range(batch_size):
            if cs_index[batch_id]==1:
                old_state[batch_id] = h_sen[batch_id]
                if self.hier_strategy == 4:
                    old_att[batch_id] = att[batch_id]
        topic_vector = old_state
        prev_h_word = state[0][3]
        watt_lstm_input = torch.cat([xt, fc_feats, prev_h_word, topic_vector], 1)
        h_watt, c_watt = self.watt_lstm(watt_lstm_input, (state[0][2], state[1][2]))

        if self.hier_strategy == 1:
            watt = self.wattention(h_watt, att_feats, p_att_feats, att_masks)
            word_lstm_input = torch.cat([watt, h_watt], 1)
        elif self.hier_strategy == 2:
            word_lstm_input = torch.cat([h_watt], 1)
        elif self.hier_strategy == 3:
            word_lstm_input = torch.cat([att,h_watt], 1)
        elif self.hier_strategy == 4:
            word_lstm_input = torch.cat([old_att, h_watt], 1)
        h_word, c_word = self.word_lstm(word_lstm_input, (state[0][3], state[1][3]))

        output = F.dropout(h_word, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_sen, h_watt, h_word]), torch.stack([c_att, c_sen, c_watt, c_word]))

        return output, state, old_state, old_att


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res


class HTopDownModel(HierModel):
    def __init__(self, opt):
        super(HTopDownModel, self).__init__(opt)
        self.num_layers = 4
        self.core = TopDownCore(opt)


#########################################use self-attention to build rela module################
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def compute_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(2)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = compute_attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Rela_Mod(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(Rela_Mod, self).__init__()
        c = copy.deepcopy
        h = 8
        N = getattr(opt, 'rela_mod_layer', 2)
        dropout = 0.1
        d_model = opt.att_feat_size
        d_ff = opt.att_feat_size

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)

    def forward(self, att_feats, att_masks):
        rela_fc = self.model(att_feats, att_masks)
        return rela_fc
