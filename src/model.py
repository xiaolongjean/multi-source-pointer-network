# !/usr/bin/python3
# -*-encoding:utf-8-*-


# ---------------------------------------------------------------------------------
# File: Multi-Scale Feature Pointer Network.
#
# Desc: Multi-Scale Feature Pointer Network Utilizing multi-scale feature and bi-source 
#       input to formulate a summary extraction task as selecting keywords from the 
#       bi-source input via a local pointer mechanism.
#           
#       Some new features are as follows:
#       1. Multi-Scale Embedding: To avoid multi OOV tokens sharing the same embedding.
#       2. Transformer Endoder: Enhancing the capability.
#       3. Local Token Indexing: Addressing the #UNK# tokens in the predicted sequence.
#       4. Masked Beam Search: Avoiding the first predicted token being 'EOS'.
# -----------------------------------------------------------------------------------


import time
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import Parameter
import torch.nn as nn
from search import BeamSearch
from utils import color
from utils import weighted_sum, logsumexp
from data import TestDataUtils
from nltk.translate.bleu_score import corpus_bleu


class PositionalEmbedding(nn.Module):
    
    def __init__(self, d_model, max_seq_len):
        super(PositionalEmbedding, self).__init__()
        
        position_encoding = np.array([
          [pos / pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)])
        
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.tensor(position_encoding).float()

        pad_row = torch.zeros([1, d_model])  # .double()
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)  # .to(device)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)  # .to(device)
        self.max_seq_len = max_seq_len

    def forward(self, input_len):
        device = input_len.device
        input_pos = torch.LongTensor([list(range(1, length + 1)) + [0]
                                      * (self.max_seq_len - length.item()) for length in input_len]).to(device)
        return self.position_encoding(input_pos)


class AdditiveAttention(nn.Module):
    """
    vector_dim : ``int``
        The dimension of the vector, ``x``, described above.  This is ``x.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_dim : ``int``
        The dimension of the matrix, ``y``, described above.  This is ``y.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    normalize : ``bool``, optional (default: ``True``)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """
    def __init__(self,
                 vector_dim: int,
                 matrix_dim: int):
        super().__init__()
        self._w_matrix = Parameter(torch.Tensor(vector_dim, vector_dim))
        self._u_matrix = Parameter(torch.Tensor(matrix_dim, vector_dim))
        self._v_vector = Parameter(torch.Tensor(vector_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._w_matrix)
        nn.init.xavier_uniform_(self._u_matrix)
        nn.init.xavier_uniform_(self._v_vector)

    def forward(self, vector, matrix):
        intermediate = vector.matmul(self._w_matrix).unsqueeze(1) + matrix.matmul(self._u_matrix)
        intermediate = torch.tanh(intermediate)
        return intermediate.matmul(self._v_vector).squeeze(2)


class MS_Pointer(nn.Module):
    """
    MS-Pointer Network: (Multiple source pointer network), In this demo, utilizing two sources, 
    """
    def __init__(self, args, word2index, char2index, device):
        super(MS_Pointer, self).__init__()
        self.args = args
        self.device = device
        self.lr = args.lr
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.embedding_dim = args.emb_dim
        self.target_embedding_dim = self.embedding_dim
        self.dropout_rate = 1.0 - args.dropout_keep_prob

        self.word2index = word2index
        self.char2index = char2index
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.index2char = {v: k for k, v in self.char2index.items()}
        self.char_num = len(self.char2index)
        self.word_num = len(self.word2index)

        self.max_seq_len = args.max_seq_len
        self.max_token_len = args.max_token_len
        self.max_decoding_steps = args.max_decoding_steps

        self.flag_use_layernorm = args.flag_use_layernorm
        self.flag_use_dropout = args.flag_use_dropout
        self.flag_use_position_embedding = args.use_position_emb

        self.encoder_output_dim_1 = args.encoder_output_dim_1
        self.encoder_output_dim_2 = args.encoder_output_dim_2
        self.cated_encoder_out_dim = self.encoder_output_dim_1 + self.encoder_output_dim_2

        self.decoder_output_dim = args.decoder_output_dim
        self.decoder_input_dim = self.encoder_output_dim_1 + self.encoder_output_dim_2 + self.target_embedding_dim

        # Word Embedding, Char Embedding and Positional Embeddings.
        self.char_embeddings = nn.Embedding(self.char_num, self.embedding_dim, padding_idx=self.args.PAD_idx)
        self.word_embeddings = nn.Embedding(self.word_num, self.embedding_dim, padding_idx=self.args.PAD_idx)
        if self.flag_use_position_embedding:
            self.position_embeddings = PositionalEmbedding(self.embedding_dim, self.max_seq_len)

        # Char encoder layer and word encoder layer for source1 and source2 respectively.
        self.source1_words_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4)
        self.source1_chars_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4)
        self.source2_words_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4)
        self.source2_chars_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4)

        # Char transformer layer and word transformer layer for source1 and source2 respectively.
        self.source1_words_transformer_encoder = nn.TransformerEncoder(self.source1_words_encoder_layer, num_layers=2)
        self.source1_chars_transformer_encoder = nn.TransformerEncoder(self.source1_chars_encoder_layer, num_layers=1)
        self.source2_words_transformer_encoder = nn.TransformerEncoder(self.source2_words_encoder_layer, num_layers=2)
        self.source2_chars_transformer_encoder = nn.TransformerEncoder(self.source2_chars_encoder_layer, num_layers=1)

        self.source1_attention_layer = AdditiveAttention(self.hidden_dim, self.encoder_output_dim_1)
        self.source2_attention_layer = AdditiveAttention(self.hidden_dim, self.encoder_output_dim_2)

        self.source1_dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.source2_dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.encoder_out_projection_layer = nn.Linear(in_features=self.cated_encoder_out_dim,
                                                      out_features=self.decoder_output_dim)

        self.gate_projection_layer = torch.nn.Linear(in_features=self.decoder_output_dim + self.decoder_input_dim,
                                                     out_features=1, bias=False)
        self.decoder_cell = nn.modules.LSTMCell(input_size=self.decoder_input_dim,
                                                hidden_size=self.decoder_output_dim, bias=True)
        self.beam_search = BeamSearch(self.max_seq_len * 2-1, max_steps=self.max_decoding_steps,
                                      beam_size=self.args.beam_size)
        self.test_data_utils = TestDataUtils(self.args, self.word2index, self.char2index)
        
        if self.flag_use_layernorm:
            self.source1_encoder_layernorm = nn.LayerNorm(normalized_shape=[self.max_seq_len, self.embedding_dim])
            self.source2_encoder_layernorm = nn.LayerNorm(normalized_shape=[self.max_seq_len, self.embedding_dim])
            self.decoder_hidden_layernorm = nn.LayerNorm(normalized_shape=self.decoder_output_dim)
            self.decoder_cell_layernorm = nn.LayerNorm(normalized_shape=self.decoder_output_dim)

    def get_target_token_embeddings(self, target_token_ids):
        """
        args:
            target_token_ids: (tuple), (target_token_word_ids, target_token_char_ids)
        return:
            target_token_embeddings: (torch.tensor),  Batch x Length x Dim
        """
        def get_char_idx_from_words(target_word_ids):
            target_char_ids = []

            for single_instance_word in target_word_ids.tolist():
                word = self.index2word[single_instance_word]

                if word in [self.args.PAD_token, self.args.BOS_token, self.args.EOS_token, self.args.OOV_token]:
                    char_ids = self.max_token_len * [self.char2index.get(word, self.args.OOV_idx)]
                else:
                    char_ids = [self.char2index.get(char, self.args.OOV_idx) for char in word]
                    char_ids = char_ids[0:self.max_token_len] if self.max_token_len < len(char_ids) \
                        else char_ids + (self.max_token_len - len(char_ids)) * [self.args.PAD_idx]
                target_char_ids.append(char_ids)

            return torch.LongTensor(target_char_ids).to(self.device)

        if isinstance(target_token_ids, tuple):
            target_words_ids, target_chars_ids = target_token_ids
        else:
            target_words_ids = target_token_ids
            target_chars_ids = get_char_idx_from_words(target_words_ids)

        target_word_embeddings = self.word_embeddings(target_words_ids)
        target_char_embeddings = self.char_embeddings(target_chars_ids).sum(1)
        target_token_embeddings = target_word_embeddings + target_char_embeddings
        return target_token_embeddings

    def encode(self, batch_input):
        source1_input_words_ids = batch_input["source1_input_words_ids"]
        source1_input_chars_ids = batch_input["source1_input_chars_ids"]
        source2_input_words_ids = batch_input["source2_input_words_ids"]
        source2_input_chars_ids = batch_input["source2_input_chars_ids"]
        source1_input_seq_len = batch_input["source1_input_seq_len"]
        source2_input_seq_len = batch_input["source2_input_seq_len"]

        source1_words_embs = self.word_embeddings(source1_input_words_ids)
        source1_chars_embs = self.char_embeddings(source1_input_chars_ids).sum(2)
        source2_words_embs = self.word_embeddings(source2_input_words_ids)
        source2_chars_embs = self.char_embeddings(source2_input_chars_ids).sum(2)

        if self.flag_use_position_embedding:
            source1_words_embs = source1_words_embs + self.position_embeddings(source1_input_seq_len)
            source1_chars_embs = source1_chars_embs + self.position_embeddings(source1_input_seq_len)
            source2_words_embs = source2_words_embs + self.position_embeddings(source2_input_seq_len)
            source2_chars_embs = source2_chars_embs + self.position_embeddings(source2_input_seq_len)

        source1_words_transformer_output = self.source1_words_transformer_encoder(source1_words_embs)
        source1_chars_transformer_output = self.source1_chars_transformer_encoder(source1_chars_embs) 
        source2_words_transformer_output = self.source2_words_transformer_encoder(source2_words_embs)
        source2_chars_transformer_output = self.source2_chars_transformer_encoder(source2_chars_embs)

        source1_encoder_output = source1_words_transformer_output + source1_chars_transformer_output
        source2_encoder_output = source2_words_transformer_output + source2_chars_transformer_output
        
        if self.flag_use_layernorm:
            source1_encoder_output = self.source1_encoder_layernorm(source1_encoder_output)
            source2_encoder_output = self.source2_encoder_layernorm(source2_encoder_output)

        if self.flag_use_dropout:
            source1_encoder_output = self.source1_dropout_layer(source1_encoder_output)
            source2_encoder_output = self.source2_dropout_layer(source2_encoder_output)
        
        initial_decoder_hidden_state = torch.tanh(self.encoder_out_projection_layer(
                                       torch.cat([source1_encoder_output[:, 0, :],
                                                  source2_encoder_output[:, 0, :]], dim=-1)))
        return source1_encoder_output, source2_encoder_output, initial_decoder_hidden_state

    def get_initial_model_state(self, batch_input):

        model_state = {}
        model_state["merged_source_global_ids"] = batch_input["merged_source_global_ids"]
        model_state["merged_source_local_ids"] = batch_input["merged_source_local_ids"]
        model_state["source1_local_words_ids"] = batch_input["source1_local_words_ids"]
        model_state["source2_local_words_ids"] = batch_input["source2_local_words_ids"]

        batch_size = batch_input["source1_input_words_ids"].shape[0]

        source1_encoder_output, source2_encoder_output, initial_decoder_hidden = self.encode(batch_input)
        # initial_decoder_cell = torch.rand(batch_size, self.decoder_output_dim)
        initial_decoder_cell = initial_decoder_hidden.new_zeros(batch_size, self.decoder_output_dim)
        
        model_state["decoder_hidden_state"] = initial_decoder_hidden
        model_state["decoder_hidden_cell"] = initial_decoder_cell
        model_state["source1_encoder_output"] = source1_encoder_output
        model_state["source2_encoder_output"] = source2_encoder_output

        initial_source1_decoder_attention = self.source1_attention_layer(initial_decoder_hidden,
                                                                         source1_encoder_output[:, 0:, :])
        initial_source2_decoder_attention = self.source2_attention_layer(initial_decoder_hidden,
                                                                         source2_encoder_output[:, 0:, :])
        
        initial_source1_decoder_attention_score = torch.softmax(initial_source1_decoder_attention, -1)
        initial_source2_decoder_attention_score = torch.softmax(initial_source2_decoder_attention, -1)

        initial_source1_weighted_context = weighted_sum(source1_encoder_output, initial_source1_decoder_attention_score)
        initial_source2_weighted_context = weighted_sum(source2_encoder_output, initial_source2_decoder_attention_score)
        model_state["source1_weighted_context"] = initial_source1_weighted_context
        model_state["source2_weighted_context"] = initial_source2_weighted_context

        return model_state

    def decode_step(self, previous_token_ids, model_state): 
        # Fetch last timestep values.
        previous_source1_weighted_context = model_state["source1_weighted_context"]
        previous_source2_weighted_context = model_state["source2_weighted_context"]
        previous_decoder_hidden_state = model_state["decoder_hidden_state"]
        previous_decoder_hidden_cell = model_state["decoder_hidden_cell"]
        previous_token_embedding = self.get_target_token_embeddings(previous_token_ids)
        
        # update decoder hidden state of current timestep
        current_decoder_input = torch.cat((previous_token_embedding, previous_source1_weighted_context, 
                                           previous_source2_weighted_context), dim=-1)
        decoder_hidden_state, decoder_hidden_cell = self.decoder_cell(current_decoder_input,
                                                                      (previous_decoder_hidden_state,
                                                                       previous_decoder_hidden_cell))
        # print(decoder_hidden_state.shape, decoder_hidden_cell.shape)
        if self.flag_use_layernorm:
            decoder_hidden_state = self.decoder_hidden_layernorm(decoder_hidden_state)
            decoder_hidden_cell = self.decoder_cell_layernorm(decoder_hidden_cell)
        model_state["decoder_hidden_state"] = decoder_hidden_state
        model_state["decoder_hidden_cell"] = decoder_hidden_cell

        # Computing decoder's attention score on encoder output.
        source1_encoder_output = model_state["source1_encoder_output"]
        source2_encoder_output = model_state["source2_encoder_output"]
        source1_decoder_attention_output = self.source1_attention_layer(decoder_hidden_state, source1_encoder_output)
        source2_decoder_attention_output = self.source2_attention_layer(decoder_hidden_state, source2_encoder_output)
        
        # print("attention dim: ", source1_decoder_attention_output.shape)
        source1_decoder_attention_score = torch.softmax(source1_decoder_attention_output, -1)
        source2_decoder_attention_score = torch.softmax(source2_decoder_attention_output, -1)
        model_state["source1_decoder_attention_score"] = source1_decoder_attention_score
        model_state["source2_decoder_attention_score"] = source2_decoder_attention_score
        
        # context vector of source1 and source2, weighted sum of (source encoder output) * decoder attention score.
        # source1_weighted_context = weighted_sum(source1_encoder_output[:,1:, :], source1_decoder_attention_score)
        # source2_weighted_context = weighted_sum(source2_encoder_output[:,1:, :], source2_decoder_attention_score)
        source1_weighted_context = weighted_sum(source1_encoder_output, source1_decoder_attention_score)
        source2_weighted_context = weighted_sum(source2_encoder_output, source2_decoder_attention_score)
        model_state["source1_weighted_context"] = source1_weighted_context
        model_state["source2_weighted_context"] = source2_weighted_context
        
        # Computing current gate socre.
        gate_input = torch.cat((previous_token_embedding, source1_weighted_context, 
                                source2_weighted_context, decoder_hidden_state), dim=-1)
        gate_projected = self.gate_projection_layer(gate_input).squeeze(-1)
        gate_score = torch.sigmoid(gate_projected)
        model_state["gate_score"] = gate_score

        return model_state

    def get_batch_loss(self, batch_input):
        # source1_token_mask = batch_input["source1_input_seq_mask"]
        # source2_token_mask = batch_input["source2_input_seq_mask"]
        target_words_ids = batch_input["target_words_ids"]
        target_chars_ids = batch_input["target_chars_ids"]
        # target_mask = batch_input["target_seq_mask"]
        
        batch_size, target_seq_len = target_words_ids.size()
        num_decoding_steps = target_seq_len - 1
        model_state = self.get_initial_model_state(batch_input)

        step_log_likelihoods = []  # 存放每个时间步，目标词的log似然值
        for timestep in range(num_decoding_steps):
            previous_token_ids = (target_words_ids[:, timestep], target_chars_ids[:, timestep, :])
            
            model_state = self.decode_step(previous_token_ids, model_state)

            target_to_source1 = (batch_input["source1_input_words_ids"] ==
                                 target_words_ids[:, timestep+1].unsqueeze(-1))
            target_to_source2 = (batch_input["source2_input_words_ids"] ==
                                 target_words_ids[:, timestep+1].unsqueeze(-1))

            step_log_likelihood = self.get_negative_log_likelihood(model_state["source1_decoder_attention_score"],
                                                                   model_state["source2_decoder_attention_score"],
                                                                   target_to_source1, target_to_source2,
                                                                   model_state["gate_score"])

            step_log_likelihoods.append(step_log_likelihood.unsqueeze(-1))

        # 将各个时间步的对数似然合并成一个tensor
        # shape: (batch_size, num_decoding_steps = target_seq_len - 1)
        log_likelihoods = torch.cat(step_log_likelihoods, -1)

        # 去掉第一个，不会作为目标词的START
        # shape: (batch_size, num_decoding_steps = target_seq_len - 1)
        # target_mask = target_mask[:, 1:].float()

        # 将各个时间步上的对数似然tensor使用mask累加，得到整个时间序列的对数似然
        # log_likelihood = (log_likelihoods * target_mask)  # .sum(dim=-1)
        log_likelihood = log_likelihoods.sum(dim=-1)
        batch_loss = - log_likelihood.sum()
        mean_loss = batch_loss / batch_size

        return {"mean_loss": mean_loss, "batch_loss": batch_loss}

    def get_negative_log_likelihood(self, source1_decoder_attention_score, source2_decoder_attention_score,
                                    target_to_source1, target_to_source2, gate_score):

        # shape: (batch_size, seq_max_len_1)
        combined_log_probs_1 = ((source1_decoder_attention_score * target_to_source1.float()).sum(-1) + 1e-20).log()

        # shape: (batch_size, seq_max_len_2)
        combined_log_probs_2 = ((source2_decoder_attention_score * target_to_source2.float()).sum(-1) + 1e-20).log()

        # 计算 log(p1 * gate + p2 * (1-gate))
        log_gate_score_1 = (gate_score + 1e-20).log()  # shape: (batch_size,)
        log_gate_score_2 = (1 - gate_score + 1e-20).log()  # shape: (batch_size,)
        
        item_1 = (log_gate_score_1 + combined_log_probs_1).unsqueeze(-1)  
        item_2 = (log_gate_score_2 + combined_log_probs_2).unsqueeze(-1)  
        step_log_likelihood = logsumexp(torch.cat((item_1, item_2), -1))
        return step_log_likelihood

    def merge_final_log_probs(self,
                              source1_decoder_attention_score,
                              source2_decoder_attention_score,
                              source1_local_words_ids, 
                              source2_local_words_ids, 
                              gate_score):
        """
        根据三个概率，计算全词表上的对数似然。
        """
        # 获取group_size和两个序列的长度
        group_size, seq_max_len_1 = source1_decoder_attention_score.size()
        group_size, seq_max_len_2 = source2_decoder_attention_score.size()

        # 需要和source1相乘的gate概率，shape: (group_size, seq_max_len_1)
        gate_1 = gate_score.expand(seq_max_len_1, -1).t()
        # 需要和source2相乘的gate概率，shape: (group_size, seq_max_len_2)
        gate_2 = (1 - gate_score).expand(seq_max_len_2, -1).t()

        # 加权后的source1分值，shape: (group_size, seq_max_len_1)
        source1_decoder_attention_score = source1_decoder_attention_score * gate_1
        # 加权后的source2分值，shape: (group_size, seq_max_len_2)
        source2_decoder_attention_score = source2_decoder_attention_score * gate_2

        # shape: (group_size, seq_max_len_1)
        log_probs_1 = (source1_decoder_attention_score + 1e-45).log()
        # shape: (group_size, seq_max_len_2)
        log_probs_2 = (source2_decoder_attention_score + 1e-45).log()
        
        # 初始化全词表上的概率为全0, shape: (group_size, target_vocab_size)
        final_log_probs = (source1_decoder_attention_score.new_zeros((group_size, 2 * self.max_seq_len)) + 1e-45).log()

        for i in range(seq_max_len_1):  # 遍历source1的所有时间步
            # 当前时间步的预测概率，shape: (group_size, 1)
            log_probs_slice = log_probs_1[:, i].unsqueeze(-1)
            # 当前时间步的token ids，shape: (group_size, 1)
            source_to_target_slice = source1_local_words_ids[:, i].unsqueeze(-1)

            # 选出要更新位置，原有的词表概率，shape: (group_size, 1)
            # print(source_to_target_slice.shape,"\t",final_log_probs.shape)
            selected_log_probs = final_log_probs.gather(-1, source_to_target_slice)
            # 更新后的概率值（原有概率+更新概率，混合），shape: (group_size, 1)
            combined_scores = logsumexp(torch.cat((selected_log_probs, log_probs_slice), dim=-1)).unsqueeze(-1)
            # 将combined_scores设置回final_log_probs中
            final_log_probs = final_log_probs.scatter(-1, source_to_target_slice, combined_scores)
        
        # 对source2也同样做一遍
        for i in range(seq_max_len_2):
            log_probs_slice = log_probs_2[:, i].unsqueeze(-1)
            source_to_target_slice = source2_local_words_ids[:, i].unsqueeze(-1)
            selected_log_probs = final_log_probs.gather(-1, source_to_target_slice)
            combined_scores = logsumexp(torch.cat((selected_log_probs, log_probs_slice), dim=-1)).unsqueeze(-1)
            final_log_probs = final_log_probs.scatter(-1, source_to_target_slice, combined_scores)
        
        return final_log_probs

    def take_search_step(self, previous_token_ids, model_state):
        # 更新一步decoder状态
        # model_state = self.get_initial_model_state(batch_input)
        model_state = self.decode_step(previous_token_ids, model_state)
        
        # 计算两个source的对数似然的合并结果
        final_log_probs = self.merge_final_log_probs(model_state["source1_decoder_attention_score"],
                                                     model_state["source2_decoder_attention_score"],
                                                     model_state["source1_local_words_ids"], 
                                                     model_state["source2_local_words_ids"], 
                                                     model_state["gate_score"])
        return final_log_probs, model_state

    def forward_beam_search(self, batch_input, model_state):
        source1_input_words_ids = batch_input["source1_input_words_ids"]
        # merged_source_local_ids = batch_input["merged_source_local_ids"]

        batch_size = source1_input_words_ids.size()[0]
        start_token_ids = source1_input_words_ids.new_full((batch_size,), fill_value=self.args.BOS_idx)
        
        all_top_k_predictions, log_probabilities = self.beam_search.search(start_token_ids, batch_input, model_state,
                                                                           self.take_search_step)

        return {"predicted_log_probs": log_probabilities,
                "predicted_token_ids": all_top_k_predictions}

    def get_predicted_tokens(self, predicted_token_ids, merged_source_word_list):

        word_list_len = merged_source_word_list.shape[1]
        batch_size, beam_size, target_len = predicted_token_ids.shape

        expanded_word_list = merged_source_word_list.reshape(batch_size, 1, word_list_len)
        expanded_word_list = np.tile(expanded_word_list, (1, beam_size, 1))

        dim0_indexer = np.tile(np.array(range(batch_size)).reshape(batch_size, 1, 1), (1, beam_size, target_len))
        dim1_indexer = np.tile(np.array(range(beam_size)).reshape(1, beam_size, 1), (batch_size, 1, target_len))
        dim2_indexer = predicted_token_ids.cpu()

        predicted_tokens = expanded_word_list[dim0_indexer, dim1_indexer, dim2_indexer]

        return predicted_tokens

    def predict_single_instance(self, instance):
        """
        instance: List, [[source1 words], [source2 words]]
        """
        raise NotImplementedError

    def predict_single_batch(self, batch_input):
        self.eval()

        with torch.no_grad():
            model_state = self.get_initial_model_state(batch_input)
            pred_result = self.forward_beam_search(batch_input, model_state)
            predicted_token_ids = pred_result["predicted_token_ids"]
            predicted_log_probs = pred_result["predicted_log_probs"]
            
            # merged_source_word_list = batch_input["merged_source_word_list"]
            predicted_tokens = self.get_predicted_tokens(predicted_token_ids, batch_input["merged_source_word_list"])

        return predicted_tokens, predicted_log_probs

    def predict(self, raw_test_data):
        """
        args:
            raw_test_data
                raw_test_data should be a three-order list:
                [
                    [[instance-1 source-1 words], [instance-1 source-2 words]],
                    [[instance-2 source-1 words], [instance-2 source-2 words]],
                    ... ...
                    [[instance-n source-1 words], [instance-n source-2 words]],
                ]
        return:
            all_batch_predicted_tokens:
                a three-order list with shape (batch_size, beam_size, target_length)
            all_batch_predicted_probs :
                a three-order list with shape (batch_size, beam_size, target_length)
        """

        # Set self.training as 'False' when predict to stop updating running variables of normalization or dropout 
        self.eval()

        all_batch_test_data = self.test_data_utils.get_batch_formatted_test_data(raw_test_data, device=self.device)
        with torch.no_grad():
            all_batch_predicted_tokens = []
            all_batch_predicted_probs = []

            for batch_input in all_batch_test_data:
                predicted_tokens, predicted_log_probs = self.predict_single_batch(batch_input)
                all_batch_predicted_tokens += predicted_tokens.tolist()
                all_batch_predicted_probs += predicted_log_probs.softmax(-1).tolist()

        return all_batch_predicted_tokens, all_batch_predicted_probs

    def valid_single_batch(self, batch_input, need_pred_result):
        self.eval()
        with torch.no_grad():
            valid_loss = self.get_batch_loss(batch_input)

            if need_pred_result:
                predicted_tokens, predicted_log_probs = self.predict_single_batch(batch_input)
            else:
                predicted_tokens, predicted_log_probs = None, None

        return valid_loss, predicted_tokens, predicted_log_probs

    def validation(self, all_batch_data, need_pred_result):
        all_batch_predicted_tokens = []
        all_batch_predicted_probs = []
        batch_size = len(all_batch_data)
        all_batch_loss = 0.0
        all_batch_bleu = 0.0

        batch_generator = tqdm(all_batch_data, ncols=100)

        if need_pred_result:
            for idx, batch in enumerate(batch_generator):
                batch_start_time = time.time()
                valid_loss, predicted_tokens, predicted_log_probs = self.valid_single_batch(batch, need_pred_result)

                mean_loss = valid_loss["mean_loss"].detach().cpu().item()
                all_batch_predicted_tokens += predicted_tokens.tolist()
                all_batch_predicted_probs += predicted_log_probs.tolist()
                all_batch_loss += valid_loss["batch_loss"].detach().cpu().item()

                pred_corpus = [[word_list[0: 2]] for word_list in predicted_tokens[:, 0, :].tolist()]
                bleu_score = corpus_bleu(pred_corpus, batch["target_word_list"], weights=[0.5, 0.5])
                all_batch_bleu += bleu_score
                batch_elapsed_time = round(time.time() - batch_start_time, 2)

                info = f"{color('[Valid]', 1)} Batch:{color(idx, 2)}  BLEU:{color(round(bleu_score, 5), 1)} " \
                       f"Loss:{color(round(mean_loss, 5), 1)} Time:{color(batch_elapsed_time, 2)}"
                batch_generator.set_description(desc=info, refresh=True)
        else:
            for idx, batch in enumerate(batch_generator):
                batch_start_time = time.time()
                valid_loss, predicted_tokens, predicted_log_probs = self.valid_single_batch(batch, need_pred_result)
                mean_loss = valid_loss["mean_loss"].detach().cpu().item()
                all_batch_loss += valid_loss["batch_loss"].detach().cpu().item()

                bleu_score = "None"
                batch_elapsed_time = round(time.time() - batch_start_time, 2)

                info = f"{color('[Valid]', 1)} Batch:{color(idx, 2)}  BLEU:{color(bleu_score, 1)} " \
                       f"Loss:{color(round(mean_loss, 5), 1)} Time:{color(batch_elapsed_time, 2)}"
                batch_generator.set_description(desc=info, refresh=True)

        mean_blue = all_batch_bleu/batch_size
        return all_batch_loss, mean_blue, all_batch_predicted_tokens, all_batch_predicted_probs

    def __call__(self, raw_test_data):
        """
        Call the predict function.
        """
        return self.predict(raw_test_data)

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return
