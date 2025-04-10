import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from einops import rearrange
from causal_conv1d import causal_conv1d_fn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class TFAF4SR(SequentialRecommender):
    def __init__(self, config, dataset):
        super(TFAF4SR, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        
        # Hyperparameters for Attention block
        self.n_heads = config["n_heads"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        
        # Hyperparameters for Mamba block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.layers = nn.ModuleList([Layer(d_model=self.hidden_size, d_state=self.d_state, d_conv=self.d_conv,
                     expand=self.expand, n_heads= self.n_heads, dropout=self.dropout_prob, 
                     att_dropout=self.attn_dropout_prob, num_layers=self.num_layers) for _ in range(self.num_layers)])
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        seq_emb = self.item_embedding(item_seq)
        seq_emb = self.LayerNorm(self.dropout(seq_emb))
        for i in range(self.num_layers):
            seq_emb = self.layers[i](seq_emb)
        seq_output = self.gather_indexes(seq_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention " 
                             "heads (%d)" % (hidden_size, n_heads))

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)   #row-wise
        self.softmax_col = nn.Softmax(dim=-2)   #column-wise
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.scale = np.sqrt(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size,)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Our Elu Norm Attention
        elu = nn.ELU()
        # relu = nn.ReLU()
        elu_query = elu(query_layer)
        elu_key = elu(key_layer)
        query_norm_inverse = 1/torch.norm(elu_query, dim=3,p=2) #(L2 norm)
        key_norm_inverse = 1/torch.norm(elu_key, dim=2,p=2)
        normalized_query_layer = torch.einsum('mnij,mni->mnij',elu_query,query_norm_inverse)
        normalized_key_layer = torch.einsum('mnij,mnj->mnij',elu_key,key_norm_inverse)
        context_layer = torch.matmul(normalized_query_layer, torch.matmul(normalized_key_layer,value_layer))/self.sqrt_attention_head_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class FilterLayer(nn.Module):
    def __init__(self, d_model, seq_length, dropout):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, seq_length//2 + 1, d_model, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape # [B L D]
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho') # [B L//2+1 D]
        weight = torch.view_as_complex(self.complex_weight) # [1 L//2+1 D]
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU() # nn.ReLU() only for Gowalla
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        #hidden_states = self.dropout(hidden_states) # only for Beauty
        
        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Layer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, n_heads, dropout, att_dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=16, chunk_size=32) 
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.filter = FilterLayer(d_model=d_model, seq_length=200, dropout=dropout)
        
        #self.alpha = nn.Parameter(torch.randn(d_model))  
        #self.beta = nn.Parameter(torch.randn(d_model)) 
        self.alpha = nn.Parameter(torch.rand(d_model)) #only for ML-1M
        self.beta = nn.Parameter(torch.rand(d_model)) #only for ML-1M
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)

    def forward(self, input_tensor):
        h1 = self.mamba(input_tensor)
        h1 = self.LayerNorm(self.dropout(h1) + input_tensor)
        h2 = self.filter(input_tensor)
        h = self.alpha * h1 + self.beta * h2
        hidden_states = self.ffn(h)
        return hidden_states



