from transformers import AutoConfig
# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch
from transformers import ViTModel, ViTImageProcessor
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .VAE_model import VAE

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='bert',
        logits_mode=1,
    ):
        super().__init__()
        
        if config is None:
            config = AutoConfig.from_pretrained('bert-base-uncased')
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size
        with th.no_grad():
            self.lm_head = VAE(64, 768)
            self.lm_head.load_state_dict(torch.load('vae-checkpoint/VAE_model.pt'))
            for param in self.lm_head.parameters():
                param.requires_grad=False

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )
        
        '''if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))'''

        init_pretrained = 'bert'
        
        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            #self.word_embedding = temp_bert.embeddings.word_embeddings
            #with th.no_grad():
                #self.lm_head.weight = self.word_embedding.weight
            # self.lm_head.weight.requires_grad = False
            # self.word_embedding.weight.requires_grad = False
            
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(64, config.hidden_size) #nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        
        else:
            assert False, "invalid type of init_pretrained"
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        '''if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))'''
        
        #self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        '''with th.no_grad():
            self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')'''
        


    def get_embeds(self, input_ids):
        return self.lm_head.Encoder(input_ids)

    def get_logits(self, hidden_repr):
        #logits = self.lm_head.Decoder(hidden_repr)
        return self.lm_head.Decoder(hidden_repr)#torch.tensor(logits, dtype=torch.int32)
    
        '''if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError'''


    def forward(self, txt, timesteps, pixel_values, encoded_txt):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        #pixel_values = self.processor(images=[p for p in pixel_values], return_tensors="pt")
        hidden_state_img = pixel_values.cuda()#self.image_encoder(pixel_values).last_hidden_state
        emb_txt = encoded_txt.cuda()
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        
        concat = th.concat([emb_txt, hidden_state_img], dim=1)
        input_length = concat.size(1)
        position_ids = self.position_ids[:, : input_length ]
        emb_inputs = self.position_embeddings(position_ids) + concat + emb_t.unsqueeze(1).expand(-1, input_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        '''if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:'''
        h = input_trans_hidden_states
        h = h.type(txt.dtype)
        
        hidden_state_img_diff = hidden_state_img - input_trans_hidden_states[:,emb_txt.size(1):,:]

        return h, hidden_state_img_diff