import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.n_vars = configs.enc_in

        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m", trust_remote_code=True
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

    def get_hidden_states(self, x):
        B, L, N = x.shape
        last_hidden_states = []

        for i in range(N):
            x_i = x[:, :, i]
            attention_mask = torch.ones(B, L, device=x.device)
            outputs = self.pretrained_model(
                input_ids=x_i,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                revin=True,
            )
            last_hidden_states.append(outputs.hidden_states[-1])

        return torch.stack(last_hidden_states, dim=2)

    def forward(
        self, x, x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None
    ):
        B, L, N = x.shape
        x_enc = self.get_hidden_states(x)
        print(x_enc.shape)
        return x_enc
