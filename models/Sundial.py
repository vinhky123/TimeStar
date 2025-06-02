import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer
import numpy as np


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(
            self.cross_attention(
                x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )
        x_glb_attn = torch.reshape(
            x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])
        ).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x_glb

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == "MS" else configs.enc_in
        # Embedding

        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m", trust_remote_code=True
        )

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.w_refine = nn.Parameter(torch.randn(1, 6, configs.d_model))

        # Encoder-only architecture
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

        self.glb_token = nn.Parameter(torch.randn(1, self.n_vars, 1, configs.d_model))

        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(
            configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout
        )

    def get_hidden_states(self, x):
        B, L, N = x.shape
        # Reshape input: [B, L, N] -> [B * N, L]
        x_reshaped = x.permute(0, 2, 1).reshape(B * N, L)
        w_refine = self.w_refine.repeat(B, 1, 1)

        # Gọi pretrained model một lần duy nhất
        outputs = self.pretrained_model(
            input_ids=x_reshaped,
            return_dict=True,
            output_hidden_states=True,
            revin=True,
        )

        # Lấy hidden states và reshape lại: [B * N, patch_num, d_model] -> [B, patch_num, N, d_model]
        hidden_states = outputs.hidden_states[0]  # [B * N, patch_num, d_model]
        hidden_states = hidden_states.view(B, N, -1, hidden_states.shape[-1]).permute(
            0, 2, 1, 3
        )

        # Áp dụng w_refine
        hidden_states = hidden_states * w_refine.unsqueeze(2)  # Broadcasting w_refine

        return hidden_states  # [B, patch_num, N, d_model]

    def get_pretrained_result(self, x):
        B, L, N = x.shape
        x = torch.reshape(x, (B * N, L))

        outputs = self.pretrained_model(
            input_ids=x,
            return_dict=True,
            output_hidden_states=True,
            revin=True,
        )
        output = outputs.logits
        print(output.shape)

        return output

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, N = x_enc.shape
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape

        pretrained_hidden_states = self.get_hidden_states(x_enc).permute(
            0, 2, 1, 3
        )  # [B, patch_num, n_vars, d_model 786]

        en_embed = torch.cat(
            [pretrained_hidden_states, self.glb_token.repeat(B, 1, 1, 1)], dim=2
        )
        en_embed = torch.reshape(
            en_embed,
            (
                en_embed.shape[0] * en_embed.shape[1],
                en_embed.shape[2],
                en_embed.shape[3],
            ),
        )

        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, self.n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            if self.features == "M":
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len :, :]  # [B, L, D]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        elif self.task_name == "pretrained":
            return self.get_pretrained_result(x_enc)
        else:
            return None
