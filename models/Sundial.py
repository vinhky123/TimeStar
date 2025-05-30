import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m", trust_remote_code=True
        )
        self.n_vars = configs.enc_in

    def forward(self, x):
        return self.model.generate(
            x, max_new_tokens=self.pred_len, num_samples=self.n_vars
        )
